"""
Vera Bot — magicpin AI Challenge
FastAPI + In-Memory Store + Gemini 2.0 Flash
"""

import os
import time
import json
import re
import uuid
import hashlib
from datetime import datetime
from typing import Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import google.generativeai as genai

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set!")
genai.configure(api_key=GEMINI_API_KEY)

SYSTEM_INSTRUCTION = """You are Vera, magicpin's AI assistant for merchant growth in India.
You compose WhatsApp messages for merchants across 5 categories: dentists, salons, restaurants, gyms, pharmacies.

RULES (mandatory):
1. Be SPECIFIC: use real numbers, prices, dates, names from the context. Never generic.
2. Category voice:
   - dentists: clinical, peer tone, technical OK. Use "Dr." prefix. No "cure/guaranteed".
   - salons: warm, friendly, practical.
   - restaurants: operator-to-operator tone.
   - gyms: coaching, motivational.
   - pharmacies: trustworthy, precise, compliance-aware.
3. ONE clear CTA per message. Binary (YES/STOP or reply YES) for action triggers.
4. Hindi-English mix when merchant languages include "hi" — match their preference.
5. NO fake data. Only use facts explicitly in the context.
6. Keep messages concise — 2-4 sentences max for WhatsApp.
7. Lead with the hook, end with the CTA.
8. Use compulsion levers: specificity, loss aversion, social proof, curiosity, reciprocity.
9. Never re-introduce yourself after first message.
10. Respond ONLY with a JSON object, no markdown fences."""

model = genai.GenerativeModel(
    "gemini-2.0-flash",
    system_instruction=SYSTEM_INSTRUCTION
)

START_TIME = time.time()

# ─────────────────────────────────────────────
# IN-MEMORY STATE  (scope, context_id) → {version, payload}
# ─────────────────────────────────────────────
contexts: dict[tuple[str, str], dict] = {}
conversations: dict[str, list] = {}          # conv_id → list of turns
sent_messages: dict[str, set] = {}           # conv_id → set of sent bodies (anti-repeat)

# ─────────────────────────────────────────────
# BOOT: Load seed data from dataset/ folder
# ─────────────────────────────────────────────
DATASET_DIR = Path(__file__).parent / "dataset"

def _load_seed_data():
    """Pre-load all seed data into memory at startup."""
    count = 0

    # Categories
    cat_dir = DATASET_DIR / "categories"
    if cat_dir.exists():
        for f in cat_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                slug = data.get("slug", f.stem)
                contexts[("category", slug)] = {"version": 1, "payload": data}
                count += 1
            except Exception as e:
                print(f"[WARN] Could not load category {f.name}: {e}")

    # Merchants seed
    m_path = DATASET_DIR / "merchants_seed.json"
    if m_path.exists():
        data = json.loads(m_path.read_text(encoding="utf-8"))
        for m in data.get("merchants", []):
            mid = m.get("merchant_id")
            if mid:
                contexts[("merchant", mid)] = {"version": 1, "payload": m}
                count += 1

    # Customers seed
    c_path = DATASET_DIR / "customers_seed.json"
    if c_path.exists():
        data = json.loads(c_path.read_text(encoding="utf-8"))
        for c in data.get("customers", []):
            cid = c.get("customer_id")
            if cid:
                contexts[("customer", cid)] = {"version": 1, "payload": c}
                count += 1

    # Triggers seed
    t_path = DATASET_DIR / "triggers_seed.json"
    if t_path.exists():
        data = json.loads(t_path.read_text(encoding="utf-8"))
        for t in data.get("triggers", []):
            tid = t.get("id")
            if tid:
                contexts[("trigger", tid)] = {"version": 1, "payload": t}
                count += 1

    print(f"[BOOT] Loaded {count} seed contexts into memory")

_load_seed_data()

# ─────────────────────────────────────────────
# AUTO-REPLY DETECTION
# ─────────────────────────────────────────────
AUTO_REPLY_PATTERNS = [
    "thank you for contacting",
    "aapki jaankari ke liye",
    "i am currently unavailable",
    "i'll get back to you",
    "automated",
    "auto-reply",
    "out of office",
    "we have received your message",
    "will respond shortly",
    "ek automated assistant",
    "main ek automated",
]

def is_auto_reply(message: str) -> bool:
    msg_lower = message.lower()
    return any(p in msg_lower for p in AUTO_REPLY_PATTERNS)

def is_hostile(message: str) -> bool:
    hostile_words = ["stop", "spam", "useless", "don't message", "do not message",
                     "unsubscribe", "block", "annoying", "leave me alone", "not interested"]
    msg_lower = message.lower()
    return any(w in msg_lower for w in hostile_words)

def is_commitment(message: str) -> bool:
    commit_words = ["ok let's do it", "okay let's do it", "let's do it", "go ahead",
                    "sounds good", "i want to join", "mujhe judrna hai", "yes proceed",
                    "chaliye shuru", "kar do", "do it", "yes go", "confirm", "haan karo"]
    msg_lower = message.lower()
    return any(w in msg_lower for w in commit_words)

# ─────────────────────────────────────────────
# COMPOSE ENGINE
# ─────────────────────────────────────────────

# System prompt is now baked into the model via system_instruction in the constructor above.

def compose(category: dict, merchant: dict, trigger: dict, customer: dict | None = None,
            conv_history: list | None = None) -> dict:
    """Core compose function — calls Gemini and returns structured message."""

    identity = merchant.get("identity", {})
    owner = identity.get("owner_first_name", identity.get("name", "there"))
    lang = identity.get("languages", ["en"])
    lang_note = "Use Hindi-English code-mix (hi-en)" if "hi" in lang else "Use English"

    perf = merchant.get("performance", {})
    signals = merchant.get("signals", [])
    offers = [o["title"] for o in merchant.get("offers", []) if o.get("status") == "active"]
    active_offers_str = ", ".join(offers) if offers else "No active offers"
    trigger_kind = trigger.get("kind", "general")
    trigger_payload = json.dumps(trigger.get("payload", {}))
    urgency = trigger.get("urgency", 1)
    suppression_key = trigger.get("suppression_key", f"vera:{merchant.get('merchant_id')}:{trigger_kind}")

    cat_voice = category.get("voice", {})
    peer_stats = category.get("peer_stats", {})
    digest_items = category.get("digest", [])
    digest_str = json.dumps(digest_items[:3]) if digest_items else "None"

    customer_block = ""
    send_as = "vera"
    if customer:
        cid = customer.get("identity", {})
        crel = customer.get("relationship", {})
        cstate = customer.get("state", "unknown")
        customer_block = f"""
Customer context:
  Name: {cid.get('name')}
  Language: {cid.get('language_pref')}
  State: {cstate}
  Last visit: {crel.get('last_visit')}
  Visits total: {crel.get('visits_total')}
  Services received: {crel.get('services_received', [])}
  Preferences: {json.dumps(customer.get('preferences', {}))}
  Consent scope: {customer.get('consent', {}).get('scope', [])}
"""
        send_as = "merchant_on_behalf"

    history_block = ""
    if conv_history:
        recent = conv_history[-4:]
        history_block = "\nRecent conversation:\n" + "\n".join(
            f"  [{t['from']}]: {t['msg']}" for t in recent
        )

    prompt = f"""Compose a WhatsApp message for Vera.

MERCHANT:
  Name: {identity.get('name')}
  Owner: {owner}
  City: {identity.get('city')} / {identity.get('locality')}
  Category: {merchant.get('category_slug')}
  Language: {lang} — {lang_note}
  Subscription: {json.dumps(merchant.get('subscription', {}))}
  Performance (30d): views={perf.get('views')}, calls={perf.get('calls')}, CTR={perf.get('ctr')}, leads={perf.get('leads')}
  Delta 7d: {json.dumps(perf.get('delta_7d', {}))}
  Active offers: {active_offers_str}
  Signals: {signals}
  Customer aggregate: {json.dumps(merchant.get('customer_aggregate', {}))}
  Review themes: {json.dumps(merchant.get('review_themes', [])[:3])}

CATEGORY ({merchant.get('category_slug')}):
  Voice: {cat_voice}
  Peer stats: {json.dumps(peer_stats)}
  Digest items: {digest_str}
  Seasonal beats: {json.dumps(category.get('seasonal_beats', [])[:2])}
  Trend signals: {json.dumps(category.get('trend_signals', [])[:2])}

TRIGGER:
  Kind: {trigger_kind}
  Urgency: {urgency}/5
  Payload: {trigger_payload}
  Suppression key: {suppression_key}
{customer_block}{history_block}

Return ONLY this JSON (no markdown, no explanation):
{{
  "body": "<the WhatsApp message>",
  "cta": "<open_ended | binary_yes_stop | none>",
  "send_as": "{send_as}",
  "suppression_key": "{suppression_key}",
  "rationale": "<1 sentence: why this message, what trigger, what compulsion lever>"
}}"""

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=512,
            ),
        )
        raw = response.text.strip()
        # Strip markdown fences if any
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        result = json.loads(raw)
        return result
    except Exception as e:
        print(f"[ERROR] compose failed: {e}")
        # Fallback: minimal valid response
        return {
            "body": f"Hi {owner}, quick update from Vera on your {merchant.get('category_slug', 'business')} — shall we connect?",
            "cta": "open_ended",
            "send_as": send_as,
            "suppression_key": suppression_key,
            "rationale": f"Fallback due to compose error: {str(e)[:80]}"
        }


def compose_reply(merchant_id: str, conv_id: str, message: str, turn_number: int,
                  merchant: dict | None = None) -> dict:
    """Handle a reply from the merchant/customer — return next action."""

    history = conversations.get(conv_id, [])

    # ── Edge case routing ──────────────────────────────────────────────
    # Count auto-replies in this conversation
    auto_count = sum(1 for t in history if t.get("from") == "merchant" and t.get("is_auto_reply", False))

    if is_auto_reply(message):
        auto_count += 1
        if auto_count >= 2:
            return {
                "action": "end",
                "rationale": "Detected automated WhatsApp Business auto-reply (2+ identical pattern). Gracefully exiting to avoid spam."
            }
        else:
            # Try once more with a direct human appeal
            return {
                "action": "send",
                "body": f"Looks like I might have caught an auto-reply! If you're available, I had a quick useful update for you. Just reply YES to hear it. 🙂",
                "cta": "binary_yes_stop",
                "rationale": "Detected potential auto-reply on turn 1 — making one human-directed attempt before exiting."
            }

    if is_hostile(message):
        return {
            "action": "end",
            "rationale": "Merchant expressed disinterest/hostility. Ending conversation gracefully to respect their preference."
        }

    if is_commitment(message):
        # Immediate action mode — no more qualifying questions
        owner_name = ""
        if merchant:
            owner_name = merchant.get("identity", {}).get("owner_first_name", "")
        return {
            "action": "send",
            "body": f"Got it{', ' + owner_name if owner_name else ''}! Starting right away. I'll have that ready for you in a moment — sit tight! ✅",
            "cta": "open_ended",
            "rationale": "Merchant committed — switching from pitch/qualify mode to immediate action mode."
        }

    # ── LLM reply composition ──────────────────────────────────────────
    if merchant is None:
        return {
            "action": "send",
            "body": "Understood! Let me look into that and get back to you shortly.",
            "cta": "open_ended",
            "rationale": "No merchant context available — safe generic acknowledgement."
        }

    identity = merchant.get("identity", {})
    owner = identity.get("owner_first_name", "there")
    lang = identity.get("languages", ["en"])
    lang_note = "Use Hindi-English code-mix" if "hi" in lang else "Use English"

    history_str = "\n".join(f"  [{t['from']}]: {t['msg']}" for t in history[-6:])

    prompt = f"""You are Vera, magicpin's AI assistant. A merchant just replied to you.

Merchant: {identity.get('name')} ({identity.get('city')})
Owner: {owner}
Language: {lang_note}
Turn number: {turn_number}

Conversation so far:
{history_str}
  [merchant]: {message}

Respond naturally. Be brief (1-3 sentences). Match their energy.
If they asked a question, answer it directly.
If they gave partial info, ask for the next specific piece.
If they want to stop or unsubscribe, end gracefully.

Return ONLY this JSON:
{{
  "action": "send" | "wait" | "end",
  "body": "<your reply — only if action=send>",
  "wait_seconds": <int — only if action=wait>,
  "cta": "open_ended",
  "rationale": "<1 sentence why>"
}}"""

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.1, max_output_tokens=400),
        )
        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        result = json.loads(raw)
        return result
    except Exception as e:
        print(f"[ERROR] compose_reply failed: {e}")
        return {
            "action": "send",
            "body": "Got it! Let me take a look and get back to you.",
            "cta": "open_ended",
            "rationale": f"Fallback reply: {str(e)[:60]}"
        }


# ─────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────
app = FastAPI(title="Vera Bot — magicpin AI Challenge", version="1.0.0")


# ── GET /v1/healthz ───────────────────────────
@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        if scope in counts:
            counts[scope] += 1
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts,
    }


# ── GET /v1/metadata ─────────────────────────
@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Vera-Flash",
        "team_members": ["Challenger"],
        "model": "gemini-2.0-flash",
        "approach": "4-context Gemini composer with trigger routing, auto-reply detection, and intent-transition handling",
        "contact_email": "challenger@example.com",
        "version": "1.0.0",
        "submitted_at": datetime.utcnow().isoformat() + "Z",
    }


# ── POST /v1/context ─────────────────────────
class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str

@app.post("/v1/context")
async def push_context(body: CtxBody):
    valid_scopes = {"category", "merchant", "customer", "trigger"}
    if body.scope not in valid_scopes:
        return JSONResponse(status_code=400, content={
            "accepted": False, "reason": "invalid_scope",
            "details": f"scope must be one of {valid_scopes}"
        })

    key = (body.scope, body.context_id)
    existing = contexts.get(key)

    if existing and existing["version"] > body.version:
        return JSONResponse(status_code=409, content={
            "accepted": False, "reason": "stale_version",
            "current_version": existing["version"]
        })

    if existing and existing["version"] == body.version:
        # Idempotent — same version, no-op
        return {
            "accepted": True,
            "ack_id": f"ack_{body.context_id}_v{body.version}_noop",
            "stored_at": datetime.utcnow().isoformat() + "Z"
        }

    contexts[key] = {"version": body.version, "payload": body.payload}
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.utcnow().isoformat() + "Z"
    }


# ── POST /v1/tick ─────────────────────────────
class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []

@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []
    seen_merchants = set()  # one action per merchant per tick

    # Sort triggers by urgency descending
    trigger_items = []
    for tid in body.available_triggers:
        t_ctx = contexts.get(("trigger", tid))
        if t_ctx:
            urgency = t_ctx["payload"].get("urgency", 1)
            trigger_items.append((urgency, tid, t_ctx["payload"]))
    trigger_items.sort(key=lambda x: -x[0])

    for urgency, tid, trigger in trigger_items:
        if len(actions) >= 20:
            break

        merchant_id = trigger.get("merchant_id")
        if not merchant_id or merchant_id in seen_merchants:
            continue

        merchant_ctx = contexts.get(("merchant", merchant_id))
        if not merchant_ctx:
            continue
        merchant = merchant_ctx["payload"]

        category_slug = merchant.get("category_slug", "")
        category_ctx = contexts.get(("category", category_slug))
        if not category_ctx:
            continue
        category = category_ctx["payload"]

        # Customer context (optional)
        customer = None
        customer_id = trigger.get("customer_id")
        if customer_id:
            c_ctx = contexts.get(("customer", customer_id))
            if c_ctx:
                customer = c_ctx["payload"]

        suppression_key = trigger.get("suppression_key", f"vera:{merchant_id}:{tid}")

        try:
            composed = compose(category, merchant, trigger, customer)
        except Exception as e:
            print(f"[ERROR] tick compose error for {merchant_id}: {e}")
            continue

        body_text = composed.get("body", "")
        if not body_text:
            continue

        conv_id = f"conv_{merchant_id}_{tid}_{uuid.uuid4().hex[:6]}"

        # Anti-repetition: check if same body was sent before in any conversation
        all_sent = set()
        for s in sent_messages.values():
            all_sent.update(s)
        if body_text in all_sent:
            # Slightly tweak to avoid identical repeat
            body_text = body_text + " 🔔"

        sent_messages.setdefault(conv_id, set()).add(body_text)
        seen_merchants.add(merchant_id)

        actions.append({
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": composed.get("send_as", "vera"),
            "trigger_id": tid,
            "template_name": f"vera_{trigger.get('kind', 'generic')}_v1",
            "template_params": [
                merchant.get("identity", {}).get("name", ""),
                trigger.get("kind", ""),
                body_text[:50]
            ],
            "body": body_text,
            "cta": composed.get("cta", "open_ended"),
            "suppression_key": suppression_key,
            "rationale": composed.get("rationale", ""),
        })

    return {"actions": actions}


# ── POST /v1/reply ────────────────────────────
class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int

@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv_id = body.conversation_id
    message = body.message

    # Store turn in conversation history
    is_auto = is_auto_reply(message)
    conversations.setdefault(conv_id, []).append({
        "from": body.from_role,
        "msg": message,
        "ts": body.received_at,
        "is_auto_reply": is_auto,
    })

    # Look up merchant context
    merchant = None
    if body.merchant_id:
        m_ctx = contexts.get(("merchant", body.merchant_id))
        if m_ctx:
            merchant = m_ctx["payload"]

    result = compose_reply(
        merchant_id=body.merchant_id or "",
        conv_id=conv_id,
        message=message,
        turn_number=body.turn_number,
        merchant=merchant,
    )

    # If we're sending, store bot's reply too + anti-repeat check
    if result.get("action") == "send":
        reply_body = result.get("body", "")
        existing = sent_messages.get(conv_id, set())
        if reply_body in existing:
            reply_body += " (follow-up)"
            result["body"] = reply_body
        sent_messages.setdefault(conv_id, set()).add(reply_body)
        conversations[conv_id].append({
            "from": "vera",
            "msg": reply_body,
            "ts": datetime.utcnow().isoformat() + "Z",
        })

    return result


# ── POST /v1/teardown (optional) ─────────────
@app.post("/v1/teardown")
async def teardown():
    contexts.clear()
    conversations.clear()
    sent_messages.clear()
    return {"cleared": True}


# ── Root health ───────────────────────────────
@app.get("/")
async def root():
    return {"message": "Vera Bot is running. See /v1/healthz for status."}
