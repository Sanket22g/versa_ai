---
title: Vera Bot - magicpin AI Challenge
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: true
---

# Vera Bot — magicpin AI Challenge

A high-quality merchant engagement bot built with **FastAPI + Gemini 2.0 Flash**.

## How It Works

### Core Architecture

```
POST /v1/context  →  In-memory store (scope, context_id) → {version, payload}
POST /v1/tick     →  Compose messages using 4-context framework
POST /v1/reply    →  Handle replies with routing logic
GET  /v1/healthz  →  Liveness probe
GET  /v1/metadata →  Bot identity
```

### Composition Strategy

Every message follows the **4-Context Framework**:
1. **Category** — voice, peer stats, digest items, seasonal beats
2. **Merchant** — identity, performance, offers, signals
3. **Trigger** — why now (kind + urgency + payload)
4. **Customer** (optional) — for merchant-on-behalf sends

### Key Design Decisions

- **Gemini 2.0 Flash** at temperature=0.1 for deterministic, grounded output
- **In-memory state** for sub-millisecond context lookup (no external deps)
- **Auto-reply detection** — exits after 2 consecutive auto-reply patterns
- **Intent routing** — detects commitment signals → switches from pitch to action immediately
- **Hostile message handling** — graceful exit with empathy
- **Anti-repetition** — tracks sent bodies per conversation, modifies if repeat detected
- **Trigger priority** — higher urgency triggers compose first

### Category Voice

| Category | Voice | Key |
|---|---|---|
| Dentists | Clinical, peer-to-peer, technical OK | "Dr." prefix, no "cure/guaranteed" |
| Salons | Warm, friendly, practical | Visual, service+price |
| Restaurants | Operator-to-operator | Volume, timing, local |
| Gyms | Coaching, motivational | Goals, retention |
| Pharmacies | Trustworthy, precise | Compliance, supply alerts |

## API Endpoints

### GET /v1/healthz
```json
{ "status": "ok", "uptime_seconds": 1234, "contexts_loaded": {"category": 5, "merchant": 50, ...} }
```

### GET /v1/metadata
```json
{ "team_name": "Vera-Flash", "model": "gemini-2.0-flash", ... }
```

### POST /v1/context
Push category/merchant/customer/trigger contexts. Idempotent by (scope, context_id, version).

### POST /v1/tick
Returns proactive message actions for currently active triggers (up to 20/tick, sorted by urgency).

### POST /v1/reply
Handles merchant/customer replies with:
- Auto-reply detection → graceful exit after 2nd pattern
- Hostile message detection → immediate end
- Commitment detection → instant action mode
- LLM reply composition for all other cases

## Tradeoffs

- **In-memory vs Redis**: Faster reads, acceptable for 60-min test window. Would use Redis/Postgres for production.
- **Gemini Flash vs Pro**: Flash is faster (sub-5s) and free-tier compatible. Pro would give marginally better quality on edge cases.
- **Temperature 0.1**: Near-deterministic for same inputs. Slight variation intentional to avoid anti-repeat flags.
