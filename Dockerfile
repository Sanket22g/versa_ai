# Vera Bot — Dockerfile for HuggingFace Spaces
FROM python:3.11-slim

# HuggingFace Spaces requires port 7860
ENV PORT=7860
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Expose HuggingFace port
EXPOSE 7860

# Run the bot
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
