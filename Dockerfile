# 
# Dockerfile - Auto-Scaling Infrastructure Agent
# Meta x Scaler OpenEnv AI Hackathon
# 
#
# Build:
#   docker build -t autoscaling-agent .
#
# Run server (HF Space / validator):
#   docker run -p 7860:7860 autoscaling-agent
#
# Run baseline (no API key needed):
#   docker run autoscaling-agent python baseline.py
#
# Run rule-based agent:
#   docker run autoscaling-agent python inference.py --agent rule
#
# Run LLM agent:
#   docker run -e HF_TOKEN=your_key \
#              -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
#              autoscaling-agent python inference.py --agent llm
# 

FROM python:3.10-slim

LABEL maintainer="Auto-Scaling Team"
LABEL description="OpenEnv Auto-Scaling Infrastructure Agent"
LABEL version="1.0.0"

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

WORKDIR /app

# Install dependencies first (Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY tasks.py .
COPY environment.py .
COPY graders.py .
COPY inference.py .
COPY baseline.py .
RUN mkdir -p server
COPY server/__init__.py server/__init__.py
COPY server/app.py server/app.py
COPY openenv.yaml .
COPY README.md .
COPY pyproject.toml .
COPY uv.lock .

# Smoke test at build time - validates all three core files
RUN python tasks.py && \
    python environment.py && \
    python graders.py

# Expose HF Space port
EXPOSE 7860

# Default: run FastAPI server so /reset responds with 200
# (required by automated validator)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]