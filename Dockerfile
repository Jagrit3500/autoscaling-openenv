FROM python:3.10-slim

LABEL maintainer="Auto-Scaling Team"
LABEL description="OpenEnv Auto-Scaling Infrastructure Agent"
LABEL version="1.0.0"

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY tasks.py .
COPY environment.py .
COPY graders.py .
COPY inference.py .
COPY baseline.py .
COPY openenv.yaml .

RUN python tasks.py && \
    python environment.py && \
    python graders.py

CMD ["python", "baseline.py"]