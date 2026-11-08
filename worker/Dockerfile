# Teacher-labelling worker. Build from the repository root:
#   docker build -f worker/Dockerfile -t cc-kd/worker:1.0 .
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY worker/requirements.txt worker/requirements.txt
RUN pip install --no-cache-dir -r worker/requirements.txt
COPY prompts/teacher_prompt.txt prompts/teacher_prompt.txt
COPY data/schema.json data/schema.json
COPY worker/teacher.py worker/worker.py worker/
USER 1000
CMD ["python", "worker/worker.py"]
