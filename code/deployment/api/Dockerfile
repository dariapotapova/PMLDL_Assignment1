FROM python:3.14-slim

WORKDIR /app

# Dependency layer first: it stays cached as long as requirements.txt is
# unchanged, so 5-minute rebuilds only refresh the lightweight code/model layers.\

COPY code/deployment/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY code/deployment/api/main.py .

# The model is re-copied on every pipeline cycle — this image must be built
# AFTER Stage 2 has produced models/model.pkl.

COPY models/model.pkl model/model.pkl

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]