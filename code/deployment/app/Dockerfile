FROM python:3.14-slim

WORKDIR /app

COPY code/deployment/app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY code/deployment/app/app.py .
COPY code/deployment/app/img ./img

EXPOSE 8501
# --server.address=0.0.0.0: without it Streamlit listens on loopback only
# and the host port mapping would not work.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]