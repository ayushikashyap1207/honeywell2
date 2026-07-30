FROM python:3.11-slim
RUN pip install --no-cache-dir --timeout 120 --retries 10 mlflow==2.14.1
WORKDIR /mlflow
EXPOSE 5000
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--default-artifact-root", "/mlflow/artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
