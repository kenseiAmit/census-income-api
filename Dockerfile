FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt setup.py ./
COPY income_classifier/ ./income_classifier/
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-build-isolation -e .

COPY main.py ./
COPY model/ ./model/

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
