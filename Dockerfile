FROM python:3.10.10 as builder

WORKDIR /wheels

COPY requirements.txt requirements.txt
RUN pip install wheel && pip wheel -r requirements.txt --wheel-dir=/wheels

FROM python:3.10.10-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True
ENV PORT 8000

WORKDIR /app

COPY --from=builder /wheels /app/wheels
COPY requirements.txt requirements.txt
RUN pip install --no-index --find-links=/app/wheels -r requirements.txt

COPY contracts/ contracts/
RUN ape compile

COPY src/ src/

ENTRYPOINT uvicorn --host "0.0.0.0" --port $PORT "src.main:app"
