FROM python:3.10.10

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True
ENV PORT 8000

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY contracts/ contracts/
RUN ape compile

COPY src/ src/

ENTRYPOINT uvicorn --host "0.0.0.0" --port $PORT "src.main:app"
