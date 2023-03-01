FROM python:3.10.10

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY contracts/ contracts/
RUN ape compile

COPY src/ src/

ENTRYPOINT [ "python", "src/main.py" ]
