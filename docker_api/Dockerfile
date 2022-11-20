#FROM ubuntu:latest
FROM python:3.8.1
COPY . /exp/
COPY ./requirements.txt /exp/requirements.txt
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
WORKDIR /exp
ARG FLASK_APP api/app.py
CMD ["flask","run"]
