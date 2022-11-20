#FROM ubuntu:latest
FROM python:3.8.1
COPY ./*.py /exp/
COPY ./requirements.txt /exp/requirements.txt
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
WORKDIR /exp
ARG arg1
ARG arg2
CMD ["python3","./endsem.py", "--clf_name=", arg1, "--random", arg2]
