FROM python:3.10.11

RUN groupadd -g 10009 -o privategpt && useradd -m -u 10009 -g 10009 -o -s /bin/bash privategpt
USER privategpt
WORKDIR /home/privategpt

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY constants.py .
COPY api.py .
COPY consumer.py .
COPY loader.py .

#ENTRYPOINT ["/usr/bin/python", "/privateGPT/privateGTP.py"]
