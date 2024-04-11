FROM ubuntu:22.04

ADD code /app
RUN chmod +x /app/main_web.py
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip

WORKDIR /app
RUN pip3 install -r requirements.txt


CMD ["python3", "-u", "main_web.py"]

