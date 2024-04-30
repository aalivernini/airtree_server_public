FROM ubuntu:22.04

ADD code/airtree_server.py /app/
ADD code/requirements.txt /app/
ADD code/.env /app/
RUN chmod +x /app/airtree_server.py
RUN apt-get update && apt-get install -y --no-install-recommends python3
RUN apt-get install -y --no-install-recommends python3-pip

WORKDIR /app
RUN pip3 install -r requirements.txt
CMD ["uvicorn", "airtree_server:app", "--host", "0.0.0.0", "--port", "5000"]


