FROM python:3.10-slim

WORKDIR /app

# Changing folder to default user
RUN chown -R 8888 /app
RUN apt update

# Install python packages
COPY requirements.txt requirements.txt
COPY check_cpu_config.py check_cpu_config/check_cpu_config.py

RUN pip install -r requirements.txt

VOLUME  /home/bobbycaliber44/docker_app:/app/data

ENTRYPOINT ["python", "./check_cpu_config/check_cpu_config.py"]