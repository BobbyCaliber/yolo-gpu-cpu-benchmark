FROM nvidia/cuda:12.6.3-runtime-ubuntu20.04

WORKDIR /app

RUN apt update
RUN apt install software-properties-common -y
RUN apt install libgl1 -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update
RUN apt install python3.10 python3.10-venv python3-pip python3.10-dev -y

# Changing folder to default user
RUN chown -R 8888 /app
RUN apt update
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y libglib2.0-0

# Install python packages
COPY requirements.txt requirements.txt
COPY yolo_predict.py yolo_predict/yolo_predict.py
COPY images images

RUN pip3 install -r requirements.txt

VOLUME  /home/bobbycaliber44/docker_app:/app/data

ENTRYPOINT [ "python3", "./yolo_predict/yolo_predict.py"]


