FROM bryankp/pycuda:latest

WORKDIR /app

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN set -xe && apt-get -yqq update && apt-get -yqq install python3-pip && pip3 install --upgrade pip

# Changing folder to default user
RUN chown -R 8888 /app
RUN apt update

# Install python packages
COPY requirements.txt requirements.txt
COPY check_gpu_config.py check_gpu_config/check_gpu_config.py

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

VOLUME  /home/bobbycaliber44/docker_app:/app/data

ENTRYPOINT [ "python3", "./check_gpu_config/check_gpu_config.py"]


