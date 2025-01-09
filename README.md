# yolo-gpu-cpu-benchmark
#### First clone repository
#### Make sure you have Docker installed
#### In docker-compose.yml change volumes lines to your easy to access home linux/wsl folder in all three services:
![image](https://github.com/user-attachments/assets/01e06aa9-c449-428f-87d9-bf741bbe5a9b)

#### Also change volume lines in Dockerfile in three services according to docker-compose.yml:
![image](https://github.com/user-attachments/assets/778ec59a-10ce-4da0-ad0c-2ae776b37791)

#### In this folder csv data will be stored after calculation

#### Then run (VPN may be required to build images):
```
docker compose up --build check_cpu_config
```
```
docker compose up --build check_gpu_config
```
```
docker compose up --build yolo_predict
```
#### If this error pops up when building then VPN is required:
![image](https://github.com/user-attachments/assets/16cc143e-ba8d-4957-be2d-50a347f5925e)

#### * All three images are about 15 GB in sum
#### * It is advised to not use PC when running containers
