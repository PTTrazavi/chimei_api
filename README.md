# chimei_api
## How to use Docker on 201
1. in the directory of Dockerfile
```bash
docker build . -t chimei_api_flask  
docker run -itd --gpus all --shm-size 16g --rm --name chimei_api -v /home/jeremylai/docker_projects/chimei_api/app:/workspace/chimei -p 15002:15002 chimei_api_flask
```
2. go into the running container bash  
```bash
docker exec -it [container ID] bash
```
3. start the flask  
```bash
python3 bin/run.py
```
4. if you want to execute flask when container is run, edit Dockerfile to:  
```bash
# ENTRYPOINT bash
CMD ["python", "/workspace/chimei/bin/run.py"]
```
if you want to execute flask by entering the container, edit Dockerfile to:  
```bash
ENTRYPOINT bash
# CMD ["python", "/workspace/chimei/bin/run.py"]
```

## the API will work on this URL
https://api.openaifab.com:15002/chimei_api  
