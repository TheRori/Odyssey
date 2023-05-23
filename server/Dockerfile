# syntax=docker/dockerfile:1

FROM jjanzic/docker-python3-opencv

WORKDIR /server

ADD requirements.txt .

RUN apt-get update && apt-get install -y python3-opencv

RUN pip install opencv-python

RUN pip install -r requirements.txt

ADD . .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
