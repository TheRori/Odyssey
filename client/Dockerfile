# syntax=docker/dockerfile:1

FROM node:lts-alpine

RUN npm install -g http-server

WORKDIR /client

COPY package*.json ./

RUN npm install

COPY . .

CMD [ "npm", "run" , "serve", "--host 0.0.0.0"]
