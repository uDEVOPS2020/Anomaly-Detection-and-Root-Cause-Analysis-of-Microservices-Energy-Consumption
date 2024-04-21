#!/bin/bash
docker-compose -f vuDevOps/microservices-demo/deploy/docker-compose/docker-compose.scaphandre.yml up -d

docker-compose -f vuDevOps/microservices-demo/deploy/docker-compose/docker-compose.cadvisor.yml up -d

docker-compose -f vuDevOps/microservices-demo/deploy/docker-compose/docker-compose.yml up -d

