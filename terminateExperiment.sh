#!/bin/bash

docker-compose -f vuDevOps/microservices-demo/deploy/docker-compose/docker-compose.scaphandre.yml down

docker-compose -f vuDevOps/microservices-demo/deploy/docker-compose/docker-compose.yml down

docker-compose -f vuDevOps/microservices-demo/deploy/docker-compose/docker-compose.cadvisor.yml down
