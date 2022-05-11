#!/usr/bin/env bash


REGISTRY='registry.gitlab-external.parc.com:8443'
REPO='hydra/polycraft_experiment'
RELEASE='month30'

docker login $REGISTRY
docker build -t $REGISTRY/$REPO/$RELEASE .
docker save -o hydra.tar $REGISTRY/$REPO/$RELEASE
