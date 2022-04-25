#!/usr/bin/env bash
set -e

REGISTRY='registry.gitlab-external.parc.com:8443'
REPO='hydra/cartpole_experiment'
RELEASE='month30'

docker login $REGISTRY
docker build -t $REGISTRY/$REPO/$RELEASE .
docker push $REGISTRY/$REPO/$RELEASE
