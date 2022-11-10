#!/usr/bin/env bash
set -e

REGISTRY='registry.gitlab-external.parc.com:8443'
REPO='hydra/experiment'
RELEASE='month36'

docker login $REGISTRY
docker build -t $REGISTRY/$REPO/$RELEASE .
docker push $REGISTRY/$REPO/$RELEASE
