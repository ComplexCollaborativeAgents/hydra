#!/bin/bash

domain=$1
problem=$2
plan=$3

echo $domain

Val-dev-Linux/bin/Validate -t 0.001 -v $domain $problem $plan
