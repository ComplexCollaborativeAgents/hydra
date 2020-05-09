#!/bin/bash

domain=$1
problem=$2
memory=$3
delta_t=$4

echo $domain

# while IFS= read -r line; do echo $line; done < $domain

../../bin/upmc $domain $problem --custom $delta_t 7 7 --force

#./"${domain%%.*}"_planner -print

timeout 10s ./"${domain%%.*}"_planner -m$memory -pi10000 -print -format:pddlvv

