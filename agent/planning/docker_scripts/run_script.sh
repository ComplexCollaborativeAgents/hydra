#!/bin/bash

domain=$1
problem=$2
memory=$3

echo $domain

# while IFS= read -r line; do echo $line; done < $domain

../../bin/upmc $domain $problem --custom 0.05 7 7 --force

#./"${domain%%.*}"_planner -print

./"${domain%%.*}"_planner -m$memory -pi10000 -print -format:pddlvv

