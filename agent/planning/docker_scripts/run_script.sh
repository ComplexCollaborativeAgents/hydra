#!/bin/bash

domain=$1
problem=$2

echo $domain

# while IFS= read -r line; do echo $line; done < $domain

../../bin/upmc $domain $problem --custom 0.05 7 7 --force

#./"${domain%%.*}"_planner -print

./"${domain%%.*}"_planner -m10 -pi10000 -print -format:pddlvv

