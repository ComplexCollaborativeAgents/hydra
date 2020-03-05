#!/bin/bash

domain=$1
problem=$2

echo $domain

# while IFS= read -r line; do echo $line; done < $domain

../../bin/upmc $domain $problem --custom 0.1 7 7 --force

#./"${domain%%.*}"_planner -print

./"${domain%%.*}"_planner -m250 -print -format:pddlvv

#target="/home/UPMurphi/ex/science_birds"
#let count=0
#for f in "$target"/*
#do
#    echo $(basename $f)
#done
