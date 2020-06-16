#!/bin/bash

domain=$1
problem=$2
memory=$3
delta_t=$4
time_out=$5

echo $domain

# while IFS= read -r line; do echo $line; done < $domain

time ../../bin/upmc $domain $problem --custom $delta_t 7 7 --force

echo ""
echo ""
echo ""
echo "=======================COMPILATION COMPLETED==========================="
echo ""
echo ""
echo ""
echo "=========================STARTING PLANNING============================="
echo ""
echo ""
echo ""
#./"${domain%%.*}"_planner -print

timeout $time_out ./"${domain%%.*}"_planner -m$memory -pi10000 -print -format:pddlvv

