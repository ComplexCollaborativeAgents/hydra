#!/bin/bash

find ./pal/shared_novelty/POGO -name "*.zip" | while read filename; do unzip -o -d "`dirname "$filename"`" "$filename"; done;

