#!/bin/bash
# To be used once the pal.zip has been unzipped to unpack the level files within

find ./pal/shared_novelty/POGO -name "*.zip" | while read filename; do unzip -o -d "`dirname "$filename"`" "$filename"; done;

