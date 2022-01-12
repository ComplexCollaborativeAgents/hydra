#!/bin/bash
## Run on Batch using AutoUser (root) - will not be able to get the dpkg lock files to install packages otherwise.

echo "Update apt"
sudo apt-get update
sudo apt-get upgrade -y

# Optional Line - not needed if user has a graphics card & display installed
echo "Install linux dependencies"
sudo apt-get -y install xvfb mesa-utils x11-xserver-utils xdotool gosu
sudo apt-get install zip unzip build-essential -y

# needed for pyODBC: for unixODBC development headers
sudo apt-get install unixodbc-dev -y
sudo apt-get install python3-dev -y
sudo apt-get install python3-pip -y
