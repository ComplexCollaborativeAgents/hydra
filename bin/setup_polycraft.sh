#!/bin/bash
## Run on Batch using AutoUser (root) - will not be able to get the dpkg lock files to install packages otherwise.

apt-get update &&  apt-get upgrade -y

# Install adoptopenjdk8-hotspot
sudo wget -qO - https://adoptopenjdk.jfrog.io/adoptopenjdk/api/gpg/key/public |  apt-key add -
sudo add-apt-repository --yes https://adoptopenjdk.jfrog.io/adoptopenjdk/deb/
apt-get update
sudo apt-get install adoptopenjdk-8-hotspot -V -y
sed -i -e '/^assistive_technologies=/s/^/#/' /etc/java-*-openjdk/accessibility.properties

# Optional Line - not needed if user has a graphics card & display installed
apt-get -y install xvfb mesa-utils x11-xserver-utils xdotool gosu
apt-get install zip unzip build-essential -y

# needed for pyODBC: for unixODBC development headers
apt-get install unixodbc-dev -y
apt-get install python3-dev -y
apt-get install python3-pip -y
