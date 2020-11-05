#!/usr/bin/env bash

CONFIG='../../data/science_birds/config/test_reinit.xml'
FLAGS='--headless --dev'

cd linux && java -jar game_playing_interface.jar --config-path $CONFIG $FLAGS