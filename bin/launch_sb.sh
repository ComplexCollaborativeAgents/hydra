# This is just for the headless launch
cd "${0%/*}"
ScienceBirds_Linux/science_birds_linux.x86_64 -batchmode -nographics &
java -jar game_playing_interface.jar


