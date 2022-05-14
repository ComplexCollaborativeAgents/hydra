for i in {1..4}
do
    echo "welcome $i random"
    python utils/generate_observations.py -c 0_5_novelty_level_1_type6_non-novelty_type222.xml
    kill -9 $(pgrep -f 9001.x86_64)
    sleep 1m
done