
for i in {1..4}
do
    echo "welcome $i random"
    python utils/generate_observations.py --random
    killall 9001.x86_64
    sleep 1m
done

for i in {1..4}
do
    echo "welcome $i"
    python utils/generate_observations.py
    killall 9001.x86_64
    sleep 1m
done
