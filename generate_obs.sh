for i in {1..4}
do
    echo "welcome $i random"
    python utils/generate_observations.py --random --config-file all_level_0_novelties.xml
    killall 9001.x86_64
    sleep 1m
done

for i in {1..4}
do
    echo "welcome $i"
    python utils/generate_observations.py --non-random --config-file all_level_0_novelties.xml
    killall 9001.x86_64
    sleep 1m
done


for i in {1..4}
do
    echo "welcome $i random"
    python utils/generate_observations.py --random --config-file 100_level_1_novelties.xml
    killall 9001.x86_64
    sleep 1m
done

for i in {1..4}
do
    echo "welcome $i"
    python utils/generate_observations.py --non-random --config-file 100_level_1_novelties.xml
    killall 9001.x86_64
    sleep 1m
done

for i in {1..4}
do
    echo "welcome $i random"
    python utils/generate_observations.py --random --config-file 100_level_2_novelties.xml
    killall 9001.x86_64
    sleep 1m
done

for i in {1..4}
do
    echo "welcome $i"
    python utils/generate_observations.py --non-random --config-file 100_level_2_novelties.xml
    killall 9001.x86_64
    sleep 1m
done

for i in {1..4}
do
    echo "welcome $i random"
    python utils/generate_observations.py --random --config-file 100_level_3_novelties.xml
    killall 9001.x86_64
    sleep 1m
done

for i in {1..4}
do
    echo "welcome $i"
    python utils/generate_observations.py --non-random --config-file 100_level_3_novelties.xml
    killall 9001.x86_64
    sleep 1m
done
