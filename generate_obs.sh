for i in {1..4}
do
    echo "welcome $i random"
    python utils/generate_observations.py -c 50_level_1_type_9_novelties_r1.xml
    kill -9 $(pgrep -f 9001.x86_64)
    sleep 1m
done

for i in {1..4}
do
    echo "welcome $i"
    python utils/generate_observations.py -c 50_level_1_type_9_novelties_r2.xml
    kill -9 $(pgrep -f 9001.x86_64)
    sleep 1m
done


for i in {1..4}
do
    echo "welcome $i random"
    python utils/generate_observations.py -c 50_level_1_type_9_novelties_r3.xml
    kill -9 $(pgrep -f 9001.x86_64)
    sleep 1m
done

for i in {1..4}
do
    echo "welcome $i"
    python utils/generate_observations.py -c 50_level_1_type_9_novelties_r4.xml
    kill -9 $(pgrep -f 9001.x86_64)
    sleep 1m
done

for i in {1..4}
do
    echo "welcome $i random"
    python utils/generate_observations.py -c 50_level_1_type_10_novelties_r1.xml
    kill -9 $(pgrep -f 9001.x86_64)
    sleep 1m
done

for i in {1..4}
do
    echo "welcome $i"
    python utils/generate_observations.py -c 50_level_1_type_10_novelties_r2.xml
    kill -9 $(pgrep -f 9001.x86_64)
    sleep 1m
done

for i in {1..4}
do
    echo "welcome $i random"
    python utils/generate_observations.py -c 50_level_1_type_10_novelties_r3.xml
    kill -9 $(pgrep -f 9001.x86_64)
    sleep 1m
done

for i in {1..4}
do
    echo "welcome $i"
    python utils/generate_observations.py -c 50_level_1_type_10_novelties_r4.xml
    kill -9 $(pgrep -f 9001.x86_64)
    sleep 1m
done
