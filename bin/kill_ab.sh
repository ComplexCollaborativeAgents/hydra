username=$(whoami)
for i in "science_birds" "Xvfb" "java"; do
    echo $i
    pid=$(ps -u "$username" | grep -i $i | head -n 1 | awk '{print $1;}')
    echo $pid
    if [ -n "$pid" ]
       then
	   kill $pid
    fi
done
