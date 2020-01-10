username=$(whoami)
pid=$(ps -u "$username" | grep -i "Science Birds" | head -n 1 | awk '{print $2;}')
kill "$pid"
