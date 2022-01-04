#!/bin/bash

"c:\users\Afrooz\anaconda3\envs\Capstone\python.exe" server.py &
sleep 10 # Sleep for 2s to give the server enough time to start

for i in `seq 0 9`; do
    echo "Starting client $i"
    "c:\users\Afrooz\anaconda3\envs\Capstone\python.exe" client.py --partition=${i} &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait