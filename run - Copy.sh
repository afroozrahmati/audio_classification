#!/bin/bash

declare -i one=1
declare -i client_counts=10

z=$(( client_counts - one ))
"c:\users\Afrooz\anaconda3\envs\Capstone\python.exe" m_server.py   &
sleep 60 # Sleep for 2s to give the server enough time to start

for i in `seq 0 $z`; do
    echo "Starting client $i"
    "c:\users\Afrooz\anaconda3\envs\Capstone\python.exe" m_client.py ${i}  &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait