#!/bin/bash
while true
do
    # Run the app, but force stop after 24 hours
    timeout 24h python3.10 app2.py
    echo "Restarting app after 24h..."
done
