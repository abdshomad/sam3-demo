#!/bin/bash
# restart.sh - Safely stop and restart background services

echo "=== Restarting background services ==="
./stop.sh
./start.sh
exit 0
