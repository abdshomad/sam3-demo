#!/bin/bash
# stop.sh - Safely stop the running Docker Compose services

echo "=== Stopping background services via Docker Compose ==="

docker compose down

echo "Services stopped cleanly."
exit 0
