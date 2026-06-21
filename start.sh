#!/bin/bash
# start.sh - Launch the Docker Compose services in the background

echo "=== Starting background services via Docker Compose ==="

# Read PORT from .env
PORT=3058
if [ -f ".env" ]; then
    PORT_VAL=$(grep "^PORT=" .env | cut -d'=' -f2 | xargs)
    if [ ! -z "$PORT_VAL" ]; then
        PORT=$PORT_VAL
    fi
fi

# Start services
docker compose up -d

echo "Services started and running in background."
echo "Application is served on port: $PORT"
exit 0
