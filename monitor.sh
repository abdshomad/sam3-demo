#!/bin/bash
# monitor.sh - Check service status and view logs of Docker Compose services

echo "=== Service Status ==="
docker compose ps

echo ""
echo "=== Last 10 Log Entries (Frontend) ==="
docker compose logs --tail=10 frontend

echo ""
echo "=== Last 10 Log Entries (Backend) ==="
docker compose logs --tail=10 backend

echo ""
echo "=== Last 10 Log Entries (Qwen3-VL) ==="
docker compose logs --tail=10 qwen3-vl

echo "========================="
exit 0
