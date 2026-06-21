import time
import signal
import sys

running = True

def handle_signal(signum, frame):
    global running
    running = False

# Register signal handlers
signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)

print("Background service started.", flush=True)
while running:
    print(f"Heartbeat: Service is active at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    try:
        # Sleep in short increments to respond to signals faster
        for _ in range(5):
            if not running:
                break
            time.sleep(1)
    except IOError:
        pass

print("Background service stopped cleanly.", flush=True)
sys.exit(0)
