import socket
import time
import random

HOST = "127.0.0.1"
PORT = 49721

print("👁 Blink sender started.")
print(f"Sending UDP to {HOST}:{PORT}")
print("Press Ctrl+C to exit.\n")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

try:
    while True:
        time.sleep(random.uniform(3.0, 6.0))
        strength = random.uniform(0.45, 0.60)  # 温和不狠
        msg = f"BLINK:{strength:.2f}"
        sock.sendto(msg.encode("utf-8"), (HOST, PORT))
        print("[SEND]", msg)
except KeyboardInterrupt:
    print("\nBye")
finally:
    sock.close()
