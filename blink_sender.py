import socket
import time
import random

HOST = "127.0.0.1"
PORT = 49721

# UDP 目标地址：与主程序监听地址保持一致。


def build_blink_message() -> str:
    """Build a natural blink command payload."""
    strength = random.uniform(0.45, 0.60)
    return f"BLINK:{strength:.2f}"


def main() -> None:
    """Continuously send randomized blink UDP messages."""
    # 固定间隔范围内随机发送眨眼强度，模拟自然眨眼。
    print("Blink sender started.")
    print(f"Sending UDP to {HOST}:{PORT}")
    print("Press Ctrl+C to exit.\n")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        while True:
            time.sleep(random.uniform(3.0, 6.0))
            msg = build_blink_message()
            sock.sendto(msg.encode("utf-8"), (HOST, PORT))
            print("[SEND]", msg)
    except KeyboardInterrupt:
        print("\nBye")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
