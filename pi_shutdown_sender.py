#!/usr/bin/env python3
"""
Raspberry Pi Remote Shutdown Sender
-----------------------------------
A simple snippet to send a "shutdown" signal to the Jetson over the LAN.
"""

import socket

def send_shutdown_signal(jetson_ip: str, port: int = 9999):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(5.0) # 5 second timeout
            s.connect((jetson_ip, port))
            # Send the agreed upon secret
            secret_command = "JETSON_SHUTDOWN_NOW"
            s.sendall(secret_command.encode('utf-8'))
            print(f"✅ Successfully sent shutdown signal to Jetson at {jetson_ip}:{port}")
    except ConnectionRefusedError:
        print(f"❌ Connection refused. Is the listener script running on the Jetson ({jetson_ip})?")
    except Exception as e:
        print(f"❌ Failed to send shutdown signal: {e}")

if __name__ == "__main__":
    # Replace with the actual IP address of your Jetson Orin
    JETSON_IP = "192.168.1.100" 
    
    print(f"Attempting to shut down Jetson at {JETSON_IP}...")
    send_shutdown_signal(JETSON_IP)
