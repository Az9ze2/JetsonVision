#!/usr/bin/env python3
"""
Remote Shutdown Listener
------------------------
Listens on a specific TCP port for a secret shutdown command from a remote device (e.g., Raspberry Pi).
Once the command is received, it executes a system shutdown using the provided sudo password.
"""

import socket
import subprocess
import logging

# Configuration
HOST = '0.0.0.0'       # Listen on all network interfaces
PORT = 9999            # Port to listen on
SECRET = "JETSON_SHUTDOWN_NOW"
SUDO_PASSWORD = "123"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Allow immediate port reuse after restart
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        logging.info(f"Listening for shutdown commands on port {PORT}...")
        
        while True:
            try:
                conn, addr = s.accept()
                with conn:
                    data = conn.recv(1024).decode('utf-8').strip()
                    if data == SECRET:
                        logging.warning(f"Valid shutdown command received from {addr[0]}. Initiating shutdown!")
                        
                        # Execute sudo shutdown using the -S flag to read password from stdin
                        cmd = f"echo '{SUDO_PASSWORD}' | sudo -S shutdown -h now"
                        subprocess.run(cmd, shell=True)
                        break
                    else:
                        logging.warning(f"Invalid command received from {addr[0]}: {data}")
            except Exception as e:
                logging.error(f"Error handling connection: {e}")

if __name__ == "__main__":
    main()
