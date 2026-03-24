# Remote Shutdown Deployment Guide

This guide details how to install the `shutdown_listener.py` script as a persistent background service on your Nvidia Jetson Orin so it's always ready to receive shutdown requests from the Raspberry Pi.

## 1. Prerequisites
- The `shutdown_listener.py` script must be located on the Jetson Orin
- It assumes the Jetson password is `123`, which you've configured inside the script.

## 2. Test It Manually First
Before creating the service, ensure it works.
1. Run the listener on the Jetson:
   ```bash
   python3 /path/to/JetsonVision/shutdown_listener.py
   ```
2. Run the sender script on the Pi (update the Jetson IP inside it):
   ```bash
   python3 pi_shutdown_sender.py
   ```
3. If it successfully triggers a shutdown, proceed below to install the service.

## 3. Create the Systemd Service
To ensure the socket always listens on startup, we will add it as a systemd background service.

1. SSH into the Jetson Orin.
2. Open a new service definition file using nano:
   ```bash
   sudo nano /etc/systemd/system/remote-shutdown.service
   ```
3. Paste the following configuration. Replace `/home/jetson/JetsonVision` with the actual path to your scripts:

   ```ini
   [Unit]
   Description=Jetson Remote Shutdown Listener
   After=network.target network-online.target

   [Service]
   Type=simple
   # Update this to your Jetson username
   User=jetson
   Group=jetson

   # Update to your script directory
   WorkingDirectory=/home/jetson/JetsonVision

   # Python 3 command to execute the listener
   ExecStart=/usr/bin/python3 /home/jetson/JetsonVision/shutdown_listener.py

   Restart=on-failure
   RestartSec=5
   
   # Log settings
   StandardOutput=journal
   StandardError=journal

   [Install]
   WantedBy=multi-user.target
   ```
4. Save (`Ctrl+O`, `Enter`) and Exit (`Ctrl+X`).

## 4. Enable and Run the Service

1. Reload the systemd daemon:
   ```bash
   sudo systemctl daemon-reload
   ```
2. Enable it to run automatically on boot:
   ```bash
   sudo systemctl enable remote-shutdown.service
   ```
3. Start the service immediately:
   ```bash
   sudo systemctl start remote-shutdown.service
   ```

## 5. Verify Listener is Running
You can check logs running through the service any time:
```bash
sudo systemctl status remote-shutdown.service
```

You should see: "Listening for shutdown commands on port 9999..."
