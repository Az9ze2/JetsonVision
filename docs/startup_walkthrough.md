# Visual Jetson Async Startup Walkthrough

This guide details how to set up `visual_jetson_async.py` to run automatically as a background service on startup on an Ubuntu-based Nvidia Jetson Orin.

By using `systemd`, the industry-standard init system for Ubuntu/Debian, we can ensure the vision pipeline starts automatically when the Jetson boots, restarts if it crashes, and logs output efficiently without requiring a desktop environment.

## 1. Prerequisites

- You should have tested the pipeline manually and confirmed it works.
- Have the **absolute path** to your `JetsonVision` repository ready. For this example, we'll assume it's located at `/home/jetson/JetsonVision`.
- Note the Python virtual environment (if you are using one), otherwise we will use the global Python interpreter (e.g., `/usr/bin/python3`).

*Note: The script itself has been updated to automatically resolve required model and database paths relative to its own location, regardless of the systemd working directory.*

## 2. Create the Systemd Service File

1. Open a terminal on your Jetson Orin (via SSH or directly).
2. Use `sudo` to create a new systemd service file:
   ```bash
   sudo nano /etc/systemd/system/vision-pipeline.service
   ```
3. Paste the following configuration, adjusting the `ExecStart`, `WorkingDirectory`, and `User` lines to match your Jetson's specific setup.

   ```ini
   [Unit]
   Description=RobotAI Jetson Vision Pipeline
   # Start after the network is up
   After=network.target network-online.target

   [Service]
   Type=simple
   # IMPORTANT: Change 'jetson' to your actual Jetson username
   User=jetson
   Group=jetson

   # Working directory (change to where you cloned JetsonVision)
   WorkingDirectory=/home/jetson/JetsonVision

   # The command to execute:
   # Update the python3 path if using a virtual environment (e.g., /home/jetson/JetsonVision/venv/bin/python)
   # Adjust the connection flags based on your needs (--ws-enabled, --ws-uri, etc.)
   ExecStart=/usr/bin/python3 /home/jetson/JetsonVision/visual_jetson_async.py --no-display --ws-enabled --ws-uri ws://192.168.1.100:8765

   # Automatically restart if the script crashes
   Restart=on-failure
   RestartSec=5

   # Log settings
   StandardOutput=journal
   StandardError=journal

   [Install]
   WantedBy=multi-user.target
   ```

4. Save the file and exit the editor (`Ctrl+O`, `Enter`, `Ctrl+X` in nano).

## 3. Enable and Start the Service

Once the service file is saved, you need to tell `systemd` to pick it up and turn it on.

1. **Reload systemd daemon** to read the new service file:
   ```bash
   sudo systemctl daemon-reload
   ```
2. **Enable the service** to start on boot:
   ```bash
   sudo systemctl enable vision-pipeline.service
   ```
3. **Start the service** right now (to test without rebooting):
   ```bash
   sudo systemctl start vision-pipeline.service
   ```

## 4. Verify the Logs and Status

To check if the pipeline is running correctly, use the following commands:

- **Check current status:**
  ```bash
  sudo systemctl status vision-pipeline.service
  ```
  *(Look for `Active: active (running)` in green).*

- **View the live logs output:**
  ```bash
  journalctl -u vision-pipeline.service -f
  ```
  *(Press `Ctrl+C` to exit the log view).*

## 5. Helpful Commands

- **Stop the service:** `sudo systemctl stop vision-pipeline.service`
- **Restart the service:** `sudo systemctl restart vision-pipeline.service`
- **Disable starting on boot:** `sudo systemctl disable vision-pipeline.service`

Since the script natively intercepts the Linux termination signals (`SIGTERM`/`SIGINT`), standard `systemctl stop` and `systemctl restart` commands will cleanly close the camera and threads before shutting down.
