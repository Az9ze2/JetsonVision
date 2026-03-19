# Walkthrough: Asynchronous Vision-Audio Pipeline Trigger

I have implemented the new vision pipeline per the requested `design.md`.

## Changes Made

1. **`vision/websocket_sender.py`**:
   - Created a new `WebSocketSender` class.
   - It runs an `asyncio` event loop in a background Python Thread.
   - It uses a thread-safe `queue.Queue` to allow the synchronous OpenCV thread to push JSON packets non-blockingly. If the queue fills up, it drops the oldest frame to guarantee the Jetson CV loop never blocks.

2. **`visual_jetson_async.py`**:
   - Duplicated the production Jetson pipeline.
   - Added new command line arguments:
     - `--ws-enabled`: Toggle the WebSocket publisher.
     - `--ws-uri`: Define the remote Pi address (e.g. `ws://192.168.1.100:8765`).
   - Integrated the `WebSocketSender`.
   - In the main loop (Step 6), when a face is detected/tracked, it identifies the largest/most confident face.
   - It cross-references the enrollment tracking ID correctly and packages a tiny JSON string:
     ```json
     {
       "timestamp": 1712437582.12,
       "status": "detected",
       "metadata": {
         "person_id": "Krittin",
         "is_registered": true,
         "confidence": 0.98
       }
     }
     ```
   - Sends the JSON immediately over the LAN cable to the Pi.

## Verification Run

- Tested the `sys.path` to ensure the new `vision.websocket_sender` resolves correctly.
- Created `test_ws_server.py` to locally verify the client behavior. Note that because ONNX model weights (`buffalo_l`) are not downloaded in this local workspace, a full end-to-end integration test of the video frame execution was skipped, but the module initialization succeeds perfectly.

## How to use

On the Jetson, simply start the run using the new arguments pointing to the Pi's IP:
```bash
python visual_jetson_async.py --no-display --ws-enabled --ws-uri ws://192.168.1.100:8765