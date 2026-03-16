import asyncio
import websockets
import json
import time

async def vision_sender():
    uri = "ws://<RASPI_IP_ADDRESS>:8765"
    async with websockets.connect(uri) as websocket:
        print("Connected to Raspberry Pi Receiver")
        
        while True:
            # --- MODEL INFERENCE PLACEHOLDER ---
            # Replace this with your actual detection/recognition output
            detected_name = "Krittin" # or "Unknown"
            is_registered = True
            # -----------------------------------

            data = {
                "timestamp": time.time(),
                "status": "detected",
                "metadata": {
                    "person_id": detected_name,
                    "is_registered": is_registered,
                    "confidence": 0.98
                }
            }

            await websocket.send(json.json.dumps(data))
            await asyncio.sleep(0.1)  # 10Hz update rate

if __name__ == "__main__":
    asyncio.run(vision_sender())