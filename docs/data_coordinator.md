# Data Receiver & Coordinator (Raspberry Pi 5)
## Role Description

The Data Receiver acts as the central brain or the "Main Loop" on the Pi. It connects to the Jetson vision client via WebSockets to receive face tracking triggers, and coordinates the Audio STT service and the LLM execution based on this trigger.

## Architecture

**WebSocket Server**
A standard `asyncio` loop running `websockets.serve(..., port=8765)` continuously listens to the Jetson over the LAN connection.

**System Trigger Logic**
When the receiver gets a JSON burst indicating a person is detected, it acts upon it:
- If `"person_id": "Unknown"`, it signals the generative agent to prompt a Guest Enrollment flow.
- If `"is_registered": true`, it retrieves the local JSON database profile for the known person to insert contextual background data into the LLM logic, enabling a personalized greeting.

## Workflow Execution Steps

1. Wait for Jetson `"status": "detected"` JSON packet.
2. Tell the `Audio & STT Worker` to transcribe the recent 3 seconds of sound.
3. Determine person context (Known User vs. Unknown User).
4. **LLM Prompt Generation:** Assemble the transcribed text *and* the visual context together into a generative prompt ("You are the robot, you see Krittin, he just said 'Hello'").
5. Return speech synthesis response file via TTS (Text-to-Speech) to the speaker.

### Example Logic (Python)
```python
import websockets
import json

async def vision_listener(websocket, path):
    async for message in websocket:
        data = json.loads(message)
        if data.get("status") == "detected":
            student_id = data["metadata"]["person_id"]
            
            # Execute STT and LLM fusion loop
            transcription = trigger_stt()
            response = trigger_llm(student_id, transcription)
            synthesize_speech(response)
```
