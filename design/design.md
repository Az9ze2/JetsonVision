# Agentic Robot Design: Decoupled Vision-Audio Pipeline

## Architecture Strategy
This design solves the "Recognition Lag" problem. Instead of a linear flow (Face -> Mic -> Text), we use an **Asynchronous Parallel Flow**. 

- **The Microphones** are always "hot," filling a circular buffer (Rolling Window).
- **The Vision Pipeline** acts as a "Trigger" that tells the system *who* to associate with the audio currently in that window.

## Components for Implementation
1. **Producer (Jetson Orin):** A WebSocket client sending high-frequency JSON packets containing `person_id` and `is_registered` status.
2. **Buffer (Pi 5):** A `collections.deque` object that holds the last 2-3 seconds of raw audio frames.
3. **Processor (Pi 5):** A coordinator that, upon receiving a "detected" signal, performs a Thai STT conversion on the buffer content.

## LLM Agent Instructions
When implementing this in a generative agent:
- **Prompt:** "Create a system where audio is captured continuously. Use a WebSocket server to receive vision updates. If an 'Unknown' user is detected, the agent should initiate a 'Guest Enrollment' Thai dialogue. If a known user is detected, use their stored profile to personalize the LLM response."
- **Key Constraint:** Do not block the main loop. Audio buffering must happen in a separate thread from the WebSocket receiver to ensure no frames are dropped.

## Data Schema
```json
{
  "status": "detected" | "searching",
  "metadata": { "person_id": "string", "is_registered": "bool" }
}