import asyncio
import websockets
import json

async def handler(websocket):
    print(f"Client connected: {websocket.remote_address}")
    try:
        async for message in websocket:
            data = json.loads(message)
            print(f"Received JSON: {json.dumps(data, indent=2)}")
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")

async def main():
    print("Starting Dummy WebSocket Server on ws://localhost:8765...")
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
