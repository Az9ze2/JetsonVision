import asyncio
import json
import threading
import queue
import time
from typing import Dict, Any, Optional
from loguru import logger
import websockets

class WebSocketSender:
    """
    A non-blocking WebSocket sender that runs an asyncio event loop
    in a background thread. This allows synchronous code (like an OpenCV loop)
    to push messages to a queue without stalling.
    """
    def __init__(self, uri: str = "ws://127.0.0.1:8765"):
        self.uri = uri
        self._queue: queue.Queue = queue.Queue(maxsize=100)
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Starts the background thread and asyncio loop."""
        if self._running:
            return
        logger.info(f"Starting WebSocketSender thread connecting to {self.uri}")
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Signals the background thread to stop and waits for it."""
        if not self._running:
            return
        logger.info("Stopping WebSocketSender thread")
        self._running = False
        if self._thread:
            # We push a dummy to wake up the queue if it's blocking
            try:
                self._queue.put({"_terminate": True}, block=False)
            except queue.Full:
                pass
            self._thread.join(timeout=2.0)
            self._thread = None

    def send(self, data: Dict[str, Any]):
        """
        Pushes data to the queue. If the queue is full, the oldest item is dropped
        to ensure the main loop never blocks.
        """
        if not self._running:
            return

        try:
            self._queue.put_nowait(data)
        except queue.Full:
            # Drop the oldest packet (lossy) to keep the pipeline moving
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(data)
            except (queue.Empty, queue.Full):
                pass

    def _run_loop(self):
        """The target function for the background thread."""
        asyncio.run(self._async_sender())

    async def _async_sender(self):
        """The actual async function that connects and sends messages forever."""
        while self._running:
            try:
                async with websockets.connect(self.uri) as websocket:
                    logger.info(f"WebSocket connected to {self.uri}")
                    while self._running:
                        try:
                            # We use run_in_executor to not block the async loop
                            # while waiting for the synchronous queue
                            data = await asyncio.get_event_loop().run_in_executor(
                                None, self._queue.get, True, 0.5
                            )
                            if data.get("_terminate"):
                                break
                            
                            payload = json.dumps(data)
                            await websocket.send(payload)
                        except queue.Empty:
                            # Timeout to let us check self._running
                            continue
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning("WebSocket connection closed, reconnecting...")
                            break
                        except Exception as e:
                            logger.error(f"Error sending message: {e}")
                            break
            except Exception as e:
                # If connection fails, wait a bit before trying again
                if self._running:
                    logger.debug(f"WebSocket connection failed ({e}), retrying in 2s...")
                    await asyncio.sleep(2.0)
