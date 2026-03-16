# Audio & STT Worker (Raspberry Pi 5)
## Role Description

The Audio & STT Working Node operates strictly on the Pi. It is responsible for continuously capturing ambient audio in a local rolling buffer, keeping the system "hot," to ensure that no conversational frames are lost before the vision system triggers an interaction.

## Architecture

**Continuous Rolling Buffer**
The microphones record endlessly. Instead of saving large wav files to disk, the `pyaudio` byte stream feeds into a synchronized `collections.deque` object simulating a 2-3 second sliding window of recorded sound.

**Speech-to-Text Processing**
Upon receiving a signal from the Data Coordinator, a snapshot of the buffer is passed into the Whisper STT service (e.g. `whisper.cpp` or faster-whisper) tailored for Thai/English hybrid speech recognition.

## Integration Flow

1. The Audio script starts up and begins populating the Ring Buffer dynamically.
2. The Data Coordinator triggers the script to stop inserting and take a "snapshot" of the current audio history.
3. The chunk is processed into text via the STT module.
4. The transcription is pushed back to the Coordinator to fuse into the Generative LLM prompt.

### Key Example Logic (Python)
```python
import pyaudio
import collections

# A rolling buffer of roughly ~3 seconds of audio frames
audio_buffer = collections.deque(maxlen=150) 

def record_audio_continuously():
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    while recording_active:
        data = stream.read(1024, exception_on_overflow=False)
        audio_buffer.append(data)
```
