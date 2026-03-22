# Vision Pipeline – Run Commands

All commands must be run from the project root:
```
cd /home/rairobodog/Desktop/Tuatuang_Capstone/final
```

---

## Enrollment (run this first before recognition testing)

```bash
cd /home/rairobodog/Desktop/Tuatuang_Capstone/final
python enroll_student.py
```

### What it does
1. Asks for student details in the terminal (ID, full name EN/TH, nickname EN/TH)
2. Opens the RealSense camera window
3. Guides you through 5 angles: **STRAIGHT → LEFT → RIGHT → UP → DOWN**
4. Press `SPACE` to capture each angle, `R` to retake, `Q` to quit
5. Saves embeddings directly to `data/enrollments.json`

> Enroll all 4 students fresh before running the vision pipeline test.

---

## visual_jetson_async.py (latest)

### Basic test – with display (debug mode, CPU only)
```bash
python visual_jetson_async.py --cpu-only
```

### Headless – no display, CPU only
```bash
python visual_jetson_async.py --no-display --cpu-only
```

### GPU mode – with TensorRT (production)
```bash
python visual_jetson_async.py --no-display
```

### With WebSocket trigger (sends recognition events to audio Pi)
```bash
python visual_jetson_async.py --no-display --ws-enabled --ws-uri ws://192.168.1.20:8765
```

### With ZMQ LAN stream (sends frames to downstream device)
```bash
python visual_jetson_async.py --no-display --stream-enabled --stream-host 192.168.1.20 --stream-port 5555
```

### Full production (WebSocket + ZMQ stream + GPU)
```bash
python visual_jetson_async.py \
  --no-display \
  --ws-enabled --ws-uri ws://192.168.1.20:8765 \
  --stream-enabled --stream-host 192.168.1.20 --stream-port 5555
```

---

## Key flags

| Flag | Default | Description |
|---|---|---|
| `--no-display` | off | Headless mode (no cv2 window) |
| `--cpu-only` | off | Skip TensorRT/CUDA (faster startup for testing) |
| `--skip-frames` | 10 | Run detector every N frames |
| `--db-path` | `data/enrollments.json` | Enrollment database |
| `--det-model` | `models/buffalo_l/det_10g.onnx` | SCRFD face detector model |
| `--rec-model` | `models/buffalo_l/w600k_r50.onnx` | ArcFace recognition model (must match model used during enrollment) |
| `--ws-enabled` | off | Enable WebSocket sender |
| `--ws-uri` | `ws://127.0.0.1:8765` | WebSocket target URI |
| `--stream-enabled` | off | Enable ZMQ LAN stream |
| `--stream-host` | `0.0.0.0` | ZMQ bind address |
| `--stream-port` | 5555 | ZMQ port |
| `--log-level` | `INFO` | Log verbosity (DEBUG, INFO, WARNING, ERROR) |

---

## Enrolled students (data/enrollments.json)

| Student ID | Name | Nickname |
|---|---|---|
| 65011386 | Natcha Rungruang | Fern |
| 65011356 | Krittin Sakharin | Palm |
| 65011333 | Kirawut Chalermkitpaisan | Sainam |
| 65011402 | Natwasa Manomaiwiboon | Fernny |

Each student has 5 embeddings × 512 dimensions (ArcFace).

---

## Controls (when display is on)

| Key | Action |
|---|---|
| `q` | Quit |
| `s` | Save screenshot |
| `r` | Reset tracker |
