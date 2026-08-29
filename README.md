# CCTV Plank Inspection System

> **Real-Time Conveyor Belt Plank Counting, Defect Detection & Video Question Answering**

A computer vision system for continuously monitoring wooden planks on a conveyor belt using a CCTV/RTSP camera stream. The system uses YOLOv8 for real-time plank and defect detection, tracking for persistent plank IDs, virtual line crossing for accurate counting, automatic defect snapshot generation, continuous logging, Docker-based deployment, and a Video Question Answering (Video-QA) engine for querying inspection results using natural language.

---

## Overview

The CCTV Plank Inspection System is designed for industrial conveyor-belt inspection.

The system continuously receives video from a CCTV camera and processes the stream using a YOLOv8-based computer vision pipeline.

For every plank moving through the conveyor:

- Detects the plank.
- Assigns a persistent tracking ID.
- Tracks the plank across video frames.
- Counts the plank when it crosses a virtual counting line.
- Detects visible defects such as cracks, knots, holes, rot, and splits.
- Stores defective plank snapshots.
- Places the Plank ID, defect type, and timestamp directly on the snapshot.
- Maintains a continuous count and defect log.
- Maintains a real-time summary of production statistics.
- Stores all inspection data in persistent Docker-mounted storage.
- Allows users to ask natural-language questions about the inspection data.
- Uses a Vision-Language Model for visual questions about saved plank images.

The system is intended to operate continuously and can automatically reconnect if the CCTV stream temporarily disconnects.

---

## System Architecture

```text
                     ┌─────────────────────────┐
                     │     CCTV / IP Camera    │
                     │       RTSP Stream       │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │    RTSP Stream Reader   │
                     │  Low-Latency Frame Grab │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │       YOLOv8 Model      │
                     │                         │
                     │  Plank Detection        │
                     │  Defect Detection       │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │   Object Tracking       │
                     │                         │
                     │  Persistent Plank IDs   │
                     │  ByteTrack / BoT-SORT   │
                     └────────────┬────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
          ┌──────────────────────┐     ┌──────────────────────┐
          │ Virtual Line Counter │     │ Defect Identification│
          │                      │     │                      │
          │ Count each plank     │     │ Crack / Knot / Rot   │
          │ once                 │     │ Hole / Split / etc.  │
          └──────────┬───────────┘     └──────────┬───────────┘
                     │                            │
                     └─────────────┬──────────────┘
                                   │
                                   ▼
                     ┌─────────────────────────┐
                     │   Inspection Logger     │
                     │                         │
                     │   CSV Count Log         │
                     │   JSON Live Summary     │
                     │   Defect Snapshots      │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │      Video-QA Engine    │
                     │                         │
                     │ Telemetry Questions     │
                     │ Visual Questions        │
                     │ Plank-specific Queries  │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │   Vision Language Model │
                     │                         │
                     │   Gemini / Qwen-VL      │
                     └─────────────────────────┘
```

---

## Key Features

### 1. Live CCTV Processing

The system can process:

- IP camera RTSP streams
- CCTV streams
- Webcam input for development/testing
- Recorded video files for testing

Example RTSP URL:

```text
rtsp://username:password@192.168.1.100:554/stream1
```

The stream processing is designed for continuous operation.

### 2. YOLOv8 Detection

YOLOv8 is used as the core computer vision model.

Depending on the trained dataset, the model can detect:

- `plank`
- `crack`
- `knot`
- `hole`
- `rot`
- `split`
- `mold`
- `other_defect`

A custom-trained model should eventually replace the initial pretrained YOLO weights.

Example:

```text
models/
└── best.pt
```

### 3. Plank Tracking

Detection alone cannot reliably determine whether an object is the same plank across multiple frames.

Therefore, object tracking is used to assign a persistent ID.

Example:

```text
Frame 1 → Plank #12
Frame 2 → Plank #12
Frame 3 → Plank #12
Frame 4 → Plank #12
```

The same physical plank therefore maintains the same tracking ID while it remains visible.

The system can use tracking algorithms supported by Ultralytics, such as:

- ByteTrack
- BoT-SORT

### 4. Plank Counting

A virtual line is placed across the conveyor belt.

Example:

```text
                 Conveyor Direction
                         ↓

        ┌──────────────────────────────┐
        │                              │
        │       ┌────────────┐         │
        │       │  PLANK #15 │         │
        │       └────────────┘         │
        │                              │
        │──────────────────────────────│
        │        COUNTING LINE         │
        │──────────────────────────────│
        │                              │
        └──────────────────────────────┘
```

When the center of a tracked plank crosses the line, the system records one plank.

A set of already-counted tracking IDs prevents duplicate counting.

Example:

```text
Plank #1 → counted
Plank #2 → counted
Plank #3 → counted
Plank #4 → counted
```

### 5. Defect Detection

The YOLO model identifies defects on the plank.

Example:

```text
Plank #27

Detected:
- crack
- knot
```

The defect information is associated with the corresponding tracked plank.

The system can accumulate defects observed across multiple frames before the plank is counted.

This is important because a defect may not be visible with equal confidence in every frame.

### 6. Automatic Defect Snapshots

When a defective plank is detected, the system automatically saves a snapshot.

Example filename:

```text
plank_0027_20260829_145519_crack_knot.jpg
```

The snapshot contains:

```text
┌───────────────────────────────────────┐
│ PLANK #27 | DEFECT: crack, knot       │
│ Time: 2026-08-29 14:55:19             │
├───────────────────────────────────────┤
│                                       │
│              PLANK IMAGE              │
│                                       │
│                                       │
└───────────────────────────────────────┘
```

This provides visual evidence for every defective plank.

### 7. Continuous Logging

The system does not create only hourly summary files.

Instead, it continuously records every plank as it crosses the counting line.

The main log is:

```text
continuous_count_log.csv
```

Example:

```csv
Plank_ID,Timestamp,Status,Defect_Types,Running_Total_Count,Running_Defect_Count,Running_Clean_Count,Defect_Snapshot_Path
1,2026-08-29 14:55:01,CLEAN,None,1,0,1,
2,2026-08-29 14:55:05,DEFECTIVE,crack,2,1,1,plank_0002_20260829_145505_crack.jpg
3,2026-08-29 14:55:10,CLEAN,None,3,1,2,
4,2026-08-29 14:55:14,CLEAN,None,4,1,3,
5,2026-08-29 14:55:19,DEFECTIVE,knot;rot,5,2,3,plank_0005_20260829_145519_knot_rot.jpg
```

Every count is written immediately.

This allows the system to maintain a continuous audit trail while running.

### 8. Live Summary

A continuously updated JSON file provides the current production state.

```text
live_count_summary.json
```

Example:

```json
{
  "last_updated": "2026-08-29 14:55:19",
  "running_total_count": 5,
  "running_clean_count": 3,
  "running_defective_count": 2,
  "defect_rate_percentage": 40.0,
  "last_counted_plank_id": 5,
  "last_status": "DEFECTIVE"
}
```

This can later be consumed by the web dashboard.

### 9. Persistent Docker Storage

All important inspection data is stored under:

```text
/app/data/
```

Inside the container:

```text
/app/data/
├── continuous_count_log.csv
├── live_count_summary.json
└── defect_snapshots/
    ├── plank_0002_20260829_145505_crack.jpg
    ├── plank_0005_20260829_145519_knot_rot.jpg
    └── ...
```

Docker Compose mounts this directory to the host:

```text
./conveyor_data:/app/data
```

Therefore, inspection data survives container restarts.

The Docker container can restart without losing previously recorded inspection data.

### 10. Automatic RTSP Reconnection

The monitoring process is designed for continuous operation.

If the camera connection is lost:

```text
Camera disconnected
        ↓
      Wait
        ↓
Attempt reconnect
        ↓
Camera available
        ↓
Resume inspection
```

The container itself is also configured with:

```yaml
restart: always
```

This allows Docker to automatically restart the application if the process terminates unexpectedly.

### 11. Video Question Answering

The Video-QA engine allows users to ask questions using natural language.

Instead of manually opening CSV files or searching through snapshots, the user can ask questions such as:

- What is the total number of planks?
- How many defective planks were detected?
- What is the current defect rate?
- Describe the defect on plank #27.
- What defects were detected on plank #5?
- Compare plank #2 and plank #5.

---

## Video-QA Architecture

The Video-QA system uses a hybrid approach.

```text
                   User Question
                         │
                         ▼
                   ┌──────────────┐
                   │  QA Router   │
                   └──────┬───────┘
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
      Telemetry Query            Visual Query
             │                         │
             ▼                         ▼
      CSV / JSON Logs          Defect Snapshot
             │                         │
             │                         ▼
             │                  Vision-Language
             │                      Model
             │                         │
             └────────────┬────────────┘
                          ▼
                     Final Answer
```

### Telemetry Questions

Questions involving numerical inspection data are answered directly from structured logs.

For example:

> How many planks have been inspected?

The system reads:

```text
live_count_summary.json
```

and returns the current count.

This avoids unnecessarily sending simple numerical questions to an LLM.

### Visual Questions

Questions requiring visual understanding use a Vision-Language Model.

For example:

> Describe the defect on plank #27.

The system:

1. Extracts plank ID 27.
2. Searches the defect snapshot directory.
3. Loads the corresponding image.
4. Provides the image and YOLO metadata to the VLM.
5. Generates a visual explanation.

Possible VLM providers include:

- Google Gemini
- Qwen-VL
- Other compatible multimodal models

---

## Project Structure

```text
conveyor-defect-inspector/
│
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
│
├── data/
│   ├── continuous_count_log.csv
│   ├── live_count_summary.json
│   └── defect_snapshots/
│
├── models/
│   └── best.pt
│
└── src/
    ├── app.py
    ├── qa_engine.py
    ├── vlm_provider.py
    ├── cli_qa.py
    └── stream_reader.py
```

---

## Component Responsibilities

### `src/app.py`

Main inspection application.

Responsibilities:

- Load YOLO model.
- Connect to CCTV/RTSP stream.
- Run detection.
- Track planks.
- Detect defects.
- Count planks.
- Save defective snapshots.
- Write continuous logs.
- Update live summary.
- Handle stream reconnection.

### `src/stream_reader.py`

Handles video ingestion.

Responsibilities:

- RTSP connection.
- Frame capture.
- Low-latency processing.
- Dropping stale frames.
- Stream reconnection support.

### `src/qa_engine.py`

Handles natural-language questions.

Responsibilities:

- Query classification.
- Telemetry lookup.
- Plank ID extraction.
- Snapshot retrieval.
- VLM routing.
- Final response generation.

### `src/vlm_provider.py`

Provides a unified interface to multimodal models.

Possible providers:

- Gemini
- Qwen-VL
- Other VLM

The rest of the application should not need to know the implementation details of the selected VLM provider.

### `src/cli_qa.py`

Provides a command-line interface for testing Video-QA.

Example:

```bash
python src/cli_qa.py
```

Then:

```text
> What is the total plank count?

> How many defective planks were detected?

> Describe the defect on plank #14.
```

---

## Technology Stack

### Computer Vision

- Python
- OpenCV
- YOLOv8
- Ultralytics
- NumPy

### Object Tracking

- ByteTrack
- BoT-SORT

### Video

- RTSP
- CCTV/IP Camera
- OpenCV VideoCapture
- FFmpeg

### AI / VLM

- Gemini
- Qwen-VL
- Other multimodal models

### Backend

Planned:

- FastAPI
- WebSockets
- REST APIs

### Deployment

- Docker
- Docker Compose
- NVIDIA GPU support

### Storage

- CSV
- JSON
- JPEG snapshots
- Docker persistent volumes

---

## Installation

### Prerequisites

Install:

- Python 3.10+
- Docker
- Docker Compose
- Git
- NVIDIA drivers if GPU acceleration is required

For local development, verify:

```bash
python --version
docker --version
docker compose version
git --version
```

---

## Environment Configuration

The application uses environment variables for deployment configuration.

Example:

```yaml
environment:
  - RTSP_URL=rtsp://admin:password@192.168.1.100:554/stream1
  - MODEL_WEIGHTS=/app/models/best.pt
  - DATA_DIR=/app/data
  - LINE_Y=350
```

### Configuration Variables

| Variable | Purpose |
|---|---|
| `RTSP_URL` | CCTV/RTSP video source |
| `MODEL_WEIGHTS` | YOLO model path |
| `DATA_DIR` | Persistent data directory |
| `LINE_Y` | Virtual counting line position |
| `VLM_PROVIDER` | Selected VLM provider |
| `GEMINI_API_KEY` | Gemini API authentication |
| `VLM_BASE_URL` | Local VLM endpoint |
| `VLM_MODEL_NAME` | Local VLM model name |

---

## Docker Data Layout

The host machine will contain:

```text
conveyor-defect-inspector/
│
└── conveyor_data/
    ├── continuous_count_log.csv
    ├── live_count_summary.json
    └── defect_snapshots/
        ├── plank_0001_20260829_145501_crack.jpg
        ├── plank_0004_20260829_145512_knot.jpg
        └── ...
```

This directory is persistent.

Deleting/recreating the Docker container does not delete the mounted host data.

---

## Defect Snapshot Naming

Snapshots follow the format:

```text
plank_<ID>_<TIMESTAMP>_<DEFECT>.jpg
```

Example:

```text
plank_0015_20260829_151025_crack.jpg
```

Multiple defects:

```text
plank_0021_20260829_151130_knot_rot.jpg
```

The filename provides a quick machine-readable reference.

The image itself also contains:

- Plank ID
- Defect Type
- Timestamp

---

## Continuous Inspection Data

Each counted plank creates one inspection record.

Example:

```text
Plank #18
Timestamp: 2026-08-29 15:20:10
Status: DEFECTIVE
Defects: crack
Snapshot: plank_0018_20260829_152010_crack.jpg
```

This creates a traceable relationship between:

```text
Plank ID
   │
   ├── Detection
   ├── Tracking
   ├── Count
   ├── Defect
   ├── Snapshot
   └── Log Entry
```

---

## Production Metrics

The system maintains:

- Total Planks
- Clean Planks
- Defective Planks
- Defect Rate
- Last Counted Plank
- Last Status

### Defect Rate

```text
Defect Rate =
(Defective Planks / Total Planks) × 100
```

Example:

```text
Total:        1000
Clean:         930
Defective:      70

Defect Rate = 7%
```

---

## Dataset and Model Training

The initial system architecture supports a custom-trained YOLO model.

A recommended dataset contains classes such as:

```text
0: plank
1: crack
2: knot
3: hole
4: rot
5: split
6: mold
```

The dataset should be divided into:

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
└── labels/
    ├── train/
    ├── val/
    └── test/
```

A YOLO dataset configuration can define:

```yaml
names:
  0: plank
  1: crack
  2: knot
  3: hole
  4: rot
  5: split
  6: mold
```

The model should be trained using images representative of the actual conveyor environment.

---

## Model Improvement

The pretrained YOLO model is only a starting point.

For production deployment, the system should use a model trained specifically on:

- The actual wood species.
- Actual conveyor lighting.
- Actual camera angle.
- Different plank sizes.
- Different plank orientations.
- Real manufacturing defects.
- Motion blur.
- Dust and debris.
- Shadows.
- Reflections.
- Occlusions.

Model performance should be evaluated using:

- Precision
- Recall
- mAP@50
- mAP@50-95
- Confusion Matrix
- False Positive Rate
- False Negative Rate
- Inference FPS

---

## GPU Acceleration

For NVIDIA GPU deployment, the model can eventually be optimized using:

- ONNX
- TensorRT
- FP16
- INT8

Example export:

```bash
yolo export model=best.pt format=onnx
```

or TensorRT:

```bash
yolo export model=best.pt format=engine
```

GPU acceleration is particularly useful when processing high-resolution CCTV streams or multiple camera feeds.

---

## Reliability and 24/7 Operation

The system is designed as a continuously running inspection service.

Important reliability features include:

### Automatic Camera Reconnect

If RTSP fails:

```text
Disconnect
   ↓
Retry
   ↓
Reconnect
   ↓
Resume processing
```

### Persistent Storage

Inspection records are written to persistent storage.

### Immediate Logging

Counts are recorded when planks cross the counting line rather than waiting for an hourly batch process.

### Container Restart

Docker Compose uses:

```yaml
restart: always
```

to recover from application/container failures.

---

## Video-QA Examples

The QA engine is intended to support questions such as:

### Production Questions

- How many planks have been inspected?
- How many defective planks were found?
- What is the current defect rate?

### Defect Questions

- What defects were detected recently?
- How many planks had cracks?
- How many planks had knots?

### Plank-Specific Questions

- What defect does plank #27 have?
- Describe the defect on plank #27.
- Show the information recorded for plank #15.

### Visual Analysis

- Describe the visible crack on plank #27.
- Does plank #15 appear severely damaged?
- Compare plank #12 and plank #18.

> The final visual assessment should be treated as an AI-generated interpretation and should not replace human quality-control decisions where required.

---

## Planned Web Dashboard

The next major component is a browser-based monitoring dashboard.

### Planned Architecture

```text
                     Browser
                        │
                        ▼
                 ┌────────────────┐
                 │ React Frontend │
                 └───────┬────────┘
                         │
                  WebSocket / REST
                         │
                         ▼
                 ┌────────────────┐
                 │ FastAPI Backend│
                 └───────┬────────┘
                         │
              ┌──────────┼───────────┐
              │          │           │
              ▼          ▼           ▼
           YOLO       QA Engine     Logs
          Pipeline        │
                          ▼
                         VLM
```

The dashboard is expected to provide:

- Live CCTV playback.
- YOLO bounding boxes.
- Plank IDs.
- Defect highlighting.
- Total plank count.
- Clean count.
- Defective count.
- Defect percentage.
- Recent defective snapshots.
- Inspection history.
- Natural-language Video-QA chat.
- System health status.
- Camera connection status.

---

## Security Considerations

The production system should not expose CCTV credentials directly in source code.

Use environment variables or secure secrets.

Avoid committing:

```text
.env
*.pt
camera passwords
API keys
raw CCTV footage
production logs
```

These should be excluded through `.gitignore`.

Example:

```gitignore
.env
*.pt
*.onnx
*.engine
data/
conveyor_data/
__pycache__/
*.pyc
```

---

## Important Design Principle

The system separates deterministic inspection data from AI-generated interpretation.

For example:

> `"Total planks = 1,245"`

should come directly from the inspection logs.

The VLM should not be responsible for calculating production counts when structured data already exists.

Conversely:

> `"Describe the visible defect on plank #27."`

is a visual reasoning task and can be handled by a Vision-Language Model.

This hybrid design improves reliability, reduces unnecessary LLM usage, and makes numerical answers auditable.

---

## Current Status

**Project Status: Beta**

The current implementation establishes the core computer-vision and Video-QA architecture.

### Currently Implemented

- ✓ CCTV/RTSP ingestion
- ✓ YOLOv8 detection
- ✓ Object tracking
- ✓ Plank IDs
- ✓ Virtual line counting
- ✓ Defect identification
- ✓ Defect snapshots
- ✓ Continuous count logging
- ✓ Live JSON statistics
- ✓ Persistent Docker storage
- ✓ RTSP reconnect logic
- ✓ Video-QA routing
- ✓ Telemetry QA
- ✓ Plank-specific QA
- ✓ Multimodal VLM architecture

### Currently Under Development

- → Web dashboard
- → Live browser CCTV playback
- → FastAPI API
- → WebSocket communication
- → Custom wood-defect model
- → Production optimization

---

## Goal

The long-term goal of the project is to provide a complete AI-powered industrial inspection platform capable of:

```text
LIVE CCTV
    ↓
REAL-TIME AI INSPECTION
    ↓
PLANK COUNTING
    ↓
DEFECT DETECTION
    ↓
TRACKABLE PLANK ID
    ↓
AUTOMATIC EVIDENCE SNAPSHOT
    ↓
CONTINUOUS AUDIT LOG
    ↓
NATURAL-LANGUAGE VIDEO-QA
    ↓
WEB MONITORING DASHBOARD
```

The final system should allow an operator to monitor the conveyor in real time while retaining a complete machine-readable history of every inspected plank and its detected defects.
