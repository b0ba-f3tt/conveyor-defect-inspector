# Solid Paint Conveyor Plank Counter

Real-time industrial computer vision system for automatically detecting and counting solid wooden planks moving on a manufacturing conveyor. The system uses OpenCV-based image processing, ROI occupancy analysis, contour filtering, a finite-state counting system, CCTV timestamp OCR, and automatic CSV/Excel production logging.

The system is designed for live CCTV/RTSP monitoring and does not require YOLO, machine learning, model training, or object-detection models.

## Key Features

1. **Real-Time CCTV/RTSP Monitoring**
   - Connects directly to an IP CCTV camera through RTSP.
   - Uses a background capture thread to continuously maintain the latest camera frame.
   - Automatically reconnects when the RTSP connection is interrupted.
   - Prevents frame backlog by processing the latest available frame.

2. **OpenCV-Based Plank Detection**
   - Uses a configurable Region of Interest (ROI).
   - Combines HSV color segmentation and intensity thresholding.
   - Uses morphological filtering to clean the detection mask.
   - Detects plank candidates using contours.
   - Filters false detections using:
     - Minimum area
     - Minimum width and height
     - Rectangularity
     - Solidity
     - Aspect ratio

3. **Conveyor Counting State Machine**
   - Uses a finite-state machine to control plank counting.
   - Includes movement validation and cooldown protection.
   - Prevents repeated counting of the same detected plank.
   - Tracks conveyor states such as:
     - CLEAR
     - ENTERING
     - COUNTED
     - EXITING

4. **CCTV Date and Time Detection**
   - Reads the timestamp displayed on the CCTV video.
   - Uses Tesseract OCR to extract the CCTV date and time.
   - OCR runs in a background worker so it does not block the main vision-processing loop.
   - Validates detected timestamps to prevent stale or invalid OCR results.

5. **Industrial Monitoring HUD**
   - Displays the current production count.
   - Displays the current conveyor detection state.
   - Displays ROI occupancy.
   - Displays runtime processing information.
   - Displays the detected CCTV timestamp.
   - Displays the configured counting ROI.
   - Optionally displays detected plank bounding boxes.

6. **Automatic CSV Production Logging**
   - Records every successful plank count.
   - Stores production information including:
     - Date
     - Time
     - Shift
     - Total count

7. **Automatic Excel Production Logging**
   - Updates the Excel production log after every successful count.
   - Maintains daily production information.
   - Maintains shift totals.
   - Maintains a whole-day production total.
   - Uses the configured shift schedule.

8. **Performance Optimizations**
   - Background RTSP frame capture.
   - Latest-frame processing architecture.
   - Background CCTV OCR.
   - Cached ROI calculations.
   - Reduced unnecessary memory allocations.
   - Optimized contour processing.
   - Reduced visualization overhead.
   - Optimized CSV/Excel update operations.
   - Designed to minimize processing latency and prevent RTSP frame queues.

9. **Configurable System**
   - Detection thresholds are controlled through `config/config.yaml`.
   - ROI coordinates can be adjusted for different camera positions.
   - CCTV OCR settings can be configured.
   - Production shifts can be configured.
   - CSV and Excel logging behavior can be configured.

## System Architecture

```text
CCTV Camera
     |
     | RTSP
     v
RTSP Capture Thread
     |
     | Latest Frame
     v
CCTV Timestamp OCR
     |
     v
ROI Extraction
     |
     v
OpenCV Detection
     |
     +--> HSV Segmentation
     |
     +--> Intensity Thresholding
     |
     +--> Morphological Filtering
     |
     +--> Contour Detection
     |
     +--> Shape Filtering
     |
     v
Plank Bounding Box
     |
     v
Conveyor State Machine
     |
     v
Count Event
     |
     +-----------> CSV Logger
     |
     +-----------> Excel Logger
     |
     v
Industrial HUD
     |
     v
Live Display
