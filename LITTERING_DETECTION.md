# Littering Detection System - Implementation Guide

## Overview

The waste tracking system has been implemented to detect when a person throws waste (separates hand from waste object) and automatically tags it as **littered** after a threshold time period.

## New Features

### 1. **WasteTracker Class** (`objectdetection/waste_tracking.py`)

A new module that tracks waste objects and detects littering behavior.

#### Key Components:

- **Separation Detection**: Monitors the distance between detected hands and waste objects
- **Temporal Tracking**: Records when hand and waste become separated
- **Littering Tag**: Automatically marks waste as "littered" after exceeding the time threshold
- **Real-time Visualization**: Color-coded bounding boxes on video frames

#### Configuration Parameters:

```python
waste_tracker = WasteTracker(
    separation_threshold=80,        # Pixel distance to consider separation (default: 50)
    littering_time_threshold=2.0    # Seconds before marking as littered (default: 2.0)
)
```

### 2. **Status States**

Each waste object can have three states:

| State | Color | Description |
|-------|-------|-------------|
| **ATTACHED** | Green | Hand and waste are together (within separation threshold) |
| **SEPARATED** | Yellow | Hand and waste are apart, but below littering threshold time |
| **LITTERED** | Red | Hand and waste have been separated for longer than threshold |

### 3. **How It Works**

```
Frame 1: Hand holds waste
  ↓
[ATTACHED state - green box]
  ↓
Frame N: Hand releases waste (distance increases beyond threshold)
  ↓
[SEPARATED state - yellow box] - Timer starts
  ↓
Frame N+M: Separation duration exceeds littering_time_threshold (2.0s by default)
  ↓
[LITTERED state - red box] - Alert printed to console
```

## Usage

### Basic Usage

The updated `main.py` automatically uses the littering detection:

```bash
python main.py
```

### Customizing Thresholds

Edit the initialization in `main.py`:

```python
waste_tracker = WasteTracker(
    separation_threshold=80,      # Increase for more lenient separation
    littering_time_threshold=2.0  # Increase for more time before tagging
)
```

### Configuration Tips

- **separation_threshold**: 
  - Lower values (50): More sensitive to small movements
  - Higher values (100+): Only detects major separation
  - Recommended: 60-100 pixels

- **littering_time_threshold**:
  - Lower values (1.0): Faster littering detection
  - Higher values (3.0+): More time to verify littering
  - Recommended: 1.5-3.0 seconds

## Output

### Console Output

When littering is detected:
```
⚠️  LITTERED! Waste #1 has been thrown!
   Separated for 2.15 seconds
```

### Visual Output

The video window displays:
- **Green rectangles**: Waste held by hand
- **Yellow rectangles**: Waste recently released (timing out)
- **Red rectangles**: Confirmed littered waste
- **Labels**: Real-time status and waste ID

## Implementation Details

### Distance Calculation

The system calculates the Euclidean distance between bounding box centers:

```
distance = √[(cx1 - cx2)² + (cy1 - cy2)²]
```

- If distance < separation_threshold → ATTACHED
- If distance ≥ separation_threshold → SEPARATED

### Tracking Algorithm

1. Extract hand and waste bounding boxes from detection results
2. Calculate distances between all hand-waste pairs
3. For each waste object:
   - If closest hand is within threshold → ATTACHED (reset timer)
   - If no hands within threshold → SEPARATED (start/continue timer)
   - If separation time > threshold → LITTERED (mark and alert)

## File Structure

```
objectdetection/
├── hands.py              # Hand detection (MediaPipe)
├── waste.py              # Waste detection (YOLO)
├── waste_tracking.py     # NEW: Littering detection system
└── aimodel/
    └── best.pt           # YOLO model weights

main.py                    # UPDATED: Integrated tracking system
```

## Dependencies

All required dependencies are in `requirements.txt`:
- opencv-python
- mediapipe==0.10.14
- ultralytics>=8.1.0

No additional installations needed.

## Performance Considerations

- **Frame Rate**: Works at normal video playback speeds
- **Multiple Hands/Waste**: System tracks all detected objects simultaneously
- **Edge Cases**: 
  - Handles multiple people and multiple waste objects
  - Resets tracking when objects leave frame
  - Automatically associates new detections with tracked objects

## Tuning for Your Use Case

### Scenario 1: Controlled Environment (Office/Lab)
```python
separation_threshold=60
littering_time_threshold=1.5
```

### Scenario 2: Real-world (Street/Park)
```python
separation_threshold=100
littering_time_threshold=3.0
```

### Scenario 3: Quick Detection
```python
separation_threshold=50
littering_time_threshold=1.0
```

## Testing

To test the system:

1. Run with sample video:
   ```bash
   python main.py
   ```

2. Watch the video and observe:
   - Green boxes when hand holds waste
   - Yellow boxes when hand releases waste
   - Red boxes and console alerts when littering is confirmed

3. Adjust thresholds based on your observations

## Future Enhancements

Possible improvements:
- Track waste trajectory to predict landing location
- Estimate waste size and type
- Generate reports/statistics
- Database logging of littering events
- Alert system for security monitoring

