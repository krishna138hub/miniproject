# System Architecture & Flow Diagram

## Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    main.py                                   │
│            Main Application Entry Point                      │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
    ┌────────┐  ┌────────┐  ┌─────────────────┐
    │ detect │  │ detect │  │  WasteTracker   │
    │ hands  │  │ waste  │  │   (NEW)         │
    └────────┘  └────────┘  └────────┬────────┘
        │            │               │
        └────────────┼───────────────┘
                     ▼
        ┌─────────────────────────────┐
        │  Frame Processing Loop       │
        │  1. Get hand bounding boxes  │
        │  2. Get waste bounding boxes │
        │  3. Calculate distances      │
        │  4. Track states             │
        │  5. Detect littering         │
        └─────────────┬───────────────┘
                      ▼
        ┌─────────────────────────────┐
        │     Output Generation       │
        ├─────────────────────────────┤
        │ • Visual: Color-coded boxes │
        │ • Console: Alert messages   │
        │ • Data: Tracking info       │
        └─────────────────────────────┘
```

## Waste Tracking State Machine

```
                    ┌──────────────┐
                    │  UNTRACKED   │
                    └───────┬──────┘
                            │
                    (waste detected)
                            │
                            ▼
                   ┌─────────────────┐
                   │    ATTACHED     │◄────┐
                   │ (Green Box)     │     │
                   └────────┬────────┘     │
                            │          (hand
                    (hand released)  attaches)
                            │             │
                            ▼             │
                   ┌─────────────────┐    │
                   │   SEPARATED     ├────┘
                   │ (Yellow Box)    │
                   │ Timer Running   │
                   └────────┬────────┘
                            │
        ┌───────────────────┴────────────────────┐
        │                                        │
   (timer < threshold)              (timer >= threshold)
        │                                        │
        │                                        ▼
        │                              ┌──────────────────┐
        │                              │    LITTERED      │
        │                              │  (Red Box)       │
        │                              │  ALERT SENT      │
        │                              └──────────────────┘
        │
   (stays in separated)
        │
        └────────────────────┬─────────────────┘
                             │
                      (until hand attaches
                       or waste leaves)
```

## Distance Calculation Formula

```
Hand & Waste Detection
        │
        ├─ Hand: (x1_h, y1_h, x2_h, y2_h) 
        │   Center: (cx_h, cy_h)
        │
        └─ Waste: (x1_w, y1_w, x2_w, y2_w)
           Center: (cx_w, cy_w)

Distance = √[(cx_h - cx_w)² + (cy_h - cy_w)²]

If Distance < separation_threshold:
    Status = ATTACHED
    Timer = Reset to None
Else:
    Status = SEPARATED
    If (current_time - timer_start) > littering_time_threshold:
        Status = LITTERED
        Send Alert
```

## Real-time Processing Pipeline

```
Video Frame
    │
    ├─────────────────────────────────────────────┐
    │                                             │
    ▼                                             ▼
Hand Detection              Waste Detection
(MediaPipe)                (YOLO model)
    │                                             │
    ▼                                             ▼
Hand Bounding Boxes        Waste Bounding Boxes
   (0-2 per frame)            (0-N per frame)
    │                                             │
    └─────────────────────┬───────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │   WasteTracker.update()          │
        ├─────────────────────────────────┤
        │ 1. Compare distances             │
        │ 2. Update state timers           │
        │ 3. Detect littering              │
        │ 4. Return tracking_data          │
        └─────────────┬───────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────┐
        │   draw_waste_status()            │
        ├─────────────────────────────────┤
        │ • Draw ATTACHED → Green Box      │
        │ • Draw SEPARATED → Yellow Box    │
        │ • Draw LITTERED → Red Box        │
        │ • Add labels with IDs            │
        └─────────────┬───────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────┐
        │   Display & Alert                │
        ├─────────────────────────────────┤
        │ • cv2.imshow() - visual          │
        │ • print() - console alert       │
        │ • return tracking_data - data   │
        └─────────────────────────────────┘
```

## Configuration Parameters Impact

```
separation_threshold (pixels)
    │
    ├─ Lower (50): ┐
    │              ├─ More sensitive, frequent alerts
    │              │ Good for controlled environments
    ├─ Default (80): ┐
    │                ├─ Balanced detection
    ├─ Higher (100+): ┐
    │                 ├─ Less sensitive, fewer false alerts
    │                 │ Good for outdoor/cluttered areas

littering_time_threshold (seconds)
    │
    ├─ Lower (1.0): ┐
    │               ├─ Fast littering detection
    │               │ May have false positives
    ├─ Default (2.0): ┐
    │                 ├─ Balanced temporal confirmation
    ├─ Higher (3.0+): ┐
    │                 ├─ Slow detection
    │                 │ Reduces false positives
```

## Data Flow Example

```
Frame 0-50:
    Hand holds Waste #1
    Distance = 30px
    Status = ATTACHED ✓ (green)
    
Frame 51:
    Hand releases Waste #1
    Distance = 120px
    Status = SEPARATED (yellow)
    Timer_start = t_51

Frame 52-100:
    Hand moves away
    Waste #1 stays in place
    Distance = 150px
    Status = SEPARATED (yellow)
    Time_elapsed = 0.5s - 1.6s (< 2.0s)
    
Frame 101:
    Time_elapsed = 2.0s
    Status = LITTERED ✓ (red)
    Console Alert:
    "⚠️  LITTERED! Waste #1 has been thrown!"
    "   Separated for 2.02 seconds"
```

## Multiple Objects Example

```
Frame N:
    ┌─────────────────────────────────────────┐
    │  Hand1 ──┐                              │
    │          └─ Waste#1 (ATTACHED, green)   │
    │                                          │
    │  Hand2 ──┐                              │
    │          └─ Waste#2 (ATTACHED, green)   │
    │                                          │
    │              Waste#3 (SEPARATED, yellow)│
    │              Timer: 1.5s                │
    │                                          │
    │              Waste#4 (LITTERED, red) ⚠️ │
    │              Separated: 2.1s            │
    └─────────────────────────────────────────┘

All tracked independently and simultaneously!
```

