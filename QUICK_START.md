# Quick Start & Troubleshooting Guide

## ⚡ Quick Start

### 1. Run the Application
```bash
python main.py
```

### 2. Watch the Output
- **Green Box**: Person holding waste
- **Yellow Box**: Person just released waste (counting down to littering alert)
- **Red Box + Console Alert**: Waste confirmed as littered!

### 3. Stop the Application
Press **ESC** key to exit

## 📝 Expected Output

```
Application to detect hand and waste with littering detection
Press ESC to exit

[Video window opens with detection boxes]

⚠️  LITTERED! Waste #1 has been thrown!
   Separated for 2.05 seconds

⚠️  LITTERED! Waste #2 has been thrown!
   Separated for 2.15 seconds
```

## ⚙️ Configuration Examples

### Example 1: Sensitive Detection (Lab/Controlled)
```python
# In main.py, change:
waste_tracker = WasteTracker(
    separation_threshold=50,        # Lower = more sensitive
    littering_time_threshold=1.0    # Fast detection
)
```

### Example 2: Robust Detection (Outdoor/Crowded)
```python
# In main.py, change:
waste_tracker = WasteTracker(
    separation_threshold=120,       # Higher = less sensitive
    littering_time_threshold=3.0    # Slower detection
)
```

### Example 3: Balanced (Default)
```python
# Already configured in main.py:
waste_tracker = WasteTracker(
    separation_threshold=80,
    littering_time_threshold=2.0
)
```

## 🎥 Changing Video Input

### Use Webcam Instead of Video File
```python
# In main.py, change line 17:
cap = cv2.VideoCapture(0)  # 0 = default webcam
```

### Use Different Video File
```python
# In main.py, change line 17:
cap = cv2.VideoCapture("path/to/your/video.mp4")
```

### Use Network Camera
```python
# In main.py, change line 17:
cap = cv2.VideoCapture("rtsp://camera-ip:port/stream")
```

## ✅ Verification Checklist

- [ ] Python 3.7+ installed
- [ ] All dependencies installed: `pip install -r requirements.txt`
- [ ] Video file exists at `photos/input4.mp4` (or update path)
- [ ] YOLO model at `objectdetection/aimodel/best.pt` exists
- [ ] No error when running `python main.py`
- [ ] Video window opens and displays detections

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'objectdetection.waste_tracking'"

**Solution**: Ensure `waste_tracking.py` exists in `objectdetection/` folder
```bash
# Check if file exists:
dir objectdetection\waste_tracking.py
```

### Issue: No Hand Detection (No Purple Boxes)

**Solutions**:
1. Ensure good lighting
2. Check MediaPipe confidence: hands.py line 6 (increase `min_detection_confidence`)
3. Get hand fully in frame
4. Try webcam instead of video: `cv2.VideoCapture(0)`

```python
# In hands.py, make it less strict:
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,  # Lowered from 0.6
    min_tracking_confidence=0.5
)
```

### Issue: No Waste Detection (No Green Boxes)

**Solutions**:
1. Ensure waste is clearly visible
2. Check YOLO model is properly trained for your waste types
3. Check confidence in waste.py

### Issue: Too Many False Littering Alerts

**Solutions**:
1. Increase `littering_time_threshold`:
   ```python
   WasteTracker(separation_threshold=80, littering_time_threshold=3.0)
   ```
2. Increase `separation_threshold`:
   ```python
   WasteTracker(separation_threshold=120, littering_time_threshold=2.0)
   ```

### Issue: Missing Littering Alerts

**Solutions**:
1. Decrease `littering_time_threshold`:
   ```python
   WasteTracker(separation_threshold=80, littering_time_threshold=1.0)
   ```
2. Decrease `separation_threshold`:
   ```python
   WasteTracker(separation_threshold=50, littering_time_threshold=2.0)
   ```

### Issue: Video Plays Too Fast/Slow

**Solution**: The application processes at natural video speed. No change needed.

### Issue: High CPU Usage

**Solutions**:
1. Lower video resolution
2. Reduce frame processing rate (add in main loop):
   ```python
   frame_count = 0
   if frame_count % 2 == 0:  # Process every 2nd frame
       hand_results = detecthands(frame)
       waste_results = detectwaste(frame)
   frame_count += 1
   ```

## 📊 Performance Metrics

| Component | CPU Usage | Notes |
|-----------|-----------|-------|
| Hand Detection | ~10-15% | MediaPipe (optimized) |
| Waste Detection | ~30-50% | YOLO (larger model) |
| Tracking | <1% | Simple distance calc |
| Rendering | ~5-10% | Drawing boxes/text |
| **Total** | **~45-75%** | Depends on resolution |

## 🧪 Testing Scenarios

### Scenario 1: Single Hand, Single Waste
```
Expected:
1. Green box when holding
2. Yellow box when released
3. Red box + alert after 2 seconds
```

### Scenario 2: Multiple Waste Items
```
Expected:
1. Each waste tracked independently
2. Alerts trigger separately for each item
3. IDs increment (Waste#1, Waste#2, etc.)
```

### Scenario 3: Hand Moves Near Waste Without Holding
```
Expected:
1. Stays green if distance < threshold
2. No false positive littering alert
```

### Scenario 4: Quick Pick Up and Drop
```
Expected:
1. Yellow timer starts
2. Resets to green if picked up within 2s
3. Only littered if held separately > 2s
```

## 📈 Tuning Guide

**Start with default settings (80, 2.0)**

If too sensitive (false positives):
- Increase separation_threshold to 100
- Increase littering_time_threshold to 3.0

If not sensitive enough (missed detections):
- Decrease separation_threshold to 60
- Decrease littering_time_threshold to 1.0

Fine-tune in small increments (±10 pixels, ±0.5 seconds)

## 🐛 Debug Mode

To add debug output, modify main.py:

```python
# Add after update() call:
print(f"Active waste objects: {len(tracking_data['active_waste'])}")
for waste_id, waste_info in tracking_data['active_waste'].items():
    print(f"  Waste#{waste_id}: {waste_info}")
```

## 💾 Saving Output

To save the annotated video:

```python
# In main.py, after cv2.VideoCapture:
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, 
                      (int(cap.get(3)), int(cap.get(4))))

# After cv2.imshow(), add:
out.write(frame)

# In cleanup section, add:
out.release()
```

## 📞 Support

For issues or questions:
1. Check the error message carefully
2. Review LITTERING_DETECTION.md for feature details
3. Review SYSTEM_ARCHITECTURE.md for how it works
4. Check troubleshooting section above
5. Adjust configuration parameters

## ✨ Success Indicators

✅ Video plays without crashes  
✅ Purple boxes appear on hands  
✅ Green boxes appear on waste  
✅ Yellow boxes appear when waste is released  
✅ Red boxes and alerts appear after 2 seconds  
✅ Console shows "⚠️ LITTERED!" messages  

If all the above work, your system is properly configured!

