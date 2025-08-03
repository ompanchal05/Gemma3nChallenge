#!/usr/bin/env python3
"""
Test Camera Before Starting Flask
"""

import cv2
import time

def test_camera_before_flask():
    print("🔍 Testing camera before starting Flask...")
    
    # Try to open camera with DirectShow
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("❌ Failed to open camera")
        return False
    
    print("✅ Camera opened successfully")
    
    # Try to read a few frames
    frame_count = 0
    for i in range(3):
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            print(f"  Frame {i+1}: {frame.shape}")
        else:
            print(f"  Frame {i+1}: Failed to read")
        time.sleep(0.1)
    
    cap.release()
    
    if frame_count > 0:
        print(f"✅ Camera test successful - {frame_count}/3 frames read")
        return True
    else:
        print("❌ Camera test failed - no frames read")
        return False

if __name__ == "__main__":
    success = test_camera_before_flask()
    if success:
        print("\n🎉 Camera is ready for Flask!")
    else:
        print("\n⚠️  Camera may not work with Flask") 