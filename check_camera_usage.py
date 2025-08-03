#!/usr/bin/env python3
"""
Check Camera Usage and Test Different Backends
"""

import cv2
import time
import threading

def test_camera_backend(backend_name, backend_id):
    print(f"\n🔍 Testing {backend_name} (ID: {backend_id})...")
    
    try:
        # Try to open camera with specific backend
        if backend_id == cv2.CAP_DSHOW:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        elif backend_id == cv2.CAP_MSMF:
            cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
        else:
            cap = cv2.VideoCapture(0, backend_id)
        
        if not cap.isOpened():
            print(f"❌ {backend_name} failed to open camera")
            return False
        
        print(f"✅ {backend_name} opened camera successfully")
        
        # Try to read a few frames
        frame_count = 0
        for i in range(5):
            ret, frame = cap.read()
            if ret:
                frame_count += 1
                print(f"  Frame {i+1}: {frame.shape}")
            else:
                print(f"  Frame {i+1}: Failed to read")
        
        cap.release()
        
        if frame_count > 0:
            print(f"✅ {backend_name} working - read {frame_count}/5 frames")
            return True
        else:
            print(f"❌ {backend_name} failed - no frames read")
            return False
            
    except Exception as e:
        print(f"❌ {backend_name} error: {e}")
        return False

def check_camera_availability():
    print("🔍 Checking Camera Availability...")
    
    # Test different backends
    backends = [
        ("Default", -1),
        ("DirectShow", cv2.CAP_DSHOW),
        ("Media Foundation", cv2.CAP_MSMF),
        ("V4L2", cv2.CAP_V4L2),
        ("GStreamer", cv2.CAP_GSTREAMER)
    ]
    
    working_backends = []
    
    for backend_name, backend_id in backends:
        if test_camera_backend(backend_name, backend_id):
            working_backends.append((backend_name, backend_id))
        time.sleep(1)  # Wait between tests
    
    print(f"\n📊 Results:")
    if working_backends:
        print("✅ Working backends:")
        for name, backend_id in working_backends:
            print(f"  - {name} (ID: {backend_id})")
    else:
        print("❌ No working backends found")
    
    return working_backends

def test_camera_exclusive():
    print("\n🔒 Testing Camera Exclusive Access...")
    
    # Try to open camera multiple times
    caps = []
    max_cameras = 3
    
    for i in range(max_cameras):
        print(f"  Opening camera {i+1}...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if cap.isOpened():
            caps.append(cap)
            print(f"    ✅ Camera {i+1} opened successfully")
        else:
            print(f"    ❌ Camera {i+1} failed to open")
            break
    
    print(f"\n📊 Opened {len(caps)} cameras")
    
    # Try to read from each camera
    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if ret:
            print(f"  Camera {i+1}: Frame shape {frame.shape}")
        else:
            print(f"  Camera {i+1}: Failed to read frame")
        cap.release()
    
    return len(caps)

if __name__ == "__main__":
    print("🎥 Camera Diagnostic Tool")
    print("=" * 40)
    
    # Check available backends
    working_backends = check_camera_availability()
    
    # Test exclusive access
    camera_count = test_camera_exclusive()
    
    print(f"\n📋 Summary:")
    print(f"  - Working backends: {len(working_backends)}")
    print(f"  - Cameras that can be opened simultaneously: {camera_count}")
    
    if camera_count == 0:
        print("\n❌ No cameras can be opened. Possible issues:")
        print("  - Camera is being used by another application")
        print("  - Camera drivers are not installed")
        print("  - Camera is not connected")
    elif camera_count == 1:
        print("\n✅ Camera is working but only one application can use it")
    else:
        print("\n✅ Camera supports multiple simultaneous connections") 