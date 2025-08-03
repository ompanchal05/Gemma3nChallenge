#!/usr/bin/env python3
"""
Optimized Camera Module for Fast Opening and Proper Visibility
"""

import cv2
import time
import logging
import threading
from typing import Optional, Tuple
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedCamera:
    """Optimized camera class for fast opening and proper visibility"""
    
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.is_initialized = False
        self.frame_width = 640
        self.frame_height = 480
        self.fps = 30
        self.backend = cv2.CAP_DSHOW  # Use DirectShow for faster opening
        
    def initialize_camera(self) -> bool:
        """Initialize camera with optimized settings for fast opening"""
        try:
            logger.info("🚀 Fast camera initialization starting...")
            
            # Try DirectShow first (fastest on Windows)
            self.cap = cv2.VideoCapture(self.camera_index, self.backend)
            
            if not self.cap.isOpened():
                logger.warning("DirectShow failed, trying default backend...")
                self.cap = cv2.VideoCapture(self.camera_index)
                
            if not self.cap.isOpened():
                logger.error("❌ Failed to open camera with all backends")
                return False
            
            # Set optimized camera properties for fast opening
            self._set_camera_properties()
            
            # Quick test to ensure camera is working
            if self._test_camera():
                self.is_initialized = True
                logger.info("✅ Camera initialized successfully with optimized settings")
                return True
            else:
                logger.error("❌ Camera test failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Camera initialization error: {e}")
            return False
    
    def _set_camera_properties(self):
        """Set optimized camera properties for fast opening and proper visibility"""
        try:
            # Set resolution for fast processing
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            
            # Set FPS for smooth video
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Optimize for speed
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for low latency
            
            # Auto-exposure and focus for better visibility
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Auto exposure
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Auto focus
            
            # Set brightness to ensure visibility (avoid black screen)
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 1)  # Maximum brightness
            self.cap.set(cv2.CAP_PROP_CONTRAST, 1)    # Maximum contrast
            self.cap.set(cv2.CAP_PROP_SATURATION, 1)  # Maximum saturation
            
            # Set auto exposure
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Enable auto exposure
            self.cap.set(cv2.CAP_PROP_EXPOSURE, 0)  # Reset exposure to auto mode
            
            logger.info("📷 Camera properties set for optimal visibility")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not set all camera properties: {e}")
    
    def _test_camera(self) -> bool:
        """Quick test to ensure camera is working properly"""
        try:
            # Read a few frames quickly to test
            for i in range(3):
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logger.error(f"❌ Frame {i+1} read failed")
                    return False
                
                # Check frame dimensions
                if frame.shape[0] == 0 or frame.shape[1] == 0:
                    logger.error(f"❌ Invalid frame dimensions: {frame.shape}")
                    return False
                
                time.sleep(0.1)  # Small delay between frames
            
            logger.info(f"✅ Camera test passed - Frame shape: {frame.shape}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Camera test error: {e}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame with error handling"""
        if not self.is_initialized or self.cap is None:
            return False, None
        
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.warning("⚠️ Frame read failed, attempting to reinitialize...")
                if self._reinitialize():
                    ret, frame = self.cap.read()
                    return ret, frame
                return False, None
            
            # Check if frame is valid (not black)
            if frame is not None and frame.size > 0:
                # Apply minimal enhancement to avoid black screen
                frame = self._enhance_frame(frame)
                return True, frame
            else:
                logger.warning("⚠️ Invalid frame received (empty or black)")
                return False, None
            
        except Exception as e:
            logger.error(f"❌ Frame read error: {e}")
            return False, None
    
    def _enhance_frame(self, frame):
        """Enhance frame for better visibility"""
        try:
            # Check if frame is too dark
            if frame.mean() < 30:  # If average pixel value is too low
                # Apply histogram equalization to improve contrast
                frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
                frame_yuv[:,:,0] = cv2.equalizeHist(frame_yuv[:,:,0])
                frame = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2BGR)
                
                # Increase brightness if still too dark
                frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)
            return frame
            
        except Exception as e:
            logger.warning(f"⚠️ Frame enhancement failed: {e}")
            return frame
    
    def _reinitialize(self) -> bool:
        """Reinitialize camera if it fails"""
        try:
            logger.info("🔄 Reinitializing camera...")
            self.release()
            time.sleep(0.5)  # Wait before reinitializing
            return self.initialize_camera()
        except Exception as e:
            logger.error(f"❌ Camera reinitialization failed: {e}")
            return False
    
    def get_frame_info(self) -> dict:
        """Get current camera frame information"""
        if not self.is_initialized or self.cap is None:
            return {"error": "Camera not initialized"}
        
        try:
            info = {
                "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": self.cap.get(cv2.CAP_PROP_FPS),
                "backend": "DirectShow" if self.backend == cv2.CAP_DSHOW else "Default",
                "is_opened": self.cap.isOpened(),
                "buffer_size": self.cap.get(cv2.CAP_PROP_BUFFERSIZE)
            }
            return info
        except Exception as e:
            return {"error": str(e)}
    
    def set_resolution(self, width: int, height: int):
        """Set camera resolution"""
        try:
            self.frame_width = width
            self.frame_height = height
            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                logger.info(f"📐 Resolution set to {width}x{height}")
        except Exception as e:
            logger.error(f"❌ Failed to set resolution: {e}")
    
    def set_fps(self, fps: int):
        """Set camera FPS"""
        try:
            self.fps = fps
            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FPS, fps)
                logger.info(f"🎬 FPS set to {fps}")
        except Exception as e:
            logger.error(f"❌ Failed to set FPS: {e}")
    
    def is_ready(self) -> bool:
        """Check if camera is ready to use"""
        return self.is_initialized and self.cap is not None and self.cap.isOpened()
    
    def release(self):
        """Release camera resources"""
        try:
            if self.cap:
                self.cap.release()
                self.cap = None
            self.is_initialized = False
            logger.info("🔒 Camera released")
        except Exception as e:
            logger.error(f"❌ Error releasing camera: {e}")

# Global camera instance
_global_camera = None

def get_camera() -> OptimizedCamera:
    """Get or create global camera instance"""
    global _global_camera
    if _global_camera is None:
        _global_camera = OptimizedCamera()
    return _global_camera

def initialize_camera() -> bool:
    """Initialize global camera instance"""
    camera = get_camera()
    return camera.initialize_camera()

def read_frame():
    """Read frame from global camera instance"""
    camera = get_camera()
    return camera.read_frame()

def release_camera():
    """Release global camera instance"""
    global _global_camera
    if _global_camera:
        _global_camera.release()
        _global_camera = None

# Fast camera test function
def test_camera_fast():
    """Fast camera test for quick verification"""
    print("🚀 Fast Camera Test")
    print("=" * 30)
    
    camera = OptimizedCamera()
    
    start_time = time.time()
    if camera.initialize_camera():
        init_time = time.time() - start_time
        print(f"✅ Camera initialized in {init_time:.2f} seconds")
        
        # Test frame reading
        start_time = time.time()
        ret, frame = camera.read_frame()
        read_time = time.time() - start_time
        
        if ret and frame is not None:
            print(f"✅ Frame read in {read_time:.3f} seconds")
            print(f"📐 Frame shape: {frame.shape}")
            
            # Get camera info
            info = camera.get_frame_info()
            print(f"📊 Camera info: {info}")
            
            camera.release()
            return True
        else:
            print("❌ Frame read failed")
            camera.release()
            return False
    else:
        print("❌ Camera initialization failed")
        return False

if __name__ == "__main__":
    # Test the camera module
    success = test_camera_fast()
    if success:
        print("\n🎉 Camera module is working perfectly!")
    else:
        print("\n⚠️ Camera module needs attention")
