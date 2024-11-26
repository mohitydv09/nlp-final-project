import cv2
import time
import threading
import numpy as np
import pyrealsense2 as rs

class RealSenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.intrinsics = None  

        self.color_frame = None
        self.depth_frame = None
        self.running = False
        self.thread = threading.Thread(target=self._update_frames, daemon=True)

    def start(self):
        """Start the camera and the thread."""
        self.running = True
        self.pipeline.start(self.config)
        self.thread.start()

    def stop(self):
        """Stop the camera and the thread."""
        self.running = False
        self.thread.join()
        self.pipeline.stop()

    def _update_frames(self):
        """Threaded function to update frames continuously."""
        while self.running:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if color_frame and depth_frame:
                # Convert frames to numpy arrays
                self.color_frame = np.asanyarray(color_frame.get_data())
                self.depth_frame = np.asanyarray(depth_frame.get_data())

                # Show the live feed
                cv2.imshow("RealSense - Color", self.color_frame)
                cv2.waitKey(1)

    def get_color_frame(self):
        """Get the current color frame."""
        return self.color_frame

    def get_depth_frame(self):
        """Get the current depth frame."""
        return self.depth_frame

if __name__ == "__main__":
    camera = RealSenseCamera()
    camera.start()

    try:
        while True:
            time.sleep(0.01)
    except KeyboardInterrupt:
        camera.stop()

