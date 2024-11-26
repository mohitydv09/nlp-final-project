from camera import RealSenseCamera
from object_detector import objectDetector

from collections import deque

def update_deque(camera: RealSenseCamera, object_detector: objectDetector, data: deque):
    """Will update the deque with the new data"""
    start_time = time.time()

    ## Get the RGB and Depth frames
    rgb_frame = camera.get_color_frame()
    depth_frame = camera.get_depth_frame()
    intrinsics = camera.intrinsics



def main():
    """Will Import all the classes and functions from the other files and run the program"""
    ## Start the camera
    camera = RealSenseCamera()
    camera.start()

    object_detector = objectDetector()
    data = deque(maxlen=30)

    



if __name__ == "__main__":
    main()