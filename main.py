import cv2

from camera import CameraInput
from object_detector import ObjectDetector
from LLM import LLM
from realsense_camera import RSCamera


def main():
    """Will Import all the classes and functions from the other files and run the program"""
    camera_input = CameraInput()  # Create an instance of the CameraInput class
    object_detector = ObjectDetector()  # Create an instance of the ObjectDetector class

    for i in range(500):
        frame = camera_input.get_rgb_frame()
        object_detector = object_detector.detect(frame)
        

if __name__ == "__main__":
    main()