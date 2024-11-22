import cv2

from camera_input import cameraInput
from object_detector import objectDetector
from llm import LLM
from realsense_camera import RSCamera


def main():
    """Will Import all the classes and functions from the other files and run the program"""
    camera_input = cameraInput()  # Create an instance of the CameraInput class
    object_detector = objectDetector(model_name="yolo11n.pt", model_folder="./models")  # Create an instance of the ObjectDetector class

if __name__ == "__main__":
    main()