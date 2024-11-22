import cv2

from camera_input import cameraInput
from object_detector import objectDetector
from llm import LLM
from realsense_camera import RSCamera
from langchain_core.tools import tool

@tool
def scene_description() -> str:
    """Will get the image from the camera and run the object detection model on it"""
    image = cameraInput.get_frame() ## Get the image from the camera
    return vlm.generate_response(user_message=image) ## Run the object detection model on the image



def main():
    """Will Import all the classes and functions from the other files and run the program"""
    camera_input = cameraInput()  # Create an instance of the CameraInput class
    object_detector = objectDetector(model_name="yolo11n.pt", model_folder="./models")  # Create an instance of the ObjectDetector class

    ## Get a frame from camera.
    frame = camera_input.get_rgb_frame()
    print("Shape of frame: ", frame.shape)

if __name__ == "__main__":
    main()