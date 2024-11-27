import cv2
import json
import time
import threading
from collections import deque
from typing import List, Tuple

from llm import LLM
from camera import RealSenseCamera
from camera_input import cameraInput ## For the camera input from stored data.
from object_detector import objectDetector

def structure_yolo_output(yolo_output_list: List[Tuple]) -> json:
    """Will structure the yolo output in the required format"""
    structured_data = {}
    for i, (labels, world_coordinates) in enumerate(yolo_output_list):
        timestamp_key = f"Time Stamp: {i + 1 - len(yolo_output_list)}"  # Create the timestamp key
        structured_data[timestamp_key] = {}  # Ensure the timestamp key exists

        for j, (label, world_coordinate) in enumerate(zip(labels, world_coordinates)):
            structured_data[timestamp_key][f"Object {j}"] = {
                "Label": label,
                "Location": f"X: {world_coordinate[0]}, Y: {world_coordinate[1]}, Z: {world_coordinate[2]}"
            }
    return structured_data


def get_llm_response(llm : LLM, yolo_output_data: deque[Tuple]) -> str:
    """Will get the response from the LLM model"""
    
    ## Cast data as a list.
    yolo_output_list = list(yolo_output_data)

    print(yolo_output_list)

    ## Structure the data in the required format.
    # structured_data = structure_yolo_output(yolo_output_list)

    return "Hello"


def data_json(data: deque):
    data_list = list(data)  # Convert deque to list
    data_dict = {}  # Initialize the dictionary

    # Output Format:
    # for i, (labels, _, world_coordinates) in enumerate(data_list):
    #     timestamp_key = f"Time Stamp: {i + 1 - len(data_list)}"  # Create the timestamp key
    #     data_dict[timestamp_key] = {}  # Ensure the timestamp key exists

    #     for j, (label, world_coordinate) in enumerate(zip(labels, world_coordinates)):
    #         data_dict[timestamp_key][f"Object {j}"] = {
    #             "Label": label,
    #             "Location": f"X: {world_coordinate[0]}, Y: {world_coordinate[1]}, Z: {world_coordinate[2]}"
    #         }

    string_data = ""
    for i, (labels, _, world_coordinates) in enumerate(data_list):
        timestamp_key = f"Time Stamp: {i + 1 - len(data_list)}"  # Create the timestamp key
        
        for j, (label, world_coordinate) in enumerate(zip(labels, world_coordinates)):
            string_data += f"{timestamp_key}, Object {j}: {label}, Location: X: {world_coordinate[0]}, Y: {world_coordinate[1]}, Z: {world_coordinate[2]}\n"

        # data_dict[timestamp_key] = {}  # Ensure the timestamp key exists

        # for j, (label, world_coordinate) in enumerate(zip(labels, world_coordinates)):
        #     data_dict[timestamp_key][f"Object {j}"] = {
        #         "Label": label,
        #         "Location": f"X: {world_coordinate[0]}, Y: {world_coordinate[1]}, Z: {world_coordinate[2]}"
        #     }

    print(string_data)
    
    yolo_output.append(data_dict)

    with open("data_new.json", "w") as f:
        json.dump(string_data, f)

    return data_dict

def update_deque(camera: RealSenseCamera, 
                 object_detector: objectDetector, 
                 yolo_output: deque,
                 visualization: bool = False):
    """Will update the deque with the new data and visualize the data if required""" 
    current_time = time.time()
    deque_lock = threading.Lock()
    while True:
        ## Get the RGB and Depth frames
        rgb_frame = camera.get_color_frame()
        depth_frame = camera.get_depth_frame()
        depth_frame = depth_frame * camera.depth_scale
        intrinsics = camera.intrinsics

        labels, pixel_coordinates, world_coordinates = object_detector.get_objects_with_location(rgb_frame, depth_frame, intrinsics)

        if visualization: ## Code visualization is slow as this only runs once per second. Maybe just run this all the time but apend only once per second.
            for label, (x,y), (X,Y,Z) in zip(labels, pixel_coordinates, world_coordinates):
                cv2.putText(rgb_frame, label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(rgb_frame, f"({X}, {Y}, {Z})", (int(x), int(y+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.circle(rgb_frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.imshow("RGB Image", rgb_frame)
            cv2.waitKey(1)

        ## Add the data to the deque.
        if(time.time() - current_time >= 1):
            with deque_lock:
                yolo_output.append((labels, world_coordinates))
            print("Len of deque: ", len(yolo_output))
            current_time = time.time()

## Implement as thread
def update_deque_no_cam(camera: RealSenseCamera, object_detector: objectDetector, data: deque[Tuple]):
    """Will update the deque with the new data""" 
    start_time = time.time()

    ## Get the RGB and Depth frames
    # rgb_frame = camera.get_color_frame()
    # depth_frame = camera.get_depth_frame()
    intrinsics_dict = camera.intrinsics
    depth_scale = camera.image_details["depth_scale"]
    rgb_frame, depth_frame = camera.get_rgbd_frame()
    depth_frame = depth_frame * depth_scale
    intrinsics = camera.intrinsics

    labels, pixel_coordinates, world_coordinates = object_detector.get_objects_with_location(rgb_frame, depth_frame, intrinsics)

    for label, (x,y), (X,Y,Z) in zip(labels, pixel_coordinates, world_coordinates):
        cv2.putText(rgb_frame, label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(rgb_frame, f"({X}, {Y}, {Z})", (int(x), int(y+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(rgb_frame, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.imshow("RGB Image", rgb_frame)

    data.append((labels, pixel_coordinates, world_coordinates))
    data_json_format = data_json(data)

    print(data_json_format)

    time.sleep(max(0, (1 - (time.time() - start_time))))

    cv2.waitKey(1)

def main():
    """Will Import all the classes and functions from the other files and run the program"""

    ## Initialize the camera and warm it up. 
    ## Keep the visualization off as we will visualize from the update_deque function.
    camera = RealSenseCamera(visualization=False)
    camera.start()
    while camera.color_frame is None: ## Will be blocking until the camera starts sending frames.
        continue

    object_detector = objectDetector(device='cuda:0')
    llm = LLM(model_name='gpt-4o-mini', temperature=0.5)

    ## Initialize the deque to store the data, 
    ## Maxlen is set to 30, so that only the last 30 seconds data is stored.
    yolo_output_data = deque(maxlen=30)

    ## Update the deque with the new data
    data_update_thread = threading.Thread(target=update_deque, 
                                          args=(camera, object_detector, yolo_output_data, True), ## Visualization is True
                                          daemon=True) ## As daemon is True, you need to clear the resources before exiting the program.
    data_update_thread.start()

    try:
        while True:
            time.sleep(0.001)
    except KeyboardInterrupt:
        data_update_thread.join()
        cv2.destroyAllWindows
        camera.stop()

if __name__ == "__main__":
    main()