import time
import cv2
from camera import RealSenseCamera
from object_detector import objectDetector
from camera_input import cameraInput
from collections import deque

# def data_json(data: deque): ## Fix Bug.
#     data = list(data)
#     data_dict = {}
#     for i,(labels, _, world_coordinates) in enumerate(data):
#         for j, (label, world_coordinate) in enumerate(zip(labels, world_coordinates)):
#             data_dict[f"Time Stamp: {len(data_dict)-i-1}"][f"Object {j}"] = {"Label: ": label, "Location: ": f"X: {world_coordinate[0]}, Y: {world_coordinate[1]}, Z: {world_coordinate[2]}"}
#     print(data_dict)

yolo_output = []

def data_json(data: deque):
    data_list = list(data)  # Convert deque to list
    data_dict = {}  # Initialize the dictionary

    for i, (labels, _, world_coordinates) in enumerate(data_list):
        timestamp_key = f"Time Stamp: {i + 1 - len(data_list)}"  # Create the timestamp key
        data_dict[timestamp_key] = {}  # Ensure the timestamp key exists

        for j, (label, world_coordinate) in enumerate(zip(labels, world_coordinates)):
            data_dict[timestamp_key][f"Object {j}"] = {
                "Label": label,
                "Location": f"X: {world_coordinate[0]}, Y: {world_coordinate[1]}, Z: {world_coordinate[2]}"
            }
    
    yolo_output.append(data_dict)



## Implement as thread
def update_deque(camera: RealSenseCamera, object_detector: objectDetector, data: deque):
    """Will update the deque with the new data""" 
    start_time = time.time()

    ## Get the RGB and Depth frames
    rgb_frame = camera.get_color_frame()
    depth_frame = camera.get_depth_frame()
    depth_frame = depth_frame * camera.depth_scale
    intrinsics = camera.intrinsics

    labels, pixel_coordinates, world_coordinates = object_detector.get_objects_with_location(rgb_frame, depth_frame, intrinsics)

    for label, (x,y), (X,Y,Z) in zip(labels, pixel_coordinates, world_coordinates):
        cv2.putText(rgb_frame, label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(rgb_frame, f"({X}, {Y}, {Z})", (int(x), int(y+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(rgb_frame, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.imshow("RGB Image", rgb_frame)

    data.append((labels, pixel_coordinates, world_coordinates))
    data_json_format = data_json(data)

    time.sleep(max(0, (1 - (time.time() - start_time))))

    cv2.waitKey(1)
    
## Implement as thread
def update_deque_no_cam(camera: RealSenseCamera, object_detector: objectDetector, data: deque):
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

    time.sleep(max(0, (1 - (time.time() - start_time))))

    cv2.waitKey(1)

def main():
    """Will Import all the classes and functions from the other files and run the program"""
    ## Start the camera
    # camera = RealSenseCamera(visualization=False)
    # camera.start()

    # while camera.color_frame is None:
    #     continue

    camera = cameraInput(from_prestored=True)

    object_detector = objectDetector()
    data = deque(maxlen=5)

    try:
        while True:
            for i in range(20):
                camera.get_rgb_frame()
            update_deque_no_cam(camera, object_detector, data)
            print("Length of Data: ",len(data))
            # time.sleep(0.01)
    except KeyboardInterrupt:
        import json

        with open("data_new.json", "w") as f:
            json.dump(yolo_output, f)
        cv2.destroyAllWindows()
        camera.stop()



if __name__ == "__main__":
    data = deque(maxlen=30)
    main()