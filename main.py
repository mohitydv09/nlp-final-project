import cv2
import json
import time
import threading
import numpy as np
from collections import deque
from typing import List, Tuple

from llm import LLM
from audio_handler import AudioHandler
from camera import RealSenseCamera
from camera_input import CameraInput ## For the camera input from stored data.
from object_detector import ObjectDetector
from image_caption import ImageCaption
from image_vqa import ImageVQA

## Global Variables
## Location Cutoffs
THETA_1 = 80
THETA_2 = 70
R_1 = 2
R_2 = 5
CLOSE_X_CUTOFF = 0.7

DEQUE_MAX_LENGTH = 15
RESPONSE_DEQUE_LENGTH = 5
DEQUE_UPDATE_FREQEUNCY = 1 ## In seconds
LLM_RESPONSE_FREQUENCY = 5 ## In seconds

LLM_MODEL_NAME = 'gpt-4o-mini'
LLM_TEMPERATURE = 0.0 ## Deterministic
WORKING_WITH_LOCAL_DATA = True
LOCAL_DATA_FILE_PATH = "data/keller_study.npz"

DEVICE = 'cuda:0' ## 'cpu' or 'cuda:0'
MODE = "NAV" ## "VQA" "NAV" or "SD"

stop_event = threading.Event()

def scenic_description(camera: RealSenseCamera, vlm: ImageCaption) -> str:
    """Will return the scenic description of the environment"""
    ## Get the RBG Frame from the camera.
    rgb_frame = camera.get_color_frame()
    prompt_for_vlm = "You are looking at"
    description = vlm.get_conditional_caption(rgb_frame, prompt_for_vlm)
    return description

def get_polar_coordinates(x:float, z:float) -> Tuple:
    """Will return the polar coordinates of the object, r in meters and theta in degrees"""
    r = (x**2 + z**2)**0.5
    theta = np.arctan2(z, x)
    return r, np.degrees(theta)

def get_position_label(theta:float, r:float, x:float) -> str:
    """Will return the position of the object based on the polar coordinates"""
    ## Setup Cutoffs
    ### Not considering the Y-axis as it will not be required.
    if r > R_2:
        ## Far objects
        if(theta < THETA_2): #5
            return"to your right"
        elif(theta > 180-THETA_2): #1
            return "to your left"
        elif(theta < THETA_1): #4
            return "slightly to your right"
        elif(theta > 180-THETA_1): #2
            return "slightly to your left"
        else: #3
            return "in front of you"
    elif r > R_1:
        ## Medium dist objects
        if(theta < THETA_2):
            return "to your right"
        elif(theta > 180-THETA_2):
            return "to your left"
        elif(theta < THETA_1):
            return "slightly to your right"
        elif(theta > 180-THETA_1):
            return "slightly to your left"
        else:
            return "in front of you"
    else:
        ## Close objects
        if(x > CLOSE_X_CUTOFF):
            return "to your right"
        elif(x < -CLOSE_X_CUTOFF):
            return "to your left"
        else:
            return "right in front of you"

def structure_yolo_output(yolo_output_list: List[Tuple]) -> str:
    """Will structure the yolo output data in the required format"""
    output_string = ""
    for i, (labels, world_coordinates) in enumerate(yolo_output_list):
        if i == 0:
            output_string += f"Observation at Current TimeStep:\n"
        else:
            output_string += f"Observation {i} second ago:\n"
        for j, (label, world_coordinate) in enumerate(zip(labels, world_coordinates)):
            if world_coordinate[2] == 0.0:
                output_string += f"\t{j}. '{label}' is in the frame but we don't know where\n"
            else:
                r, theta = get_polar_coordinates(world_coordinate[0], world_coordinate[2])
                position = get_position_label(theta, r, world_coordinate[0])
                output_string += f"\t{j}. '{label}' is {position} and {round(r,1)} meters away\n"
    return output_string

def structure_yolo_output_json(yolo_output_list: List[Tuple]) -> json:
    """Will structure the yolo output in the required JSON format"""
    json_data = {}
    json_data['observations'] = []
    for i, (labels, world_coordinates) in enumerate(yolo_output_list):
        timestamp = -i
        json_data['observations'].append({
            "timestamp": timestamp,
            "objects": []
        })
        for j, (label, world_coordinate) in enumerate(zip(labels, world_coordinates)):
            if world_coordinate[2] == 0.0:
                json_data['observations'][i]['objects'].append({
                    "label": label,
                    "position": None,
                    "distance": None
                })
            else:
                r, theta = get_polar_coordinates(world_coordinate[0], world_coordinate[2])
                position = get_position_label(theta, r, world_coordinate[0])
                json_data['observations'][i]['objects'].append({
                    "label": label,
                    "position": position,
                    "distance": round(r,1)
                })
        if json_data['observations'][i]['objects'] == []:
            json_data['observations'][i]['objects'] = None
    return json_data

def get_llm_response(llm : LLM, yolo_output_data: deque[Tuple], llm_response_data: deque[str], audio_handler: AudioHandler) -> str:
    """Will get the response from the LLM model"""
    
    ## Cast data as a list.
    yolo_output_list = list(yolo_output_data)

    ## This is for Making the data in human readable format.
    # structured_data = structure_yolo_output(yolo_output_list)
    # print(structured_data)

    ## Get the JSON data and save it to a file.
    json_data = structure_yolo_output_json(yolo_output_list)
    # with open("./utils/yolo_output_trial.json", "w") as f:
    #     json.dump(json_data, f, indent=4)

    ## Make the System Message:
    system_message = ""
    ## Load the system message from file
    with open("utils/nav_system_prompt_3.txt", "r") as f:
        header_text = f.read()

    with open("utils/example1.json", "r") as f:
        example1 = json.load(f)

    with open("utils/example2.json", "r") as f:
        example2 = json.load(f)

    previous_message = """
    The previous LLM responses with the most recent being first in the list are given:
    """ 
    previous_responses = {}
    previous_responses['previous_AI_responses'] = []
    for i, prev_response in enumerate(list(llm_response_data)):
        previous_responses['previous_AI_responses'].append({
            "timestamp": f"T-{(i+1)*LLM_RESPONSE_FREQUENCY}",
            "response": prev_response
        })

    system_message = header_text + "\n\n" + json.dumps(example1, indent=4) + "\n\n" + json.dumps(example2, indent=4) + "\n\n" +  previous_message + "\n" + json.dumps(previous_responses, indent=4)

    user_message = json.dumps(json_data, indent=4)

    llm_response = llm.generate_response(
        system_message=system_message, 
        user_message=user_message
    )
    # audio_handler.speak(llm_response)
    llm_response_data.appendleft(llm_response)

    return llm_response

def update_deque(camera: RealSenseCamera, 
                 object_detector: ObjectDetector, 
                 yolo_output: deque,
                 llm_response_data: deque,
                 update_frequency: float = 1,
                 visualization: bool = False):
    """Will update the deque with the new data and visualize the data if required""" 
    current_time = time.time()
    deque_lock = threading.Lock()
    while not stop_event.is_set():
        ## Get the RGB and Depth frames
        rgb_frame = camera.get_color_frame()
        depth_frame = camera.get_depth_frame()
        depth_frame = depth_frame * camera.depth_scale
        intrinsics = camera.intrinsics

        labels, pixel_coordinates, world_coordinates = object_detector.get_objects_with_location(rgb_frame, depth_frame, intrinsics)

        if visualization: ## Code visualization is slow as this only runs once per second. Maybe just run this all the time but apend only once per second.
            for label, (x,y), (X,Y,Z) in zip(labels, pixel_coordinates, world_coordinates):
                cv2.putText(rgb_frame, label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                if Z != 0.0:
                    cv2.putText(rgb_frame, f"({X}, {Y}, {Z})", (int(x), int(y+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.circle(rgb_frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            last_llm_response = llm_response_data[0] if llm_response_data else "No Response Yet"
            black_bottom = np.zeros((100, rgb_frame.shape[1], 3), dtype=np.uint8)
            llm_response_list = last_llm_response.split(" ")
            first_line = "LLM Response: " + " ".join(llm_response_list[:8])
            second_line = " ".join(llm_response_list[8:18])
            third_line = " ".join(llm_response_list[18:])
            cv2.putText(black_bottom, first_line, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(black_bottom, second_line, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(black_bottom, third_line, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            # cv2.putText(black_bottom, f"LLM Response: {last_llm_response}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            rgb_frame = np.vstack((rgb_frame, black_bottom))
            cv2.imshow("RGB Image", rgb_frame)
            try: ## If running from local file then wait for 35ms, else wait for 1ms.
                if camera.idx:
                    cv2.waitKey(50)
            except:
                cv2.waitKey(1)

        ## Deque structure
        # 0th element is latest data, and the last element is the oldest data.
        # [(labels, world_coordinates), (labels, world_coordinates), (labels, world_coordinates), ...]

        ## Add the data to the deque.
        if(time.time() - current_time >= update_frequency):
            with deque_lock: ## Will append to left.
                yolo_output.appendleft((labels, world_coordinates))
            current_time = time.time()

def navigation_mode(llm: LLM, yolo_output_data: deque[Tuple], llm_response_data: deque[str], audio_handler: AudioHandler) -> None:
    """Will run the navigation mode"""
    try:
        while True:
            ## Get the LLM response
            start_time = time.time()
            llm_response = get_llm_response(llm, yolo_output_data, llm_response_data, audio_handler)
            print(llm_response)
            time.sleep(max(0, LLM_RESPONSE_FREQUENCY - (time.time() - start_time)))
    except KeyboardInterrupt:
        return

def interactive_vqa(image_vqa, llm, camera, image_caption):
    """
    Continuously prompt the user for input, process the query using BLIP and LLM,
    and provide responses until the user decides to stop.

    Parameters:
    - image_vqa (imageVqa): Instance of the imageVqa class for processing queries.
    - llm (LLM): Instance of the LLM class for generating responses.
    - camera: Camera object to get frames.
    - image_caption (str): Description of the scene provided by the camera.
    """
    # Initialize query history
    query_hist = []

    # Get the RGB frame from the camera
    rgb_frame = camera.get_color_frame()

    while True:
        # Prompt the user for a query
        user_query = input("Enter your question about the scene (type 'stop' to quit): ").strip()

        if user_query.lower() == "stop":
            print("Exiting the program.")
            break

        # Process the query using BLIP
        start_time = time.time()
        blip_response = image_vqa.get_query_response(rgb_frame, user_query)
        print(f"Time taken for query response: {time.time() - start_time}")

        # Generate response using LLM, including the history of queries and responses
        vqa_response = image_vqa.vqa_llm_response(
            llm=llm,
            vlm=blip_response,
            image_caption=image_caption,
            user_query=user_query,
            query_hist=query_hist  # Pass the query history
        )

        # Add the current query and BLIP response to the history
        query_hist.append((user_query, blip_response))

        # Display responses
        print(f"blip_response : {blip_response}")
        print(f"vqa_response : {vqa_response}")

def main():
    """Will Import all the classes and functions from the other files and run the program"""

    ## Initialize the camera and warm it up. 
    ## Keep the visualization off as we will visualize from the update_deque function.
    if WORKING_WITH_LOCAL_DATA:
        camera = CameraInput(from_prestored=True, data_path=LOCAL_DATA_FILE_PATH)
    else:
        camera = RealSenseCamera(visualization=False)
        camera.start()
        while camera.color_frame is None: ## Will be blocking until the camera starts sending frames.
            continue
    audio_handler = AudioHandler(pause_threshold=2.5)
    object_detector = ObjectDetector(device=DEVICE)
    llm = LLM(model_name=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)
    vlm = ImageCaption()
    image_vqa = ImageVQA()

    ## Initialize the deque to store the data, 
    ## Maxlen is set to 30, so that only the last 30 seconds data is stored.
    yolo_output_data = deque(maxlen=DEQUE_MAX_LENGTH)
    llm_response_data = deque(maxlen=RESPONSE_DEQUE_LENGTH)

    ## Update the deque with the new data
    data_update_thread = threading.Thread(target=update_deque, 
                                          args=(camera, 
                                                object_detector, 
                                                yolo_output_data,
                                                llm_response_data, 
                                                DEQUE_UPDATE_FREQEUNCY,     ## Update Frequency
                                                True),                     ## Visualization
                                          daemon=True) ## As daemon is True, you need to clear the resources before exiting the program.
    data_update_thread.start()

    if MODE == "NAV":
        navigation_mode(llm, yolo_output_data, llm_response_data, audio_handler)
    elif MODE == "VQA":
        interactive_vqa(image_vqa, llm, camera, vlm)
    elif MODE == "SD":
        print("Caption generation from the VLM: ", scenic_description(camera, vlm))

    stop_event.set()
    data_update_thread.join()
    cv2.destroyAllWindows
    if not WORKING_WITH_LOCAL_DATA:
        camera.stop()
    

if __name__ == "__main__":
    main()