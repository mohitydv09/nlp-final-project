import cv2
import json
import time
import threading
import numpy as np
## import inflect
from collections import deque
from typing import List, Tuple

from llm import LLM
from camera import RealSenseCamera
from camera_input import cameraInput ## For the camera input from stored data.
from object_detector import objectDetector
from image_caption import imageCaption

## Global Variables
## Location Cutoffs
THETA_1 = 80
THETA_2 = 70
R_1 = 2
R_2 = 5
CLOSE_X_CUTOFF = 0.7

DEQUE_MAX_LENGTH = 30
DEQUE_UPDATE_FREQEUNCY = 1 ## In seconds
LLM_RESPONSE_FREQUENCY = 5 ## In seconds

LLM_MODEL_NAME = 'gpt-4o-mini'
LLM_TEMPERATURE = 0.5

DEVICE = 'cuda:0' ## 'cpu' or 'cuda:0'
stop_event = threading.Event()

def scenic_description(camera: RealSenseCamera, vlm: imageCaption) -> str:
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
        elif(theta > THETA_1): #4
            return "slightly to your right"
        elif(theta < 180-THETA_1): #2
            return "slightly to your left"
        else: #3
            return "in front of you"
    elif r > R_1:
        ## Medium dist objects
        if(theta < THETA_2):
            return "to your right"
        elif(theta > 180-THETA_2):
            return "to your left"
        elif(theta > THETA_1):
            return "slightly to your right"
        elif(theta < 180-THETA_1):
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

def get_llm_response(llm : LLM, yolo_output_data: deque[Tuple]) -> str:
    """Will get the response from the LLM model"""
    
    ## Cast data as a list.
    yolo_output_list = list(yolo_output_data)

    ## Structure the data in the required format.
    structured_data = structure_yolo_output(yolo_output_list)
    print(structured_data)

    ## Get the JSON data and save it to a file.
    json_data = structure_yolo_output_json(yolo_output_list)
    with open("./utils/yolo_output_trial.json", "w") as f:
        json.dump(json_data, f, indent=4)

    # ## Load the system message from file
    # with open("utils/nav_system_prompt_1.txt", "r") as f:
    #     system_message = f.read()

    # ## Create User Message from the structured data
    # user_message = f"Hello {structured_data}"

    # llm_response = llm.generate_response(
    #     system_message=system_message, 
    #     user_message=user_message
    # )

    # return llm_response

def update_deque(camera: RealSenseCamera, 
                 object_detector: objectDetector, 
                 yolo_output: deque,
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
                cv2.putText(rgb_frame, f"({X}, {Y}, {Z})", (int(x), int(y+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.circle(rgb_frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.imshow("RGB Image", rgb_frame)
            cv2.waitKey(1)

        ## Deque structure
        # 0th element is latest data, and the last element is the oldest data.
        # [(labels, world_coordinates), (labels, world_coordinates), (labels, world_coordinates), ...]

        ## Add the data to the deque.
        if(time.time() - current_time >= update_frequency):
            with deque_lock: ## Will append to left.
                yolo_output.appendleft((labels, world_coordinates))
            current_time = time.time()

        ## If Running from local file then slow down the loop at 30fps.
        ## Won't be perfect 30fps, but will be close.
        if camera.idx is not None: ## Means running from local file.
            time.sleep(max(0, (1/30 - (time.time() - current_time))))

def navigation_mode(llm: LLM, yolo_output_data: deque[Tuple]) -> None:
    """Will run the navigation mode"""
    try:
        while True:
            ## Get the LLM response
            llm_response = get_llm_response(llm, yolo_output_data)
            time.sleep(LLM_RESPONSE_FREQUENCY)
    except KeyboardInterrupt:
        return

def main():
    """Will Import all the classes and functions from the other files and run the program"""

    ## Initialize the camera and warm it up. 
    ## Keep the visualization off as we will visualize from the update_deque function.
    working_with_local_data = True
    if working_with_local_data:
        camera = cameraInput(from_prestored=True)
    else:
        camera = RealSenseCamera(visualization=False)
        camera.start()
        while camera.color_frame is None: ## Will be blocking until the camera starts sending frames.
            continue

    object_detector = objectDetector(device=DEVICE)
    llm = LLM(model_name=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)
    vlm = imageCaption()

    ## Initialize the deque to store the data, 
    ## Maxlen is set to 30, so that only the last 30 seconds data is stored.
    yolo_output_data = deque(maxlen=DEQUE_MAX_LENGTH)

    ## Update the deque with the new data
    data_update_thread = threading.Thread(target=update_deque, 
                                          args=(camera, 
                                                object_detector, 
                                                yolo_output_data, 
                                                DEQUE_UPDATE_FREQEUNCY,     ## Update Frequency
                                                False),                     ## Visualization
                                          daemon=True) ## As daemon is True, you need to clear the resources before exiting the program.
    data_update_thread.start()

    ## Run the navigation mode
    # navigation_mode(llm, yolo_output_data)

    ## Run the Scene Description
    description = scenic_description(camera, vlm)
    print(description)

    ## Clean Up the resources
    stop_event.set()
    data_update_thread.join()
    cv2.destroyAllWindows
    if not working_with_local_data:
        camera.stop()


from langchain.agents import create_react_agent, Tool, AgentExecutor
from langchain.prompts import PromptTemplate

def sum_fn(x,y):
    return x+y

def product_fn(x,y):
    return x*y

def agent_main():
    LLM_TEMPERATURE = 0.01
    llm_wrapper = LLM(model_name=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)
    llm = llm_wrapper.model

    sumTool = Tool(
        name='sum',
        func=sum_fn,
        description='Sum two numbers',
    )

    multiplyTool = Tool(
        name='multiply',
        func=product_fn,
        description='Multiply two numbers',
    )
    
    tools = [sumTool, multiplyTool]

    prompt = PromptTemplate(
        prompt="I don't know what is the sum of 2 and 3"
    )

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )
    agent_executer = AgentExecutor(agent, tools=tools)

    agent_executer.invoke(prompt)

    print(agent_executer.response)

if __name__ == "__main__":
    agent_main()