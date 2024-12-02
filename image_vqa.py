from re import T
import time
import torch
import numpy as np
from llm import LLM
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration

from camera_input import cameraInput
from image_caption import imageCaption

LLM_MODEL_NAME = 'gpt-4o-mini'
LLM_TEMPERATURE = 0.5
# Initialize the LLM model
llm = LLM(model_name=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)

def vqa_llm_response(llm: LLM, vlm: str, user_query: str) -> str:
    """modify vqa response based on user query using LLM model"""
    
    # Combine the scene description and user query
    user_message = f"The following scene was described by vlm: '{vlm}'. The user wants to know: {user_query}"

    ## Make the System Message:
    system_message = """
    You are an assistive AI designed to help a visually impaired person navigate and understand their surroundings. 
    You receive a description of what the camera installed in their eyeglasses sees, along with the user's query about the scene.

    Your primary goal is to:
    1. Interpret the description of the scene (provided as the user_message) and tailor your response to the user's specific query.
    2. Be concise, clear, and relevant to the user's question. Avoid unnecessary details unless the user asks for more information.
    3. Provide actionable and accessible information for someone who cannot see, such as describing objects, actions, or hazards in a way that helps the user understand their environment.

    The description from the camera might not directly answer the user's query. In such cases:
    - Extract useful insights from the scene description.
    - Deduce and infer additional relevant information to respond meaningfully.

    For example:
    - If the scene description mentions "a long hallway with a blue railing" and the user asks, "What's going on in front of me?", your response might be, "You are in a hallway with blue railings; it's empty and safe to walk forward."

    Always remain empathetic and supportive, remembering that the user relies entirely on your responses to navigate and interact with the world.
    """

    vqa_response = llm.generate_response(
        system_message=system_message, 
        user_message=user_message,
    )

    return vqa_response

if __name__ == "__main__":

    ## Get the image from the camera
    camera_input = cameraInput()
    rgb_frame = camera_input.get_color_frame()

    ## Get the image caption
    image_caption = imageCaption()

    start_time = time.time()
    user_query = "what is going on in front of me?"
    unconditional_caption = image_caption.get_unconditional_caption(rgb_frame)
    print(f"Time taken for unconditional caption: {time.time() - start_time}")
    vqa_response = vqa_llm_response(
        llm=llm,
        vlm=unconditional_caption,
        user_query=user_query
    )
    print(f"unconditional_caption : {unconditional_caption}")
    print(f"vqa_response : {vqa_response}")
