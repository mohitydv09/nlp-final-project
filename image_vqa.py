import time
import torch
import numpy as np
from llm import LLM
from transformers import AutoProcessor, BlipForQuestionAnswering

from camera_input import CameraInput
from image_caption import ImageCaption

class ImageVQA():
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(self.device)

    def get_query_response(self, rgb_image: np.ndarray, user_query: str) -> str:
        inputs = self.processor(images=rgb_image, text=user_query, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=100)
        return self.processor.decode(output[0], skip_special_tokens=True)

    def vqa_llm_response(self, llm: LLM, vlm: str, user_query: str, image_caption: str, query_hist: list) -> str:
        """Modify VQA response based on user query using LLM model."""

        # Include query history in the message
        history = "\n".join([f"Query: {q}, Response: {r}" for q, r in query_hist])

        # Combine the scene description, history, and user query
        user_message = f"""
        The description of the space the user is looking at right now is: {image_caption}.
        The history of previous queries and vlm responses are:
        {history}
        Knowing the description, the user wants to know: {user_query}.
        The following response was provided by vlm: {vlm}.
        """

        ## Make the System Message:
        system_message = """
        You are an assistive AI designed to help a visually impaired person (blind person) navigate and understand their surroundings. 
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

        Always remain empathetic and supportive, remembering that the user is blind and relies entirely on your responses to navigate and interact with the world.
        """

        vqa_response = llm.generate_response(
            system_message=system_message,
            user_message=user_message,
        )

        return vqa_response


if __name__ == "__main__":

    LLM_MODEL_NAME = 'gpt-4o-mini'
    LLM_TEMPERATURE = 0.5
    WORKING_WITH_LOCAL_DATA = True
    llm = LLM(model_name=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)
    
    # Initialize variables
    camera_input = CameraInput()
    rgb_frame = camera_input.get_color_frame()

    image_vqa = ImageVQA()
    image_caption = "you are looking at a man standing in a room with a whiteboard"
    query_hist = []  # List to store the history of queries and responses

    while True:
        # Continuously prompt the user for input
        user_query = input("Enter your question about the scene (type 'stop' to quit): ").strip()

        if user_query.lower() == "stop":
            print("Exiting the program.")
            break

        start_time = time.time()
        blip_response = image_vqa.get_query_response(rgb_frame, user_query)
        print(f"Time taken for query response: {time.time() - start_time}")

        vqa_response = image_vqa.vqa_llm_response(
            llm=llm,
            vlm=blip_response,
            image_caption=image_caption,
            user_query=user_query,
            query_hist=query_hist
        )

        # Add query and response to history
        query_hist.append((user_query, blip_response))

        # Display responses
        print(f"blip_response : {blip_response}")
        print(f"vqa_response : {vqa_response}")
