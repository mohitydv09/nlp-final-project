import os 
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import time

class LLM():
    def __init__(self, model_name: str = 'gpt-4o-mini', temperature:float=0.5) -> None:
        self.openai_api_key = os.environ["OPENAI_API_KEY"]
        if self.openai_api_key is None:
            print("Please set the OPENAI_API_KEY environment variable.")
            exit()
        self.model = ChatOpenAI(
            model=model_name, 
            temperature=temperature,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=self.openai_api_key,
        )
        
    def generate_response(self, system_message: str="", user_message: str="") -> str:
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message)
        ]
        response = self.model.invoke(messages)
        return response.content

if __name__ == "__main__":
    llm = LLM(model_name='gpt-4o-mini', temperature=0.5)

    system_message = """
    You are an assistant designed to provide concise and precise spatial guidance
    for visually impaired individuals. Your task is to describe the location 
    and inferred motion of objects around the user based on time-series data. 
    Follow these rules:

    Processing Instructions:
    Infer Motion:
    Compare object positions across consecutive timestamps.
    Determine motion based on distance changes:
    If an object is getting closer, label it as "approaching."
    If an object is getting farther, label it as "moving away."
    If the position remains nearly constant, label it as "static."
    Prioritize Relevance:
    Focus on objects closest to the user or those approaching.
    Ignore static or distant objects unless critical to the user's surroundings.
    Simplify Directions:
    Use terms like "in front," "to your left," "to your right," and "behind."
    Reference relative directions based on the x, y, z coordinates.
    Be Brief and Dynamic:
    Limit descriptions to one or two sentences per object.
    Highlight only changes or critical information for the user.
    Group Similar Objects:
    If multiple objects of the same type (e.g., "chair") are present, summarize their locations.
    Input Format:
    You will receive a nested JSON structure with timestamps, object labels,
      and their 3D positions over time (e.g., {"Object 0": {"Label": "chair", "Location": "X: 0.5, Y: 0.2, Z: 0.8"}}).

    Output Format:
    Provide concise guidance by describing:

    The object type.
    Its position relative to the user.
    Its inferred motion (approaching, moving away, static).
    Example:

    Input:
    {
    "Time Stamp: -1": {
        "Object 0": {"Label": "chair", "Location": "X: 0.5, Y: 0.2, Z: 2.0"},
        "Object 1": {"Label": "dog", "Location": "X: -1.0, Y: 0.0, Z: 1.5"}
    },
    "Time Stamp: 0": {
        "Object 0": {"Label": "chair", "Location": "X: 0.4, Y: 0.2, Z: 1.8"},
        "Object 1": {"Label": "dog", "Location": "X: -1.0, Y: 0.0, Z: 1.2"}
    }
    }
    Output:
    "A chair is approaching slightly from in front of you. A dog is quickly approaching from your left."
    """

    user_message = """
        Time Stamp: -4, Object 0: chair, Location: X: -0.7, Y: 1.1, Z: 6.9\nTime Stamp: -4, Object 1: clock, Location: X: -1.6, Y: -1.3, Z: 8.7\nTime Stamp: -4, Object 2: chair, Location: X: 0.9, Y: 0.7, Z: 2.0\nTime Stamp: -3, Object 0: chair, Location: X: -0.5, Y: 0.6, Z: 4.3\nTime Stamp: -3, Object 1: clock, Location: X: -1.6, Y: -1.6, Z: 8.2\nTime Stamp: -3, Object 2: tv, Location: X: 0.0, Y: -0.0, Z: 0\nTime Stamp: -2, Object 0: chair, Location: X: -0.3, Y: 0.7, Z: 3.5\nTime Stamp: -2, Object 1: clock, Location: X: -1.1, Y: -1.5, Z: 7.7\nTime Stamp: -2, Object 2: tv, Location: X: 1.1, Y: -0.5, Z: 3.7\nTime Stamp: -1, Object 0: chair, Location: X: -0.1, Y: 0.8, Z: 3.2\nTime Stamp: -1, Object 1: clock, Location: X: -0.6, Y: -1.3, Z: 7.3\nTime Stamp: 0, Object 0: chair, Location: X: -0.1, Y: 0.6, Z: 2.7\nTime Stamp: 0, Object 1: clock, Location: X: -1.0, Y: -1.5, Z: 6.5\nTime Stamp: 0, Object 2: tv, Location: X: 0.0, Y: -0.0, Z: 0\n
    """

    start_time = time.time()

    response = llm.generate_response(
        system_message=system_message,
        user_message=user_message
    )

    end_time = time.time()
    print("LLM Response: ",response)
    print("Time taken: ", end_time-start_time)