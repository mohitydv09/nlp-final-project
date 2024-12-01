import os 
import time
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

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
    llm = LLM(model_name='gpt-3.5-turbo', temperature=0.5)

    system_message = """
        You are an agent trying to determine which function to call based on a query of what someone would like to do. 
        Your options of functions is listed below:
        1. HelpGrabAnObject() - When this funciton is called, audio plays to help the user grab the closest object
        2. Navigation()- This plays audio every 5 seconds to explain the persons scene and help them navigate the world
        3. QuestionAnswering() - Answers questions about what the person is seeing.

        Your job is to return a one-hot vector for which function to call; possible responses are:
        1. [1, 0, 0] for HelpGrabAnObject()
        2. [0, 1, 0] for Navigation()
        3. [0, 0, 1] for QuestionAnswering()

        Only return the one-hot vector
    """

    user_message = """
        Help me walk down the hallway
    """
    start_time = time.time()
    response = llm.generate_response(
        system_message=system_message,
        user_message=user_message
    )

    print("LLM Response: ",response)