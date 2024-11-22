import os 
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

class LLM():
    def __init__(self, model_name: str = 'gpt-3.5-turbo', temperature:float=0.5) -> None:
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

    system_message = "You are an Indian person. Reply back in Hinglish."
    user_message = "Hey, how are you doing today?"

    response = llm.generate_response(
        system_message=system_message,
        user_message=user_message
    )
    print("ChatGPT-3.5 Response: ",response)