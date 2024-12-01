import os
from llm import LLM

def classify_request_with_llm(user_input, llm):
    """
    Use the ChatGPT API to classify the user input into one of the predefined classes:
    0: VQM (Visual Quality Management)
    1: Help reach an object in a small environment
    2: Navigation on the sidewalk
    If the input does not match any of the classes, return None.
    """
    system_message = """
    You are an assistant designed to classify user requests into one of the following categories:
    - Class 0: VQM (Visual Quality Management) — requests related to describing surroundings or spatial details.
    - Class 1: Help reach an object in a small environment — requests related to finding or guiding to specific objects in confined spaces.
    - Class 2: Navigation on the sidewalk — requests related to navigating outdoors, particularly on sidewalks or paths.
    If the user's input doesn't match any of these categories, respond with "Out of scope".
    Provide the classification as one of the following responses:
    - "Class 0"
    - "Class 1"
    - "Class 2"
    - "Out of scope"
    """
    # Use the LLM to generate a classification
    response = llm.generate_response(system_message=system_message, user_message=user_input)
    return response.strip()

def main():
    input_file = "data/question.txt"
    output_file = "data/answer.txt"

    # Ensure the input file exists
    if not os.path.exists(input_file):
        print("Input file not found.")
        return

    # Read the user's input
    with open(input_file, "r") as file:
        user_input = file.read().strip()

    # Initialize the LLM
    llm = LLM(model_name='gpt-3.5-turbo', temperature=0.5)

    # Classify the input using LLM
    classification = classify_request_with_llm(user_input, llm)

    # Handle the classification result
    if classification in ["Class 0", "Class 1", "Class 2"]:
        # Extract the class number from the response
        class_number = int(classification.split()[-1])
        print(f"Classified as: {class_number}")
    else:
        # Write the response to the output file
        with open(output_file, "w") as file:
            file.write("The request is beyond this application's capability at this moment.")
        print("Request is beyond the application's capability.")

if __name__ == "__main__":
    main()
