from gtts import gTTS
import os

def text_to_speech(answer_path):
    # Read the answer
    with open(answer_path, "r") as file:
        answer = file.read().strip()

    print(f"Speaking: {answer}")

    # Convert text to speech
    tts = gTTS(text=answer, lang="en")
    tts.save("data/answer.mp3")

    # Play the audio file
    os.system("start data/answer.mp3" if os.name == "nt" else "afplay data/answer.mp3")

if __name__ == "__main__":
    answer_path = "data/answer.txt"  # Path to the answer text file
    text_to_speech(answer_path)
