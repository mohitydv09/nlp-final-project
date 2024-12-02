import os
import subprocess
from gtts import gTTS
from playsound import playsound
import speech_recognition as sr

class Talker:
    def __init__(self, pause_threshold: float = 1) -> None:
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = pause_threshold
        self.recognizer.non_speaking_duration = 2.0
        self.recognizer.dynamic_energy_threshold = True

    def speak(self, text: str) -> None:
        tts = gTTS(text=text, lang='en')
        audio_file_path = "utils/temp.mp3"
        tts.save(audio_file_path)
        subprocess.run(['mpg123', audio_file_path], 
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        
    def listen(self) -> str:
        with sr.Microphone() as source:
            print("Adjusting for ambient noise... Please wait.")
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            print("Listening for your question. Press Ctrl+C to stop.")
            try:
                print("Start speaking...")
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=10)
                print("Processing speech...")
                question = self.recognizer.recognize_google(audio)
                print(f"You asked: {question}")
                with open("data/question.txt", "w") as file:
                    file.write(question)
                return question
            except sr.UnknownValueError:
                print("Sorry, I could not understand the audio. Please try again.")
            except sr.RequestError as e:
                print(f"Speech Recognition Error: {e}")
            except KeyboardInterrupt:
                print("\nStopped listening.")
                return None

if __name__ == '__main__':
    talker = Talker(pause_threshold=2.5)
    talker.speak("Hello, I am a speaker.")
    talker.listen()