import speech_recognition as sr

def live_speech_to_text():
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 2.5  # Default is 0.8 seconds
    recognizer.non_speaking_duration = 2.0  # Default is 0.5 seconds
    recognizer.dynamic_energy_threshold = True
    with sr.Microphone() as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Listening for your question. Press Ctrl+C to stop.")

        try:
            print("Start speaking...")
            # Listen until the user pauses
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
            print("Processing speech...")
            
            # Convert audio to text
            question = recognizer.recognize_google(audio)
            print(f"You asked: {question}")

            # Save the question to a file
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

if __name__ == "__main__":
    live_speech_to_text()
