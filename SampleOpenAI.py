# import openai, if it is not installed, run "pip install openai==0.28.0"
# this exact method was depricated after openai==0.28.0, so if you have a newer version, you will have to use the new method
# you should be able to Chat-GPT the new method if this one will not work.
import openai

# check the version of the openai library
print(openai.__version__)

# Importing ALex's Key
openai.api_key = "blah blah blah"

# !openai migrate

input_text = "I am happy"

response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "system", "content": " Given text, predict the sentiment of the text. The predicted label should be either Positive, or Negative."},
        {"role": "user", "content": "Input: " + str(input_text) + "; Output:"},
        ]
    )

print(response.choices[0].message.content)