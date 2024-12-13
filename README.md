# Scene Descriptions for the Visually Impaired
## CSCI 5541 Natural Language Processing Final Project
### Group Name: Sentimantals
#### Members: Mohit Yadav, Alex Besch, Abbas Booshehrain, Ruolei Zeng

For summary or our work please visit our Project WebPage [here](https://mohitydv09.github.io/nlpfinalprojectwebsite/).

For detailed analysis and methodology of our work, see the project final report [here](https://mohitydv09.github.io/nlpfinalprojectwebsite/static/pdfs/report.pdf).

## Demo Output from our Method:

## Installaton

This code was tested with Python 3.12.

Clone this repo in your machine by running the following command in terminal:

```shell
git clone https://github.com/mohitydv09/nlp-final-project.git
cd nlp-final-project
```

Create the conda env
```shell
conda env create -f environment.yml
```

This repo uses OpenAI's ChatGPT for inference and hence OpenAI API Key is required to be stored as evn variable `OPENAI_API_KEY`.

This can be done via following command:
```shell
export OPENAI_API_KEY="YOUR OPENAI API KEY"
```

Check the correct setting of the evn variable by running:
```shell
echo $OPENAI_API_KEY
```

### Downloading sample data to run the code without Depth Camera

Download the test data from [here](https://drive.google.com/drive/folders/1V4nHudH28UQZuzTyz9ETtODgYE5Jx2lJ?usp=share_link) and copy it into the `data/` folder.

Note: The code will automatically download the object detection model and the Vision Language model locally.


## Setting Parameters
The following variables control how the code fuctions. Open the file `main.py` and adjust accordingly:

```shell
## Choose the OpenAI's LLM Model to be used
LLM_MODEL_NAME = 'gpt-4o-mini'

## Set Model Temperature
LLM_TEMPERATURE = 0.0 ## Deterministic

## Set the data input stream
WORKING_WITH_LOCAL_DATA = True 
    # True - Uses local data in ./data folder
    # False - Requires Intel RealSense camera to be connected

## Choose the recoreded data file to run from the downloaded data.
LOCAL_DATA_FILE_PATH = "data/keller_study.npz" 

## Set device
DEVICE = 'cuda' ## 'cpu' or 'cuda'

## Select the Mode of the Product.
MODE = "NAV" 
    # NAV - Navigation assistance
    # VQA - Visual Question Answering
    # SD - Scene Descriptions
```

## Running the code
First, activate your environment. By default, the name is nlp if the environment was created from environment.yml
```
conda activate nlp
```

Run the main python file with the following command once the environment has been activated
```
python3 main.py
```

## Troubleshooting
OpenCV only can be run in the main thread in MacOS. This limitation means the code will run, but the user cannot see the camera output on MacOS.

## Acknowledgements

This project uses several open-source repositories:

- Jocher, G., Qiu, J., & Chaurasia, A. (2023). Ultralytics YOLO (Version 8.0.0) [Computer software]. https://github.com/ultralytics/ultralytics

- Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Ma, C., Jernite, Y., Plu, J., Xu, C., Le Scao, T., Gugger, S., Drame, M., Lhoest, Q., & Rush, A. M. (2020). Transformers: State-of-the-Art Natural Language Processing [Conference paper]. 38–45. https://www.aclweb.org/anthology/2020.emnlp-demos.6

- Intel Corporation. (2024). *librealsense* (Version 2.55.1). GitHub. https://github.com/IntelRealSense/librealsense

- Chase, H. (2022). LangChain [Computer software]. https://github.com/langchain-ai/langchain