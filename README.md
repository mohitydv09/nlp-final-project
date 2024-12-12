# Scene Descriptions for the Visually Impaired
## CSCI 5541 Natural Language Processing Final Project
### Group Name: Sentimantals
#### Members: Mohit Yadav, Alex Besch, Abbas Boosherain, Ruolei Zeng

[paper](https://www.overleaf.com/project/65b17a76caac86c24bbaae5d), [project page](https://mohitydv09.github.io/nlpfinalprojectwebsite/)

## Demos:
[![Sample Run 1](https://img.youtube.com/vi/bQnBfadSGAU/0.jpg)](https://www.youtube.com/watch?v=bQnBfadSGAU)

[![Sample Run 2](https://img.youtube.com/vi/WyWYLxGRPOc/0.jpg)](https://www.youtube.com/watch?v=WyWYLxGRPOcU)

# Running the code
## Downloading all the required files
In your working directory, ensure you have the following folders:
- data
    - fill the data with the folder located [here](https://drive.google.com/drive/folders/1V4nHudH28UQZuzTyz9ETtODgYE5Jx2lJ?usp=sharing) (you must be logged into a @umn.edu email).
- models
    - Additional models will download as necessary, such as yolov11n

## Preparing your environment
Create the conda env
```
conda env create -f environment.yml
```

Ensure your openai key is stored to your environment by typing the following into terminal:
```
export OPENAI_API_KEY="{OPENAI API KEY}"
```

## Setting parameters
The following variables will control how the code funcitons
```shell
LLM_MODEL_NAME = 'gpt-4o-mini' 
    # Choose any chagpt model

LLM_TEMPERATURE = 0.0 ## Deterministic

WORKING_WITH_LOCAL_DATA = True 
    # True - Uses local data in ./data folder
    # False - Requires Intel RealSense camera be connected

LOCAL_DATA_FILE_PATH = "data/keller_study.npz" 
    # Choose any file within ./data/ folder

DEVICE = 'cuda:0' ## 'cpu' or 'cuda:0'
    # CPU for computer without Nvidia GPU
    # cuda:0 for computer with Nvidia GPU

MODE = "NAV" 
    # NAV - Navigation assistance
    # VQA - Visual Question Answering
    # SD - Scene Descriptions
```

## Motivation:
This project seeks to address challenges faced by visually impaired individuals by providing real-time, contextually relevant scene descriptions that enable better spatial awareness and navigation. Leveraging advancements in Vision and Language Models (VLMs), our goal is to develop a system that translates visual information into accessible language, prioritizing functional details within the user's immediate surroundings. This real-time scene description system will provide users with audible updates on important elements within their environment, such as nearby people or obstacles, supporting safer and more confident navigation.

