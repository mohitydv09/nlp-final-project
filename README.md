# Scene Descriptions for the Visually Impaired
## CSCI 5541 Natural Language Processing Final Project
### Group Name: Sentimantals
#### Members: Mohit Yadav, Alex Besch, Abbas Booshehrain, Ruolei Zeng

[paper](https://www.overleaf.com/project/65b17a76caac86c24bbaae5d), [project page](https://mohitydv09.github.io/nlpfinalprojectwebsite/)

## Demos:

<div align="center">
  <a href="https://www.youtube.com/watch?v=bQnBfadSGAU">
    <img src="https://img.youtube.com/vi/bQnBfadSGAU/0.jpg" alt="Sample Run 1">
  </a>
  <br>
  <a href="https://www.youtube.com/watch?v=WyWYLxGRPOcU">
    <img src="https://img.youtube.com/vi/WyWYLxGRPOc/0.jpg" alt="Sample Run 2">
  </a>
</div>


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
The following variables will control how the code funcitons. Open the file `main.py` and adjust the following parameters on lines 30-36

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

## Motivation:
This project seeks to address challenges faced by visually impaired individuals by providing real-time, contextually relevant scene descriptions that enable better spatial awareness and navigation. Leveraging advancements in Vision and Language Models (VLMs), our goal is to develop a system that translates visual information into accessible language, prioritizing functional details within the user's immediate surroundings. This real-time scene description system will provide users with audible updates on important elements within their environment, such as nearby people or obstacles, supporting safer and more confident navigation.

# Acknowldgements 
We would like to extend our sincere gratitude to the following individuals for their support:
- Dr. Karthik, for generously allowing us to utilize his laboratory facilities, providing us with a conducive environment to conduct our research.
- Grant Besch, for granting us access to his computer resources, which significantly facilitated our project's computational requirements.

Their contributions have been invaluable to our project, and we appreciate their kindness and willingness to support our endeavors.

# References
```
@misc{li2022blipbootstrappinglanguageimagepretraining,
      title={BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation}, 
      author={Junnan Li and Dongxu Li and Caiming Xiong and Steven Hoi},
      year={2022},
      eprint={2201.12086},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2201.12086}, 
}

@misc{intelrealsense,
  author = {Intel Corporation},
  title = {Intel RealSense SDK},
  year = {2022},
  howpublished = {\url{https://www.intelrealsense.com/developers/}}
}

@misc{openai_chatgpt_2024,
    author = {OpenAI},
    title = {ChatGPT-4},
    year = {2024},
    howpublished = {\url{https://openai.com}},
    note = {Accessed: 2024-12-05}
}

@misc{yolo_creation,
      title={You Only Look Once: Unified, Real-Time Object Detection}, 
      author={Joseph Redmon and Santosh Divvala and Ross Girshick and Ali Farhadi},
      year={2016},
      eprint={1506.02640},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1506.02640}, 
}

@article{opencv,
  title={The OpenCV Library},
  author={Gary Bradski},
  journal={Dr. Dobb's Journal of Software Tools},
  year={2000}
}
```