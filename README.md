# Scene Descriptions for the Visually Impaired
## CSCI 5541 Natural Language Processing Final Project
### Group Name: Sentimantals
#### Members: Mohit Yadav, Alex Besch, Abbas Boosherain, Ruolei Zeng

## Introduction and Motivation:
This project seeks to address challenges faced by visually impaired individuals by providing real-time, contextually relevant scene descriptions that enable better spatial awareness and navigation. Leveraging advancements in Vision and Language Models (VLMs), our goal is to develop a system that translates visual information into accessible language, prioritizing functional details within the user's immediate surroundings. This real-time scene description system will provide users with audible updates on important elements within their environment, such as nearby people or obstacles, supporting safer and more confident navigation.

## Problem Formulation

We aim to develop a language descriptor system capable of processing RGB-D (Red, Green, Blue, Depth) images and generating concise, relevant descriptions in natural language. This system will operate with limited latency, enhancing its suitability for real-time, assistive applications. The intended output will be a seamless narrative that emphasizes objects and people within the user’s immediate vicinity, thus enhancing situational awareness and spatial context.

Below image shows the mathematical modeling of our project:

<img width="1402" alt="Screenshot 2024-10-31 at 18 28 19" src="https://github.com/user-attachments/assets/ada3581d-5699-44c8-a184-4f40738c068a">

## Current Work

To work on the project we needed a continrous video stream from the RGBD camera, as the hardware in not available to everyone all the time we created a sample dataset with a small RGBD video for anyone to use. To mimic a video stream we created a Camera Input class which can be called in code to mimic a video stream from sample data in ```camera_input.py```. We then use this input to and implement a YOLO object detector('''object_detector.py''') on each frame to get label and bounding boxes on the objects that are present in the scene. Using the bounding boxes we randomly sample 25 points inside the bounding box and get the correspoiding depths for the points. We take the median of these depths to get the depth of the object. Given the depth and the pixel coordinates of the ojects we project them into the 3D scene using camera intrinsics matrix. 

We obtain a label and 3D locaiton of the item w.r.t to the camera, a sample of output is shown below.

https://github.com/user-attachments/assets/ee370484-6318-4173-ba56-5a4658194edb

We have also implemented a BLIP Image captioning model to give a general descption of the scnee in ```image_caption.py```. A sample input-output of the same is given below:

Input Image to VLM(Human in Image is Blurred):

![image-2](https://github.com/user-attachments/assets/adc742af-810f-4515-903f-49a36d6106ad)

VLM Output:

**Unconditional caption:** there is a man standing in a room with a whiteboard

**Conditional caption:** _a photography of_ a man standing in a room with a whiteboard


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


