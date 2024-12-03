import time
import torch
import numpy as np
import cv2

from camera_input import CameraInput

from transformers import BlipProcessor, BlipForConditionalGeneration

class ImageCaption():
    def __init__(self)->None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(self.device)

    def get_unconditional_caption(self, rgb_image:np.ndarray)->str:
        inputs = self.processor(rgb_image, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=100)
        return self.processor.decode(output[0], skip_special_tokens=True)
    
    def get_conditional_caption(self, rgb_image:np.ndarray, text:str="a photography of")->str:
        inputs = self.processor(rgb_image, text, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=100)
        return self.processor.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":

    ## Get the image from the camera
    camera_input = CameraInput()
    rgb_frame = camera_input.get_color_frame()

    ## Get the image caption
    image_caption = ImageCaption()

    start_time = time.time()    
    unconditional_caption = image_caption.get_unconditional_caption(rgb_frame)
    print(f"Time taken for unconditional caption: {time.time() - start_time}")
    print(f"Unconditional caption: {unconditional_caption}")

    start_time = time.time()
    conditional_caption = image_caption.get_conditional_caption(rgb_frame)
    print(f"Time taken for conditional caption: {time.time() - start_time}")
    print(f"Conditional caption: {conditional_caption}")
