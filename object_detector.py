
class ObjectDetector:
    def __init__(self):
        raise NotImplementedError
    
if __name__ == "__main__":
    pass


## TESTING ##
import cv2
import numpy as np

data_dict = np.load('data/data.npy', allow_pickle=True).item()

# print(data_dict.keys()) 

# first_frame = (data_dict["data"][100,:,:,:3]/255).astype(np.uint8)
first_frame = (data_dict["data"][100, :, :, :3]).astype(np.uint8)
depth_frame = data_dict["data"][0,:,:,3]

# print("Shape of data: ", data_dict["data"].shape)

# print(np.max(first_frame))
# print(np.min(first_frame))

# print(np.max(depth_frame))
# print(np.min(depth_frame))
# print(np.median(depth_frame))


# cv2.imshow("frame", first_frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

##############################################################################################
# Importing the YOLO v11 shit to detect objects
# Going to use a standard model, not fine tune another
from ultralytics import YOLO
from PIL import Image
from urllib.request import urlopen
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import time

class objectDetector:
    def __init__(self, model_name, model_folder):
        # Ensure model directory exists
        os.makedirs(model_folder, exist_ok=True)

        # Construct the model path
        model_path = os.path.join(model_folder, model_name)
        print(f"Model path: {model_path}")

        # Check if model file exists; if not, download and save it
        if not os.path.exists(model_path):
            try:
                print(f"Model {model_name} not found. Downloading...")
                self.model = YOLO(model_name)  # Download model by name
                self.model.save(model_path)  # Save the model to the specified path
                print("Model download complete.")
            except Exception as e:
                print(f"Error downloading model: {e}")
                raise FileNotFoundError(f"The model {model_name} could not be downloaded.")
        else:
            print(f"Using existing model: {model_name}")
            self.model = YOLO(model_path)  # Load the existing model

    def get_bounding_boxes(self, rgb_image, conf=0.2, show_plt=False, printouts=False):
        t_start = time.time()

        # Predict bounding boxes for the provided image
        results = self.model.predict(rgb_image, conf=conf)

        detected_objects = []

        # Annotate and display results
        for result in results:
            if show_plt:
                annotated_frame = result.plot()
                plt.imshow(annotated_frame)
                plt.axis('off')
                plt.show()
            
        
            for box in result.boxes:
                class_id = int(box.cls.item())  # Convert tensor to int
                # Inspect box.xyxy structure
                # print(f"box.xyxy: {box.xyxy}, type: {type(box.xyxy)}")

                # Convert coordinates to integers
                coordinates = box.xyxy.flatten().tolist() if hasattr(box.xyxy, 'flatten') else box.xyxy.tolist()
                obj_data = {
                    'label_id': class_id,
                    'label_name': self.model.names[class_id],
                    'confidence': box.conf.item(),
                    'coordinates': [int(coord) for coord in coordinates],  # Convert coordinates to integers
                    'depth': None # Convert coordinates to integers
                }
                # print(obj_data)
                detected_objects.append(obj_data)
        
        t_end = time.time()
        if printouts:
            print(f"Time to extract bounding boxes: {t_end - t_start:.2f} seconds")
        
        return detected_objects

    def get_depth_data(self, depth_image, bounding_boxes):
        for obj in bounding_boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = obj['coordinates']
            
            # Ensure the bounding box coordinates are within the image dimensions
            x1, x2 = max(0, x1), min(depth_image.shape[1], x2)
            y1, y2 = max(0, y1), min(depth_image.shape[0], y2)
            
            # Extract depth data for the bounding box region
            depth_values = depth_image[y1:y2, x1:x2]

            # Filter for valid depth values greater than 0
            valid_depth_values = depth_values[depth_values > 0]
            
            if valid_depth_values.size > 0:  # Check if there are valid depth values
                # Grab 25 random depth values from the region
                depth_values_sample = np.random.choice(valid_depth_values, size=min(25, valid_depth_values.size))
                # Compute the median depth value for the region
                median_depth = np.median(depth_values_sample)
                obj['depth'] = median_depth * 0.00025 # Convert to meters
            else:
                obj['depth'] = None  # Or set to a default value, like 0

        return bounding_boxes
        

# Test with a sample image from the internet
# url = 'https://ultralytics.com/images/zidane.jpg'
# img = Image.open(urlopen(url))
# rgb_frame = np.array(img)
rgb_frame = np.array(first_frame)
depth_frame = np.array(depth_frame)

object_detector = objectDetector(model_name="yolo11n.pt", model_folder="./models")

objects = object_detector.get_bounding_boxes(rgb_frame, printouts=True)

objects = object_detector.get_depth_data(depth_frame, objects)

print(objects)
