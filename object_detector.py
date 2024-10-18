
class ObjectDetector:
    def __init__(self):
        raise NotImplementedError
    
if __name__ == "__main__":
    pass


## TESTING ##
import cv2
import numpy as np

data_dict = np.load('data/data.npy', allow_pickle=True).item()

print(data_dict.keys()) 

# first_frame = (data_dict["data"][100,:,:,:3]/255).astype(np.uint8)
first_frame = (data_dict["data"][100, :, :, :3]).astype(np.uint8)
depth_frame = data_dict["data"][0,:,:,3]

print("Shape of data: ", data_dict["data"].shape)

print(np.max(first_frame))
print(np.min(first_frame))

print(np.max(depth_frame))
print(np.min(depth_frame))
print(np.median(depth_frame))


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

# Test with a sample image from the internet
# url = 'https://ultralytics.com/images/zidane.jpg'
# img = Image.open(urlopen(url))
# rgb_frame = np.array(img)
rgb_frame = np.array(first_frame)

# Check image shape
print("Image shape:", rgb_frame.shape)

# Load the pre-trained YOLOv8 model
model = YOLO('./models/yolov8n.pt')

# Get class names from the model
class_names = model.names

# Run object detection with a lower confidence threshold to detect more objects
results = model.predict(rgb_frame)

# Check for detections and draw bounding boxes if any
for result in results:
    if result.boxes is None or len(result.boxes) == 0:
        print("No detections found")
    else:
        print(f"Number of detections: {len(result.boxes)}")
        for box in result.boxes:
            # Extract box coordinates (converted to int for drawing)
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            confidence = box.conf[0].cpu().numpy()  # Confidence score
            class_id = int(box.cls[0].cpu().numpy())  # Class label

            # Get the class name using the class ID
            class_name = class_names[class_id]

            print(f"Bounding box: [{x1}, {y1}, {x2}, {y2}], Confidence: {confidence}, Class Name: {class_name}")

            # Draw bounding box on the image
            cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box

            # Add label with class name and confidence score
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(rgb_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image with bounding boxes
cv2.imshow('YOLO Object Detection', rgb_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
