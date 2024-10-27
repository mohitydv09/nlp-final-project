import os
import numpy as np

from ultralytics import YOLO

from camera_input import cameraInput

class objectDetector:
    def __init__(self, model_name='yolo11n.pt', model_folder='./models')->None:
        self.model = self.load_model(model_name='yolo11n.pt', model_folder='./models')

    def load_model(self, model_name='yolo11n.pt', model_folder='./models')->YOLO:
        os.makedirs(model_folder, exist_ok=True)
        model_path = os.path.join(model_folder, model_name)

        if not os.path.exists(model_path):
            try:
                print(f"Model {model_name} not found in {model_folder}. Downloading...")
                model = YOLO(model_name)  # Download model by name
                model.save(model_path)  # Save the model to the specified path
                print("Model download complete.")
            except Exception as e:
                print(f"Error downloading model: {e}")
                raise FileNotFoundError(f"The model {model_name} could not be downloaded.")
        else:
            print(f"Using existing model: {model_name}")
            model = YOLO(model_path)
        return model

    def detect_objects(self, 
                       rgb_image:np.ndarray, 
                       device = 'cuda:0', 
                       verbose=False, 
                       visualization=False, 
                       save_image=False)->tuple:
        """Get bounding boxes from the image using the YOLO model.
        Input Image should be as UltraLytics expects (HWC format): https://docs.ultralytics.com/modes/predict/#inference-sources
        """
        result = self.model.predict(rgb_image,
                                    verbose=verbose, 
                                    conf = 0.6, 
                                    iou=0.4, 
                                    device = device, 
                                    half=True)[0]
         ## Result is a list of size as many images are passed, we only will pass one so we need one result.
        class_ids = result.boxes.cls
        labels = [result.names[int(cls_id)] for cls_id in class_ids] ## List auto sends it to CPU
        bboxes_xyxyn = result.boxes.xyxyn.tolist()  ## List auto sends it to CPU
        if visualization: result.show()
        if save_image: result.save()
        return labels, bboxes_xyxyn

    def get_depth_data(self, 
                       depth_image:np.ndarray, 
                       bounding_boxes:list)->list:
        depth_values = []
        for bbox in bounding_boxes:
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            ## Sample random points in the bounding box, and scale to get pixel vales
            x_random = np.random.uniform(x1, x2, 5)*depth_image.shape[1]
            y_random = np.random.uniform(y1, y2, 5)*depth_image.shape[0]
            ## Get depth values at random points
            random_depths = [depth_image[int(y), int(x)] for x, y in zip(x_random, y_random)]
            random_depths = [depth for depth in random_depths if depth > 0]  ## Remove zero values
            if len(random_depths) > 0:
                depth_values.append(np.median(random_depths))
            else:
                print("This is a bug, do something about this.")
                depth_values.append(0)
        return depth_values
    
    def pixel2world(self, 
                    pixel_coordinates:list, 
                    depth:list, 
                    ppx:float,
                    ppy:float,
                    fx:float,
                    fy:float)->list:
        """Convert pixel coordinates to world coordinates using depth information."""
        world_coordinates = []
        for i in range(len(depth)):
            # Convert pixel coordinates to normalized device coordinates
            x_ndc = (pixel_coordinates[i][0] - ppx) / fx
            y_ndc = (pixel_coordinates[i][1] - ppy) / fy

            # Get depth value
            z = depth[i]

            # Convert to world coordinates
            x_world = z * x_ndc
            y_world = z * y_ndc
            world_coordinates.append((int(x_world), int(y_world), int(z)))
        return world_coordinates
    
    def get_objects_with_location(self,
                                   rgb_image:np.ndarray, 
                                   depth_image:np.ndarray, 
                                   intrinsics:dict)->tuple:
        """Get objects with their 3D locations."""
        labels, bboxes = self.detect_objects(rgb_image)
        depth_values = self.get_depth_data(depth_image, bboxes)

        pixel_coordinates = [((bbox[0]+bbox[2])//2 , (bbox[1]+bbox[3])//2) for bbox in bboxes] 
        world_coordinates = self.pixel2world(pixel_coordinates, depth_values, 
                                              intrinsics['ppx'], intrinsics['ppy'], 
                                              intrinsics['fx'], intrinsics['fy'])
        return labels, pixel_coordinates, world_coordinates
        
if __name__ == "__main__":
    object_detector = objectDetector()  # Create an instance of the ObjectDetector class
    camera_input = cameraInput()  # Create an instance of the CameraInput class
    intrinsics_dict = camera_input.intrinsics
    depth_scale = camera_input.image_details["depth_scale"]
    for i in range(5):
        frame = camera_input.get_frame()
        rgb_frame = frame[:,:,:3].astype(np.uint8)
        depth_frame = frame[:,:,3] * depth_scale

        labels, pixel_coordinates, world_coordinates = object_detector.get_objects_with_location(rgb_frame, depth_frame, intrinsics_dict)
        print("Labels:", labels)
        print("World Coordinates:", world_coordinates)  # Print the world coordinates of detected objects
