import numpy as np

class CameraInput:
    def __init__(self, from_prestored = True, data_path: str = 'data/keller_stairway.npz')-> None:
        if from_prestored:
            self.data = self.load_data(data_path)
            self.color_frames = self.data["color_frames"]
            self.depth_frames = self.data["depth_frames"]
            self.idx = 0
            self.num_frames = self.color_frames.shape[0]
            self.intrinsics = self.data["intrinsics"].item()
            self.image_details = self.data["image_details"].item()
            self.depth_scale = self.image_details["depth_scale"]
        else:
            print("Real-time camera input not implemented yet.")
            raise NotImplementedError

    def load_data(self, data_path:str)-> dict:
        return np.load(data_path, allow_pickle=True)
    
    def get_color_frame(self):
        """Generator that yields RGB frames each time it is called.
            Example usage:
            while i_want_to_detect_objects:
                frame = camera_input.get_color_frame()
                yolo_result = yolo_model.detect(frame)
            """
        color_frame = self.color_frames[self.idx,:,:]
        self.idx = (self.idx + 1) % self.num_frames
        return color_frame
    
    def get_depth_frame(self):
        """Generator that yields RGBD frames each time it is called.
            Example usage:
            while i_want_to_detect_objects:
                frame = camera_input.get_color_frame()
                yolo_result = yolo_model.detect(frame)
            """
        depth_frame = self.depth_frames[self.idx,:,:]
        self.idx = (self.idx + 1) % self.num_frames
        return depth_frame

if __name__ == "__main__":
    camera_input = CameraInput()  # Create an instance of the CameraInput class
    for i in range(500):
        frame = camera_input.get_color_frame()
        print(frame[0,0,0])
