import numpy as np

class cameraInput:
    def __init__(self, from_prestored = True)-> None:
        if from_prestored:
            self.data = self.load_data('data/lab_walking.npz')
            self.color_frames = self.data["color_frames"]
            self.depth_frames = self.data["depth_frames"]
            self.idx = 0
            self.num_frames = self.color_frames.shape[0]
            self.intrinsics = self.data["intrinsics"].item()
            self.image_details = self.data["image_details"].item()
        else:
            print("Real-time camera input not implemented yet.")
            raise NotImplementedError

    def load_data(self, data_path:str)-> dict:
        return np.load(data_path, allow_pickle=True)
    
    def get_color_frame(self):
        """Generator that yields RGB frames each time it is called.
            Example usage:
            while i_want_to_detect_objects:
                frame = camera_input.get_rgb_frame()
                yolo_result = yolo_model.detect(frame)
            """
        color_frame = self.color_frames[self.idx,:,:]
        self.idx = (self.idx + 1) % self.num_frames
        return color_frame
    
    def get_depth_frame(self):
        """Generator that yields RGBD frames each time it is called.
            Example usage:
            while i_want_to_detect_objects:
                frame = camera_input.get_rgb_frame()
                yolo_result = yolo_model.detect(frame)
            """
        depth_frame = self.depth_frames[self.idx,:,:]
        self.idx = (self.idx + 1) % self.num_frames
        return depth_frame

if __name__ == "__main__":
    camera_input = cameraInput()  # Create an instance of the CameraInput class
    for i in range(500):
        frame = camera_input.get_rgb_frame()
        print(frame[0,0,0])
