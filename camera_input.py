import numpy as np

class CamearInput:
    def __init__(self, from_prestored = True)-> None:
        if from_prestored:
            self.data = self.load_data('data/data.npy')
            self.frames = self.data["data"]
            self.idx = 0
            self.num_frames = self.frames.shape[0]
            self.intrinsics = self.data["intrinsics"]
            self.image_details = self.data["image_details"]
        else:
            print("Real-time camera input not implemented yet.")
            raise NotImplementedError

    def load_data(self, data_path:str)-> dict:
        return np.load(data_path, allow_pickle=True).item()
    
    def get_rgb_frame(self):
        """Generator that yields RGB frames each time it is called.
            Example usage:
            while i_want_to_detect_objects:
                frame = camera_input.get_rgb_frame()
                yolo_result = yolo_model.detect(frame)
            """
        frame = self.frames[self.idx,:,:,:3]/255
        self.idx = (self.idx + 1) % self.num_frames
        return frame

if __name__ == "__main__":
    camera_input = CamearInput()  # Create an instance of the CameraInput class
    for i in range(500):
        frame = camera_input.get_rgb_frame()
        print(frame[0,0,0])
