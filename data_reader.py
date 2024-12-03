import cv2
import numpy as np

data_dict = np.load('data/empty_hallway.npz', allow_pickle=True)

print("Keys in data dictionary: ", data_dict.files)

## Use this to extrac the data from the dictionary
color_frames = data_dict['color_frames']
depth_frames = data_dict['depth_frames']
intrinsics = data_dict['intrinsics'].item()
image_details = data_dict['image_details'].item()


##Example of how to use the data
print("Intinsics: ", intrinsics)
print("Image details: ", image_details)
print("Color frames shape: ", color_frames.shape)
print("Depth frames shape: ", depth_frames.shape)

## Display the first frame
cv2.imshow("frame", color_frames[0,:,:])
cv2.imshow("depth", depth_frames[0])
cv2.waitKey(0)
cv2.destroyAllWindows()