import cv2
import numpy as np

data_dict = np.load('data/data.npy', allow_pickle=True).item()

print(data_dict.keys()) 

first_frame = data_dict["data"][100,:,:,:3]/255
depth_frame = data_dict["data"][0,:,:,3]

print("Shape of data: ", data_dict["data"].shape)

print(np.max(first_frame))
print(np.min(first_frame))

print(np.max(depth_frame))
print(np.min(depth_frame))
print(np.median(depth_frame))


cv2.imshow("frame", first_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
