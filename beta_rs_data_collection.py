import numpy as np
import cv2
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

path = "data/"

data = np.empty((2000, 480, 640, 4), dtype=np.uint16)

pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

i = 0
while True:
    frames = pipeline.wait_for_frames()

    aligned_frames = align.process(frames)

    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if i == 5:
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

    data[i,:,:,:3] = np.asanyarray(color_frame.get_data())
    data[i,:,:,3] = np.asanyarray(depth_frame.get_data())

    cv2.imshow("frame", data[i,:,:,:3])
    cv2.imshow("depth", data[i,:,:,3])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    i += 1

pipeline.stop() 
cv2.destroyAllWindows()

## Remove the empty frames
data = data[:i,:,:,:]

""" This is causing issues with data extraction, as it needs pyrealsense2
Next time extract the values and store them as non pyrealsense2 objects. """
intrinsics_dict = {
    "width": intrinsics.width,
    "height": intrinsics.height,
    "ppx": intrinsics.ppx,
    "ppy": intrinsics.ppy,
    "fx": intrinsics.fx,
    "fy": intrinsics.fy,
    "model": intrinsics.model,
    "coeffs": intrinsics.coeffs
}

image_details = {
    "width": 640,
    "height": 480,
    "channels": 4,
    "dtype": "uint16",
    "depth_scale": 0.00025
}

## Save the intrinsics matrix with data in a dictionary.
data_dict = {"data": data, "intrinsics": intrinsics_dict, "image_details": image_details}

np.save(path + "data.npy", data_dict)