import numpy as np
import cv2
import time
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

path = "data/"

color_data = np.empty((4000, 480, 640, 3), dtype=np.uint8)
depth_data = np.empty((4000, 480, 640), dtype=np.uint16)

pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

i = 0
start_time = time.time()
while True:
    frames = pipeline.wait_for_frames()

    aligned_frames = align.process(frames)

    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if i == 5:
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

    color_data[i,:,:,:] = np.asanyarray(color_frame.get_data())
    depth_data[i,:,:] = np.asanyarray(depth_frame.get_data())

    cv2.imshow("frame", color_data[i,:,:,:])
    # cv2.imshow("depth", depth_data[i,:,:])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    print("Time: ", round(time.time() - start_time))

    i += 1

pipeline.stop() 
cv2.destroyAllWindows()

## Remove the empty frames
color_data = color_data[:i,:,:,:]
depth_data = depth_data[:i,:,:]

intrinsics_dict = {
    "width": intrinsics.width,
    "height": intrinsics.height,
    "ppx": intrinsics.ppx,
    "ppy": intrinsics.ppy,
    "fx": intrinsics.fx,
    "fy": intrinsics.fy,
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
data_dict = {"color_frames": color_data, 
             "depth_frames": depth_data,
              "intrinsics": intrinsics_dict, 
              "image_details": image_details}

print("Saving data")
np.savez_compressed(path + f"data_{time.time()}.npz", **data_dict)