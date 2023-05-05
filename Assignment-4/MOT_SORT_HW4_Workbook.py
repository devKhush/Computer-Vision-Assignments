# %% [markdown]
# ## **CV HW4: Multi-object Tracking (MOT) with Detection**
# **Detection**: YOLOv5,
# **Tracking**: Simple Online Realtime Tracking (SORT)
#
# ---
#
#

# %% [markdown]
# ## **1. Unzip data folder**

# %%
import glob
import numpy as np
import os
from sort import *
from collections import namedtuple, OrderedDict
from google.colab.patches import cv2_imshow
import matplotlib
import sys
import cv2
import torchvision
import torch
from google.colab import drive
drive.mount("/content/drive")

# %%
# Change the path according to your setup
# !unzip '/content/drive/MyDrive/ECE344: CV (Computer Vision)/Assignments/Assignment-4/hw4/sort-master.zip'
# !unzip '/content/drive/MyDrive/ECE344: CV (Computer Vision)/Assignments/Assignment-4/hw4/KIT  TI_17_images.zip'

# %% [markdown]
# # **2. Install requirements**

# %%
# !pip install -r sort-master/requirements.txt
# !pip install cv

# %%
# !pip install filterpy
# !pip install scikit-image
# !pip install lap

# %% [markdown]
# # **3. Import libraries**

# %%
sys.path.insert(0, './sort-master/')

# %% [markdown]
# # **4. Load YOLOv5 detector from torch hub**

# %%
yolov5_detector = torch.hub.load(
    'ultralytics/yolov5', 'yolov5s', pretrained=True)
yolov5_detector.float()
yolov5_detector.eval()

# %% [markdown]
# # **5. Import SORT library**

# %%

# %% [markdown]
# #**6. Perform tracking with detection**

# %%
# Write your code here to perform tracking with detection using the provided YOLOv5 model and the SORT implementation

# %%
# Define the paths to the input images and output files
output_video_file_path = '/content/drive/MyDrive/ECE344: CV (Computer Vision)/Assignments/Assignment-4/output_video.mp4'
output_ground_truth_file_path = '/content/drive/MyDrive/ECE344: CV (Computer Vision)/Assignments/Assignment-4/output_boxes_file.txt'

# Load the input images
all_image_paths = sorted(glob.glob('/content/KITTI_17_images/*.jpg'))

# Create a SORT object
mot_tracker = Sort()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolov5_detector.to(device)

# Video resolution and frame rate for writing into the .mp4 file
frame_rate = 11
frame_size = (1224, 370)

# Create a Video writer
video_writer = cv2.VideoWriter(
    output_video_file_path, cv2.VideoWriter_fourcc(*'DIVX'), frame_rate, frame_size)

# Detect persons in each image of the directory
image_number = 0

# Store the colors used for bounding box for each unique person
all_colors = {}

with open(output_ground_truth_file_path, 'w') as f:
    pass

for image_path in all_image_paths:
    image = cv2.imread(image_path)
    # print(image.shape)
    image = cv2.resize(image, (1224, 370))
    # print(image.shape)

    results = yolov5_detector(image)

    # Extract the bounding box coordinates, class labels and confidence scores of the detected objects
    boundaries_scores = results.xyxy[0].cpu().numpy()[:, :4]
    class_labels = []
    for label in results.xyxy[0].cpu().numpy()[:, 5]:
        class_labels.append(results.names[label])
    confidence_scores = results.xyxy[0].cpu().numpy()[:, 4]

    # Filter the detected objects to only include pedestrians
    detected_persons = []
    for i in range(len(boundaries_scores)):
        if class_labels[i] == 'person' and confidence_scores[i] > 0:
            detected_persons.append(boundaries_scores[i])
    detected_persons = np.array(detected_persons)

    # Using Inbuilt SORT Algorithm to detect/track the persons
    person_trackers = mot_tracker.update(detected_persons)

    # Plot the track IDs and draw the bounding boxes for every person in the image
    for i, person_tracker in enumerate(person_trackers):
        x1, y1, x2, y2, track_id = person_tracker.astype(np.int32)
        if track_id in all_colors.keys():
            color = all_colors[track_id]
        else:
            color = list(np.random.random(size=3) * 256)
            all_colors[track_id] = color
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, 'ID: '+str(track_id), (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # In the ground truth output file, save all the bounding box coordinates
        with open(output_ground_truth_file_path, 'a') as f:
            f.write('{},{},{},{},{},{},-1,-1,-1\n'.format(image_number +
                    1, track_id, x1, y1, x2 - x1, y2 - y1))

    # Plot the image and write the image-frame to the output video file
    video_writer.write(image)
    cv2_imshow(image)

    # Increment  frame count for every image
    image_number += 1

# Save the video
video_writer.release()

# %% [markdown]
# # **7. Report Evaluation Metrics**

# %%
# Use the Track-Eval kit to report the complete set of performance and accuracy metrics
# Comment on and interpret MOTA and MOTP values

# %%
# !unzip '/content/drive/MyDrive/ECE344: CV (Computer Vision)/Assignments/Assignment-4/hw4/TrackEval-master.zip'
