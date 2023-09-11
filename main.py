import os
import cv2
from os import listdir
from cv_functions import get_image_dimensions, get_d32, resize_save

# for every image in input folder, resize and divide into 4 and save in output folder.
# folder_dir = "datasets_sorted/asphalt-tiles"
# output_dir = "output/asphalt_tiles_resize_1"

# for image in os.listdir(folder_dir):
#     resize_save(folder_dir + f"/{image}", output_dir, 1632, 1216)
#     print(image)

# img_path = "datasets_sorted/asphalt-tiles/IMG_9083.JPG"
# h, w = get_image_dimensions(image_path=img_path)
# get_d32(h, w, new_width=1600)