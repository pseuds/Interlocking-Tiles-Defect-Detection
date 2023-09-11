import os
from os import listdir
from cv_functions import edge_detector_save, resize_save

folder_dir = "images/asphalt-tiles"
output_dir = "output/asphalt_tiles_resize_1"

for image in os.listdir(folder_dir):
    resize_save(folder_dir + f"/{image}", output_dir, 1632, 1216)
    print(image)

# input_img = "images/asphalt_tiles/IMG_9083.JPG"
# output_folder = "output/asphalt_tiles"
# t1 = 350
# t2 = 475

# edge_detector_save(input_img, output_folder, t1, t2)

# input_img = "images/asphalt_tiles/IMG_9084.JPG"
# output_folder = "output/asphalt_tiles"

# edge_detector_save(input_img, output_folder, t1, t2)

