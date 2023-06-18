
from data_augmentation.fda import  apply_fda
import sys
import os
import random
from glob import glob
from tqdm import tqdm
from data_augmentation.post_processing_night_image import apply_post_processing


#target_images = os.listdir()



source_directory = sys.argv[1]
target_directory = sys.argv[2]
output_directory = sys.argv[3]

if not os.path.isdir(output_directory):
    os.makedirs(output_directory)


source_images_paths = sorted(glob(f"{source_directory}/**/*.jpg", recursive=True))
target_images_paths = glob(f"{target_directory}/*.jpg", recursive=True)
for source_filename in tqdm(source_images_paths):
    source_internal_dir, source_image_name = source_filename.split("\\")[-2], source_filename.split("\\")[-1]
    synthetic_image_name = source_image_name.split('\\')[-1].replace('@','night@',1)
    random_index = int(random.uniform(0,105))
    fdaed_image = apply_fda(source_filename,target_images_paths[random_index])
    night_image = apply_post_processing(fdaed_image, True)
    if not os.path.isdir(os.path.join(output_directory, source_internal_dir)):
        os.makedirs(os.path.join(output_directory, source_internal_dir))
    night_image.save(os.path.join(output_directory, source_internal_dir, synthetic_image_name))






        



