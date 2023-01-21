import numpy as np
from PIL import Image
from utils import FDA_source_to_target
import torch
import older_scipy
import os
import random
from glob import glob

source_directory = 'small/test/queries_v1/'
target_directory = 'small/train/'
output_directory = 'small/test/queries_v1_delighted/'
#target_images = os.listdir()



targets_images = os.listdir(target_directory)

def transform(source: str, target, dir):
    im_src = Image.open(source).convert('RGB')
    im_trg = Image.open(target).convert('RGB')

    w,h = im_src.size


    im_src = im_src.resize( (1024,512), Image.BICUBIC )
    im_trg = im_trg.resize( (1024,512), Image.BICUBIC )

    im_src = np.asarray(im_src, np.float32)
    im_trg = np.asarray(im_trg, np.float32)

    im_src = im_src.transpose((2, 0, 1))
    im_trg = im_trg.transpose((2, 0, 1))

    im_src = torch.from_numpy(im_src).unsqueeze(0)
    im_trg = torch.from_numpy(im_trg).unsqueeze(0)

    src_in_trg = FDA_source_to_target( im_src, im_trg, L=0.01 )

    src_in_trg = torch.Tensor.numpy(src_in_trg.squeeze(0))

    src_in_trg = src_in_trg.transpose((1,2,0))
    src_in_trg = src_in_trg.astype(int)
    #Image.fromarray(np.uint8(src_in_trg)).save('demo_images/src_in_tar.png')
    name = source.split('/')[-1].replace('@','day@',1)
    
    older_scipy.toimage(src_in_trg, cmin=0.0, cmax=255.0).resize((w,h)).save(f'{dir}/{name}')





if not os.path.isdir(output_directory):
    os.makedirs(output_directory)
source_images_paths = sorted(glob(f"{source_directory}/*.jpg", recursive=True))
targets_images_paths = sorted(glob(f"{target_directory}/**/*.jpg", recursive=True))
for source_image in source_images_paths:
    index = int(random.uniform(0, len(targets_images_paths)))
    
    transform(source_image, targets_images_paths[index], output_directory)
    

        



