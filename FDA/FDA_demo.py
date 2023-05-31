import numpy as np
from PIL import Image
from utils import FDA_source_to_target
import scipy.misc
import torch
import older_scipy

im_src = Image.open("demo_images/day.jpg").convert('RGB')
im_trg = Image.open("demo_images/night.jpg").convert('RGB')

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
older_scipy.toimage(src_in_trg, cmin=0.0, cmax=255.0).resize((480,853)).save('demo_images/src_in_tar.png')

