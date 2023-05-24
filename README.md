
# Rethinking Visual Geo-localization for Large-Scale Applications

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-pittsburgh-250k)](https://paperswithcode.com/sota/visual-place-recognition-on-pittsburgh-250k?p=rethinking-visual-geo-localization-for-large)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-pittsburgh-30k)](https://paperswithcode.com/sota/visual-place-recognition-on-pittsburgh-30k?p=rethinking-visual-geo-localization-for-large)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-tokyo247)](https://paperswithcode.com/sota/visual-place-recognition-on-tokyo247?p=rethinking-visual-geo-localization-for-large)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-mapillary-val)](https://paperswithcode.com/sota/visual-place-recognition-on-mapillary-val?p=rethinking-visual-geo-localization-for-large)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-st-lucia)](https://paperswithcode.com/sota/visual-place-recognition-on-st-lucia?p=rethinking-visual-geo-localization-for-large)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-sf-xl-test-v1)](https://paperswithcode.com/sota/visual-place-recognition-on-sf-xl-test-v1?p=rethinking-visual-geo-localization-for-large)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-sf-xl-test-v2)](https://paperswithcode.com/sota/visual-place-recognition-on-sf-xl-test-v2?p=rethinking-visual-geo-localization-for-large)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/image-classification-on-amstertime)](https://paperswithcode.com/sota/image-classification-on-amstertime?p=rethinking-visual-geo-localization-for-large)	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/image-retrieval-on-amstertime)](https://paperswithcode.com/sota/image-retrieval-on-amstertime?p=rethinking-visual-geo-localization-for-large)


This is the official repository for the CVPR 2022 paper [Rethinking Visual Geo-localization for Large-Scale Applications](https://arxiv.org/abs/2204.02287).
The paper presents a new dataset called San Francisco eXtra Large (SF-XL, go [_here_](https://forms.gle/wpyDzhDyoWLQygAT9) to download it), and a highly scalable training method (called CosPlace), which allows to reach SOTA results with compact descriptors.

The images below represent respectively:
1) the map of San Francisco eXtra Large
2) a visualization of how CosPlace Groups (read datasets) are formed
3) results with CosPlace vs other methods on Pitts250k (CosPlace trained on SF-XL, others on Pitts30k)
<p float="left">
  <img src="https://github.com/gmberton/gmberton.github.io/blob/main/images/SF-XL%20map.jpg" height="150" />
  <img src="https://github.com/gmberton/gmberton.github.io/blob/main/images/map_groups.png" height="150" /> 
  <img src="https://github.com/gmberton/gmberton.github.io/blob/main/images/backbones_pitts250k_main.png" height="150" />
</p>



## Train
After downloading the SF-XL dataset, simply run 

`$ python3 train.py --dataset_folder path/to/sf-xl/processed`

the script automatically splits SF-XL in CosPlace Groups, and saves the resulting object in the folder `cache`.
By default training is performed with a ResNet-18 with descriptors dimensionality 512 is used, which fits in less than 4GB of VRAM.

To change the backbone or the output descriptors dimensionality simply run 

`$ python3 train.py --dataset_folder path/to/sf-xl/processed --backbone resnet50 --fc_output_dim 128`

You can also speed up your training with Automatic Mixed Precision (note that all results/statistics from the paper did not use AMP)

`$ python3 train.py --dataset_folder path/to/sf-xl/processed --use_amp16`

Run `$ python3 train.py -h` to have a look at all the hyperparameters that you can change. You will find all hyperparameters mentioned in the paper.

#### Dataset size and lightweight version

The SF-XL dataset is about 1 TB.
For training only a subset of the images is used, and you can use this subset for training, which is only 360 GB.
If this is still too heavy for you (e.g. if you're using Colab), but you would like to run CosPlace, we also created a small version of SF-XL, which is only 5 GB.
Obviously, using the small version will lead to lower results, and it should be used only for debugging / exploration purposes.
More information on the dataset and lightweight version are on the README that you can find on the dataset download page (go [_here_](https://forms.gle/wpyDzhDyoWLQygAT9) to find it).

#### Reproducibility
Results from the paper are fully reproducible, and we followed deep learning's best practices (average over multiple runs for the main results, validation/early stopping and hyperparameter search on the val set).
If you are a researcher comparing your work against ours, please make sure to follow these best practices and avoid picking the best model on the test set.


## Test
You can test a trained model as such

`$ python3 eval.py --dataset_folder path/to/sf-xl/processed --backbone resnet50 --fc_output_dim 128 --resume_model path/to/best_model.pth`

You can download plenty of trained models below.

## Model Zoo

In the table below are links to models with different backbones and dimensionality of descriptors, trained on SF-XL.
If you want to use these weights in your own code, make sure that the model is the same as ours: CNN backbone -> L2 -> GeM -> FC -> L2.

<table>
  <tr>
    <th rowspan=2>Model</th>
    <th colspan=7>Dimension of Descriptors</th>
  </tr>
  <tr>
    <td>32</td>
    <td>64</td>
    <td>128</td>
    <td>256</td>
    <td>512</td>
    <td>1024</td>
    <td>2048</td>
  </tr>
  <tr>
    <td>ResNet-18</td>
    <td><a href="https://drive.google.com/file/d/1tfT8r2fBeMVAEHg2bVfCql5pV9YzK620/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1-d_Yi3ly3bY6hUW1F9w144FFKsZtYBL4/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1HaQjGY5x--Ok0RcspVVjZ0bwrAVmBvrZ/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1hjkogugTsHTQ6GTuW3MHqx-t4cXqx0uo/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1rQAC2ZddDjzwB2OVqAcNgCFEf3gLNa9U/view?usp=sharing">link</a></td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>ResNet-50</td>
    <td><a href="https://drive.google.com/file/d/18AxbLO66CO0kG05-1YrRb1YwqN7Wgp6Z/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1F2WMt7vMUqXBjsZDIwSga3N0l0r9NP2s/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/14U3jsoNEWC-QsINoVCWZaHFUGE20fIgZ/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1Q2sZPEJfHAe19JaZkdgeFotUYwKbV_x2/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1LgDaxCjbQqQWuk5qrPogfg7oN8Ksl1jh/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1VBLUiQJfmnZ4kVQIrXBW-AE1dZ3EnMv2/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1yNzxsMg34KO04UJ49ncANdCIWlB3aUGA/view?usp=sharing">link</a></td>
  </tr>
  <tr>
    <td>ResNet-101</td>
    <td><a href="https://drive.google.com/file/d/1a5FqhujOn0Pr6duKrRknoOgz8L8ckDSE/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/17C8jBQluxsbI9d8Bzf67b5OsauOJAIuX/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1w37AztnIyGVklBMtm-lwkajb0DWbYhhc/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1G5_I4vX4s4_oiAC3EWbrCyXrCOkV8Bbs/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1uBKpNfMBt6sLIjCGfH6Orx9eQdQgN-8Z/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/12BU8BgfqFYzGLXXNaKLpaAzTHuN5I9gQ/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1PF7lsSw1sFMh-Bl_xwO74fM1InyYy1t8/view?usp=sharing">link</a></td>
  </tr>
  <tr>
    <td>ResNet-152</td>
    <td><a href="https://drive.google.com/file/d/12pI1FToqKKt8I6-802CHWXDP-JmHEFSW/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1rTjlv_pNtXgxY8VELiGYvLcgXiRa2zqB/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1q5-szPBn4zL8evWmYT04wFaKjen66mrk/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1sCQMA_rsIjmD-f381I0f2yDf0At4TnSx/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1ggNYQfGSfE-dciKCS_6SKeQT76O0OXPX/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/15vBWuHVqEMxkAWWrc7IrkGsQroC65tPc/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1AlF7xPSswDLA1TdhZ9yTVBkfRnJm0Hn8/view?usp=sharing">link</a></td>
  </tr>
  <tr>
    <td>VGG-16</td>
    <td>-</td>
    <td><a href="https://drive.google.com/file/d/1YJTBwagC0v50oPydpKtsTnGZnaYOV0z-/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1vgw509lGBfJR46cGDJGkFcdBTGhIeyAH/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1-4JtACE47rkXXSAlRBFIbydimfKemdo7/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1F6CT-rnAGTTexdpLoQYncn-ooqzJe6wf/view?usp=sharing">link</a></td>
    <td>-</td>
    <td>-</td>
  </tr>
</table>

Or you can download all models at once at [this link](https://drive.google.com/drive/folders/1WzSLnv05FLm-XqP5DxR5nXaaixH23uvV?usp=sharing)

## Issues
If you questions regarding our code or dataset, feel free to open an issue or send an email to berton.gabri@gmail.com

## Acknowledgements
Parts of this repo are inspired by the following repositories:
- [CosFace implementation in PyTorch](https://github.com/MuggleWang/CosFace_pytorch/blob/master/layer.py)
- [CNN Image Retrieval in PyTorch](https://github.com/filipradenovic/cnnimageretrieval-pytorch) (for the GeM layer)
- [Visual Geo-localization benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark) (for the evaluation / test code)

## Cite
Here is the bibtex to cite our paper
```
@inProceedings{Berton_CVPR_2022_cosPlace,
  author = {Berton, Gabriele and Masone, Carlo and Caputo, Barbara},
  title = {Rethinking Visual Geo-localization for Large-Scale Applications}, 
  booktitle = {CVPR},
  month = {June}, 
  year = {2022}, }
```

## Results

<table>
  <tr>
    <th>Model</th>
    <th>sf-xs</th>
    <th>tokyo-xs</th>
    <th>tokyo-night</th>
  </tr>
  <tr>
    <td>Cosface</td>
    <td>52.7 / 65.5 / 70.8</td>
    <td>70.2 / 84.1 / 89.5</td>
    <td>52.4 / 70.5 / 80.0</td>
  </tr>
  <tr>
    <td>Arcface (s = 64, m = 0.5)</td>
    <td>48.4/ 61.1/ 66.0</td>
    <td>70.2/ 84.4/ 87.9</td>
    <td>51.4 / 74.3 / 80.0</td> 
  </tr>
  
  <tr>
    <td>Sphereface (s = 30, m = 1.5)</td>
    <td>50.4 / 63.7 / 68.6</td>
    <td>71.7 / 84.4 / 87.9</td>
    <td>56.2 / 70.5 / 77.1</td> 
  </tr>
  
  <tr>
    <td>FDA+PP+DA (alpha = 0.1)</td>
    <td>50.9 / 63.4 / 69.1</td>
    <td>73.3 / 85.4 / 88.6</td>
    <td>61.0 / 81.0 / 83.8</td> 
  </tr>
  <tr>
    <td>AE</td>
    <td>52.8 / 66.0 / 71.0</td>
    <td>73.0 / 86.3 /91.1</td>
    <td>53.3 / 76.2 / 87.6</td> 
  </tr>
  <tr>
    <td>FDA+PP</td>
    <td>51.0 / 65.9 / 71.0</td>
    <td>69.8 / 84.1 / 88.6</td>
    <td>53.3 / 74.3 / 77.1</td> 
  </tr>
  <tr>
    <td>FDA</td>
    <td>51.3 / 65.2 / 69.8</td>
    <td>71.7 / 83.8 / 88.3</td>
    <td>53.3 / 69.5 / 77.1</td> 
  </tr>
  <tr>
    <td>PP</td>
    <td>48.6 / 61.9 / 67.2</td>
    <td>67.6 / 82.2 / 88.3</td>
    <td>50.5 / 71.4 / 81.9</td> 
  </tr>
  <tr>
    <td>FDA on test</td>
    <td>52.1 / 66.6 / 70.6</td>
    <td>70.2 / 84.4 / 88.6</td>
    <td>56.2 / 71.4 / 81.0</td> 
  </tr>
  <tr>
    <td>DA</td>
    <td>52.7 / 68.0 / 73.0</td>
    <td>71.7 / 86.7 / 91.1,</td>
    <td>56.2 / 78.1 / 84.8</td> 
  </tr><tr>
    <td>FDA+PP+DA (alpha = 0.01)</td>
    <td>50.4 / 63.9 / 69.0</td>
    <td>71.4 / 86.0 / 89.8</td>
    <td>59.0 / 79.0 / 83.8</td> 
  </tr>
  <tr>
    <td>FDA+PP+DA (alpha = 1)</td>
    <td>47.2 / 62.5 / 67.7</td>
    <td>58.1 / 71.4 / 78.1</td>
    <td>68.3 / 82.9 / 87.3</td> 
  </tr>
   <tr>
    <td>FDA+PP+DA (beta = 0.09)</td>
    <td>46.5 / 61.8 / 68.3 </td>
    <td> 71.4 / 84.8 / 89.5 </td>
    <td> 64.8 / 74.3  / 81.0  </td> 
  </tr>
  <tr>
    <td>Soup Face</td>
    <td>51.6 / 65.1 / 70.1 </td>
    <td>68.9 / 82.5 / 87.9 </td>
    <td>52.4 / 71.4  / 77.1  </td> 
  </tr>
  <tr>
    <td>EMMS l=1</td>
    <td>52.9 / 66.5 / 71.1 </td>
    <td>69.2 / 81.6 / 86.3 </td>
    <td>52.4 / 70.5  / 77.1  </td> 
  </tr>
  <tr>
    <td>EMMS l=1.5</td>
    <td>51.7 / 66.0 / 71.1 </td>
    <td>69.8 / 83.5 / 87.6 </td>
    <td>54.3 / 72.4  / 77.1  </td> 
  </tr>
  <tr>
    <td>EMMS l=2</td>
    <td>50.5 / 65.4 / 70.4 </td>
    <td>70.5 / 83.5 / 87.6 </td>
    <td>52.4 / 72.4  / 77.1  </td> 
  </tr>
  <tr>
    <td>FDA+PP+colorjitter+EMMS</td>
    <td>50.3 / 65.2 / 70.9 </td>
    <td>70.5 / 86.3 / 90.5 </td>
    <td>54.3 / 79.0  / 84.8  </td> 
  </tr>
  <tr>
    <td>Places 365 pretrain</td>
    <td> 50.3 / 65.5 / 70.3</td>
    <td> 64.4 / 81.6 / 87.6 </td>
    <td> 42.9 / 68.6 / 77.1 </td> 
  </tr>
  <tr>
    <td>Different Alphas</td>
    <td> 49.8 / 64.7 / 69.8</td>
    <td> 72.1 / 84.1 / 89.2 </td>
    <td> 57.1 / 78.1 /83.8 </td> 
  </tr>
  <tr>
    <td>Face soup + EMMS</td>
    <td> 51.6 / 66.8 / 70.7</td>
    <td> 67.6 / 84.1 / 89.5</td>
    <td> 50.5 / 73.3 / 81.0 </td> 
  </tr>
  <tr>
    <td> FDA PP DA soup</td>
    <td>  51.4 / 65.3 / 70.3 </td>
    <td>  70.8 / 84.1 / 88.9 </td>
    <td>  55.2 / 75.2 / 81.9 </td> 
  </tr>
  
  




</table>
