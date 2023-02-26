
import torch
from typing import Tuple, Union
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random


class DeviceAgnosticColorJitter(T.ColorJitter):
    def __init__(self, brightness: float = 0., contrast: float = 0., saturation: float = 0., hue: float = 0.):
        """This is the same as T.ColorJitter but it only accepts batches of images and works on GPU"""
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        # Applies a different color jitter to each image
        color_jitter = super(DeviceAgnosticColorJitter, self).forward
        augmented_images = [color_jitter(img).unsqueeze(0) for img in images]
        augmented_images = torch.cat(augmented_images)
        assert augmented_images.shape == torch.Size([B, C, H, W])
        return augmented_images


class DeviceAgnosticRandomResizedCrop(T.RandomResizedCrop):
    def __init__(self, size: Union[int, Tuple[int, int]], scale: float):
        """This is the same as T.RandomResizedCrop but it only accepts batches of images and works on GPU"""
        super().__init__(size=size, scale=scale)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        # Applies a different color jitter to each image
        random_resized_crop = super(DeviceAgnosticRandomResizedCrop, self).forward
        augmented_images = [random_resized_crop(img).unsqueeze(0) for img in images]
        augmented_images = torch.cat(augmented_images)
        return augmented_images


class DeviceAgnosticColorJitter(T.ColorJitter):
    def __init__(self, brightness: float = 0., contrast: float = 0., saturation: float = 0., hue: float = 0.):
        """This is the same as T.ColorJitter but it only accepts batches of images and works on GPU"""
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        # Applies a different color jitter to each image
        color_jitter = super(DeviceAgnosticColorJitter, self).forward
        augmented_images = [color_jitter(img).unsqueeze(0) for img in images]
        augmented_images = torch.cat(augmented_images)
        assert augmented_images.shape == torch.Size([B, C, H, W])
        return augmented_images

class DeviceAgosticAdjustBrightness(): 
    def __init__(self, brightness_factor: float = 0.65):
        self.brightness_factor = brightness_factor

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        # Applies a different color jitter to each image
        offset = random.uniform(-0.5, 0.5)
        augmented_images = []
        for img in images:
            #transform with probability 50%
            if random.random() < 0.5:
                augmented_img = TF.adjust_brightness(img, self.brightness_factor + offset).unsqueeze(0)
                augmented_images.append(augmented_img)
            else:
                augmented_images.append(img.unsqueeze(0))
        augmented_images = torch.cat(augmented_images)
        assert augmented_images.shape == torch.Size([B, C, H, W])
        return augmented_images
    
class DeviceAgnosticContrast(): 
    def __init__(self, contrast_factor: float = 1.15):
        self.contrast_factor = contrast_factor

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        # Applies a different color jitter to each image
        offset = random.uniform(-0.05, 0.05)
        augmented_images = []
        for img in images:
            #transform with probability 50%
            if random.random() < 0.5:
                augmented_img = TF.adjust_contrast(img, self.contrast_factor + offset ).unsqueeze(0)
                augmented_images.append(augmented_img)
            else:
                augmented_images.append(img.unsqueeze(0))
        augmented_images = torch.cat(augmented_images)
        assert augmented_images.shape == torch.Size([B, C, H, W])
        return augmented_images
    
class DeviceAgosticAdjustBrightnessContrastSaturation():
    def __init__(self, brightness_factor: float = 0.65, contrast_factor: float = 1.15, saturation_factor: float = 0.85):
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.saturation_factor = saturation_factor

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        # Applies a different color jitter to each image
        offsetBright = random.uniform(-0.5, 0.5)
        offsetContrast = random.uniform(-0.05, 0.05)
        offsetSaturation = random.uniform(-0.05, 0.05)
        augmented_images = []
        for img in images:
            #transform with probability 50%
            if random.random() < 0.5:
                augmented_img = TF.adjust_brightness(img, self.brightness_factor + offsetBright).unsqueeze(0)
                augmented_img = TF.adjust_saturation(augmented_img, self.saturation_factor + offsetSaturation ).unsqueeze(0)
                augmented_img = TF.adjust_contrast(augmented_img, self.contrast_factor + offsetContrast ).unsqueeze(0)
                augmented_images.append(augmented_img)
            else:
                augmented_images.append(img.unsqueeze(0))
        augmented_images = torch.cat(augmented_images)
        assert augmented_images.shape == torch.Size([B, C, H, W])
        return augmented_images
    
class DeviceAgosticAdjustSaturation():
    def __init__(self, saturation_factor: float = 0.85):
        self.saturation_factor = saturation_factor

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        # Applies a different color jitter to each image
        offset = random.uniform(-0.05, 0.05)
        augmented_images = []
        for img in images:
            #transform with probability 50%
            if random.random() < 0.5:
                augmented_img = TF.adjust_saturation(img, self.saturation_factor + offset).unsqueeze(0)
                augmented_images.append(augmented_img)
            else:
                augmented_images.append(img.unsqueeze(0))
        augmented_images = torch.cat(augmented_images)
        assert augmented_images.shape == torch.Size([B, C, H, W])
        return augmented_images
    
    
if __name__ == "__main__":
    """
    You can run this script to visualize the transformations, and verify that
    the augmentations are applied individually on each image of the batch.
    """
    from PIL import Image
    # Import skimage in here, so it is not necessary to install it unless you run this script
    from skimage import data
    
    # Initialize DeviceAgnosticRandomResizedCrop
    brightness = 0.5
    augs = DeviceAgosticAdjustBrightness()
    # Create a batch with 2 astronaut images
    pil_image = Image.fromarray(data.astronaut())
    tensor_image = T.functional.to_tensor(pil_image).unsqueeze(0)
    images_batch = torch.cat([tensor_image, tensor_image])
    # Apply augmentation (individually on each of the 2 images)
    augmented_batch = augs(images_batch)
    # Convert to PIL images
    augmented_image_0 = T.functional.to_pil_image(augmented_batch[0])
    augmented_image_1 = T.functional.to_pil_image(augmented_batch[1])
    # Visualize the original image, as well as the two augmented ones
    pil_image.show()
    augmented_image_0.show()
    augmented_image_1.show()
