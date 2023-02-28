from PIL import Image, ImageDraw, ImageEnhance, ImageColor, ImageFilter
import numpy as np
from random import gauss
import os
from glob import glob
from tqdm import tqdm
import sys
import random

def brightness(image, factorMin, factorMax):
    enhancer = ImageEnhance.Brightness(image)
    factor = np.random.uniform(factorMin, factorMax)
    return enhancer.enhance(factor)

def desaturate(image, factorMin, factorMax):
    enhancer = ImageEnhance.Color(image)
    factor = np.random.uniform(factorMin, factorMax)
    return enhancer.enhance(factor)

def blue_tint(image, factorMin, factorMax):
    # Loop through all pixels and add a blue tint
    factor = np.random.uniform(factorMin, factorMax)
    for i in range(image.width):
        for j in range(image.height):
            r, g, b = image.getpixel((i, j))
            # Increase the blue channel by 30%
            image.putpixel((i, j), (r, g, int(b * factor)))
    
    return image

def contrast(image, factorMin, factorMax):
    enhancer = ImageEnhance.Contrast(image)
    factor = np.random.uniform(factorMin, factorMax)
    return enhancer.enhance(factor)

def generate_and_apply_gradient(image, factorMin, factorMax):
    """ Add a gradient to the image """
    # Create a new image with a black to transparent vertical gradient
    gradient = Image.new("RGBA", (image.width, image.height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(gradient)
    for y in range(image.height):
        # Set the transparency of the pixel based on its position in the image
        # The top of the image will be fully opaque (255), and the bottom will be fully transparent (0)
        transparency = int(y / (image.height*1.2) * 255)
        draw.line((0, y, image.width, y), fill=(0, 0, 0, transparency))

    #rotate the gradient
    gradient = gradient.rotate(180)

    # Paste the gradient onto the image
    image.paste(gradient, (0, 0), gradient)

    return image

def dark_mask(image):
    """ Create random circular masks"""
    mask = Image.new("RGBA", (image.width, image.height), (0,0,0, 100))
    draw = ImageDraw.Draw(mask)
    for i in range(8):
      x = np.random.randint(0, image.width)
      y = np.random.randint(image.height/4, image.height)
      radius = np.random.randint(image.height/5, image.height/3)
      # Draw the circle
      draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(0,0,0,0))

    # blur the mask
    mask = mask.filter(ImageFilter.GaussianBlur(radius=80))

    # Paste the mask onto the image
    image.paste(mask, (0, 0), mask)
    return image

def add_gaussian_noise(image):
    mean = 0
    stddev = 5
    for i in range(image.width):
        for j in range(image.height):
            r, g, b = image.getpixel((i, j))
            # Increase the blue channel by 30%
            r = r + int(gauss(mean, stddev))
            g = g + int(gauss(mean, stddev))
            b = b + int(gauss(mean, stddev))
            r = max(0, min(r, 255))
            g = max(0, min(g, 255))
            b = max(0, min(b, 255))

            image.putpixel((i, j), (r, g, b))
    
    return image

def apply_post_processing(filename, fda_applied = True):
    # Open the image
    if fda_applied:
        brightness_min = 0.8
        brightness_max = 0.9
        desaturate_min = 0.9
        desaturate_max = 0.95
        blue_tint_min = 1.05
        blue_tint_max = 1.15
        contrast_min = 1.05
        contrast_max = 1.1
        gradient_min = 1.1
        gradient_max = 1.3
    else:
        brightness_min = 0.7
        brightness_max = 0.8
        desaturate_min = 0.8
        desaturate_max = 0.9
        blue_tint_min = 1.15
        blue_tint_max = 1.2
        contrast_min = 1.1
        contrast_max = 1.2
        gradient_min = 1.2
        gradient_max = 1.4
    image = Image.open(filename)
    image = brightness(image, brightness_min, brightness_max)
    image = desaturate(image, desaturate_min, desaturate_max)
    image = blue_tint(image, blue_tint_min, blue_tint_max)
    image = contrast(image, contrast_min, contrast_max)
    image = generate_and_apply_gradient(image, gradient_min, gradient_max)
    image = dark_mask(image)
    image = add_gaussian_noise(image)
    return image

if __name__ == "__main__":
    directory = sys.argv[1]
    output_directory = sys.argv[2]
    if not os.path.isdir(output_directory):
      os.makedirs(output_directory)
    images_paths = sorted(glob(f"{directory}/**/*.jpg", recursive=True))
    for filename in tqdm(images_paths):
        internal_dir, image_name = filename.split("/")[-2], filename.split("/")[-1]
        if random.random() < 0.5:
            night_image = apply_post_processing(filename, False)
        else:
            night_image = Image.open(filename)
        if not os.path.isdir(os.path.join(output_directory, internal_dir)):
            os.makedirs(os.path.join(output_directory, internal_dir))
        night_image.save(os.path.join(output_directory,internal_dir, image_name))