import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from scipy import fftpack

def _low_high_pass(im, radius, low=1):
    '''
    input: im is a numpy 3 channel image with first two dimensions and last nr of channels 
           radius of circle in shifted-fft image blocked/passed
           low default 1 is low pass, 0 is high pass

    returns: tuple with low pass and high pass filtered image
    '''
    c = []
    for i in range(3):
        im_np = im[:, :, i] / 255.0
        # fft of image
        fft1 = fftpack.fftshift(fftpack.fft2(im_np))
        # Create a low pass filter image
        x, y = im_np.shape[0], im_np.shape[1]
        # size of circle
        e_x, e_y = radius, radius
        # create a box 
        bbox = ((x / 2) - (e_x / 2), (y / 2) - (e_y / 2), (x / 2) + (e_x / 2), (y / 2) + (e_y / 2))
        
        low_pass = Image.new("L", (im_np.shape[0], im_np.shape[1]), color=low)

        draw1 = ImageDraw.Draw(low_pass)
        draw1.ellipse(bbox, fill= 1 - low)

        low_pass_np = np.transpose(np.array(low_pass), (1, 0))
        
        # multiply both the images
        filtered_low = np.multiply(fft1, low_pass_np)

        # inverse fft
        ifft2_low = np.real(fftpack.ifft2(fftpack.ifftshift(filtered_low)))
        ifft2_low = np.maximum(0, np.minimum(ifft2_low, 255))

        # clip data outside range [0...1]
        ifft2_low[ifft2_low > 1] = 1

        c.append(ifft2_low)

    final = np.transpose(np.array(c), (1, 2, 0))
    return final

def low_pass(image, radius):
    return Image.fromarray(np.uint8(_low_high_pass(np.array(image), radius, 1) * 255))

def high_pass(image, radius):
    return Image.fromarray(np.uint8(_low_high_pass(np.array(image), radius, 0) * 255))

def low_high_pass_mean(image, radius):
    arr = np.array(image)
    return Image.fromarray(np.uint8(((_low_high_pass(arr, radius, 0) + _low_high_pass(arr, radius, 0)) / 2) * 255))

def hue_rotation(image, rot):
    img = np.array(image.convert(mode="HSV"))
    img[..., 0] = rot % 256
    img[..., 1] = 96
    return Image.fromarray(img, "HSV").convert("RGB")

def saturation_rotation(image, rot, separate_final=True):
    img = np.array(image.convert(mode="HSV"))
    if not separate_final or rot < 255:
        img[..., 1] = rot
    return Image.fromarray(img, "HSV").convert("RGB")

def value_rotation(image, rot):
    img = np.array(image.convert(mode="HSV"))
    img[..., 2] = img[..., 2] * (rot / 255)
    return Image.fromarray(img, "HSV").convert("RGB")

def gaussian_blur(image, radius):
    return image.filter(ImageFilter.GaussianBlur(radius))

def posterize(image, bits):
    if bits > 8:
        return image
    return ImageOps.posterize(image, bits)
