import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import IPython.display as display

# Constants used for preprocessing images (see https://pytorch.org/hub/pytorch_vision_inception_v3/)
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

def preprocess(img):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return preprocess(img).unsqueeze(0)

def postprocess(img):
    inv_normalize = transforms.Normalize([-m/s for m, s in zip(mean, std)], [1/s for s in std])
    convert = transforms.ToPILImage()
    img = convert(inv_normalize(img).squeeze())
    return img

def clip_to_valid_range(img):
    ranges = [((0-m)/s, (1-m)/s) for m, s in zip(mean, std)]
    img[:, 0, :, :] = torch.clip(img[:, 0, :, :], ranges[0][0], ranges[0][1])
    img[:, 1, :, :] = torch.clip(img[:, 1, :, :], ranges[1][0], ranges[1][1])
    img[:, 2, :, :] = torch.clip(img[:, 2, :, :], ranges[2][0], ranges[2][1])
    return img

def gaussian_pyramid(img, num_octaves):
    org_shape = torch.tensor(img.shape[2:]).float()
    factor = 1.25
    pyramid = []

    for i in range(num_octaves):
        octave = scale(img, 1/factor**i)
        pyramid.append(octave)
    
    return pyramid[::-1]

def scale(img, factor):
    org_shape = torch.tensor(img.shape[2:]).float()
    new_shape = org_shape * factor
    resize = transforms.Resize(new_shape.int().tolist())
    return resize(img)

def jitter(img, shift_y=None, shift_x=None):
    if shift_y is None:
        shift_y = np.random.randint(-img.shape[2] // 2, img.shape[2] // 2)
    if shift_x is None:
        shift_x = np.random.randint(-img.shape[3] // 2, img.shape[3] // 2)
    img = torch.roll(img, (shift_y, shift_x), (2, 3))
    return img, shift_y, shift_x

def zoom(img):
    w, h = img.size
    img = img.crop((w*0.01, h*0.01, w-w*0.01, h-h*0.01))
    return img.resize((w, h), Image.LANCZOS)

def random_noise(height, width):
    noise = np.random.randint(0, 256, (height, width, 3))
    return Image.fromarray(noise.astype("uint8"))

def show(img):
    display.display(img)
