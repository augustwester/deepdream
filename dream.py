import torch
import torchvision
import IPython.display as display
from helpers import Model, preprocess, postprocess, gaussian_pyramid, jitter, clip_to_valid_range, resize, zoom, show
from functools import partial

#activations = []

"""
def load_model(settings):
    model = torchvision.models.inception_v3(pretrained=True)
    model.eval()
    register_hooks(model, settings)
    return model

def hook(channels, module, input, output):
    if channels == "all":
        activations.append(output)
    elif type(channels) == tuple:
        activations.append(output[:, channels, :, :])
    
def register_hooks(model, settings):
    for (layer_name, channels) in settings.items():
        if channels == "all" or type(channels) == tuple:
            module = getattr(model, layer_name)
            module.register_forward_hook(partial(hook, channels))
"""

def compute_loss(activations):
    losses = [act.sum() for act in activations]
    loss = torch.sum(torch.stack(losses, dim=0))
    return loss

def gradient_ascent(img, loss, step_size):
    loss.backward(inputs=img)
    grad = img.grad
    img.requires_grad_(False)
    grad = (grad - grad.mean()) / (grad.std() + 1e-8)
    img = img + step_size * grad
    img = clip_to_valid_range(img)
    return img

def optimize(model, img, num_iter, step_size):
    for _ in range(num_iter):
        img, shift_y, shift_x = jitter(img)
        img.requires_grad_(True)
        model(img)
        loss = compute_loss(model.activations)
        img = gradient_ascent(img, loss, step_size)
        model.zero_grad()
        img, _, _ = jitter(img, -shift_y, -shift_x)
    return img

def dream(img, num_octaves, steps_per_octave, settings, step_size=0.01, model=None):
    model = model if model is not None else Model(settings)
    img = preprocess(img)
    octaves = gaussian_pyramid(img, num_octaves)
    dream, detail = None, None
    
    for idx, octave in enumerate(octaves):
        print("Processing octave #{} with shape {}".format(idx+1, octave.shape))

        if dream is None:
            dream = optimize(model, octave, steps_per_octave, step_size)
            detail = dream - octave
        else:
            rescaled_detail = resize(detail, octave.shape[2:])
            combined_img = clip_to_valid_range(octave + rescaled_detail)
            dream = optimize(model, combined_img, steps_per_octave, step_size)
            detail = dream - octave
        
        display.clear_output(wait=True)
        show(postprocess(dream))
            
    return postprocess(dream)

def zoom_dream(img, num_frames, steps_per_frame, settings):
    model = Model(settings)
    frames = [img]
    for i in range(num_frames-1):
        img = zoom(img)
        img = dream(img, num_octaves=1, steps_per_octave=steps_per_frame, settings=settings, model=model)
        frames.append(img)
    return frames
