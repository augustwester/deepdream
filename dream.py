import torchvision
import IPython.display as display
from torch.autograd import Variable
from helpers import preprocess, postprocess, gaussian_pyramid, jitter, clip_to_valid_range, scale, zoom, show
from functools import partial

module_names = {}
activations = []

def load_model(settings):
    model = torchvision.models.inception_v3(pretrained=True)
    model.eval()
    register_hooks(model, settings)
    return model

def hook(settings, module, input, output):
    module_name = module_names[module]
    channels = settings[module_name]
    if channels == "all":
        activations.append(output)
    elif type(channels) == tuple:
        activations.append(output[:, channels, :, :])
    
def register_hooks(model, settings):
    for (layer_name, channels) in settings.items():
        if channels == "all" or type(channels) == tuple:
            module = getattr(model, layer_name)
            module.register_forward_hook(partial(hook, settings))
            module_names[module] = layer_name

def compute_loss(activations):
    loss = activations[0].mean()
    activations.clear()
    return loss

def gradient_ascent(img, loss, lr):
    loss.backward()
    grad = img.grad
    grad = (grad - grad.mean()) / (grad.std() + 1e-8)
    img = img + lr * grad
    img = clip_to_valid_range(img)
    return img

def optimize(model, img, steps, lr):
    for i in range(steps):
        img, shift_y, shift_x = jitter(img)
        img = Variable(img, requires_grad=True)
        model(img)
        loss = compute_loss(activations)
        img = gradient_ascent(img, loss, lr)
        model.zero_grad()
        img, _, _ = jitter(img, -shift_y, -shift_x)
    return img

def dream(img, num_octaves, steps_per_octave, settings, lr=0.01, model=None):
    model = model if model is not None else load_model(settings)
    img = preprocess(img)
    octaves = gaussian_pyramid(img, num_octaves)
    dream, detail = None, None
    
    for idx, octave in enumerate(octaves):
        print("Processing octave #{} with shape {}".format(idx+1, octave.shape))

        if dream is None:
            dream = optimize(model, octave, steps_per_octave, lr)
            detail = dream - octave # Extracting the changes to the original image
        else:
            # Upscaling changes applied to the previous (smaller) octave to same size as current octave
            new_width = octave.shape[-1]
            old_width = dream.shape[-1]
            factor = new_width / old_width
            rescaled_detail = scale(detail, factor)

            # Combining previous dream and current octave; then running GA on *that*.
            combined_img = octave + rescaled_detail
            combined_img = clip_to_valid_range(combined_img)
            dream = optimize(model, combined_img, steps_per_octave, lr)
            detail = dream - octave
        
        display.clear_output(wait=True)
        show(postprocess(dream))
            
    return postprocess(dream)

def zoom_dream(img, num_frames, steps_per_frame, settings):
    model = load_model(settings)
    frames = [img]
    for i in range(num_frames-1):
        img = zoom(img)
        img = dream(img, num_octaves=1, steps_per_octave=steps_per_frame, settings=settings, model=model)
        frames.append(img)
    return frames
