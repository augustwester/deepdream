from torch import nn
from torchvision.models import inception_v3
from functools import partial

class Model(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.model = inception_v3(pretrained=True)
        self.model.eval()
        self.settings = settings
        self._activations = []
        self.register_hooks()
    
    def forward(self, x):
        return self.model(x)
        
    def register_hooks(self):
        def hook(channels, module, input, output):
            if channels == "all":
                self._activations.append(output)
            elif type(channels) == tuple:
                self._activations.append(output[:, channels, :, :])

        for (layer_name, channels) in self.settings.items():
            if channels == "all" or type(channels) == tuple:
                module = getattr(self.model, layer_name)
                module.register_forward_hook(partial(hook, channels))

    @property
    def activations(self):
        acts = self._activations.copy()
        self._activations.clear()
        return acts
