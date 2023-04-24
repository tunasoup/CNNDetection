import torch
from torchvision import transforms

from .networks.resnet import resnet50


class Detector(torch.nn.Module):
    def __init__(self):
        super(Detector, self).__init__()

        self._transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._model = resnet50(num_classes=1)

    def load_pretrained(self, weights_path):
        state_dict = torch.load(weights_path, map_location="cpu")
        self._model.load_state_dict(state_dict["model"])

    def forward(self, img):
        img = self._transform(img)
        sig = self._model(img).sigmoid()
        label = torch.round(sig).to(torch.int)
        return label, sig
