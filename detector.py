from pathlib import Path
from typing import Optional

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

    def load_pretrained(self, weights_path: Path) -> None:
        state_dict = torch.load(weights_path, map_location='cpu')
        self._model.load_state_dict(state_dict['model'])

    def configure(self, device: Optional[str], training: Optional[bool] = None, **kwargs) -> None:
        if device is not None:
            self.to(device)
            self._model.to(device)

        if training is None:
            return

        if training:
            self.train()
            self._model.train()
        else:
            self.eval()
            self._model.eval()

    def forward(self, img_batch: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        img_batch = self._transform(img_batch)
        sig = self._model(img_batch).sigmoid()
        label = torch.round(sig).to(torch.int)
        return label, sig
