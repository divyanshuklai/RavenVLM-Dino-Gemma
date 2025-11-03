import torch
import torch.nn as nn

class Raven1(nn.Module):
    def __init__(self, language_model, vision_model, adapter,
                 freeze_lm=True):
        """
        Wireframe for vision and language models.
        """
        super().__init__()

        self.gemma = language_model
        self.vit = vision_model
        self.adapter = adapter

        for param in self.vit.parameters():
            param.requires_grad = False
        for param in self.gemma.parameters():
            param.requires_grad = not freeze_lm

    def forward(self, image, captions):
        


    