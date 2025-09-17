import types
import torch
import torch.nn as nn

from src.utils.training import LitCaptioner


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Add a dummy parameter so optimizers have something to optimize
        self._w = nn.Parameter(torch.tensor(0.0))

    def forward(self, images, captions):
        # Return loss in HF style object
        return types.SimpleNamespace(loss=self._w + torch.tensor(0.0, requires_grad=True))


def test_litcaptioner_basic():
    lm = LitCaptioner(TinyModel(), optimizer_cfg={"lr": 1e-3})
    opt = lm.configure_optimizers()
    assert isinstance(opt, torch.optim.Optimizer)

    images = [object(), object()]
    captions = ["a", "b"]
    loss = lm.training_step((images, captions), 0)
    assert loss.requires_grad
