"""HERNet with optional training-only structural prediction heads."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .architecture import Encoder, HERModule, Decoder


def center_pad_crop(x, shape):
    """Match the original convention: the leading side receives the odd pixel."""
    h, w = x.shape[-2:]
    th, tw = shape
    dh, dw = max(th-h, 0), max(tw-w, 0)
    if dh or dw:
        x = F.pad(x, (dw-dw//2, dw//2, dh-dh//2, dh//2), mode='replicate')
    h, w = x.shape[-2:]
    top, left = (h-th)-(h-th)//2, (w-tw)-(w-tw)//2
    return x[..., top:top+th, left:left+tw]


class HERNet(nn.Module):
    def __init__(self, auxiliary=False):
        super().__init__()
        self.auxiliary = auxiliary
        # Names retained to allow strict loading of the archived trained tensors.
        self.fe_block = Encoder(1, 64)
        self.tbsf_block = HERModule(256, (2,2,2,2,2), 16, auxiliary=auxiliary)
        self.re_block = Decoder(64)

    def forward(self, x, return_auxiliary=False):
        if x.ndim != 4 or x.shape[1] != 1 or any(v % 32 for v in x.shape[-2:]):
            raise ValueError('Expected Bx1xHxW with H,W divisible by 32; use enhance() for arbitrary sizes.')
        x = (x-x.mean((-2,-1),keepdim=True))/(x.std((-2,-1),keepdim=True)+1e-9)
        outputs = self.tbsf_block(self.fe_block(x))
        enhanced = self.re_block(outputs[0])
        if return_auxiliary:
            if not self.auxiliary:
                raise ValueError('Construct HERNet(auxiliary=True) to request prediction heads.')
            return enhanced, outputs[1], outputs[2]
        return enhanced

    def enhance(self, x):
        """Enhance one 1xHxW float image without resizing or changing polarity."""
        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError('Expected a single-channel 1xHxW tensor in [0,1].')
        shape = x.shape[-2:]
        aligned = tuple((v+31)//32*32 for v in shape)
        x = center_pad_crop(x, aligned)
        x = center_pad_crop(x, (aligned[0]+64, aligned[1]+64))
        return center_pad_crop(self(x[None])[0], shape)


def load_model(checkpoint, device='cpu'):
    """Load an inference-only tensor checkpoint; no pickled model execution."""
    path = Path(checkpoint)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    state = payload.get('state_dict', payload)
    model = HERNet(auxiliary=False)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
