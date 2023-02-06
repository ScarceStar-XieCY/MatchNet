import cv2
import numpy as np
import torch

from torch import nn
from torch.nn import functional as F

from matchnet.code.ml.models.base import BaseModel
from matchnet.code.ml.models.fcn import FCNet
from matchnet.code.utils.misc import anglepie
from tools.matrix import gen_rot_mtx_anticlockwise

from torchvision import transforms

trans = transforms.ToTensor()


class CorrespondenceNet(BaseModel):
    """The correspondence prediction network.

    Attributes:
        num_channels: (int) The number of channels in the input tensor.
        num_descriptor: (int) The dimension of the descriptor space.
        num_rotations: (int) The number of rotations to perform on the source heightmap.
    """

    def __init__(self, num_channels, num_descriptor, num_rotations):
        super().__init__()

        self.num_channels = num_channels
        self.num_descriptor = num_descriptor
        self.num_rotations = num_rotations

        self._rotations = anglepie(num_rotations, False)

        self._fcn = FCNet(num_channels, num_descriptor)

    def forward(self, x, uc, vc):
        """Forwards the target and rotated sources through the network.
        """
        batch_size = len(x)
        device = torch.device("cuda" if x.is_cuda else "cpu")

        # split input tensor into source and target tensors
        xs, xt = x[:, :self.num_channels, :, :], x[:, self.num_channels:, :, :]

        xs = xs.detach().cpu().numpy().squeeze()

        # apply `num_rotations` to source tensors
        all_rotate = []
        for angle in self._rotations:
            aff_1 = np.eye(3)
            aff_1[:2, 2] = [-vc, -uc]
            aff_2 = gen_rot_mtx_anticlockwise(np.radians(angle))
            aff_3 = np.eye(3, 3)
            aff_3[:2, 2] = [vc, uc]
            affine = aff_3 @ aff_2 @ aff_1
            affine = affine[:2, :]
            one_rotate= [cv2.warpAffine(xs[i], affine, (xs.shape[2], xs.shape[1]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE) for i in range(self.num_channels)]
            all_rotate.append(np.stack(one_rotate))

        # concatenate all rotated sources and target into single tensor along batch dim
        # ins = [np.stack(all_rotate[i]) for i in range(self._rotations)]
        if x.is_cuda:
            all_rotate = [torch.FloatTensor(i).cuda().unsqueeze(0) for i in all_rotate]
        else:
            all_rotate = [torch.FloatTensor(i).unsqueeze(0) for i in all_rotate]
        all_rotate.append(xt)
        all_rotate = torch.cat(all_rotate, dim=0) # (21,4,W//2,H)

        # forward pass
        out = self._fcn(all_rotate) # (21,64,W//2,H)

        # separate into source and target outputs per batch
        out_s = []
        for b in range(batch_size):
            sources = [out[i * batch_size + b] for i in range(self.num_rotations)]
            out_s.append(torch.stack(sources, dim=0))
        out_s = torch.stack(out_s, dim=0) #(1,20,64,240,212)
        out_t = out[
            batch_size * self.num_rotations : batch_size * self.num_rotations + batch_size
        ]

        return out_s, out_t

    @property
    def rotations(self):
        return self._rotations
