from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianData:
    xyz: np.ndarray
    rot: np.ndarray
    scale: np.ndarray
    opacity: np.ndarray
    sh: np.ndarray

    def flat(self) -> np.ndarray:
        # check if scale second dimension is 3
        if self.scale.shape[-1] != 3:
            # (N, 1) -> (N, 3)
            scale = np.repeat(self.scale, 3, axis=1)
        else:
            scale = self.scale
        ret = np.concatenate(
            [self.xyz, self.rot, scale, self.opacity, self.sh], axis=-1
        )
        return np.ascontiguousarray(ret)

    def __len__(self):
        return len(self.xyz)

    @property
    def sh_dim(self):
        return self.sh.shape[-1]
