import numpy as np
from scipy.io import loadmat
from scipy.linalg import block_diag, norm

from isofit.configs import Config

from ..core.common import svd_inv
from .surface_multicomp import MultiComponentSurface

class MultiComponentSurface_Water(MultiComponentSurface):

    def __init__(self, full_config: Config):

        super().__init__(full_config)

        self.ewt_initial = 0.0
        self.ewt_bounds = [-0.5, 0.5]            # @jean -- should account for the fact that EWT could be negative depending on priors?

        rmin, rmax = 0, 2.0
        self.statevec_names = ["RFL_%04i" % int(w) for w in self.wl] + ["Water_Thickness"]
        self.bounds = [[rmin, rmax] for w in self.wl] + [self.ewt_bounds]
        self.scale = [1.0 for w in self.wl] + [1.0]
        self.init = [0.15 * (rmax - rmin) + rmin for v in self.wl] + [self.ewt_initial]
        self.idx_lamb = np.arange(self.n_wl)
        self.n_state = len(self.statevec_names) + 1

