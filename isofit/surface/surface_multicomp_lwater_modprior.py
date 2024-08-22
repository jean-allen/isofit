import numpy as np
from scipy.io import loadmat
from scipy.linalg import block_diag, norm
import os
import pandas as pd

from isofit.core.common import get_refractive_index

from isofit.configs import Config

from ..core.common import svd_inv
from .surface_multicomp import MultiComponentSurface



# load in water extinction coefficients
isofit_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
path_k = os.path.join(isofit_path, "data", "iop", "k_liquid_water_ice.xlsx")
k_wi = pd.read_excel(io=path_k, sheet_name="Sheet1", engine="openpyxl")
wl_water, k_water = get_refractive_index(
        k_wi=k_wi, a=0, b=982, col_wvl="wvl_6", col_k="T = 20°C"
    )

# window across which to fit the liquid water aborption feature
l_shoulder = 350
r_shoulder = 1365


class MultiComponentSurface_Water(MultiComponentSurface):

    def __init__(self, full_config: Config):

        super().__init__(full_config)

        self.ewt_initial = 0.1
        self.ewt_bounds = [0, 0.5]            # @jean -- should account for the fact that EWT could be negative depending on priors?
        self.ewt_sigma = 100

        self.idx_ewt = self.idx_lamb[-1] + 1

        rmin, rmax = 0, 2.0
        self.statevec_names = ["RFL_%04i" % int(w) for w in self.wl] + ["Water_Thickness"]
        self.bounds = [[rmin, rmax] for w in self.wl] + [self.ewt_bounds]
        self.scale = [1.0 for w in self.wl] + [1.0]
        self.init = [0.15 * (rmax - rmin) + rmin for v in self.wl] + [self.ewt_initial]
        # self.idx_lamb = np.arange(self.n_wl)    # unchanged from original
        self.n_state = len(self.statevec_names)


    

    # modified from surface_multicomp
    def calc_lamb(self, x_surface, geom):
        """Lambertian reflectance. Modified to account for attenuation by water."""

        # Partition state vector into reflectance and water thickness
        rfl = x_surface[self.idx_lamb]
        ewt = x_surface[self.idx_ewt]

        # Calculate attenuation due to water
        attenuation = self.calc_attenuation(ewt)

        return rfl * attenuation
    
    def Sa(self, x_surface, geom):
        """Covariance of prior distribution, calculated at state x. We find
        the covariance in a normalized space (normalizing by z) and then un-
        normalize the result for the calling function."""

        lamb = self.calc_lamb(x_surface, geom)
        lamb_ref = lamb[self.idx_ref]
        ci = self.component(x_surface, geom)
        Cov = self.components[ci][1]
        Cov = Cov * (self.norm(lamb_ref) ** 2)
        
        # append water thickness covariance
        Cov = block_diag(Cov, self.ewt_sigma ** 2)

        # If there are no other state vector elements, we're done.
        if len(self.statevec_names) == len(self.idx_lamb):
            return Cov

        # Embed into a larger state vector covariance matrix
        nprefix = self.idx_lamb[0]
        nsuffix = len(self.statevec_names) - self.idx_lamb[-1] - 2   # changed to 2
        Cov_prefix = np.zeros((nprefix, nprefix))
        Cov_suffix = np.zeros((nsuffix, nsuffix))
        return block_diag(Cov_prefix, Cov, Cov_suffix)
    
    # brand new!
    def calc_abs_vector(self, x_surface):
        """Calculate absorption coefficient of water at each wavelength"""
        ewt = x_surface[self.idx_ewt]

        return self.calc_abs(ewt)
    
    # brand new!
    def calc_abs(self, ewt):
        """Calculate absorption coefficient of water at each wavelength"""
        # Subset wavelengths based on liquid water fit window and interpolate extinction coefficients
        wl_sel = self.wl[(self.wl >= l_shoulder) & (self.wl <= r_shoulder)]
        kw = np.interp(x=wl_sel, xp=wl_water, fp=k_water)
        # Calculate absorption coefficient
        abs_co_w = 4 * np.pi * kw / wl_sel

        # Fill in absorption coerricient of zero for all other wavelengths
        abs_co = np.zeros_like(self.wl)
        abs_co[(self.wl >= l_shoulder) & (self.wl <= r_shoulder)] = abs_co_w

        return abs_co
    

    # brand new!
    def calc_attenuation(self, ewt):
        """Calculate attenuation due to water at each wavelength"""
        abs_co = self.calc_abs(ewt)
        attenuation = np.exp(-ewt * 1e7 * abs_co)

        return attenuation
    

    # modified from surface_multicomp
    def dlamb_dsurface(self, x_surface, geom):
        """Partial derivative of Lambertian reflectance with respect to
        state vector, calculated at x_surface."""

        # Partition state vector into reflectance and water thickness
        rfl = x_surface[self.idx_lamb]
        ewt = x_surface[self.idx_ewt]

        # Calculate absorption coefficient of water at each wavelength
        alpha = self.calc_abs(ewt)
        atten = np.exp(-ewt * 1e7 * alpha)

        # Build matrix of partial derivatives
        # Takes format of identity matrix with attenuation coefficient in the place of the diagonal
        dlamb = np.diag(atten)

        # partial derivative of measured radiance in x_surface with respect to water thickness
        datten_dewt = -1e7 * alpha * rfl * np.exp(-ewt * 1e7 * alpha)

        # Concatenate the partial derivative of attenuation with respect to water thickness
        # onto the end of the matrix
        dlamb = np.hstack((dlamb, datten_dewt[:, np.newaxis]))

        nprefix = self.idx_lamb[0]
        nsuffix = self.n_state - self.idx_lamb[-1] - 2   # changed to 2 to account for the water thickness
        prefix = np.zeros((self.n_wl, nprefix))
        suffix = np.zeros((self.n_wl, nsuffix))
        return np.concatenate((prefix, dlamb, suffix), axis=1)