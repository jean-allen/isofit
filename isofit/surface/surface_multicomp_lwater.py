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
l_shoulder = 300
r_shoulder = 2500


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
        # self.idx_lamb = np.arange(self.n_wl)    # unchanged from original
        self.n_state = len(self.statevec_names)


    # unchanged from surface_multicomp
    def component(self, x, geom):
        """We pick a surface model component using the Mahalanobis distance.

        This always uses the Lambertian (non-specular) version of the
        surface reflectance. If the forward model initialize via heuristic
        (i.e. algebraic inversion), the component is only calculated once
        based on that first solution. That state is preserved in the
        geometry object.
        """

        if self.n_comp <= 1:
            return 0
        elif hasattr(geom, "surf_cmp_init"):
            return geom.surf_cmp_init
        elif self.select_on_init and hasattr(geom, "x_surf_init"):
            x_surface = geom.x_surf_init
        else:
            x_surface = x

        # Get the (possibly normalized) reflectance
        lamb = self.calc_lamb(x_surface, geom)
        lamb_ref = lamb[self.idx_ref]
        lamb_ref = lamb_ref / self.norm(lamb_ref)

        # Mahalanobis or Euclidean distances
        mds = []
        for ci in range(self.n_comp):
            ref_mu = self.mus[ci]
            ref_Cinv = self.Cinvs[ci]
            if self.selection_metric == "Mahalanobis":
                md = (lamb_ref - ref_mu).T.dot(ref_Cinv).dot(lamb_ref - ref_mu)
            else:
                md = sum(pow(lamb_ref - ref_mu, 2))
            mds.append(md)
        closest = np.argmin(mds)

        if (
            self.select_on_init
            and hasattr(geom, "x_surf_init")
            and (not hasattr(geom, "surf_cmp_init"))
        ):
            geom.surf_cmp_init = closest
        return closest
    
    # unchanged from surface_multicomp
    def xa(self, x_surface, geom):
        """Mean of prior distribution, calculated at state x. We find
        the covariance in a normalized space (normalizing by z) and then un-
        normalize the result for the calling function. This always uses the
        Lambertian (non-specular) version of the surface reflectance."""

        lamb = self.calc_lamb(x_surface, geom)
        lamb_ref = lamb[self.idx_ref]
        mu = np.zeros(self.n_state)
        ci = self.component(x_surface, geom)
        lamb_mu = self.components[ci][0]
        lamb_mu = lamb_mu * self.norm(lamb_ref)
        mu[self.idx_lamb] = lamb_mu
        return mu


    # unchanged from surface_multicomp
    def Sa(self, x_surface, geom):
        """Covariance of prior distribution, calculated at state x. We find
        the covariance in a normalized space (normalizing by z) and then un-
        normalize the result for the calling function."""

        lamb = self.calc_lamb(x_surface, geom)
        lamb_ref = lamb[self.idx_ref]
        ci = self.component(x_surface, geom)
        Cov = self.components[ci][1]
        Cov = Cov * (self.norm(lamb_ref) ** 2)

        # If there are no other state vector elements, we're done.
        if len(self.statevec_names) == len(self.idx_lamb):
            return Cov

        # Embed into a larger state vector covariance matrix
        nprefix = self.idx_lamb[0]
        nsuffix = len(self.statevec_names) - self.idx_lamb[-1] - 1
        Cov_prefix = np.zeros((nprefix, nprefix))
        Cov_suffix = np.zeros((nsuffix, nsuffix))
        return block_diag(Cov_prefix, Cov, Cov_suffix)
    
    # unchanged from surface_multicomp
    def fit_params(self, rfl_meas, geom, *args):
        """Given a reflectance estimate, fit a state vector."""

        x_surface = np.zeros(len(self.statevec_names))
        if len(rfl_meas) != len(self.idx_lamb):
            raise ValueError("Mismatched reflectances")
        for i, r in zip(self.idx_lamb, rfl_meas):
            x_surface[i] = max(
                self.bounds[i][0] + 0.001, min(self.bounds[i][1] - 0.001, r)
            )
        return x_surface

    # unchanged from surface_multicomp
    def calc_rfl(self, x_surface, geom):
        """Non-Lambertian reflectance."""

        return self.calc_lamb(x_surface, geom)

    # modified from surface_multicomp
    def calc_lamb(self, x_surface, geom):
        """Lambertian reflectance. Modified to account for attenuation by water."""

        # Partition state vector into reflectance and water thickness
        rfl = x_surface[self.idx_lamb]
        ewt = x_surface[-1]

        # Calculate attenuation due to water
        attenuation = self.calc_attenuation(ewt)

        return rfl * attenuation
    
    # brand new!
    def calc_abs_vector(self, x_surface):
        """Calculate absorption coefficient of water at each wavelength"""
        ewt = x_surface[-1]

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
    
    # unchanged from surface_multicomp
    def dlamb_dsurface(self, x_surface, geom):
        """Partial derivative of Lambertian reflectance with respect to
        state vector, calculated at x_surface."""

        dlamb = np.eye(self.n_wl, dtype=float)
        nprefix = self.idx_lamb[0]
        nsuffix = self.n_state - self.idx_lamb[-1] - 1
        prefix = np.zeros((self.n_wl, nprefix))
        suffix = np.zeros((self.n_wl, nsuffix))
        return np.concatenate((prefix, dlamb, suffix), axis=1)