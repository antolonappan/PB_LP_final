import os
import warnings
import numpy as np
import healpy as hp
from power_spectrum import read_camb_cl, bin_cl
from fgbuster import MixingMatrix, CMB, Dust, Synchrotron

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), 'templates')
template = lambda x: os.path.join(TEMPLATES_DIR, x)

R = 0.
A_LENS = 1.
A_DUST = 0
#A_DUST = 0.035
ALPHA_DUST = -0.29
BETA_DUST = 1.6
A_SYNC = 0.
#A_SYNC = 0.035
ALPHA_SYNC = -0.1
BETA_SYNC = -3.
PL_AMPLITUDE = A_DUST
PL_TILT = ALPHA_DUST

is_positive_parameter = dict(r=False, A_lens=True,
                             A_dust=False, alpha_dust=False, beta_dust=True,
                             A_sync=True, alpha_sync=False, beta_sync=False,
                             amplitude=True, tilt=False)


def dl_tensor(lmax=None):
    return read_camb_cl(template('cambonline_default_Dl_tensor.txt'), lmax=lmax)


def dl_scalar(lmax=None):
    return read_camb_cl(template('cambonline_default_Dl_scalar.txt'), lmax=lmax)


def _add_zero_TB_EB(spectrum):
    assert spectrum.shape[-2] == 4  # TT EE BB EB
    return np.append(spectrum, np.zeros_like(spectrum[..., :2, :]), -2)


def dl_cmb_evaluator(bin_edges, spectra_slice, n_ell=None):
    if isinstance(bin_edges, str):
        bin_edges = hp.mrdfits(bin_edges)[0]
    assert len(spectra_slice.shape) == 1
    scalar = bin_cl(dl_scalar(bin_edges[-1]), bin_edges) #XXX
    scalar = _add_zero_TB_EB(scalar)[spectra_slice]
    tensor = bin_cl(dl_tensor(bin_edges[-1]), bin_edges) #XXX
    tensor = _add_zero_TB_EB(tensor)[spectra_slice]
    try:
        bb_index = int(np.where(np.arange(6)[spectra_slice] == 2)[0])
    except TypeError:
        bb_index = None
    if bb_index == 0 and len(scalar.shape) == 1:
        bb_index = np.s_[:]
    def dl_cmb(r=R, A_lens=A_LENS):
        result = scalar.copy()
        if bb_index is not None:
            result[bb_index] *= A_lens
        result += tensor * r
        if n_ell is None:
            return result
        else:
            return result + n_ell
    return dl_cmb


def dl_power_law_evaluator(bin_edges, spectra_slice, ell0, n_ell=None):
    if isinstance(bin_edges, str):
        bin_edges = hp.mrdfits(bin_edges)[0]
    assert len(spectra_slice.shape) == 1
    ell0_6 = np.empty(6)
    ell0_6[:] = ell0
    ells = np.arange(bin_edges[-1])
    ells[0] = 1.
    spectrum = ells / ell0_6[:, np.newaxis]
    #factor = ells * (ells + 1) / (2 * np.pi) # XXX Actually it returns Cls
    factor = 1.
    spectrum = spectrum[spectra_slice]
    def dl_power_law(amplitude=PL_AMPLITUDE, tilt=PL_TILT):
        tilt = np.atleast_1d(tilt)[:, np.newaxis]
        amplitude = np.atleast_1d(amplitude)[:, np.newaxis]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', message='divide by zero encountered in power')
            dl = np.squeeze(amplitude * spectrum**tilt)
            dl /= factor #XXX
        result = bin_cl(dl, bin_edges)
        if n_ell is None:
            return result
        else:
            return result + n_ell
    return dl_power_law


def dl_cross_cmb_dust_evaluator(bin_edges, spectra_slice, freqs,
                                dust_freq0, dust_temp, dust_ell0, n_ell=None):
    if isinstance(bin_edges, str):
        bin_edges = hp.mrdfits(bin_edges)[0]
    assert len(spectra_slice.shape) == 1
    mm_ev = MixingMatrix(CMB(), Dust(dust_freq0, dust_temp)).evaluator(freqs)
    dl_cmb = dl_cmb_evaluator(bin_edges, spectra_slice)
    dl_dust = dl_power_law_evaluator(bin_edges, spectra_slice, dust_ell0)
    shape = dl_dust(1, 1).shape
    if n_ell is not None:
        n_maps = n_ell.shape[0]
        auto_slice = np.array([i + i * n_maps for i in range(n_maps)])
    def dl_cross_cmb_dust(r=R, A_lens=A_LENS, A_dust=A_DUST,
                          alpha_dust=ALPHA_DUST, beta_dust=BETA_DUST):
        dl_comp = np.empty((2,)+shape, dtype=float)
        dl_comp[0] = dl_cmb(r, A_lens)
        dl_comp[1] = dl_dust(A_dust, alpha_dust)
        mm = mm_ev(beta_dust)
        result = np.einsum('fc,nc,c...->fn...', mm, mm, dl_comp)
        if n_ell is None:
            return result
        else:
            result = result.reshape((-1,) + shape)
            result[auto_slice] += n_ell
            return result.reshape((n_maps, n_maps) + shape)
    return dl_cross_cmb_dust


def dl_cross_cmb_dust_sync_evaluator(
    bin_edges, spectra_slice, freqs,
    dust_freq0, dust_temp, dust_ell0, sync_freq0, sync_ell0, n_ell=None):
    if isinstance(bin_edges, str):
        bin_edges = hp.mrdfits(bin_edges)[0]
    assert len(spectra_slice.shape) == 1
    mm_ev = MixingMatrix(CMB(), Dust(dust_freq0, dust_temp),
                         Synchrotron(sync_freq0)).evaluator(freqs)
    dl_cmb = dl_cmb_evaluator(bin_edges, spectra_slice)
    dl_dust = dl_power_law_evaluator(bin_edges, spectra_slice, dust_ell0)
    dl_sync = dl_power_law_evaluator(bin_edges, spectra_slice, sync_ell0)
    shape = dl_dust(1, 1).shape
    if n_ell is not None:
        n_maps = n_ell.shape[0]
        auto_slice = np.array([i + i * n_maps for i in range(n_maps)])
    def dl_cross_cmb_dust_sync(
            r=R, A_lens=A_LENS,
            A_dust=A_DUST, alpha_dust=ALPHA_DUST, beta_dust=BETA_DUST,
            A_sync=A_SYNC, alpha_sync=ALPHA_SYNC, beta_sync=BETA_SYNC):
        dl_comp = np.empty((3,)+shape, dtype=float)
        dl_comp[0] = dl_cmb(r, A_lens)
        dl_comp[1] = dl_dust(A_dust, alpha_dust)
        dl_comp[2] = dl_sync(A_sync, alpha_sync)
        mm = mm_ev(np.array([beta_dust, beta_sync]))
        result = np.einsum('fc,nc,c...->fn...', mm, mm, dl_comp)
        if n_ell is None:
            return result
        else:
            result = result.reshape((-1,) + shape)
            result[auto_slice] += n_ell
            return result.reshape((n_maps, n_maps) + shape)
    return dl_cross_cmb_dust_sync
