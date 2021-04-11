import os
from math import ceil
import numpy as np
import healpy as hp

NAMES = ['TT', 'EE', 'BB', 'TE', 'TB', 'EB']

def read_camb_cl(filename, lmax=None):
    theory_cl = np.loadtxt(filename, usecols=(0,1,2,3,4), unpack=True)
    if lmax is None:
        lmax = theory_cl.shape[-1] + 2
    elif lmax < 2:
        raise ValueError('lmax must be at least 2')
    else:
        lmax = int(ceil(lmax))
    cls = np.zeros((4, lmax))
    cls[:, 2:] = theory_cl[1:, :lmax-2]
    cls[:, :2] = 0
    return cls


def read_xpure_cl(filename, bin_center=False):
    file_content =  hp.mrdfits(filename)
    dl = np.array(file_content[1:])
    if bin_center:
        return dl, file_content[0]
    else:
        return dl


def write_xpure_cl(filename, ells, spectrum, **kwargs):
    file_content = [ells] + [s for s in spectrum]
    colnames = ['Ell'] + NAMES
    hp.mwrfits(filename, file_content, colnames=colnames, **kwargs)


def cl_to_dl(ells):
    return ells * (ells + 1) / (2 * np.pi)


def bin_cl(cl, bin_edges, bin_center=False):
    ''' Averages within the bins
    '''
    if isinstance(bin_edges, str):
        bin_edges = hp.mrdfits(bin_edges)
    ells = np.arange(cl.shape[-1])
    bins = np.digitize(ells, bin_edges)
    bin_hits = np.bincount(bins) + 1e-16  # Avoid division by zero error
    if len(cl.shape) == 1:
        binned_cl = (np.bincount(bins, cl) / bin_hits)[1:len(bin_edges)]
    elif len(cl.shape) == 2:
        binned_cl = [np.bincount(bins, c) / bin_hits for c in cl]
        binned_cl = np.array(binned_cl)[:, 1:len(bin_edges)]
    else:
        ValueError('Unsupported cl shape %s'%str(cl.shape))
    if bin_center:
        ell_bin_center = np.bincount(bins, ells) / bin_hits
        return binned_cl, ell_bin_center[1:len(bin_edges)]
    else:
        return binned_cl

def naive_bin_cl(cl, bintab):
    bintab = bintab.astype(int)
    ntt = bintab.size - 1
    binned_cl = np.zeros(cl.shape[:-1] + (ntt,))
    for b in range(ntt):
        for l in range(bintab[b], bintab[b+1]):
            binned_cl[:, b] += l * (l + 1.) / (2.*np.pi) / (bintab[b+1] - bintab[b]) * cl[:, l]
    return binned_cl
