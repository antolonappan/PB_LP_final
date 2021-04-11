import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.stats
import pickle as pk
from tqdm import tqdm

def make_filename(exp1,exp2):
    if exp1 == exp2:
        return exp1
    else:
        if exp1 == "pb":
            return f"{exp2}_{exp1}"
        else:
            return f"{exp1}_{exp2}"

def make_keyname(exp1,exp2):
    exp1 = 'polarbear' if exp1 == 'pb' else exp1
    exp2 = 'polarbear' if exp2 == 'pb' else exp2
    
    if exp1 == 'polarbear':
        return f"{exp2}x{exp1}"
    else:
        return f"{exp1}x{exp2}"

def dump_spectra(hdf5,exp1,exp2,sim):
    name = make_filename(exp1,exp2)
    foldername = os.path.join("polar/sum_r0p00_spectra_fid_pb_fakeauto/",f"sim{sim:03}",name)
    os.makedirs(foldername,exist_ok=True)
    spectra = {}
    spectra['bins'] = hdf5['signal_fullcross']['bins'][:].flatten()
    for keys in spectra_keys:
        dust = 9.e-3*(binslmid/80)**(-0.6)* 1e-12
        Ds_dust = np.dot(hdf5['signal_fullcross/bb']['bpwf'][:],dust)
        spectra[keys] = {}
        signal_key = f"signal_fullcross/bb"
        noise_key = f"noise_fullcross/{make_keyname(exp1,exp2)}/{keys}"
        spectra[keys]['Cb'] = hdf5[signal_key]['Cb'][sim,:] + hdf5[noise_key]['Cb'][sim,:] +Ds_dust
    filename = os.path.join(foldername,'allspec.pkl')
    pk.dump(spectra,open(filename,'wb'))

def dump_sim(sim):
    for i, exp1 in enumerate(freq_name):
        for j, exp2 in enumerate(freq_name):
            if i <= j:
                dump_spectra(hdf5,exp1,exp2,sim)

if __name__ == "__main__":
    hdf5=h5py.File('/project/projectdirs/polar/data/largepatch_reanalysis/largepatch_planck_sim_set/lowell_4_20210328/final_spectra_pack_unblind.hdf5','r' )
    spectra_keys = ['tt', 'ee', 'bb', 'te', 'tb', 'eb']
    freq_name = ['planck_143','pb', 'planck_217','planck_353']
    binslmid = hdf5['signal_fullcross/tt/binslmid'][:]
    
    for i in tqdm(range(96),desc='Creating Simulations',unit='simulatoin'):
        dump_sim(i)