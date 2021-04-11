import h5py
import numpy as np
import pickle as pk
import os
from scipy.stats import gmean

def dump_noise(hdf5,name):
    name_ = 'polarbear' if name=='pb' else name

    foldername = os.path.join('polar/noise_bias',name)
    os.makedirs(foldername,exist_ok=True)
    noise_bias = {}
    for keys in spectra_keys:
        noise_bias[keys] = {}
        noise_bias[keys]['Cb'] = hdf5[f"noise_fullcross/{name_}x{name_}/{keys}"]['Cb'][:,:].mean(axis=0)
    filename = os.path.join(foldername,'allspec.pkl')

    pk.dump(noise_bias, open(filename, 'wb'))
    
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

def dump_spectra(hdf5,exp1,exp2):
    name = make_filename(exp1,exp2)
    foldername = os.path.join('polar/real_spectra_abscal_pb_fakeauto/',name)
    os.makedirs(foldername,exist_ok=True)
    spectra = {}
    spectra['bins'] = hdf5['signal_fullcross']['bins'][:].flatten()
    for keys in spectra_keys:
        spectra[keys] = {}
        key = f"real_fullcross/{make_keyname(exp1,exp2)}/{keys}"
        spectra[keys]['Cb'] = hdf5[key]['Cb'][:]
    filename = os.path.join(foldername,'allspec.pkl')
    print(f"writing {filename}")
    pk.dump(spectra,open(filename,'wb'))

def dump_dof_calculated(hdf5,exp1,exp2):
    
    
    name = make_filename(exp1,exp2)
    foldername = os.path.join('polar/dof/calculated',name)
    os.makedirs(foldername,exist_ok=True)
    spectra = {}
    spectra['bins'] = hdf5['signal_fullcross']['bins'][:].flatten()
    for keys in spectra_keys:
        spectra[keys] = {}
        #key = f"real_fullcross/{make_keyname(exp1,exp2)}/{keys}"
        
        bias = hdf5[f"noise_fullcross/{make_keyname(exp1,exp2)}/{keys}"]['Cb'][:,:].mean(axis=0)
        std = hdf5[f"noise_fullcross/{make_keyname(exp1,exp2)}/{keys}"]['Cb'][:,:].std(axis=0)
        dof = (bias/std)**2*2
        
        
        spectra[keys]['nub'] = dof
        #spectra[keys]['Cb'] = hdf5[key]['Cb'][:]
    filename = os.path.join(foldername,'allspec.pkl')
    print(f"writing {filename}")
    pk.dump(spectra,open(filename,'wb'))
    
if __name__ == "__main__":
    
    hdf5=h5py.File('/project/projectdirs/polar/data/largepatch_reanalysis/largepatch_planck_sim_set/lowell_4_20210328/final_spectra_pack_unblind.hdf5','r' )
    spectra_keys = ['tt', 'ee', 'bb', 'te', 'tb', 'eb']
    freq_name = ['planck_143','pb', 'planck_217','planck_353']
    
    # MAKE NOISE FILES
    print("MAKING NOISE FILES")
    for name in freq_name:
        dump_noise(hdf5,name)
    
    #MAKE SPECTRA FILES
    print("MAKING SPECTRA FILES")
    for i, exp1 in enumerate(freq_name):
        for j, exp2 in enumerate(freq_name):
            if i <= j:
                dump_spectra(hdf5,exp1,exp2)
    
    #MAKE DOF FILES
    print("MAKING DOF FILES")
    for i, exp1 in enumerate(freq_name):
        for j, exp2 in enumerate(freq_name):
            if i <= j:
                dump_dof_calculated(hdf5,exp1,exp2)
                
    dof = {}
    for i, exp1 in enumerate(freq_name):
        for j, exp2 in enumerate(freq_name):
            if i == j:
                dof[make_filename(exp1,exp2)] = {}
                for keys in spectra_keys:
                    bias = hdf5[f"noise_fullcross/{make_keyname(exp1,exp2)}/{keys}"]['Cb'][:,:].mean(axis=0)
                    std = hdf5[f"noise_fullcross/{make_keyname(exp1,exp2)}/{keys}"]['Cb'][:,:].std(axis=0)

                    dof[make_filename(exp1,exp2)][keys] = (bias/std)**2*2
    
    planck ={}
    for keys in spectra_keys:
        planck[keys] = {}
        planck_arr = []
        for key in dof.keys():
            if 'planck' in key:
                planck_arr.append(dof[key][keys])
        planck[keys]['nub'] = gmean(np.array(planck_arr))
    planck['bins'] = hdf5['signal_fullcross']['bins'][:].flatten()
    foldername = os.path.join('polar/dof/calculated','planck')
    filename = os.path.join(foldername,'allspec.pkl')
    print(f"writing {filename}")
    pk.dump(planck,open(filename,'wb'))
    
    planck_pb = {}
    pb_dof = dof['pb']

    for keys in spectra_keys:
        planck_pb[keys] = {}
        planck_pb[keys]['nub'] = np.sqrt(planck[keys]['nub'] * pb_dof[keys])

    planck_pb['bins'] = hdf5['signal_fullcross']['bins'][:].flatten()
    foldername = os.path.join('polar/dof/calculated','planck_pb_final')
    os.makedirs(foldername,exist_ok=True)
    filename = os.path.join(foldername,'allspec.pkl')
    print(f"writing {filename}")
    pk.dump(planck_pb,open(filename,'wb'))
    
      