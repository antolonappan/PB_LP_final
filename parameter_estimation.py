#!/usr/bin/env python
import os
import os.path as op
from copy import deepcopy
import argparse
import inspect
import warnings
import six
import h5py
import numpy as np
from scipy.special import digamma
import emcee
import healpy as hp
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from corner import corner
from likelihood import likelihood_evaluator
import power_spectrum as ps
import models as models

class _StoreArray(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, np.array(values))

SPECTRA_NAMES = ['TT', 'EE', 'BB', 'TE', 'TB', 'EB']

## emcee parameters
MC_WALKERS = 24
MC_DISCARDED_FRAC = 0.3


def _plot_spectra(ells, dl, dof=None, model=None, **kwargs):
    axs = plt.gcf().axes
    if len(dl.shape) == 3:
        n_freq = dl.shape[0]
        for i in range(n_freq):
            for j in range(n_freq):
                color = kwargs['c'] if 'c' in kwargs else 'k'
                if dof is not None:
                    sigma = np.sqrt((dl[i, i] * dl[j, j] + dl[i, j]**2) / dof)
                    axs[i + j * n_freq].fill_between(
                        ells, dl[i, j] - sigma, dl[i, j] + sigma,
                        alpha=0.2, facecolor=color)
                if i == j:
                    axs[i + j * n_freq].loglog(ells, dl[i, j], **kwargs)
                else:
                    axs[i + j * n_freq].plot(ells, dl[i, j], **kwargs)
            axs[i + (n_freq -1) * n_freq].set_xlabel('$\ell$')
            axs[i * n_freq].set_ylabel('$\ell (\ell + 1) C_{\ell} / 2 \pi\ '
                                       '[\mu K^2]$')
    elif len(dl.shape) == 1:
        plt.loglog(ells, dl, **kwargs)
        plt.xlabel('$\ell$')
        plt.ylabel('$\ell (\ell + 1) C_{\ell} / 2 \pi\ [\mu K^2]$')


def _plot_dev(ells, dl, dof, model, **kwargs):
    axs = plt.gcf().axes
    n_freq = dl.shape[0]
    for i in range(n_freq):
        for j in range(n_freq):
            color = kwargs['c'] if 'c' in kwargs else 'k'
            sigma = np.sqrt((model[i, i] * model[j, j] + model[i, j]**2) / dof)
            i_plot = i + j * n_freq
            axs[i_plot].axhline(0, color='k')
            axs[i_plot].fill_between(ells, -3, 3, alpha=0.1, facecolor='k')
            axs[i_plot].fill_between(ells, -2, 2, alpha=0.1, facecolor='k')
            axs[i_plot].fill_between(ells, -1, 1, alpha=0.1, facecolor='k')
            axs[i_plot].plot(ells, (dl[i, j] - model[i, j]) / sigma, **kwargs)
        axs[i + (n_freq -1) * n_freq].set_xlabel('$\ell$')


def _store_dev(out_file, ells, best_fit_dl, input_dl, dof=None,
               show=False, overwrite=False):
    best_fit_dl = np.squeeze(best_fit_dl)
    input_dl = np.squeeze(input_dl)
    if not op.exists(op.dirname(out_file)):
        os.makedirs(op.dirname(out_file))
    if len(input_dl.shape) == 3:
        n_freq = input_dl.shape[0]
        plt.subplots(n_freq, n_freq, sharex=True, sharey=True,
                     figsize=(3*n_freq, 2*n_freq))
    elif len(input_dl.shape) == 1:
        plt.figure()
    else:
        raise ValueError('either cross or auto, but just BB, for time being')
    label='(Input - model) / sigma'
    _plot_dev(ells, input_dl, dof, best_fit_dl, label=label, c='DarkOrange')
    plt.legend()
    plt.tight_layout()
    fig_name = op.join(out_file)
    if not op.exists(fig_name) or overwrite:
        plt.savefig(fig_name)
    if show:
        plt.show()
    else:
        plt.close()


def _store_spectra(out_file, ells, best_fit_dl, input_dl, dof=None,
                   show=False, overwrite=False):
    best_fit_dl = np.squeeze(best_fit_dl)
    input_dl = np.squeeze(input_dl)
    if not op.exists(op.dirname(out_file)):
        os.makedirs(op.dirname(out_file))
    if len(input_dl.shape) == 3:
        n_freq = input_dl.shape[0]
        plt.subplots(n_freq, n_freq, sharex=True, figsize=(3*n_freq, 2*n_freq))
    elif len(input_dl.shape) == 1:
        plt.figure()
    else:
        raise ValueError('either cross or auto, but just BB, for time being')
    label='Best fit'
    _plot_spectra(ells, best_fit_dl, label=label, dof=dof, c='k')
    label='Input'
    _plot_spectra(ells, input_dl, label=label, c='DarkOrange')
    plt.legend()
    plt.tight_layout()
    fig_name = op.join(out_file)
    if not op.exists(fig_name) or overwrite:
        plt.savefig(fig_name)
    if show:
        plt.show()
    else:
        plt.close()


def _store_mc(out_dir, chain, logL, par_tag, overwrite=False, show=False, **kwargs):
    if not op.exists(out_dir):
        os.makedirs(out_dir)
    best_fit = chain[np.argmax(logL)]

    h5flag = 'w' if overwrite else 'w-'
    with h5py.File(op.join(out_dir, 'mcmc.hdf5'), h5flag) as f:
        f['chain'] = chain
        f['logL'] = logL
        f['parameters'] = str(par_tag)
        f['best_fit'] = best_fit

    # If no dynamic range, define ranges
    ranges = [[x.min(), x.max()] for x in chain.T]
    for i in range(len(ranges)):
        if ranges[i][0] == ranges[i][1]:
            ranges[i][0] = ranges[i][0] - np.abs(ranges[i][0]) * 0.01
            ranges[i][1] = ranges[i][1] + np.abs(ranges[i][0]) * 0.01

    corner(chain, labels=par_tag, show_titles=True, range=ranges,
            title_fmt='.4f', **kwargs)
    fig_name = op.join(out_dir, 'corner.pdf')
    if not op.exists(fig_name) or overwrite:
        plt.savefig(fig_name)
    if show:
        plt.show()
    else:
        plt.close()


def _dof_per_bin(bin_edges):
    ell = np.arange(bin_edges[-1])
    dof = 2 * ell + 1
    return np.histogram(ell, weights=dof, bins=bin_edges)[0]


def _spectra_slice(spectra_names):
    spectra_slice = np.full((6,), False, dtype=bool)
    for name in spectra_names:
        spectra_slice[SPECTRA_NAMES.index(name)] = True
    return spectra_slice


def _get_model_parameters(dl_model_evaluator):
    var_names, _, _, default = inspect.getargspec(dl_model_evaluator)
    return {n: d for n, d in zip(var_names, default)}


def _build_vec_args_converters(dl_model, free_param):
    default_param = _get_model_parameters(dl_model)
    def vec2args(vec):
        kwargs = deepcopy(default_param)
        for name, val in zip(free_param, vec):
            kwargs[name] = val
        return kwargs

    def args2vec(args_dict):
        return np.array([args_dict[name] for name in free_param])
    return vec2args, args2vec


def _trim_bin_edges(bin_edges_file, bins_retained):
    bin_edges = hp.mrdfits(bin_edges_file)[0]
    bin_edges = bin_edges[np.where(bin_edges > bins_retained[0])[0][0]-1:]
    bin_edges = bin_edges[:np.where(bin_edges < bins_retained[-1])[0][-1]+2]
    return bin_edges


def _get_dof(dof_or_fsky, bin_edges, spectra_slice, bins_mask):
    try:
        fsky = float(dof_or_fsky)
    except TypeError:
        return np.array(dof_or_fsky)[spectra_slice, bins_mask]
    else:
        return _dof_per_bin(bin_edges) * fsky * np.ones((6, 1))[spectra_slice]


def _prepare_spectrum_for_likelihood(model):
    assert isinstance(model, six.string_types)
    if 'cross' in model:
        return lambda x: x.T
    else:
        return lambda x: x


def _rearrange_cross(spectra):

    n_spectra = spectra.shape[0]
    n_map_all_cross = np.sqrt(n_spectra)
    is_all_cross = n_map_all_cross.is_integer()
    n_map_all_cross = int(n_map_all_cross)
    n_map_upper_triangle = (np.sqrt(1 + 8 * n_spectra) - 1) / 2
    is_upper_triangle = n_map_upper_triangle.is_integer()
    n_map_upper_triangle = int(n_map_upper_triangle)
    if is_all_cross and is_upper_triangle:
        print ('You may have supplied either all the cross between %i maps'
               ' or the ixj with j>=i of %i maps. I decide for the former'
               %(n_map_all_cross, n_map_upper_triangle))
        is_upper_triangle = False
    if is_all_cross:
        shape = (n_map_all_cross, n_map_all_cross) + spectra.shape[1:]
        return spectra.reshape(shape)
    elif is_upper_triangle:
        shape = (n_map_upper_triangle, n_map_upper_triangle) + spectra.shape[1:]
        full_spectra = np.zeros(shape)
        full_spectra[np.triu_indices(n_map_upper_triangle)] = spectra
        for i, j in zip(*np.tril_indices(n_map_upper_triangle)):
            full_spectra[i, j] = full_spectra[j, i]
        return full_spectra
    else:
        raise ValueError('You provided %i spectra, which can not be interpreted'
                         ' as ixj for i >= j nor for all i and j')


def main(args):
    '''
    try:
        models.A_LENS = args.A_lens
    except AttributeError:
        models.A_LENS = 0.
    '''

    # Read and trim data
    dl_data = np.array(args.input)

    bin_edges_mask = (args.ell_range[0] <= args.bin_edges) * (args.bin_edges <= args.ell_range[1])
    bin_edges = args.bin_edges[bin_edges_mask]
    bins_mask = np.delete(bin_edges_mask, np.where(bin_edges_mask == True)[0][0])
    bins_mask = bins_mask[:dl_data.shape[-1]]  # bin edges > ps bins
    spectra_slice = _spectra_slice(args.spectra)
    if dl_data.shape[-1] == args.bin_edges.shape[-1]:  # ps beyond last edge
        dl_data = dl_data[..., :-1]
    dl_data = dl_data[..., spectra_slice, bins_mask]
    is_cross = 'cross' in args.model
    if is_cross:
        dl_data = _rearrange_cross(dl_data)
    dof = _get_dof(args.dof, bin_edges, spectra_slice, bins_mask)
    if args.n_ell is not None:
        args.n_ell = np.array(args.n_ell)[..., spectra_slice, bins_mask]
        if args.pb_noise_factor:
            print ('noise factor applied', args.pb_noise_factor)
            args.n_ell[1, :3] *= args.pb_noise_factor #XXX


    # Prepare likelihood-related functions
    model_evaluator = getattr(models, 'dl_%s_evaluator'%args.model)
    args_names = inspect.getargspec(model_evaluator).args[2:]
    dl_model = model_evaluator(bin_edges, spectra_slice,
                               *[getattr(args, n) for n in args_names])
    vec2args, args2vec = _build_vec_args_converters(dl_model, args.fit_for)
    prepare_spectrum = _prepare_spectrum_for_likelihood(args.model)

    if 'dust' in args.model and args.beta_prior and 'beta_dust' in args.fit_for:
        i_beta = args.fit_for.index('beta_dust')
        if args.alpha_prior:
            i_alpha = args.fit_for.index('alpha_dust')
        else:
            i_alpha = -1

        def prior(x):
            res = -((x[i_beta] - args.param0[i_beta]) / args.beta_prior)**2 / 2
            if i_alpha >= 0:
                res += -((x[i_alpha] - args.param0[i_alpha]) / args.alpha_prior)**2 / 2
            return res
    else:
        def prior(x):
            return 0.

    def dl_model_for_logL(x):
        dl = dl_model(**vec2args(x))
        return prepare_spectrum(dl)

    if args.test: #XXX
        dl_data = dl_model_for_logL(args.param0).T
        args.out_dir += '__test'
    


    logL = likelihood_evaluator(prepare_spectrum(dl_data), dl_model_for_logL,
                                prepare_spectrum(dof), is_cross)

    logL(args.param0)
    def logPost(x):  # Add priors to logL
        if not args.no_prior and np.any(x[args2vec(models.is_positive_parameter)] < 0):
            return -np.inf
        try:
            val = prior(x) + logL(x) 
        except BaseException as err:
            message = ('Setting logL to Nan after catching the following '
                       'exception:\n' + err.message)
            warnings.warn(message, RuntimeWarning)
            print(message)
            val = np.nan

        if np.isnan(val):
            par_tag = '\t'.join(args.fit_for)
            par_val = '\t'.join([str(v) for v in x])
            message = 'likelihood returned NaN for\n%s\n%s' 
            message = message % (par_tag, par_val)
            warnings.warn(message, RuntimeWarning)
            return -np.inf
        return val

    # Plot spectra with initial guess
    bins = (bin_edges[1:] + bin_edges[:-1]) / 2.
    '''
    dl_initial_fit = dl_model(**vec2args(args.param0))
    _store_spectra(op.join(args.out_dir, 'initial_freq_spectra.pdf'), bins, dl_initial_fit,
                   dl_data, overwrite=args.overwrite, show=args.show)
    '''

    # Initial conditions
    if args.mc_n_step:
        #par0s = args.param0[:, np.newaxis] * np.linspace(0.8, 1.2, num=MC_WALKERS)
        #map(np.random.shuffle, par0s)
        #par0s = par0s.T
        par0s = args.param0 + 1e-4 * np.random.randn(MC_WALKERS, 4)
    

        # MCMC
        sampler = emcee.EnsembleSampler(MC_WALKERS, len(args.param0), logPost,
                                        a=2., )
        par0s, _, _ = sampler.run_mcmc(par0s, MC_DISCARDED_FRAC * args.mc_n_step)
        sampler.reset()
        sampler.run_mcmc(par0s, args.mc_n_step)
        best_fit = sampler.flatchain[np.argmax(sampler.flatlnprobability)]
    else:
        best_fit = args.param0[:, np.newaxis]

    # Save result
    '''
    db_args = vec2args(best_fit)
    for k in db_args:
        if 'A' in k:
            db_args[k] = 0
        elif k == 'r':
            db_args[k] = 1
    fish_dl_model_dB = prepare_spectrum(dl_model(**db_args))
    fish_dl_model = prepare_spectrum(dl_model(**vec2args(best_fit)))
    for i, n_ell_f in enumerate(args.n_ell):
        fish_dl_model_dB[:, i, i] -= n_ell_f

    print db_args
    print vec2args(best_fit)
    print likelihood_fisher(fish_dl_model, fish_dl_model_dB, prepare_spectrum(dof))
    _store_spectra('./deleteme.pdf', bins, fish_dl_model_dB.T,
                   fish_dl_model.T, overwrite=False, show=args.show)
    '''
    print('BEST fit', best_fit)
    dl_best_fit = dl_model(**vec2args(best_fit))
    _store_spectra(op.join(args.out_dir, 'freq_spectra.pdf'), bins, dl_best_fit,
                   dl_data, dof=dof, overwrite=args.overwrite, show=args.show)
    _store_dev(op.join(args.out_dir, 'freq_spectra_dev.pdf'), bins, dl_best_fit,
               dl_data, dof, overwrite=args.overwrite, show=args.show)
    if args.mc_n_step:
        _store_mc(args.out_dir, sampler.flatchain, sampler.flatlnprobability,
                  args.fit_for, overwrite=args.overwrite, show=args.show,
                  truths=args.param0)
        with h5py.File(op.join(args.out_dir, 'mcmc.hdf5'), 'r+') as f:
            f['dl_data'] = dl_data
            f['dl_param0'] = dl_model(**vec2args(args.param0))
            f['dl_best_fit'] = dl_best_fit
            f['param0'] = args.param0
            f['dof'] = dof
            chi2_eff = np.max(sampler.flatlnprobability)
            chi2_eff *= -2
            chi2_param0 = -2 * logL(args.param0)
            n_freq = len(dl_data)
            chi2_exp = n_freq * np.log(dof / 2) 
            chi2_exp -= digamma((dof - np.arange(n_freq)[:, None])/2).sum(0)
            chi2_exp *= dof - n_freq - 1
            chi2_exp = chi2_exp.sum()
            chi2_dev = (chi2_eff - chi2_exp) / np.sqrt(2 * chi2_exp)
            chi2_param0_dev = (chi2_param0 - chi2_exp) / np.sqrt(2 * chi2_exp)
            f['chi2_eff'] = chi2_eff
            f['chi2_exp'] = chi2_exp
            f['chi2_dev'] = chi2_dev
            f['chi2_param0'] = chi2_param0
            f['chi2_param0_dev'] = chi2_param0_dev
            print('chi2_eff', chi2_eff)
            print('chi2_exp', chi2_exp)
            print('chi2_dev', chi2_dev)
            print('chi2_param0', chi2_param0)
            print('chi2_param0_dev', chi2_param0_dev)
            rs = np.sort(sampler.flatchain[:,0])
            rs = rs[rs>0]
            rs_95_cl = rs[int(rs.size * 0.95)]
            f['95cl'] = rs_95_cl
            print('r < %.2f 95 %s cl' % (rs_95_cl, '%'))
    '''
    if args.model == 'cross_cmb_dust_sync':  # XXX deletem ASAP
        mm = MixingMatrix(CMB(), Dust(args.dust_freq0, args.dust_temp),
                          Synchrotron(args.sync_freq0))
        mm = mm.evaluate(args.freqs, 0., 0.)
        from mapext.hpcs.likelihood import solve_for_components
        w = 1./args.n_ell.T
        dl_comp = solve_for_components(mm, w, dl_data.swapaxes(-1,-2))
        dl_comp = solve_for_components(mm, w, dl_comp.swapaxes(-1,0)).swapaxes(-1,-2)
        dl_comp_fit = solve_for_components(mm, w, dl_best_fit.swapaxes(-1,-2))
        dl_comp_fit = solve_for_components(mm, w, dl_comp_fit.swapaxes(-1,0)).swapaxes(-1,-2)
        _store_spectra(op.join(args.out_dir, 'comp_spectra.pdf'), bins,
                       dl_comp_fit, dl_comp, args.overwrite, args.show)
    '''

    return best_fit

def _build_fits_fields_loader(fields=None):
    ''' Build a loader of the specified fields, default: all
    '''
    def _fits_fields_loader(filename):
        all_fields = hp.mrdfits(filename)
        if fields is None:
            return all_fields
        else:
            try:
                return [all_fields[i] for i in fields]
            except TypeError:
                return all_fields[0]

    class LoadFitsField(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if isinstance(values, list):
                setattr(namespace, self.dest,
                        [_fits_fields_loader(v) for v in values])
            else:
                setattr(namespace, self.dest, _fits_fields_loader(values))
    return LoadFitsField

_LoadBinEdges = _build_fits_fields_loader(0)
_LoadSpectra = _build_fits_fields_loader(range(1, 7))

class _LoadDOF(argparse.Action):
    def __call__(self, parser, namespace, value, option_string=None):
        try:
            float(value)
        except ValueError:
            setattr(namespace, self.dest, hp.mrdfits(value)[1:])  # dof (fits)
        else:
            setattr(namespace, self.dest, value)  # float (fsky)


_LoadFrequencies = _StoreArray


def get_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # Common args
    parser.add_argument('--bin_edges', required=True, action=_LoadBinEdges,
                        help='Edges of the ell bins (fits)')
    parser.add_argument('--out_dir', required=True,
                        help='Output_folder (created if not existing).'
                             'It contains: hdf5 with the outcome of the MC,'
                             'corner plot of the MC chain')
    parser.add_argument('--spectra', nargs='+',
                        help='Spectra to be considered: '
                             'any choice of TT EE BB TE TB EB')
    parser.add_argument('--mc_n_step', type=int, required=True,
                        help='Number of steps in the MC chain')
    parser.add_argument('--pb_noise_factor', type=float, required=False,
                        help='Multiply PB noise by this factor')
    parser.add_argument('--ell_range', type=int, nargs=2,
                        help='Min and max ell cosidered in the likelihood')
    parser.add_argument('--input', nargs='+', action=_LoadSpectra,
                        help='Data Dl (fits)')
    parser.add_argument('--n_ell', nargs='+', action=_LoadSpectra,
                        help='Noise Dl (fits)')
    parser.add_argument('--no_prior', action='store_true',
                        help='Do not use priors on the free parameters')
    parser.add_argument('--dof', required=True, action=_LoadDOF,
                        help='Can be '
                             '1) spectrum (fits): each entry reports the dof'
                             '   for each ell and spectrum component'
                             '2) f_sky from which the dof are inferred')
    parser.add_argument('--fit_for', nargs='+',
                        help='Parameters to be estimated. They must correspond '
                             'to variables of the model evaluator')
    parser.add_argument('--param0', nargs='+', action=_StoreArray, type=float,
                        help='Starting parameters')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite output files, if they already exist')
    parser.add_argument('--test', action='store_true',
                        help='The data are replaced with the model evaluated at'
                             'param0')
    parser.add_argument('--show', action='store_true',
                        help='Show the corner plot')
    subparser = parser.add_subparsers(help='Model to be fitted to the data',
                                      dest='model')
    # CMB
    sparser = subparser.add_parser('cmb')
    # Power law
    sparser = subparser.add_parser('power_law')
    sparser.add_argument('--ell0', type=int)
    # Cross spectra containing CMB and Dust
    sparser = subparser.add_parser('cross_cmb_dust')
    sparser.add_argument('--freqs', type=float, nargs='+',
                         action=_LoadFrequencies,
                         help='Frequncies [GHz]')
    sparser.add_argument('--dust_ell0', type=int)
    sparser.add_argument('--dust_freq0', type=float)
    sparser.add_argument('--dust_temp', type=float)
    sparser.add_argument('--beta_prior', type=float,
                         help='std prior')
    sparser.add_argument('--alpha_prior', type=float,
                         help='std prior')
    # Cross spectra containing CMB, Dust and Synchrotron
    sparser = subparser.add_parser('cross_cmb_dust_sync')
    sparser.add_argument('--freqs', type=float, nargs='+', action=_StoreArray,
                         help='Frequncies [GHz]')
    sparser.add_argument('--dust_ell0', type=int)
    sparser.add_argument('--dust_freq0', type=float)
    sparser.add_argument('--dust_temp', type=float)
    sparser.add_argument('--sync_ell0', type=int)
    sparser.add_argument('--sync_freq0', type=float)
    sparser.add_argument('--A_lens', type=float)
    return parser


if __name__ == '__main__':
    parser = get_parser()
    try:
        args = parser.parse_args()
    except SystemExit as err:
        print ('\nIf you get some unexpected parsing error, try \n'
               '- putting the model-specific arguments at the end \n'
               '- putting a flag with fixed number of arguments right before '
               'the model name (e.g. mc_n_step)')
        raise err

    main(args)
