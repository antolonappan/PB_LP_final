import numpy as np
import numpy.linalg as la


def inv(m):
    result = np.array(map(np.linalg.inv, m.reshape((-1,)+m.shape[-2:])))
    return result.reshape(m.shape)

INV_MAX_COND_NUM = 1e-8

def likelihood_fisher(dl_model, dl_model_dB, dof):
    e, v = la.eigh(dl_model)

    # NOTE: the mask == True for non-degenerate eigenvalues
    mask = e > e[...,-1:] * INV_MAX_COND_NUM

    e[~mask] = 0.
    e[mask] = 1. / e[mask]
    inv_C_CdB = np.einsum('...ji,...i,...ki,...kl', v, e, v, dl_model_dB)
    return np.sum(np.einsum('...ji,...ij', inv_C_CdB, inv_C_CdB)
                  * dof[..., np.newaxis, np.newaxis])

def likelihood_evaluator(dl_data, dl_model_evaluator, dof, block_diag=False):
    ''' Provide an evaluator of logL
    '''
    
    e_data, _ = la.eigh(dl_data)
    e_regularization = np.broadcast_to(e_data[..., -1:] * INV_MAX_COND_NUM,
                                       e_data.shape)
    e_data[~(e_data > e_data[...,-1:] * INV_MAX_COND_NUM)] = 1.
    ln_det_D = np.sum(np.log(e_data) * dof[..., np.newaxis])

    dof_n_freq = dof.sum() * dl_data.shape[-1]

    def likelihood_diagonal(*args, **kwargs):
        dl_model = dl_model_evaluator(*args, **kwargs)
        logL = - 0.5 * np.sum(dof * (dl_data / dl_model + np.log(dl_model)))
        return logL

    def likelihood_block_diagonal(*args, **kwargs):
        dl_model = dl_model_evaluator(*args, **kwargs)
        e, v = la.eigh(dl_model)

        # NOTE: the mask == True for non-degenerate eigenvalues
        mask = e > e[...,-1:] * INV_MAX_COND_NUM
        regularize = not np.all(mask)
        #e[~mask] = np.abs(e[~mask]) # Avoid warning when taking log
        if regularize:
            e[~mask] = 1.

        ln_det_C = np.sum(np.log(e) * dof[..., np.newaxis])
        if regularize:
            e[~mask] = e_regularization[~mask]
        e = 1. / e
        tr_invC_D = np.einsum('...i,...ji,...jm,...mi', e, v, dl_data, v) * dof
        tr_invC_D = np.sum(tr_invC_D)

        return - 0.5 * (tr_invC_D + ln_det_C - ln_det_D - dof_n_freq)
        '''
        inv_C = np.array([np.linalg.pinv(x) for x in dl_model])
        tr_invC_D = np.einsum('...ji,...ij->...', inv_C, dl_data) * dof
        tr_invC_D = np.sum(tr_invC_D)
        e, v = la.eigh(dl_model)
        mask = e > e[...,-1:] * INV_MAX_COND_NUM
        e[~mask] = 1.
        ln_det_C = np.sum(np.log(e) * dof[..., np.newaxis])
        return - 0.5 * (tr_invC_D + ln_det_C)
        '''

    if block_diag:
        return likelihood_block_diagonal
    else:
        return likelihood_diagonal
