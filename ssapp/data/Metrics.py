import numpy as np
import torch


def relRMSE(y_pred, y_true, sample_weights=None):
    """Better relative MSE implementation than the sklearn version (which can yield NANs!)"""
    """See https://alex.miller.im/posts/linear-model-custom-loss-function-regularization-python/"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert len(y_true) == len(y_pred)

    if np.any(y_true == 0):
        idx = np.where(y_true == 0)
        y_true = np.delete(y_true, idx)
        y_pred = np.delete(y_pred, idx)
        if type(sample_weights) != type(None):
            sample_weights = np.array(sample_weights)
            sample_weights = np.delete(sample_weights, idx)

    nomin = np.power(y_true - y_pred, 2).sum()
    denom = np.power(y_true, 2).sum() + 1.0e-9
    if type(sample_weights) == type(None):
        return np.sqrt(nomin / denom)
    else:
        sample_weights = np.array(sample_weights)
        assert len(sample_weights) == len(y_true)
        return np.dot(sample_weights, np.sqrt(nomin / denom)) / sum(sample_weights)

def relRMSE_pytorch(y_pred, y_true, sample_weights=None):
    """Torch Implementation of relRMSE"""
    assert type(y_true) is type(torch.tensor([]))
    assert type(y_pred) is type(torch.tensor([]))

    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    assert len(y_true) == len(y_pred)

    if torch.any(y_true == 0):
        idx = torch.where(y_true == 0)
        y_true = torch.delete(y_true, idx)
        y_pred = torch.delete(y_pred, idx)
        if type(sample_weights) != type(None):
            sample_weights = torch.tensor(sample_weights)
            sample_weights = torch.delete(sample_weights, idx)

    nomin = ((y_true - y_pred)**2).sum()
    denom = (y_true**2).sum() + 1.0e-9
    if type(sample_weights) == type(None):
        return torch.sqrt(nomin / denom)
    else:
        sample_weights = np.array(sample_weights)
        assert len(sample_weights) == len(y_true)
        return torch.dot(sample_weights, torch.sqrt(nomin / denom)) / sum(sample_weights)


def relRMSEComplex(y_pred, y_true, sample_weights=None, co_cross_norm = lambda co,cross : np.sqrt(co**2+cross**2)):
    """Better relative MSE implementation than the sklearn version (which can yield NANs!)"""
    """See https://alex.miller.im/posts/linear-model-custom-loss-function-regularization-python/"""
    """ Expecting both y to have dimensions size (num_samples,theta,phi,2*num_complex_values)"""

    if type(y_true) is not type(np.complex) or type(y_pred) is not type(np.complex):
        y_true = np.array(y_true).view(dtype=np.complex)
        y_pred = np.array(y_pred).view(dtype=np.complex)

    print(y_true.shape)
    assert len(y_true) == len(y_pred)

    if np.any(y_true == 0):
        idx = np.where(y_true == 0)
        y_true = np.delete(y_true, idx)
        y_pred = np.delete(y_pred, idx)
        if type(sample_weights) != type(None):
            sample_weights = np.array(sample_weights)
            sample_weights = np.delete(sample_weights, idx)

    nomin = np.power(y_true - y_pred, 2).sum(axis=(0,1,2))
    denom = np.power(y_true, 2).sum(axis = (0,1,2)) + 1.0e-9

    if type(sample_weights) == type(None):
        return np.sqrt(nomin / denom)
    else:
        sample_weights = np.array(sample_weights)
        assert len(sample_weights) == len(y_true)
        return np.dot(sample_weights, np.sqrt(nomin / denom)) / sum(sample_weights)

    
    