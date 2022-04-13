import numpy as np
from MultivariateUniform import MultivariateUniform

def make_2d_link_active_stripes(shape, mu, off):
    assert len(shape) == 2+1, 'need to pass shape suitable for 2D gauge theory'
    assert shape[0] == len(shape[1:]), 'first dim of shape must be Nd'
    assert mu in (0,1), 'mu must be 0 or 1'
    mask = np.zeros(shape)#.astype(np.uint8)
    if mu == 0:
        mask[mu,:,0::4] = 1
    elif mu == 1:
        mask[mu,0::4] = 1
    nu = 1-mu
    mask = np.roll(mask, off, axis=nu+1)
    return mask


def make_single_stripes(shape, mu, off):
    assert len(shape) == 2, 'need to pass 2D shape'
    assert mu in (0,1), 'mu must be 0 or 1'
    mask = np.zeros(shape)#.astype(np.uint8)
    if mu == 0:
        mask[:,0::4] = 1
    elif mu == 1:
        mask[0::4] = 1
    mask = np.roll(mask, off, axis=1-mu)
    return mask


def make_double_stripes(shape, mu, off):
    assert len(shape) == 2, 'need to pass 2D shape'
    assert mu in (0,1), 'mu must be 0 or 1'
    mask = np.zeros(shape)#.astype(np.uint8)
    if mu == 0:
        mask[:,0::4] = 1
        mask[:,1::4] = 1
    elif mu == 1:
        mask[0::4] = 1
        mask[1::4] = 1
    mask = np.roll(mask, off, axis=1-mu)
    return mask


def make_plaq_masks(mask_shape, mask_mu, mask_off):
    mask = {}
    mask['frozen'] = make_double_stripes(mask_shape, mask_mu, mask_off+1)
    mask['active'] = make_single_stripes(mask_shape, mask_mu, mask_off)
    mask['passive'] = 1 - mask['frozen'] - mask['active']
    return mask
if __name__ == '__main__':
# For example
    _test_plaq_masks = make_plaq_masks((8,8), 0, mask_off=1)
    print('Frozen (fed into NNs)')
    print(_test_plaq_masks['frozen'])
    print('Active (driving the link update)')
    print(_test_plaq_masks['active'])
    print('Passive (updated as a result of link update)')
    print(_test_plaq_masks['passive'])
