from lib.utils_vis import SamplePoolv2, load_im, pad_list
import numpy as np
import torch

def test_pool_zero_seed():

    pool_size = 100
    num_channels = 16
    target_imgs = (load_im(), load_im("data/jmu_jdog.png"))
    seeds = [np.zeros(target.shape) for target in target_imgs] 

    pool = SamplePoolv2(pool_size, num_channels, seeds=seeds, targets=target_imgs)
    x, y = pool.sample(5)

    assert x.shape[:-1] == y.shape[:-1]
    assert x.shape[0] == 5

    assert torch.count_nonzero(x) == torch.tensor(0)

    target_imgs, _ = pad_list(target_imgs)
    target_imgs = [torch.tensor(tar).to("cuda:0") for tar in target_imgs]

    for i in range(y.shape[0]):
        assert torch.equal(y[i], target_imgs[0]) or torch.equal(y[i], target_imgs[1])

def test_pool_zero_seed():

    pool_size = 100
    num_channels = 16
    target_imgs = (load_im(), load_im("data/jmu_jdog.png"))
    seeds = target_imgs

    pool = SamplePoolv2(pool_size, num_channels, seeds=seeds, targets=target_imgs)
    x, y = pool.sample(5)

    assert x.shape[:-1] == y.shape[:-1]
    assert x.shape[0] == 5

    target_imgs, _ = pad_list(target_imgs)
    target_imgs = [torch.tensor(tar).to("cuda:0") for tar in target_imgs]

    # Seed is equal to target, so check if x = target
    for i in range(x.shape[0]):
        assert torch.equal(x[i, :, :, :4], target_imgs[0]) or torch.equal(x[i, :, :, :4], target_imgs[1])
        # All other channels should be initialized to zero
        assert torch.count_nonzero(x[i, :, :, 4:]) == torch.tensor(0)

    for i in range(y.shape[0]):
        assert torch.equal(y[i], target_imgs[0]) or torch.equal(y[i], target_imgs[1])


