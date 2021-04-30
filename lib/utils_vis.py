import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import imageio

class SamplePool:
    def __init__(self, *, _parent=None, _parent_idx=None, **slots):
        self._parent = _parent
        self._parent_idx = _parent_idx
        self._slot_names = slots.keys()
        self._size = None
        for k, v in slots.items():
            if self._size is None:
                self._size = len(v)
            assert self._size == len(v)
            setattr(self, k, np.asarray(v))

    def sample(self, n):
        idx = np.random.choice(self._size, n, False)
        batch = {k: getattr(self, k)[idx] for k in self._slot_names}
        batch = SamplePool(**batch, _parent=self, _parent_idx=idx)
        return batch

    def commit(self):
        for k in self._slot_names:
            getattr(self._parent, k)[self._parent_idx] = getattr(self, k)

class SamplePoolv2:
    def __init__(self, pool_size, num_channels, seeds=None, targets=None, seed_shape=None, device='cuda:0'):

        self.pool_size = pool_size
        if seeds is None:
            assert seed_shape is not None
            # Create seed of target shape with one middle seed
            self.seeds = make_seeds(num_channels, n=self.pool_size, shape=seed_shape)
        else:
            assert len(targets) == len(seeds)
            assert seed_shape is None
            # Create an equally weighted number of seed states
            self.seeds = make_seeds(num_channels, n=self.pool_size, seeds=seeds)
        # Send to device
        self.seeds = torch.from_numpy(self.seeds).to(device)

        # Create an equally weighted number of targets
        self.targets = np.repeat(targets, pool_size / len(targets))

    def sample(self, n):
        self.idx = np.random.choice(self.pool_size, n, False)
        return (self.seeds[self.idx], self.targets[self.idx])

    def commit(self, batch):
        self.seeds[self.idx] = batch

def to_alpha(x):
    return np.clip(x[..., 3:4], 0, 0.9999)

def to_rgb(x):
    # assume rgb premultiplied by alpha
    rgb, a = x[..., :3], to_alpha(x)
    return np.clip(1.0-a+rgb, 0, 0.9999)

def get_living_mask(x):
    return nn.MaxPool2d(3, stride=1, padding=1)(x[:, 3:4, :, :])>0.1

def make_targets(targets, n=1):

    # Pad targets to uniform shape
    targets, max_shape = pad_list(targets)

    # (n, d0, ..., dn, n_channels)
    created_targets = np.empty((n, *max_shape), dtype=np.float32)

    n_per_tar = int(n / len(targets))
    for i, target in enumerate(targets):
        target_indicies = slice(i * n_per_tar, (i + 1) * n_per_tar)
        created_targets[target_indicies] = target

    return created_targets

def make_seeds(n_channels, n=1, seeds=None, shape=None):
    created_seeds = None
    if seeds is not None:
        assert len(seeds) <= n
        assert shape is None

        # Pad seeds to uniform shape
        seeds, max_shape = pad_list(seeds)

        # (n, d0, ..., dn, n_channels)
        created_seeds = np.empty((n, *max_shape[:-1], n_channels), dtype=np.float32)

        seed_channels = max_shape[-1]
        n_per_seed = int(n / len(seeds))

        # (d0, ..., dn, n_channels - seed_channels)
        x = np.zeros([*max_shape[:-1], n_channels - seed_channels], np.float32)
        for i, seed in enumerate(seeds):
            seed_indicies = slice(i * n_per_seed, (i + 1) * n_per_seed)
            # Concatenate to  (n_per_seed, d0, ..., dn, n_channels)
            created_seeds[seed_indicies] = np.concatenate([seed, x], axis=-1)
    elif shape is not None:
        created_seeds = np.zeros([n, *shape[:-1], n_channels], np.float32)
        created_seeds[:, shape[0]//2, shape[1]//2, shape[-1]:] = 1.0
    else:
        print("Neither shape or seed are specified")
    return created_seeds 

def pad_list(lists_to_pad):
    def pad_to_shape(a, shape):
        assert len(a.shape) == len(shape)
        diff = [shape[i] - a.shape[i] for i in range(len(shape))]
        amount_to_pad = [(pad // 2, pad // 2 + pad % 2) for pad in diff]

        return np.pad(a, amount_to_pad, mode='constant')

    sizes = np.array([list_to_pad.shape for list_to_pad in lists_to_pad])
    max_shape = [max(sizes[:, i]) for i in range(sizes.shape[1])]
    padded = [pad_to_shape(lists_to_pad, max_shape) for list_to_pad in lists_to_pad]

    return padded, max_shape

def make_seed(shape, n_channels, value=1.0):
    seed = np.zeros([shape[0], shape[1], n_channels], np.float32)
    seed[shape[0]//2, shape[1]//2, 3:] = value
    return seed

def make_circle_masks(n, h, w):
    x = np.linspace(-1.0, 1.0, w)[None, None, :]
    y = np.linspace(-1.0, 1.0, h)[None, :, None]
    center = np.random.random([2, n, 1, 1])*1.0-0.5
    r = np.random.random([n, 1, 1])*0.3+0.1
    x, y = (x-center[0])/r, (y-center[1])/r
    mask = (x*x+y*y < 1.0).astype(np.float32)
    return mask

def load_im(path="data/kirby.png"):
    im = imageio.imread(path)
    im = np.array(im.astype(np.float32))
    im /= 255.0
    return im


