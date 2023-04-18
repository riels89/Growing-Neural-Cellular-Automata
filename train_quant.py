import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from lib.CAModel import CAModel, QCAModel
from lib.utils_vis import SamplePool, to_alpha, to_rgb, get_living_mask, make_seed, make_circle_masks
from lib.utils import load_emoji

device = torch.device("cpu:0")
model_path = "models/remaster_1.pth"

CHANNEL_N = 16        # Number of CA state channels
TARGET_PADDING = 16   # Number of pixels used to pad the target image border
TARGET_SIZE = 40

lr = 2e-3
lr_gamma = 0.9999
betas = (0.5, 0.5)
n_epoch = 80000

BATCH_SIZE = 8
POOL_SIZE = 1024
CELL_FIRE_RATE = 0.5

TARGET_EMOJI = 0 #@param "🦎"

EXPERIMENT_TYPE = "Regenerating"
EXPERIMENT_MAP = {"Growing":0, "Persistent":1, "Regenerating":2}
EXPERIMENT_N = EXPERIMENT_MAP[EXPERIMENT_TYPE]

USE_PATTERN_POOL = [0, 1, 1][EXPERIMENT_N]
DAMAGE_N = [0, 0, 3][EXPERIMENT_N]  # Number of patterns to damage in a batch


target_img = load_emoji(TARGET_EMOJI)
print(f"Target shape: {target_img.shape}")
p = TARGET_PADDING
pad_target = np.pad(target_img, [(p, p), (p, p), (0, 0)])
h, w = pad_target.shape[:2]
pad_target = np.expand_dims(pad_target, axis=0)
pad_target = torch.from_numpy(pad_target.astype(np.float32)).to(device)

seed = make_seed((h, w), CHANNEL_N)
pool = SamplePool(x=np.repeat(seed[None, ...], POOL_SIZE, 0))

m_f32 = QCAModel(CHANNEL_N, CELL_FIRE_RATE, device)
# ca.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

#### Start of quantization stuff
m_f32.eval()
m_f32.qconfig = torch.quantization.get_default_qconfig('x86')
m_f32_fused = torch.quantization.fuse_modules(m_f32, [['fc0', 'relu']])
m_f32_prepared = torch.quantization.prepare_qat(m_f32_fused.train())
#### End of quantization stuff, start training


#### Start of training
optimizer = optim.Adam(m_f32_prepared.parameters(), lr=lr, betas=betas)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, lr_gamma)

loss_log = []

def train(x, target, steps, optimizer, scheduler):
    x = m_f32_prepared(x, steps=steps)
    loss = F.mse_loss(x[:, :, :, :4], torch.vstack([target,] * x.shape[0]))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return x, loss

def loss_f(x, target):
    return torch.mean(torch.pow(x[..., :4]-target, 2), [-2,-3,-1])

for i in range(n_epoch+1):
    if USE_PATTERN_POOL:
        batch = pool.sample(BATCH_SIZE)
        x0 = torch.from_numpy(batch.x.astype(np.float32)).to(device)
        loss_rank = loss_f(x0, pad_target).detach().cpu().numpy().argsort()[::-1]
        x0 = batch.x[loss_rank]
        x0[:1] = seed
        if DAMAGE_N:
            damage = 1.0-make_circle_masks(DAMAGE_N, h, w)[..., None]
            x0[-DAMAGE_N:] *= damage
    else:
        x0 = np.repeat(seed[None, ...], BATCH_SIZE, 0)
    x0 = torch.from_numpy(x0.astype(np.float32)).to(device)

    x, loss = train(x0, pad_target, np.random.randint(64,96), optimizer, scheduler)
    
    if USE_PATTERN_POOL:
        batch.x[:] = x.detach().cpu().numpy()
        batch.commit()

    step_i = len(loss_log)
    loss_log.append(loss.item())
    
    print(step_i, "loss =", loss.item(), "\r")
    if step_i%100 == 0:
        # visualize_batch(x0.detach().cpu().numpy(), x.detach().cpu().numpy())
        torch.save(m_f32_prepared.state_dict(), model_path)
##### End of training


#### Convert to int8 !
m_f32_prepared.eval()
m_i8 = torch.quantization.convert(m_f32_prepared)
#### Finished converting
