import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse
import os.path

from lib.CAModel import QCAModel
from lib.utils_vis import SamplePool, make_seed, make_circle_masks, visualize_batch
from lib.utils import load_emoji, load_png

parser = argparse.ArgumentParser()
parser.add_argument("target", help="Path to target png image to train on.",
                    type=load_png)
parser.add_argument("model_path", help="Path to save/load the model")
parser.add_argument("--plot_path", help="Path to put batch vizualization.",
                    default="latest_batch.pdf")
parser.add_argument("-r", "--reload", 
                    help="""If model path already contains a trained model,
                            should training start with this loaded model.""",
                    type=bool, default=True)
parser.add_argument("-d", "--device", help="Device to train on.",
                    default="cuda", type=torch.device)
parser.add_argument("-c", "--channel_n", help="Total number of CA channels.",
                    default=16, type=int)
parser.add_argument("-pad", "--target_padding", help="Amount of padding to add around the target",
                    default=10, type=int)
parser.add_argument("-lr", "--learning_rate", help="Learning rate of optimizer.",
                    default=2e-3, type=float)
parser.add_argument("-g", "--gamma", help="Learning rate gamma value",
                    default=0.9999, type=float)
parser.add_argument("-e", "--epochs", help="Number of epochs to train for.",
                    default=80000, type=int)
parser.add_argument("-b", "--batch_size", help="Training batch size",
                    default=8, type=int)
parser.add_argument("-p", "--pool_size", help="Size of NCA pool.",
                    default=1024, type=int)
parser.add_argument("-f", "--fire_rate", help="Fire rate of CA.",
                    default=0.5, type=float)
parser.add_argument("-t", "--exp_type", help="Expirement type. 0:Growing, 1:Persistent, 2:Regenerating",
                    default=2, type=int, choices=[0,1,2])
parser.add_argument("-s", "--save_every", help="After how many epochs to save the model",
                    default=1000, type=int)
args = parser.parse_args()

device = args.device
model_path = args.model_path
parts = model_path.split("/")
parts[-1] = "f32_" + parts[-1]
f32_model_path = "/".join(parts)

CHANNEL_N = args.channel_n        # Number of CA state channels
TARGET_PADDING = args.target_padding   # Number of pixels used to pad the target image border

lr = args.learning_rate
lr_gamma = args.gamma
betas = (0.5, 0.5)
n_epoch = args.epochs

BATCH_SIZE = args.batch_size
POOL_SIZE = args.pool_size
CELL_FIRE_RATE = args.fire_rate
SAVE_EVERY = args.save_every

EXPERIMENT_TYPE = args.exp_type
USE_PATTERN_POOL = [0, 1, 1][EXPERIMENT_TYPE]
DAMAGE_N = [0, 0, 3][EXPERIMENT_TYPE]  # Number of patterns to damage in a batch


target_img = args.target
print(f"Target shape: {target_img.shape}")
p = TARGET_PADDING
pad_target = np.pad(target_img, [(p, p), (p, p), (0, 0)])
h, w = pad_target.shape[:2]
pad_target = np.expand_dims(pad_target, axis=0)
pad_target = torch.from_numpy(pad_target.astype(np.float32)).to(device)

seed = make_seed((h, w), CHANNEL_N)
pool = SamplePool(x=np.repeat(seed[None, ...], POOL_SIZE, 0))

m_f32 = QCAModel(CHANNEL_N, CELL_FIRE_RATE, device)

#### Start of quantization stuff
m_f32_prepared = m_f32.prepare()
if args.reload and os.path.isfile(f32_model_path) :
    m_f32_prepared.load_state_dict(torch.load(f32_model_path, map_location=device))
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
    
    if step_i%SAVE_EVERY == 0:
        print(step_i, "loss =", sum(loss_log[-SAVE_EVERY:]) / SAVE_EVERY)
        visualize_batch(x0.detach().cpu().numpy(), x.detach().cpu().numpy(), args.plot_path)
        torch.save(m_f32_prepared.state_dict(), f32_model_path)
##### End of training


#### Convert to int8 !
m_i8 = m_f32_prepared.convert()
parts = model_path.split("/")
parts[-1] = "int8_" + parts[-1]
i8_path = "/".join(parts)
torch.save(m_i8.state_dict(), i8_path)
#### Finished converting
