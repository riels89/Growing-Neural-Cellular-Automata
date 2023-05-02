import os
import pygame
import torch
import numpy as np

from lib.displayer import displayer
from lib.utils import mat_distance
from lib.CAModel import CAModel,QCAModel, NoGCAModel
from lib.utils_vis import to_rgb, make_seed
import glob


eraser_radius = 6

pix_size = 5
display_map_shape = (60, 60)
_map_shape = (60, 60)
CHANNEL_N = 16
CELL_FIRE_RATE = 0.5
model_paths = ["models/jmu_dog.pth", *glob.glob("new_models/*")]
model_idx = 0
model_path = model_paths[model_idx]
device = torch.device("cpu")

torch.set_grad_enabled(False)

_rows = np.arange(_map_shape[0]).repeat(_map_shape[1]).reshape([_map_shape[0],_map_shape[1]])
_cols = np.arange(_map_shape[1]).reshape([1,-1]).repeat(_map_shape[0],axis=0)
_map_pos = np.array([_rows,_cols]).transpose([1,2,0])

_map = make_seed(_map_shape, CHANNEL_N)
# seed_values = np.zeros((2, CHANNEL_N - 3))
# seed_values[:, 0] = 1
# seed_values[1, :int((CHANNEL_N - 3)/2)] = 1
# seed_values[0, int((CHANNEL_N - 3)/2)+1:] = 1
curr_target = 0

# print(seed_values.shape)
print(_map.shape)
# _map[_map.shape[0]//2, _map.shape[1]//2, 3:] = seed_values[curr_target]
display_map = np.ones([*display_map_shape, 3]) -0.00001

# Rasberry pi only has this backend
torch.backends.quantized.engine = 'qnnpack'
model = CAModel(CHANNEL_N, CELL_FIRE_RATE, device).to(device)
# model = model.prepare()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# model = model.convert()
# model = model.half()
first_input = torch.from_numpy(_map.reshape([1,_map_shape[0],_map_shape[1],CHANNEL_N]))
output = model(first_input, 1)

disp = displayer(display_map_shape, pix_size)
# hide cursor for touch screen
pygame.mouse.set_cursor((8,8),(0,0),(0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0))


isMouseDown = False
isSpaceDown = False
doubleClick = False
running = True
timer = False
dbclock = pygame.time.Clock()
N_SECONDS_CHANGE = 300
pygame.time.set_timer(pygame.USEREVENT, 1000 * N_SECONDS_CHANGE)

DOUBLECLICKTIME = 500
NO_CHANGE_AFTER = 1000
iters_no_interupt = 0
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.USEREVENT: 
            timer = True

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                isMouseDown = True
            if dbclock.tick() < DOUBLECLICKTIME:
                doubleClick = True

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                isMouseDown = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                isSpaceDown = True
            if event.key == pygame.K_ESCAPE:
                running = False

    if isMouseDown:
        try:
            mouse_pos = np.array([int((event.pos[1] - disp.offset[1]) / disp.scale_factor),
                                  int((event.pos[0] - disp.offset[0])) / disp.scale_factor])
            should_keep = (mat_distance(_map_pos, mouse_pos)>eraser_radius).reshape([_map_shape[0],_map_shape[1],1])
            output = output * torch.tensor(should_keep)
            iters_no_interupt = 0
        except AttributeError:
            pass
    elif isSpaceDown or doubleClick or (timer and iters_no_interupt > NO_CHANGE_AFTER):
        output = make_seed(_map_shape, CHANNEL_N)
        output = torch.from_numpy(output[np.newaxis])

        model_idx = (model_idx + 1) % len(model_paths)
        model_path = model_paths[model_idx]
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        isSpaceDown = False
        doubleClick = False
        timer = False
        iters_no_interupt = 0

    if iters_no_interupt < NO_CHANGE_AFTER:
        output = model(output, 1)
        _map = to_rgb(output.numpy()[0])
        display_map = _map
        disp.update(display_map)
    iters_no_interupt += 1
