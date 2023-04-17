import pygame
import numpy as np

class displayer:

    def __init__(self, _map_shape, pix_size, has_gap=False):
        """
        _map_size: tuple
        color_map: a list indicates the color to each index.
                   0 : empty block, should always white
                   1+: varies building types
        """
        pygame.init()
        clock = pygame.time.Clock()
        clock.tick(60)

        self._map_shape = _map_shape
        self.has_gap = has_gap
        self.pix_size = pix_size
        self.screen = pygame.display.set_mode((_map_shape[1]*self.pix_size,
                                               _map_shape[0]*self.pix_size))
        blank = np.zeros((*_map_shape, 3), dtype=np.uint8)
        # blank = blank.repeat(self.pix_size, axis=0).repeat(self.pix_size, axis=1)
        self.image = pygame.surfarray.make_surface(blank)
        self.scaled_image = pygame.transform.scale(self.image, (_map_shape[1]*self.pix_size,
                                            _map_shape[0]*self.pix_size))

    def update(self, _map):
        self.screen.fill((255,255,255))
        c = (_map*256).astype(np.uint8)[:, :, :3]
        c = np.transpose(c, (1, 0, 2))
        pygame.surfarray.blit_array(self.image, c)
        self.scaled_image = pygame.transform.scale(self.image, (self._map_shape[1]*self.pix_size,
                                            self._map_shape[0]*self.pix_size),
                                            self.scaled_image)
        self.screen.blit(self.scaled_image, (20,20))
        pygame.display.update()
