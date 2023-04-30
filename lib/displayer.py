import pygame
import numpy as np

"""
aspect_scale.py - Scaling surfaces keeping their aspect ratio
Raiser, Frank - Sep 6, 2k++
crashchaos at gmx.net

This is a pretty simple and basic function that is a kind of
enhancement to pygame.transform.scale. It scales a surface
(using pygame.transform.scale) but keeps the surface's aspect
ratio intact. So you will not get distorted images after scaling.
A pretty basic functionality indeed but also a pretty useful one.

Usage:
is straightforward.. just create your surface and pass it as
first parameter. Then pass the width and height of the box to
which size your surface shall be scaled as a tuple in the second
parameter. The aspect_scale method will then return you the scaled
surface (which does not neccessarily have the size of the specified
box of course)

Dependency:
a pygame version supporting pygame.transform (pygame-1.1+)
"""

def aspect_scale(img, xy, dest=None):
    """ Scales 'img' to fit into box bx/by.
     This method will retain the original image's aspect ratio """
    bx, by = xy
    ix,iy = img.get_size()
    if ix > iy:
        # fit to width
        scale_factor = bx/float(ix)
        sy = scale_factor * iy
        if sy > by:
            scale_factor = by/float(iy)
            sx = scale_factor * ix
            sy = by
        else:
            sx = bx
    else:
        # fit to height
        scale_factor = by/float(iy)
        sx = scale_factor * ix
        if sx > bx:
            scale_factor = bx/float(ix)
            sx = bx
            sy = scale_factor * iy
        else:
            sy = by

    if dest is None:
        return scale_factor, pygame.transform.scale(img, (sx,sy))
    else:
        return scale_factor, pygame.transform.scale(img, (sx,sy), dest)
    


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
        infoObject = pygame.display.Info()
        self.screen_size = (infoObject.current_w, infoObject.current_h)

        self._map_shape = _map_shape
        self.has_gap = has_gap
        self.pix_size = pix_size
        # self.screen_size =(_map_shape[1]*self.pix_size, _map_shape[0]*self.pix_size) 
        # self.screen = pygame.display.set_mode((_map_shape[1]*self.pix_size,
        #                                        _map_shape[0]*self.pix_size), pygame.FULLSCREEN)
        self.screen = pygame.display.set_mode(self.screen_size)
        blank = np.zeros((*_map_shape, 3), dtype=np.uint8)
        # blank = blank.repeat(self.pix_size, axis=0).repeat(self.pix_size, axis=1)
        self.image = pygame.surfarray.make_surface(blank)
        self.scale_factor, self.scaled_image = aspect_scale(self.image, self.screen_size)
        self.scaled_size = self.scaled_image.get_size()
        self.offset = ((self.screen_size[0] - self.scaled_size[0])/2, (self.screen_size[1] - self.scaled_size[1]) / 2)
        self.screen.fill((255,255,255))

    def update(self, _map):
        c = (_map*256).astype(np.uint8)[:, :, :3]
        c = np.transpose(c, (1, 0, 2))
        pygame.surfarray.blit_array(self.image, c)
        _, self.scaled_image = aspect_scale(self.image, self.screen_size, self.scaled_image)
        self.screen.blit(self.scaled_image, self.offset)
        pygame.display.update()
