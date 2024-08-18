import numpy as np
import cv2

class Block:
    def __init__(self, x1, x2, y1, y2, color = 255):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.color = color

    @property
    def x1(self):
        return self._x1

    @x1.setter
    def x1(self, value):
        self._x1 = value

    @property
    def y1(self):
        return self._y1

    @y1.setter
    def y1(self, value):
        self._y1 = value

    @property
    def x2(self):
        return self._x2

    @x2.setter
    def x2(self, value):
        self._x2 = value

    @property
    def y2(self):
        return self._y2

    @y2.setter
    def y2(self, value):
        self._y2 = value

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        self._color = value

    def get_mask(self, canvas_height, canvas_width):
        canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
        cv2.rectangle(canvas, (int(self.y1), int(self.x1)), (int(self.y2), int(self.x2)), self.color, -1)
        return canvas
    
    def draw_block_on_canvas_with_color_from_image(self, generated_image, original_image):
        image_height, image_width = original_image.shape

        mask = self.get_mask(image_height, image_width)
        non_zero_pixels = original_image[mask != 0]

        if non_zero_pixels.size > 0:
            lightest_shade = np.max(non_zero_pixels)
            self.color = int(lightest_shade)

        line_image = 255 * np.ones_like(mask).astype(dtype=np.int64)
        line_image[mask != 0] = self.color

        blended_image = np.minimum(generated_image, line_image)
        return blended_image, self.color