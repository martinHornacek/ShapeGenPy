import cv2
import numpy as np

class Ellipse:
    def __init__(self, x_center, y_center, major_axis_length, minor_axis_length, rotation_angle, color = 0):
        self.x_center = x_center
        self.y_center = y_center
        self.major_axis_length = major_axis_length
        self.minor_axis_length = minor_axis_length
        self.rotation_angle = rotation_angle
        self.color = color

    @property
    def x_center(self):
        return self._x_center

    @x_center.setter
    def x_center(self, value):
        self._x_center = value

    @property
    def y_center(self):
        return self._y_center

    @y_center.setter
    def y_center(self, value):
        self._y_center = value

    @property
    def major_axis_length(self):
        return self._major_axis_length

    @major_axis_length.setter
    def major_axis_length(self, value):
        self._major_axis_length = value

    @property
    def minor_axis_length(self):
        return self._minor_axis_length

    @minor_axis_length.setter
    def minor_axis_length(self, value):
        self._minor_axis_length = value

    @property
    def rotation_angle(self):
        return self._rotation_angle

    @rotation_angle.setter
    def rotation_angle(self, value):
        self._rotation_angle = value

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        self._color = value

    def get_mask(self, canvas_width, canvas_height):
        canvas = 255 * np.ones((canvas_height, canvas_width), dtype=np.uint8)  # White canvas (has to be unit8 because of cv2)
        cv2.ellipse(canvas, center=(int(self.x_center), int(self.y_center)),
                            axes=(int(self._major_axis_length), int(self._minor_axis_length)),
                            angle=int(self._rotation_angle),
                            startAngle=0,
                            endAngle=360,
                            color=0,
                            thickness=-1)
        
        mask = np.where(canvas == 0, 1, 0).astype(dtype=np.int64)
        return mask
    
    def draw_ellipse_on_canvas_with_color_from_image(self, generated_image, original_image):
        image_height, image_width = original_image.shape

        mask = self.get_mask(image_width, image_height)
        non_zero_pixels = original_image[mask != 0]

        if non_zero_pixels.size > 0:
            lightest_shade = np.max(non_zero_pixels)
            self.color = int(lightest_shade)
        else:
            self.color = 0

        # Create a new image with the same shape as the mask
        ellipse_image = 255 * np.ones_like(mask).astype(dtype=np.int64)
        ellipse_image[mask != 0] = self.color

        # Overlay the line image over the generated image and take the darker color for each pixel
        blended_image = np.minimum(generated_image, ellipse_image)
        return blended_image, self.color