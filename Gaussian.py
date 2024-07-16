import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class Gaussian:
    def __init__(self, x_mean, y_mean, x_sigma, y_sigma, rotation_angle, amplitude=255):
        self.x_mean = x_mean
        self.y_mean = y_mean
        self.x_sigma = x_sigma
        self.y_sigma = y_sigma
        self.rotation_angle = rotation_angle
        self.amplitude = amplitude

    @property
    def x_mean(self):
        return self._x_mean

    @x_mean.setter
    def x_mean(self, value):
        self._x_mean = value

    @property
    def y_mean(self):
        return self._y_mean

    @y_mean.setter
    def y_mean(self, value):
        self._y_mean = value

    @property
    def x_sigma(self):
        return self._x_sigma

    @x_sigma.setter
    def x_sigma(self, value):
        self._x_sigma = value

    @property
    def y_sigma(self):
        return self._y_sigma

    @y_sigma.setter
    def y_sigma(self, value):
        self._y_sigma = value

    @property
    def rotation_angle(self):
        return self._rotation_angle

    @rotation_angle.setter
    def rotation_angle(self, value):
        self._rotation_angle = value

    @property
    def amplitude(self):
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value):
        self._amplitude = value

    def display_matrix_as_image(self, matrix, figsize=(5, 5), cmap='viridis'):
        fig, ax = plt.subplots(figsize=figsize)
        image = ax.imshow(matrix, cmap=cmap)
        cbar = fig.colorbar(image, ax=ax)
        cbar.ax.set_ylabel('Values', rotation=-90, va="bottom")
        plt.show()

    def get_mask(self, x, y):
        image_height, image_width = y.shape[0], x.shape[1]
        
        # Create coordinate grids
        x_coords, y_coords = np.meshgrid(np.arange(image_width), np.arange(image_height))
        
        # Calculate the Gaussian values
        gaussian_x = norm.pdf(x_coords, loc=self.x_mean, scale=self.x_sigma)
        gaussian_y = norm.pdf(y_coords, loc=self.y_mean, scale=self.y_sigma)
        
        # Create 2D Gaussian
        gaussian_2d = gaussian_x * gaussian_y
        
        # Normalize
        gaussian_2d = gaussian_2d / np.max(gaussian_2d)
        
        return gaussian_2d

    def draw_gaussian_on_canvas_with_color_from_image(self, generated_image, original_image):
        image_height, image_width = original_image.shape

        x, y = np.meshgrid(np.arange(image_width), np.arange(image_height))
        normalized_gaussian = self.get_mask(x, y)
    
        # Avoid division by zero
        epsilon = 1e-10  # Adjust this value as needed
        diff = ((255 - original_image) - (255 - generated_image))
        division_result = np.where(normalized_gaussian > epsilon, diff / (normalized_gaussian + epsilon), np.inf)
        
        # Find the minimum value from the division
        scale = int(np.min(division_result[(np.isfinite(division_result)) & (division_result > 0)]))

        grayscale_gaussian = np.round(255 - (normalized_gaussian * scale)).astype(np.int64)
        self.amplitude = np.max(grayscale_gaussian)

        blended_image = self.blend_images(generated_image, grayscale_gaussian)
        return blended_image

    def blend_images(self, image1, image2):
        inverted_base = 255 - image1
        inverted_overlay = 255 - image2
        
        summed = inverted_base + inverted_overlay
        clipped = np.clip(summed, 0, 255)    
        blended_image = 255 - clipped

        return blended_image
