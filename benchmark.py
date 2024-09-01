import numpy as np
import time
import matplotlib.pyplot as plt

# Define your methods here (or import them if they're in a different module)

def draw_block_on_canvas_with_optimal_color(generated_image, original_image):
    image_height, image_width = original_image.shape

    # Create a copy of the generated image
    new_image = generated_image.copy()

    mask = get_mask(image_height, image_width)
    original_block = original_image[mask != 0]
    current_block = new_image[mask != 0]

    if original_block.size > 0:
        # Calculate the optimal color
        diff = original_block - current_block
        numerator = np.sum(diff)
        denominator = diff.size
        
        if denominator != 0:
            optimal_color = np.mean(current_block) + (numerator / denominator)
            color = int(np.clip(optimal_color, 0, 255))
        else:
            color = int(np.mean(original_block))

    # Update only the pixels where the mask is non-zero in the new image
    new_image[mask != 0] = color

    return new_image, color

def draw_block_on_canvas_with_color_from_image(generated_image, original_image):
    image_height, image_width = original_image.shape

    mask = get_mask(image_height, image_width)
    non_zero_pixels = original_image[mask != 0]

    if non_zero_pixels.size > 0:
        lightest_shade = np.max(non_zero_pixels)
        color = int(lightest_shade)

    line_image = 255 * np.ones_like(mask).astype(dtype=np.int64)
    line_image[mask != 0] = color

    blended_image = np.minimum(generated_image, line_image)
    return blended_image, color

# Dummy mask function for example purposes
def get_mask(image_height, image_width):
    mask = np.zeros((image_height, image_width), dtype=np.int64)
    mask[20:40, 20:40] = 1  # Example block in the mask
    return mask

# Generate test data
image_height, image_width = 64, 64
original_image = np.random.randint(0, 256, size=(image_height, image_width), dtype=np.int64)
generated_image = np.ones_like(original_image) * 255  # Start with a white canvas

# Number of samples
num_samples = 1000

# Benchmark the first method
times_optimal_color = []
for _ in range(num_samples):
    start_time = time.time()
    draw_block_on_canvas_with_optimal_color(generated_image, original_image)
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
    times_optimal_color.append(elapsed_time)

# Benchmark the second method
times_color_from_image = []
for _ in range(num_samples):
    start_time = time.time()
    draw_block_on_canvas_with_color_from_image(generated_image, original_image)
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
    times_color_from_image.append(elapsed_time)

# Calculate average execution time in milliseconds
average_time_optimal_color = np.mean(times_optimal_color)
average_time_color_from_image = np.mean(times_color_from_image)

# Print out the results
print(f"Priemerný čas pre draw_block_on_canvas_with_optimal_color: {average_time_optimal_color:.3f} ms")
print(f"Priemerný čas pre draw_block_on_canvas_with_color_from_image: {average_time_color_from_image:.3f} ms")

# Plot the results
methods = ['Optimálna Farba', 'Farba z Obrázka']
average_times = [average_time_optimal_color, average_time_color_from_image]

plt.bar(methods, average_times, color=['blue', 'green'])
plt.ylabel('Priemerný Čas (ms)')
plt.title('Porovnanie Výkonu Metód')
plt.show()
