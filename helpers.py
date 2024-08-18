import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

def plot_fitness_over_iterations(fitness):
    plt.style.use('seaborn-paper')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.linewidth'] = 0.1

    fig, ax = plt.subplots() # create new graph
    plt.plot(fitness, 'b', linewidth=0.5)
    plt.title('Image vectorization via genetic evolution')
    plt.xlabel('Number of generations')
    plt.ylabel('Fitness')
    plt.xlim(left=0)

    plt.box(True)
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='--', linewidth='0.5')
    ax.grid(which='minor', linestyle='-.', linewidth='0.05', alpha=0.1)

    plt.show()
    
def save_output(image_name, generated_image, data, images):
    uniq_filename = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
    base_name = image_name.rsplit('.', 1)[0] + '_' + uniq_filename
    
    iterations_folder = f"./results/iterations"
    os.makedirs(iterations_folder, exist_ok=True)
    
    out_path = f"./results/{base_name}.png"
    generated_image.save(out_path, dpi=(600,600))
    
    csv_path = f"./results/{base_name}.csv"
    np.savetxt(csv_path, data, delimiter=";")
    
    for i, img_data in enumerate(images):
        img_data.save(f"{iterations_folder}/frame_{i:06d}.png")
    
    images[0].save(f"./results/{base_name}.gif", save_all=True, append_images=images[1::10], optimize=False, duration=2, loop=0)

def clamp(value, min_val=0, max_val=1):
    return max(min_val, min(max_val, value))

def calculate_rmse(original_image, generated_image):
    squared_diff = (original_image - generated_image) ** 2.0

    mean_squared_error = np.sum(squared_diff) / (generated_image.size)
    rmse_normalized = np.sqrt(mean_squared_error) / 255.0

    return rmse_normalized

def calculate_mse(original_image, generated_image):
    squared_diff = (original_image - generated_image) ** 2.0

    mean_squared_error = np.sum(squared_diff) / (generated_image.size)
    
    return mean_squared_error

def array_to_image(generated_image):
    return Image.fromarray(generated_image.astype(np.uint8)).convert('P')