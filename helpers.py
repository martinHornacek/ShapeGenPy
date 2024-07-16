import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from PIL import Image

def plot_fitness_over_iterations(fitness):
    plt.style.use('seaborn-paper') # rendering settings (font and style)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.linewidth'] = 0.1 # frame boundaries in graphs

    fig, ax = plt.subplots() # create new graph
    plt.plot(fitness, 'b', linewidth=0.5)
    plt.title('Image vectorization via genetic evolution')
    plt.xlabel('Number of generations')
    plt.ylabel('Fitness')
    plt.xlim(left=0)

    plt.box(True) # grid and display settings
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='--', linewidth='0.5')
    ax.grid(which='minor', linestyle='-.', linewidth='0.05', alpha=0.1)

    plt.show() # display the resulting graph and list the solution found
    
def save_output_as_gif(image_name, generated_image, data, images):
    uniq_filename = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
    out_path = u"./results/{}.png".format(image_name.rsplit('.', 1)[0] + '_' + uniq_filename)
    generated_image.save(out_path, dpi=(600,600))
    np.savetxt("./results/" + image_name.rsplit('.', 1)[0] + '_' + uniq_filename + ".csv", data, delimiter=";")
    images[0].save(u"./results/{}.gif".format(image_name.rsplit('.', 1)[0] + '_' + uniq_filename), save_all=True, append_images=images[1::10], optimize=False, duration=2, loop=0)

def save_as_video(image_name, generated_images, data):
    uniq_filename = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
    
    if not os.path.exists("./results"):
        os.makedirs("./results")

    height, width = np.array(generated_images[0]).shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = f"./results/{image_name.rsplit('.', 1)[0]}_{uniq_filename}.mp4"
    out = cv2.VideoWriter(out_path, fourcc, 1, (width, height))

    for img in generated_images:
        # Convert PIL Image to numpy array and from RGB to BGR color space
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()
    return out_path

def clamp(value, min_val=0, max_val=1):
    return max(min_val, min(max_val, value))

def calculate_rmse(original_image, generated_image):
    squared_diff = (original_image - generated_image) ** 2.0

    mean_squared_error = np.sum(squared_diff) / (generated_image.size)
    rmse_normalized = np.sqrt(mean_squared_error) / 255.0

    return rmse_normalized

def array_to_image(generated_image):
    return Image.fromarray(generated_image.astype(np.uint8)).convert('P')