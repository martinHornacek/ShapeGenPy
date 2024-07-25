import datetime
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
import pandas as pd

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
    
def save_output_as_gif_and_3d(image_name, generated_image, data, images, display_3d_plot=True):
    uniq_filename = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
    base_name = image_name.rsplit('.', 1)[0] + '_' + uniq_filename
    
    # Save the generated image
    out_path = f"./results/{base_name}.png"
    generated_image.save(out_path, dpi=(600,600))
    
    # Save the data as CSV
    csv_path = f"./results/{base_name}.csv"
    np.savetxt(csv_path, data, delimiter=";")
    
    # Save the GIF
    images[0].save(f"./results/{base_name}.gif", save_all=True, append_images=images[1::10], optimize=False, duration=2, loop=0)
    
    # Create, display (if requested), and save 3D plot
    plot_3d_rectangles(csv_path, base_name, display_before_save=display_3d_plot)

def plot_3d_rectangles(csv_file, base_name, display_before_save=True):
    # Read the CSV file
    df = pd.read_csv(csv_file, delimiter=";", names=['x1', 'x2', 'y1', 'y2', 'color', 'fitness'])
    
    # Create a new 3D figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Iterate through each rectangle in the dataframe
    for _, row in df.iterrows():
        x1, x2, y1, y2 = row['x1'], row['x2'], row['y1'], row['y2']
        color = row['color'] / 255.0  # Normalize color to [0, 1]
        
        # Define the vertices of the rectangle
        vertices = [
            [x1, y1, color],
            [x2, y1, color],
            [x2, y2, color],
            [x1, y2, color]
        ]
        
        # Create a Poly3DCollection
        poly = Poly3DCollection([vertices], alpha=0.8)
        poly.set_facecolor(plt.cm.gray(color))
        ax.add_collection3d(poly)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Color (Grayscale)')
    ax.set_title('3D Visualization of Rectangles')
    
    # Set axis limits
    ax.set_xlim(df['x1'].min(), df['x2'].max())
    ax.set_ylim(df['y1'].min(), df['y2'].max())
    ax.set_zlim(0, 1)
    
    # Display the plot if requested
    if display_before_save:
        plt.show()
    
    # Save the plot
    plt.savefig(f"./results/{base_name}_3d.png", dpi=300, bbox_inches='tight')
    
    # Close the plot to free up memory
    plt.close()

def clamp(value, min_val=0, max_val=1):
    return max(min_val, min(max_val, value))

def calculate_rmse(original_image, generated_image):
    squared_diff = (original_image - generated_image) ** 2.0

    mean_squared_error = np.sum(squared_diff) / (generated_image.size)
    rmse_normalized = np.sqrt(mean_squared_error) / 255.0

    return rmse_normalized

def array_to_image(generated_image):
    return Image.fromarray(generated_image.astype(np.uint8)).convert('P')