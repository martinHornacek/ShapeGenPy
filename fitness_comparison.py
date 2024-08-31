import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Function to load CSV file
def load_csv(file_path):
    return pd.read_csv(file_path, header=None, 
                       names=['x1', 'x2', 'y1', 'y2', 'color', 'fitness'],
                       sep=';')

# Load data for both approaches
data_original = load_csv('/Users/martin/Documents/ShapeGenPy/results/lena_2024-08-18_14.56.31.728839.csv')
data_new = load_csv('/Users/martin/Documents/ShapeGenPy/results/lena_2024-08-24_18.47.46.697908.csv')

# Add iteration column (assuming each row is an iteration)
data_original['iteration'] = range(1, len(data_original) + 1)
data_new['iteration'] = range(1, len(data_new) + 1)

# Find the crossover point
crossover_index = np.argwhere(np.diff(np.sign(data_original['fitness'].values - data_new['fitness'].values))).flatten()
if len(crossover_index) > 0:
    crossover_point = crossover_index[0]
    crossover_iteration = data_original['iteration'].iloc[crossover_point]
    crossover_fitness = (data_original['fitness'].iloc[crossover_point] + data_new['fitness'].iloc[crossover_point]) / 2
else:
    crossover_point = None

# Plot fitness comparison
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]})

# Main plot
ax1.plot(data_original['iteration'], data_original['fitness'], label='Metóda dotykom', linewidth=2)
ax1.plot(data_new['iteration'], data_new['fitness'], label='Metóda priemerom', linewidth=2)

ax1.set_xlabel('Iterácia', fontsize=12)
ax1.set_ylabel('MSE', fontsize=12)
ax1.set_title('Porovnanie evolúcie fitness', fontsize=16)
ax1.legend(fontsize=10)
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

# Highlight crossover point
if crossover_point is not None:
    ax1.scatter(crossover_iteration, crossover_fitness, color='red', s=100, zorder=5)
    ax1.annotate(f'Crossover\n({crossover_iteration:.0f}, {crossover_fitness:.2f})', 
                 (crossover_iteration, crossover_fitness),
                 xytext=(10, 10), textcoords='offset points', 
                 bbox=dict(boxstyle="round,pad=0.5", fc="yellow", ec="b", alpha=0.8),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

# Zoomed plot
zoom_start = max(0, crossover_iteration - 1000)
zoom_end = min(len(data_original), crossover_iteration + 1000)
ax2.plot(data_original['iteration'], data_original['fitness'], linewidth=2)
ax2.plot(data_new['iteration'], data_new['fitness'], linewidth=2)
ax2.set_xlim(zoom_start, zoom_end)
y_min = min(data_original['fitness'].iloc[int(zoom_start):int(zoom_end)].min(),
            data_new['fitness'].iloc[int(zoom_start):int(zoom_end)].min())
y_max = max(data_original['fitness'].iloc[int(zoom_start):int(zoom_end)].max(),
            data_new['fitness'].iloc[int(zoom_start):int(zoom_end)].max())
ax2.set_ylim(y_min, y_max)
ax2.set_title('Detailný pohľad na oblasť crossover', fontsize=14)
ax2.set_xlabel('Iterácia', fontsize=12)
ax2.set_ylabel('MSE', fontsize=12)
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

# Highlight crossover point in zoomed plot
if crossover_point is not None:
    ax2.scatter(crossover_iteration, crossover_fitness, color='red', s=100, zorder=5)

plt.tight_layout()
plt.savefig('fitness_comparison_detailed.png', dpi=300)
plt.close()

print("Analysis complete. Check 'fitness_comparison_detailed.png' for the detailed visualization.")