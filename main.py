from typing import Tuple
import cv2
import copy
import numpy as np
import time
from Line import Line
from evolution import evaluate_fitness_tuple, generate_tuple_population, mutate_tuple, select_best_tuples, select_sus_tuples
from helpers import array_to_image, plot_fitness_over_iterations, save_output

DETERMINISTIC_MODE = True
DETERMINISTIC_SEED = 42
GENERATE_GIF = True
MAX_ADDMUT = 5
MUT_RATE = 20
NEvo = 10
NUMBER_OF_TUPLES = 416  # Reduced from 2500 as we're now working with pairs
TUPLE_SIZE = 6  # We're working with pairs of lines
SEARCH_SPACE_SIZE = 4

def print_best_fitness(population, fitness, start_time):    
    best, best_fitness = select_best_tuples(population, fitness, [1])
    print("Final fitness value: " + str(best_fitness[0]))
    print("--- Evolution lasted: %s seconds ---" % (time.time() - start_time))

if (DETERMINISTIC_MODE):
    np.random.seed(DETERMINISTIC_SEED)

image_name = "lena.png"
original_image = cv2.imread("images/lena.png", cv2.IMREAD_GRAYSCALE)
original_image = cv2.resize(original_image, (64, 64), interpolation=cv2.INTER_LANCZOS4)
original_image = np.asarray(original_image, dtype=np.int64)

original_image_height, original_image_width = original_image.shape[0], original_image.shape[1]
generated_image = 255 * np.ones((original_image_height, original_image_width), dtype=np.int64)

search_space = np.concatenate((np.zeros((1, SEARCH_SPACE_SIZE)),
                               np.array([[original_image_width - 1,
                                          original_image_width - 1,
                                          original_image_height - 1,
                                          original_image_height - 1
                                          ]])
                               ), axis = 0)

additive_mutation_space = search_space[1,:] * (MAX_ADDMUT / 100.0)

data = np.zeros((NUMBER_OF_TUPLES * TUPLE_SIZE, 6))  # (x1,x2,y1,y2,color,fitness)
fitness_over_iterations = []
current_fitness = None
buffer = 0
count = 1
images = []

if GENERATE_GIF:
    images.append(array_to_image(generated_image))

start_time = time.time()

while count <= NUMBER_OF_TUPLES:
    next_population = generate_tuple_population(24, search_space, additive_mutation_space, TUPLE_SIZE)
    next_population_fitness = evaluate_fitness_tuple(next_population, original_image, generated_image, current_fitness)

    for i in range(NEvo):
        previous_population = copy.deepcopy(next_population)
        previous_population_fitness = np.copy(next_population_fitness)

        next_population_best, next_population_fitness_best = select_best_tuples(previous_population, next_population_fitness, [3,2,1])
        next_population_sus, next_population_fitness_sus = select_sus_tuples(previous_population, next_population_fitness, 18)

        next_population_sus = mutate_tuple(next_population_sus, MUT_RATE / 100.0, search_space, additive_mutation_space)        
        next_population = np.concatenate((next_population_best, next_population_sus), axis=0)
        next_population_fitness = evaluate_fitness_tuple(next_population, original_image, generated_image, current_fitness)
        
        if (np.min(next_population_fitness) == np.min(previous_population_fitness)):
            buffer += 1
        else: 
            buffer = 0
        if (buffer >= 5):
            break

    best, best_fitness = select_best_tuples(next_population, next_population_fitness, [1])
    
    if current_fitness is None:
        current_fitness = 1e6
  
    if best_fitness[0] < current_fitness:
        current_fitness = best_fitness[0]
        fitness_over_iterations.append(current_fitness)

        best_tuple: Tuple[Line, ...] = best[0]
        for idx, best_individual in enumerate(best_tuple):
            generated_image, color = best_individual.draw_line_on_canvas_with_color_from_image(generated_image, original_image)
            data[(count - 1) * TUPLE_SIZE + idx] = np.array([
                int(best_individual.x1),
                int(best_individual.x2),
                int(best_individual.y1),
                int(best_individual.y2),
                color,
                current_fitness
            ])

        print("# " + str(count) + " Fitness: " + str(current_fitness))
        count += 1

        if GENERATE_GIF:
            images.append(array_to_image(generated_image))

plot_fitness_over_iterations(fitness_over_iterations)
print_best_fitness(next_population, next_population_fitness, start_time)
save_output(image_name, array_to_image(generated_image), data, images)