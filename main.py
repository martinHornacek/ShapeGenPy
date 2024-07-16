import copy
import cv2
import numpy as np
import time
from Ellipse import Ellipse
from evolution import evaluate_fitness, generate_population, mutate, select_best, select_sus
from helpers import array_to_image, plot_fitness_over_iterations, save_output_as_gif

DETERMINISTIC_MODE = True  # reproducible results [True, False]
DETERMINISTIC_SEED = 42 # seed for pseudo-random generator
GENERATE_GIF = True # generate animation [True, False]
MAJOR_AXIS_MAX_LENGTH = 15
MINOR_AXIS_MAX_LENGTH = 10
MAX_ADDMUT = 5 # [%] maximum aditive mutation range
MUT_RATE = 20 # [%] mutation rate (percentage of individuals to be mutated)
NEvo = 10 # number of evolution steps per one object 
NUMBER_OF_OBJECTS = 6000
SEARCH_SPACE_SIZE = 5

def print_best_fitness(population, fitness, start_time):    
    best, best_fitness = select_best(population, fitness, [1]) # find out the final solution
    print("Final fitness value: " + str(best_fitness[0]))
    print("--- Evolution lasted: %s seconds ---" % (time.time() - start_time))

if (DETERMINISTIC_MODE): # if deterministic mode, use specified seed for reproducible results
    np.random.seed(DETERMINISTIC_SEED)

image_name = "lena.png"
original_image = cv2.imread("images/lena.png", cv2.IMREAD_GRAYSCALE)
# Resize the image to 64 x 64 pixels
# original_image = cv2.resize(original_image, (64, 64), interpolation=cv2.INTER_AREA)
original_image = np.asarray(original_image, dtype=np.int64)

original_image_height, original_image_width = original_image.shape[0], original_image.shape[1]
generated_image = 255 * np.ones((original_image_height, original_image_width), dtype=np.int64)

search_space = np.concatenate((np.zeros((1, SEARCH_SPACE_SIZE)), # lower bound
                               np.array([[original_image_width - 1,
                                          original_image_height - 1,
                                          MAJOR_AXIS_MAX_LENGTH - 1,
                                          MINOR_AXIS_MAX_LENGTH - 1,
                                          360 - 1]])
                               ), axis = 0) # upper bound

additive_mutation_space = search_space[1,:] * (MAX_ADDMUT / 100.0) # range of changes for the additive mutation

data = np.zeros((NUMBER_OF_OBJECTS, 7)) # (center_x,center_y,major_axis_length,minor_axis_length,angle,color,fitness)
fitness_over_iterations = []
current_fitness = None # initial fitness value
buffer = 0 # auxiliary variable to stop evolution if no changes occur
count = 1 # iterator for number of objects in the final image
images = [] # list of images used for animation process

if GENERATE_GIF:
    images.append(array_to_image(generated_image))

start_time = time.time() # start the timer

while count <= NUMBER_OF_OBJECTS:
    next_population = generate_population(24, search_space)
    next_population_fitness = evaluate_fitness(next_population, original_image, generated_image, current_fitness)

    for i in range(NEvo):
        previous_population = copy.deepcopy(next_population)
        previous_population_fitness = np.copy(next_population_fitness)

        next_population_best, next_population_fitness_best = select_best(previous_population, previous_population_fitness, [3,2,1])
        next_population_sus, next_population_fitness_sus = select_sus(previous_population, previous_population_fitness, 18)

        next_population_sus = mutate(next_population_sus, MUT_RATE / 100.0, search_space, additive_mutation_space)        
        next_population = np.concatenate((next_population_best, next_population_sus), axis = 0)
        next_population_fitness = evaluate_fitness(next_population, original_image, generated_image, current_fitness)
        
        if (np.min(next_population_fitness) == np.min(previous_population_fitness)):
            buffer += 1
        else: 
            buffer = 0
        if (buffer >= 5):
            break

    best, best_fitness = select_best(next_population, next_population_fitness, [1])
    
    if current_fitness is None:
        current_fitness = 1e6
    
    if best_fitness[0] < current_fitness:
        current_fitness = best_fitness[0]
        fitness_over_iterations.append(current_fitness)

        best_individual: Ellipse = best[0]
        generated_image, color = best_individual.draw_ellipse_on_canvas_with_color_from_image(generated_image, original_image)

        print("# " + str(count) + " Fitness: " + str(current_fitness))
        data[count - 1, :] = np.concatenate(np.array(
                (int(best_individual.x_center),
                 int(best_individual.y_center),
                 int(best_individual.major_axis_length),
                 int(best_individual.minor_axis_length),
                 int(best_individual.rotation_angle),
                 color,
                 current_fitness)
            ).reshape(7,1))
        
        count += 1

        if GENERATE_GIF:
            images.append(array_to_image(generated_image))


plot_fitness_over_iterations(fitness_over_iterations)
print_best_fitness(next_population, next_population_fitness, start_time)
save_output_as_gif(image_name, array_to_image(generated_image), data, images)