import copy
import numpy as np
from Ellipse import Ellipse
from helpers import calculate_rmse, clamp
from typing import List

def select_best(population, fitness, nums):
    N = len(nums)
    fitness_sorted = np.sort(fitness)
    indices_fitness_sorted = np.argsort(fitness)
    new_population_zeros = np.zeros(N, dtype=Ellipse)
    new_population = np.zeros(int(np.sum(nums)), dtype=Ellipse)
    new_fitness = np.zeros((int(np.sum(nums),)))
    
    for i in range(N):
        new_population_zeros[i] = population[indices_fitness_sorted[i]]    
    r = 0
    for i in range(N):
        for j in range(nums[i]):
            new_population[r] = new_population_zeros[i]
            new_fitness[r] = fitness_sorted[i]
            r += 1
    return [new_population, new_fitness]

def select_sus(population, fitness, nums):
    new_population = np.zeros(nums, dtype=Ellipse)
    new_fitness = np.zeros((nums,))
    old_fitness = np.copy(fitness)

    population_size = len(population)
    fitness = fitness - np.min(fitness) + 1
    sum_fitness = np.sum(fitness).astype(np.float32)

    w0 = np.zeros((population_size + 1),)

    for i in range(population_size):
        men = fitness[i] * sum_fitness
        w0[i] = 1.0 / men # creation of inverse weights 

    w0[i + 1] = 0
    w = np.zeros(population_size + 1)
    for i in np.arange(population_size - 1, -1, -1):
        w[i] = w[i + 1] + w0[i]

    max_w = np.max(w)
    if (max_w == 0):
        max_w = 0.00001

    w = (w / max_w) * 100 # weigth vector
    pdel = 100.0/nums 
    b0 = np.random.uniform() * pdel - 0.00001
    b = np.zeros((nums,))

    for i in range(1, nums + 1):
        b[i - 1] = (i - 1) * pdel + b0

    for i in range(nums):
        for j in range(population_size):
            if(b[i] < w[j] and b[i] > w[j + 1]):
                break
        new_population[i] = population[j]
        new_fitness[i] = old_fitness[j]
    
    return [new_population, new_fitness]

def evaluate_fitness(population, original_image, generated_image, fitness):
    length_population = len(population)
    population_fitness = [] 

    for i in range(length_population):
        individual_fitness = evaluate_partial_similarity(list([original_image, generated_image, population[i], fitness]))
        population_fitness.append(individual_fitness)

    return population_fitness

def evaluate_partial_similarity(params):
    original_image = params[0]
    generated_image = params[1]
    individual: Ellipse = params[2]
    fitness = params[3] 

    if fitness is None:
        generated, color = individual.draw_ellipse_on_canvas_with_color_from_image(generated_image, original_image)       
        return calculate_rmse(original_image, generated)
    else:
        generated, color = individual.draw_ellipse_on_canvas_with_color_from_image(generated_image, original_image)  
        
        if (color == 0):
            return fitness
        
        new_fitness = calculate_rmse(original_image, generated)

        if new_fitness < fitness:
            return new_fitness
        else:
            return fitness
        
def generate_population(population_size, search_space) -> List[Ellipse]:   
    new_population = []

    dX = search_space[1,0] - search_space[0,0]
    dY = search_space[1,1] - search_space[0,1]
    dMajorAxis = search_space[1,2] - search_space[0,2]
    dMinorAxis = search_space[1,3] - search_space[0,3]
    dAngle = search_space[1,4] - search_space[0,4]

    for r in range(int(population_size)):
        x_center = search_space[0,0] + np.random.uniform() * dX
        x_center = clamp(x_center, search_space[0,0], search_space[1,0])

        y_center = search_space[0,1] + np.random.uniform() * dY
        y_center = clamp(y_center, search_space[0,1], search_space[1,1])

        major_axis_length = search_space[0,2] + np.random.uniform() * dMajorAxis
        major_axis_length = clamp(major_axis_length, search_space[0,2] + 1, search_space[1,2])
        
        minor_axis_length = search_space[0,3] + np.random.uniform() * dMinorAxis
        minor_axis_length = clamp(minor_axis_length, search_space[0,3] + 1, search_space[1,3])
        
        rotation_angle = search_space[0,4] + np.random.uniform() * dAngle

        new_population.append(Ellipse(x_center,
                                      y_center,
                                      major_axis_length,
                                      minor_axis_length,
                                      rotation_angle))

    return new_population

def mutate(population: List[Ellipse], factor, search_space, additive_mutation_space) -> List[Ellipse]:
    length_population = len(population)

    factor = clamp(factor, 0, 1)
    n = int(np.ceil(length_population * 5 * factor * np.random.uniform()))
    new_population = copy.deepcopy(population)
    
    for i in range(n):
        rN = np.random.randint(3)
        r = int(np.ceil(np.random.uniform() * length_population)) - 1

        old_ellipse = population[r]
        new_ellipse = new_population[r]

        if rN == 0:  # mutate center x, y
            mutation_x_center = np.random.normal(0, additive_mutation_space[0])
            new_x_center = old_ellipse.x_center + mutation_x_center
            new_x_center = clamp(new_x_center, search_space[0, 0], search_space[1, 0])

            mutation_y_center = np.random.normal(0, additive_mutation_space[1])
            new_y_center = old_ellipse.y_center + mutation_y_center
            new_y_center = clamp(new_y_center, search_space[0, 1], search_space[1, 1])

            new_ellipse.x_center = new_x_center
            new_ellipse.y_center = new_y_center

        elif rN == 1:  # mutate major_axis_length, minor_axis_length
            mutation_major = np.random.normal(0, additive_mutation_space[2])
            new_major_axis_length = old_ellipse.major_axis_length + mutation_major
            new_major_axis_length = clamp(new_major_axis_length, search_space[0, 2] + 1, search_space[1, 2])

            mutation_minor = np.random.normal(0, additive_mutation_space[3])
            new_minor_axis_length = old_ellipse.minor_axis_length + mutation_minor
            new_minor_axis_length = clamp(new_minor_axis_length, search_space[0, 3] + 1, search_space[1, 3])

            new_ellipse.major_axis_length = new_major_axis_length
            new_ellipse.minor_axis_length = new_minor_axis_length

        elif rN == 2:  # mutate angle
            mutation_angle = np.random.normal(0, additive_mutation_space[4])
            new_angle = (old_ellipse.rotation_angle + mutation_angle) % 360

            new_ellipse.rotation_angle = new_angle

    return new_population