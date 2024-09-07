import copy
import numpy as np
from helpers import calculate_mse, clamp
from Line import Line
from typing import List, Tuple

def select_best_tuples(population, fitness, nums):
    N = len(nums)
    fitness_sorted = np.sort(fitness)
    indices_fitness_sorted = np.argsort(fitness)
    new_population_zeros = np.zeros(N, dtype=object)
    new_population = np.zeros(int(np.sum(nums)), dtype=object)
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

def select_sus_tuples(population, fitness, nums):
    new_population = np.zeros(nums, dtype=object)
    new_fitness = np.zeros((nums,))
    old_fitness = np.copy(fitness)
    population_size = len(population)
    fitness = fitness - np.min(fitness) + 1
    sum_fitness = np.sum(fitness).astype(np.float32)

    w0 = np.zeros((population_size + 1,))

    for i in range(population_size):
        men = fitness[i] * sum_fitness
        w0[i] = 1.0 / men # creation of inverse weights 
    w0[i + 1] = 0
    w = np.zeros((population_size + 1,))
    for i in np.arange(population_size - 1, -1, -1):
        w[i] = w[i + 1] + w0[i]
    max_w = np.max(w)
    if (max_w == 0):
        max_w = 0.00001
    w = (w / max_w) * 100 # weight vector
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

def evaluate_fitness_tuple(population, original_image, generated_image, fitness):
    length_population = len(population)
    population_fitness = [] 

    for i in range(length_population):
        tuple_fitness = evaluate_partial_similarity_tuple(original_image, generated_image, population[i], fitness)
        population_fitness.append(tuple_fitness)

    return population_fitness

def evaluate_partial_similarity_tuple(original_image, generated_image, tuple_individual: Tuple[Line, ...], fitness):
    if fitness is None:
        fitness = 1e6

    new_generated_image = generated_image.copy()
    for individual in tuple_individual:
        new_generated_image, color = individual.draw_line_on_canvas_with_color_from_image(new_generated_image, original_image)
        
    new_fitness = calculate_mse(original_image, new_generated_image)

    return min(new_fitness, fitness)

def generate_tuple_population(population_size, search_space, additive_mutation_space, tuple_size) -> List[Tuple[Line, ...]]:   
    new_population = []

    dX = search_space[1,0] - search_space[0,0]
    dY = search_space[1,2] - search_space[0,2]

    for _ in range(int(population_size)):
        tuple_lines = []
        for _ in range(tuple_size):
            x1 = search_space[0,0] + np.random.uniform() * dX
            x1 = clamp(x1, search_space[0,0], search_space[1,0])

            y1 = search_space[0,2] + np.random.uniform() * dY
            y1 = clamp(y1, search_space[0,2], search_space[1,2])

            x2 = x1 + np.random.randint(2 * additive_mutation_space[1] + 1) - additive_mutation_space[1]
            x2 = clamp(x2, search_space[0,1], search_space[1,1])
              
            y2 = y1 + np.random.randint(2 * additive_mutation_space[3] + 1) - additive_mutation_space[3] 
            y2 = clamp(y2, search_space[0,3], search_space[1,3])

            tuple_lines.append(Line(x1, x2, y1, y2))

        new_population.append(tuple(tuple_lines))

    return new_population

def mutate_tuple(population: List[Tuple[Line, ...]], factor, search_space, additive_mutation_space) -> List[Tuple[Line, ...]]:
    length_population = len(population)
    tuple_size = len(population[0])

    factor = clamp(factor, 0, 1)
    n = int(np.ceil(length_population * tuple_size * 4 * factor * np.random.uniform()))
    new_population = copy.deepcopy(population)
    
    for _ in range(n):
        r = int(np.ceil(np.random.uniform() * length_population)) - 1
        line_index = np.random.randint(tuple_size)
        rN = np.random.randint(2)

        old_line = population[r][line_index]
        new_line = new_population[r][line_index]

        if rN == 0:  # x1, y1
            mutation_x1 = (2.0 * np.random.uniform() - 1) * additive_mutation_space[0]
            new_x1 = old_line.x1 + mutation_x1
            new_x1 = clamp(new_x1, search_space[0, 0], search_space[1, 0])

            mutation_y1 = (2.0 * np.random.uniform() - 1) * additive_mutation_space[2]
            new_y1 = old_line.y1 + mutation_y1
            new_y1 = clamp(new_y1, search_space[0, 2], search_space[1, 2])

            new_line.x1 = new_x1
            new_line.y1 = new_y1

        elif rN == 1:  # x2, y2
            mutation_x2 = (2.0 * np.random.uniform() - 1) * additive_mutation_space[1]
            new_x2 = old_line.x2 + mutation_x2
            new_x2 = clamp(new_x2, search_space[0, 1], search_space[1, 1])

            mutation_y2 = (2.0 * np.random.uniform() - 1) * additive_mutation_space[3]
            new_y2 = old_line.y2 + mutation_y2
            new_y2 = clamp(new_y2, search_space[0, 3], search_space[1, 3])

            new_line.x2 = new_x2
            new_line.y2 = new_y2

    return new_population