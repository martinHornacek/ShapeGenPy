import copy
import numpy as np
from helpers import calculate_rmse, clamp
from Line import Line
from typing import List

def select_best(population, fitness, nums):
    N = len(nums)
    fitness_sorted = np.sort(fitness)
    indices_fitness_sorted = np.argsort(fitness)
    new_population_zeros = np.zeros(N, dtype=Line)
    new_population = np.zeros(int(np.sum(nums)), dtype=Line)
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
    new_population = np.zeros(nums, dtype=Line)
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
    individual: Line = params[2]
    fitness = params[3]
    
    if fitness is None:
        generated, color = individual.draw_line_on_canvas_with_color_from_image(generated_image, original_image)       
        return calculate_rmse(original_image, generated)
    else:
        generated, color = individual.draw_line_on_canvas_with_color_from_image(generated_image, original_image)  
        
        if (color == 0):
            return fitness
        
        new_fitness = calculate_rmse(original_image, generated)

        if new_fitness < fitness:
            return new_fitness
        else:
            return fitness
        
def generate_population(population_size, search_space, additive_mutation_space) -> List["Line"]:   
    new_population = []

    dX = search_space[1,0] - search_space[0,0]
    dY = search_space[1,2] - search_space[0,2]

    for r in range(int(population_size)):
        x1 = search_space[0,0] + np.random.uniform() * dX
        x1 = clamp(x1, search_space[0,0], search_space[1,0])

        y1 = search_space[0,2] + np.random.uniform() * dY
        y1 = clamp(y1, search_space[0,2], search_space[1,2])

        x2 = x1 + np.random.randint(2 * additive_mutation_space[1] + 1) - additive_mutation_space[1]
        x2 = clamp(x2, search_space[0,1], search_space[1,1])
          
        y2 = y1 + np.random.randint(2 * additive_mutation_space[3] + 1) - additive_mutation_space[3] 
        y2 = clamp(y2, search_space[0,3], search_space[1,3])

        new_population.append(Line(x1, x2, y1, y2))

    return new_population

def mutate(population: List[Line], factor, search_space, additive_mutation_space) -> List[Line]:
    length_population = len(population)

    factor = clamp(factor, 0, 1)
    n = int(np.ceil(length_population * 4 * factor * np.random.uniform()))
    new_population = copy.deepcopy(population)
    
    for i in range(n):
        rN = np.random.randint(2)
        r = int(np.ceil(np.random.uniform()*length_population)) - 1

        old_line = population[r]
        new_line = new_population[r]

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