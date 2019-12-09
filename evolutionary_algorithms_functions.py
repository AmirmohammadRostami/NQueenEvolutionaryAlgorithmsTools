import random
import numpy as np
from chromosome import Chromosome
import warnings

'''
-------------------------------------
Selection Algorithms, (SA)
-------------------------------------
input: parameters (dictionary of algorithm parameters key: parameters name, value: parameters value)
return-> selected items as array
'''

def warning_data_type_check_selection_algorithms(items, probs):
    """
    :param items: (for check) Items that want to choose from them, np.array or list
    :param probs: (for check) Probabilities of each item, np.array or list
    :return: fixed items and probs
    """
    if type(items) == list:
        items = np.array(items)
    if type(probs) == list:
        probs = np.array(probs)
    if len(probs) != len(items):
        raise ValueError(
            "Length of probs and items must be equal! probs length = {} and items length = {}".format(len(probs),
                                                                                                      len(items)))
    if type(probs) != np.ndarray or type(items) != np.ndarray:
        raise ValueError(
            "Type of items and probs must be list or np.array, items type = {} and probs type = {}".format(type(items),
                                                                                                           type(probs)))
    if np.min(probs) < 0:
        raise ValueError("Probabilities can not contain negative values")

    if np.sum(probs) != 1:
        warnings.warn(
            'Sum of Probabilities array must be 1 but it is = {}, and we normalize it to reach sum equal 1'.format(
                np.sum(probs)), stacklevel=4)
        probs = probs / np.sum(probs)
    return items, probs

def roulette_wheel_selection(items, probs, n):
    """
    :param items:  Items that want to choose from them, np.array or list
    :param probs:  Probabilities of each item, np.array or list
    :param n: number of selected item(s), Integer
    :return: array of selected Items, np.array
    """
    if n == 0:
        return np.array([])
    items, probs = warning_data_type_check_selection_algorithms(items, probs)
    rnds = np.random.random(size=n)
    inds = np.zeros(n, dtype=np.int)
    cum_sum = np.cumsum(probs)
    for i, rnd in enumerate(rnds):
        inds[i] = np.argmax(cum_sum >= rnd)
    return items[inds]

def stochastic_universal_selection(items, probs, n):
    """
    :param items:  Items that want to choose from them, np.array or list
    :param probs:  Probabilities of each item, np.array or list
    :param n: number of selected item(s), Integer
    :return:
    """
    items, probs = warning_data_type_check_selection_algorithms(items, probs)
    index = np.arange(len(items))
    np.random.shuffle(index)
    items = items[index]
    probs = probs[index]
    start_index = np.random.uniform(0, 1 / n, 1)
    index_of_choose = np.linspace(0, (n - 1) / n, n) + start_index
    cum_sum = np.cumsum(probs)
    selected_items = []
    items_pointer = 0
    for index in index_of_choose:
        while cum_sum[items_pointer] < index:
            items_pointer += 1
        selected_items.append(items[items_pointer])
    return np.array(selected_items)


'''
-------------------------------------
Random Gene Generator Algorithms, RGGA
-------------------------------------
inputs: number_of_queen (n of n-Queen problem)
        and parameters( dictionary of algorithm parameterss, key: parameters name, value: parameters value)
return-> np.array (genotype of chromosome)
'''

def default_random_gene_generator(number_of_queen, parameters=None):
    """
    :param number_of_queen: Number of Queen, Integer
    :param parameters: dictionary of parameters that key = parameter name and value = parameter value
    :return: return np.array with  with len number of queen for each row
    """
    gen = np.zeros(number_of_queen)
    for i in range(number_of_queen):
        gen[i] = np.random.randint(0, number_of_queen, 1)
    return gen

def permutation_random_gene_generator(number_of_queen, parameters=None):
    """
    :param number_of_queen: Number of Queen, Integer
    :param parameters: dictionary of parameters that key = parameter name and value = parameter value
    :return: return np.array with  with len number of queen for each row
    """
    gen = np.arange(0, number_of_queen)
    np.random.shuffle(gen)
    return gen


'''
-------------------------------------
Random Evaluators Algorithms, REA
-------------------------------------
inputs: a chromosome
return-> single float number as fitness of input chromosome
'''

def default_evaluator(chromosome, parameters=None):
    """
    :param chromosome (Chromosome)
    :param parameters: dictionary of parameters that key = parameter name and value = parameter value
    :return: fitness of that chromosome, float between 0,1
    """
    danger = 0
    for i in range(len(chromosome.genotype)):
        for j in range(len(chromosome.genotype)):
            if i != j:
                if chromosome.genotype[i] == chromosome.genotype[j] or \
                        abs(chromosome.genotype[i] - chromosome.genotype[j]) == abs(i - j):
                    danger += 1
    if danger > 0:
        fitness = 1 / danger
    else:
        fitness = 1
    return fitness


'''
-------------------------------------
Mutation Algorithms, MA
-------------------------------------
inputs: single chromosome
        and parameters( dictionary of algorithm parameterss, key: parameters name, value: parameters value)
return->
'''

def default_mutation(chromosome, parameters={'prob': 0.05}):
    """
    :param chromosome: Chromosome
    :param parameters: dictionary of parameters that key = parameter name and value = parameter value
    :param prob: default 0.05, float
    :return:
    """
    prob = parameters['prob']
    for i in range(len(chromosome.genotype)):
        rand = np.random.random()
        if rand < prob:
            chromosome.genotype[i] = np.random.randint(0, len(chromosome.genotype), 1)
    return chromosome

def random_swap_mutation(chromosome, parameters={'prob': 0.05}):
    """
    :param chromosome: Chromosome
    :param parameters: dictionary of parameters that key = parameter name and value = parameter value
    :param prob: default 0.05, float
    :return:
    """
    if np.random.random() <= parameters['prob']:
        idx = np.random.choice(np.arange(len(chromosome.genotype)), 2, replace=False)
        chromosome.genotype[idx[0]], chromosome.genotype[idx[1]] = chromosome.genotype[idx[1]], chromosome.genotype[
            idx[0]]
    return chromosome


'''
-------------------------------------
Cross Over  Algorithms, COA
-------------------------------------
inputs: parent1, parent2 as two chromosomes and parameters( dictionary of algorithm parameterss, key: parameters name, value: parameters value)
return-> two chromosomes as childes
'''

def default_cross_over(parent1, parent2, parameters={'prob': 0.4}):
    """
    :param parameters: dictionary of parameters that key = parameter name and value = parameter value
    :param parent1: First parent chromosome, Gene, np.array with len [n^2,1]
    :param parent2: Second parent chromosome, Gene, np.array with len [n^2,1]
    :return: return two chromosome for each children, Chromosome
    """
    prob = parameters['prob']
    idx = int(len(parent1.genotype) / 2)
    gen1, gen2 = np.zeros(len(parent1.genotype)), np.zeros(len(parent1.genotype))
    rand = np.random.random()
    if rand <= prob:
        gen2[:idx] = parent2.genotype[:idx]
        gen1[:idx] = parent1.genotype[:idx]
        gen1[idx:] = parent2.genotype[idx:]
        gen2[idx:] = parent1.genotype[idx:]
    else:
        gen1[:idx] = parent2.genotype[:idx]
        gen2[:idx] = parent1.genotype[:idx]
        gen1[idx:] = parent1.genotype[idx:]
        gen2[idx:] = parent2.genotype[idx:]
    chromosome1, chromosome2 = Chromosome(gen1, 0), Chromosome(gen2, 0)
    return chromosome1, chromosome2

def multi_points_crossover(parent1, parent2, parameters={'prob': 0.4, 'points_count': 'middle'}):
    """
    :param parameters: dictionary of parameters that key = parameter name and value = parameter value
    :param parent1: First parent chromosome, Gene, np.array with len [n^2,1]
    :param parent2: Second parent chromosome, Gene, np.array with len [n^2,1]
    :return: return two chromosome for each children, Chromosome
    """

    if str(parameters['points_count']) == 'middle':
        return default_cross_over(parent1, parent2)
    if parameters['points_count'] > len(parent1.genotype) or parameters['points_count'] <= 0:
        warnings.warn('points must be between 1 and size of genotype. parents will be returned', stacklevel=3)
        return parent1, parent2

    crossover_points = np.sort(
        np.random.choice(np.arange(len(parent1.genotype)), replace=False, size=parameters['points_count']))
    crossover_points = np.append(crossover_points, len(parent1.genotype))
    # print('cross over points', crossover_points)
    first_idx = 0
    gen1, gen2 = np.zeros(len(parent1.genotype)), np.zeros(len(parent1.genotype))
    for last_idx in crossover_points:
        if np.random.random() <= parameters['prob']:
            gen2[first_idx: last_idx] = parent2.genotype[first_idx: last_idx]
            gen1[first_idx: last_idx] = parent1.genotype[first_idx: last_idx]
            # print('same')
        else:
            gen1[first_idx: last_idx] = parent2.genotype[first_idx: last_idx]
            gen2[first_idx: last_idx] = parent1.genotype[first_idx: last_idx]
            # print('not same')
        # print(gen1)
        # print(gen2)
        first_idx = last_idx

    chromosome1, chromosome2 = Chromosome(gen1, 0), Chromosome(gen2, 0)
    return chromosome1, chromosome2


'''
-------------------------------------
Parent Selection Algorithms, PaSA
-------------------------------------
inputs:  population (current population chromosomes list),
         n (number of selected items)
         and parameters( dictionary of algorithm parameterss key: parameters name, value: parameters value)
return-> list of selected chromosomes
'''

# our
def default_parent_selection(population, n, parameter=None):
    """
    :param parameter: dictionary of parameters that key = parameter name and value = parameter value
    :param population: list of current population Chromosomes, List
    :param n: Number of Parents that should choose, Integer and less or equal than len parents list
    :return: list of selected Parents
    """
    if n > len(population):
        print('n should be less or equal than len parents list')
        return -1
    return np.random.choice(population, size=n)


'''
-------------------------------------
Population Selection Algorithms, PoSA
-------------------------------------
inputs:  parents (list of parents chromosome),
         childes (list of children chromosome),
         n (number of selected items)
         and parameters (dictionary of algorithm parameterss key: parameters name, value: parameters value)
return-> list of selected
'''

# our
def default_population_selection(parents, children, n, parameters=None):
    """
    :param parameters: dictionary of parameters that key = parameter name and value = parameter value
    :param parents: list of Parents of current Generation, List
    :param children: list of new children of current Generation, List
    :param n: Number of remaining population, Integer
    :return: list of remained Chromosomes
    """
    indexes = np.random.randint(0, len(parents) + len(children), n)
    res = []
    for index in indexes:
        if index < len(parents):
            res.append(parents[index])
        else:
            res.append(children[index - len(parents)])
    return res

# our
def fitness_based_population_selection(parents, children, n, parameters=None):
    """
    :param parameters: dictionary of parameters that key = parameter name and value = parameter value
    :param parents: list of Parents of current Generation, List
    :param children: list of new children of current Generation, List
    :param n: Number of remaining population, Integer
    :return: list of remained Chromosomes
    """

    population = parents + children
    fitness_arr = np.array([x.fitness for x in population])
    fitness_arr = fitness_arr / np.sum(fitness_arr)
    return roulette_wheel_selection(population, fitness_arr, n)


'''
-------------------------------------
Stop Conditions, SC
-------------------------------------
input: parameters (dictionary of algorithm parameterss key: parameters name, value: parameters value)
return-> boolean (True as stop and False as keep on)
'''

# our
def default_stop_condition(generation, max_generation, parameters=None):
    """
    :param parameters: dictionary of parameters that key = parameter name and value = parameter value
    :param generation: current generation, Integer
    :param max_generation: Maximum generation, Integer
    :return: Boolean as continue (False) and stop (True)
    """
    if generation < max_generation:
        return False
    return True


if __name__ == '__main__':
    # items = ['a', 'b', 'c', 'd', 'e']
    # probs = [0.5, 0.2, 0.1, 0.1, 0.1]
    # n = 15
    # print('RW', roulette_wheel_selection(items, probs, n))
    # print('SUS', stochastic_universal_selection(items, probs, n))

    # ch1 = Chromosome(np.arange(8), 5)
    # ch2 = Chromosome(np.arange(8) * 2, 5)
    #
    # pop = []
    # for i in range(10):
    #     pop.append(Chromosome(np.arange(10), fitness=i + 1))
    # sel_pops = fitness_based_population_selection(pop[: 5], pop[5:], n=5)
    # for x in sel_pops:
    #     print(x.genotype, x.fitness)
    for i in range(10):
        print(permutation_random_gene_generator(8))
