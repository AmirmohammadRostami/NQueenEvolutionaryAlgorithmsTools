import random
import numpy as np
from chromosome import Chromosome

'''
-------------------------------------
Random Gene Generator Algorithms, RGGA
-------------------------------------
inputs: number_of_queen (n of n-Queen problem)
        and parameters( dictionary of algorithm parameterss, key: parameters name, value: parameters value)
return-> np.array (genotype of chromosome)
'''


def default_random_gene_generator(number_of_queen, parameters):
    """
    :param number_of_queen: Number of Queen, Integer
    :param parameters: dictionary of parameters that key = parameter name and value = parameter value 
    :return: return np.array with  with len number of queen for each row
    """
    gen = np.zeros(number_of_queen)
    for i in range(number_of_queen):
        gen[i] = np.random.randint(0, number_of_queen, 1)
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


'''
-------------------------------------
Parent Selection Algorithms, PaSA
-------------------------------------
inputs:  population (current population chromosomes list),
         n (number of selected items)
         and parameters( dictionary of algorithm parameterss key: parameters name, value: parameters value) 
return-> list of selected chromosomes
'''


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
    indexes = np.random.randint(0, len(population), n)
    res = []
    for index in indexes:
        res.append(population[index])
    return res


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


'''
Stop Conditions, SC
input: parameters (dictionary of algorithm parameterss key: parameters name, value: parameters value)
return-> boolean (True as stop and False as keep on)
'''


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
