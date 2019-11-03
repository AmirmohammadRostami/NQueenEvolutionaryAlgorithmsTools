import random
import numpy as np
from chromosome import Chromosome
'''
Random Gene Generator Algorithms, RGGA
'''


def default_random_gene_generator(number_of_queen):
    '''

    :param number_of_queen: Number of Queen, Integer
    :return: return np.array with  with len number of queen for each row
    '''
    gen = np.zeros(number_of_queen)
    for i in range(number_of_queen):
        gen[i] = np.random.randint(0, number_of_queen, 1)
    return gen


'''
Random Evaluators Algorithms, REA
'''


def default_evaluator(chromosome):
    '''

    :param chromosome: Chromosome
    :return: fitness of that chromosome, float between 0,1
    '''
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
Mutation Algorithms, MA
'''


def default_mutation(chromosome, prob=0.05):
    '''

    :param chromosome: Chromosome
    :param prob: default 0.05, float
    :return:
    '''
    for i in range(len(chromosome.genotype)):
        rand = np.random.random()
        if rand < prob:
            chromosome.genotype[i] = np.random.randint(0, len(chromosome.genotype), 1)
    return chromosome


'''
Cross Over  Algorithms, COA
'''


def default_cross_over(parent1, parent2, prob=0.4):
    '''
    :param parent1: First parent chromosome, Gene, np.array with len [n^2,1]
    :param parent2: Second parent chromosome, Gene, np.array with len [n^2,1]
    :param prob: Probability of choose which parent, default  = 0.4, float
    :return: return two chromosome for each children, Chromosome
    '''
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
Parent Selection Algorithms, PaSA
'''


def default_parent_selection(parents, n):
    '''

    :param parents: list of parents Chromosomes, List
    :param n: Number of Parents that should choose, Integer and less or equal than len parents list
    :return: list of selected Parents
    '''
    if n > len(parents):
        print('n should be less or equal than len parents list')
        return -1
    indexes = np.random.randint(0, len(parents), n)
    res = []
    for index in indexes:
        res.append(parents[index])
    return res


'''
Population Selection Algorithms, PoSA
'''


def default_population_selection(parents, childs, n):
    '''

    :param parents: list of Parents of current Generation, List
    :param childs: list of new childs of current Generation, List
    :param n: Number of remaining population, Integer
    :return: list of remained Chromosomes
    '''
    indexes = np.random.randint(0, len(parents) + len(childs), n)
    res = []
    for index in indexes:
        if index < len(parents):
            res.append(parents[index])
        else:
            res.append(childs[index - len(parents)])
    return res


'''
Stop Conditions, SC
'''


def default_stop_condition(generation, max_generation):
    '''

    :param generation: current generation, Integer
    :param max_generation: Maximum generation, Integer
    :return: Boolean as continue (False) and stop (True)
    '''
    if generation < max_generation:
        return False
    return True

