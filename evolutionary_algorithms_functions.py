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


def q_tournament_selection(items, probs, q, n):
    """
    :param items:  Items that want to choose from them, np.array or list
    :param probs:  Probabilities of each item, in fact the fitness values of the items
    :param q: The number of the items wwe will choose on each iteration, integer
    :param n: number of selected item(s), Integer
    :return:
    """
    assert q != 0

    if n == 0:
        return np.array([])

    else:
        items, probs = warning_data_type_check_selection_algorithms(items, probs)
        index = np.arange(len(items))
        np.random.shuffle(index)
        items = items[index]
        probs = probs[index]

        selected_items = []
        len_items = len(items)

        for i in range(n):
            indexes = np.random.choice(np.arange(len_items),q,replace = False)
            selected_items.append(items[indexes[np.argmax(probs[indexes])]])
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
        chromosome.genotype[idx[0]], chromosome.genotype[idx[1]] = \
        chromosome.genotype[idx[1]], chromosome.genotype[idx[0]]
    return chromosome

def insertion_swap_mutation(chromosome, parameters={'prob: 0.05'}):
    """
    :param chromosome: Chromosome
    :param parameters: dictionary of parameters that key = parameter name and value = parameter value
    :param prob: default 0.05, float
    :return:
    """
    if np.random.random() <= parameters['prob']:
        idx = np.random.choice(np.arange(len(chromosome.genotype)), 2, replace=False)

        # The index has been extracted from the chromosome genotype
        second = chromosome.genotype[max(idx)]

        # The new genotype is made by remvoving the second index and inserting it
        #   just after the first index
        chromosome.genotype = np.insert(np.delete(chromosome.genotype, max(idx)), min(idx)+1, second)

def neighbour_based_mutation(chromosome, parameters=None):
    """
    :param chromosome: chromosome,        Gene, np.array with shape = (1,len(parent))
    :return: return mutated chromosome , Gene, np.array with shape = (1,len(parent))
    """
    gene = chromosome.genotype.copy()
    begin_indx, end_indx = np.random.randint(len(gene)), np.random.randint(len(gene))
    if begin_indx <= end_indx:
        s = gene[begin_indx:end_indx]
        s = s[::-1]
        gene[begin_indx:end_indx] = s
    else:
        c = len(gene) - (abs(begin_indx - end_indx))
        for i in range(c):
            gene[end_indx] = chromosome.genotype[begin_indx]
            gene[begin_indx] = chromosome.genotype[end_indx]
            end_indx = (end_indx - 1) % len(gene)
            begin_indx = (begin_indx + 1) % len(gene)
    chromosome.genotype = gene
    return chromosome


def shuffle_index_mutation(chromosome, parameters={'prob': 0.05}):
    """
    :param chromosome: Chromosome
    :param parameters: dictionary of parameters that key = parameter name and value = parameter value
    :param prob: default 0.05, float
    :return:
    """
    for i in range(len(chromosome.genotype)):
        if np.random.random() <= parameters['prob']:
            swap_idx = np.random.randint(0, len(chromosome.genotype))
            chromosome.genotype[i], chromosome.genotype[swap_idx] = chromosome.genotype[swap_idx], chromosome.genotype[
                i]
    return chromosome


def scramble_mutation(chromosome, parameters=None):
    """
    :param chromosome: Chromosome
    :param parameters: dictionary of parameters that key = parameter name and value = parameter value
    :param prob: default 0.05, float
    :return:
    """
    idxs = np.random.choice(np.arange(len(chromosome.genotype)), 2, replace=False)
    if idxs[0] > idxs[1]:
        idxs = [idxs[1], idxs[0]]
    print(idxs)
    temp = np.zeros(len(chromosome.genotype))
    for i in range(idxs[0]):
        temp[i] = chromosome.genotype[i]
    scramble = chromosome.genotype[idxs[0]:idxs[1]]
    np.random.shuffle(scramble)
    for i in range(idxs[0], idxs[1]):
        temp[i] = scramble[i - idxs[0]]
    for i in range(idxs[1], len(chromosome.genotype)):
        temp[i] = chromosome.genotype[i]
    chromosome.genotype = temp
    return chromosome


def insertion_swap_mutation(chromosome, parameters={'prob: 0.05'}):
    """
    :param chromosome: Chromosome
    :param parameters: dictionary of parameters that key = parameter name and value = parameter value
    :param prob: default 0.05, float
    :return:
    """
    if np.random.random() <= parameters['prob']:
        idx = np.random.choice(np.arange(len(chromosome.genotype)), 2, replace=False)

        # The index has been extracted from the chromosome genotype
        second = chromosome.genotype[max(idx)]

        # The new genotype is made by remvoving the second index and inserting it
        #   just after the first index
        chromosome.genotype = np.insert(np.delete(chromosome.genotype, max(idx)), min(idx)+1, second)

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


def upmx_crossover(parent1, parent2, parameters={'prob': 0.33}):
    """
    :param parameters: dictionary of parameters that key = parameter name and value = parameter value
    :param parent1: First parent chromosome, Gene, np.array with len [n^2,1]
    :param parent2: Second parent chromosome, Gene, np.array with len [n^2,1]
    :return: return two chromosome for each children, Chromosome
    References for crossover UPMX --> https://arxiv.org/pdf/1203.3097.pdf
    """
    prob = parameters['prob']
    par_size = len(parent1.genotype)
    gen1, gen2 = np.zeros(par_size, dtype=np.int), np.zeros(par_size, dtype=np.int)
    for i in range(par_size):
        gen1[parent1.genotype[i]], gen2[parent2.genotype[i]] = i, i
    chromosome1, chromosome2 = parent1, parent2
    for i in range(par_size):
        if random.random() < prob:
            sel1, sel2 = int(chromosome1.genotype[i]), int(chromosome2.genotype[i])
            chromosome1.genotype[i], chromosome1.genotype[gen1[sel2]] = sel2, sel1
            chromosome2.genotype[i], chromosome2.genotype[gen2[sel1]] = sel1, sel2
            gen1[sel1], gen1[sel2] = gen1[sel2], gen1[sel1]
            gen2[sel1], gen2[sel2] = gen2[sel2], gen2[sel1]
    return chromosome1, chromosome2


def edge_crossover(parent1, parent2, parameters=None):
    """
    :param parent1: First parent chromosome,     Gene, np.array with shape = (1,len(parent))
    :param parent2: Second parent chromosome, Gene, np.array with shape = (1,len(parent))
    :return: return two chromosome for each children, Chromosome
    """

    import copy

    first_chromosome, second_chromosome = parent1.genotype, parent2.genotype
    child1, child2 = np.zeros(len(parent1.genotype)), np.zeros(len(parent1.genotype))

    # find neighbours of a gene
    def neighbour(gene, index):
        return [gene[index - 1], gene[(index + 1) % len(gene)]]

    #  whether each gen has same neighbour in both parents
    def has_superior_neighbour(list):
        for i in range(len(list) - 1):
            if list[i] in list[i + 1:]:
                return True, list[i]
        return False, None

    # remove an item in a dic
    def remover(item, dic):
        for i in range(len(dic)):
            while item in dic[i]:
                dic[i].remove(item)

    # make a dictionary of the neighbours
    def dictionary(first_chromosome, second_chromosome):
        dic = {}
        for i in range(len(first_chromosome)):
            dic[i] = []
        for i in range(len(first_chromosome)):
            dic[first_chromosome[i]] += neighbour(first_chromosome, i)
            dic[second_chromosome[i]] += neighbour(second_chromosome, i)
        return dic

    # child producer
    def child_maker(chromosome, dict):

        dic = copy.deepcopy(dict)
        chromosome[0] = np.random.randint(0, len(chromosome))
        remover(chromosome[0], dic)
        for gene in range(1, len(chromosome)):
            neighbors_list = dic[chromosome[gene - 1]]
            if len(neighbors_list) == 0:
                index = list(range(gene))
                remaining_genes = list(range(len(chromosome)))
                for i in range(len(index)):
                    remaining_genes.remove(chromosome[index[i]])
                chromosome[gene] = np.random.choice(remaining_genes)
                remover(chromosome[gene], dic)
            else:
                has_superios, superior_gene = has_superior_neighbour(neighbors_list)
                if has_superios:
                    chromosome[gene] = superior_gene
                    remover(chromosome[gene], dic)
                else:
                    list_len_nghbrs_of_nghbrs = []
                    for p in range(len(neighbors_list)):
                        list_len_nghbrs_of_nghbrs.append(len(set(dic[neighbors_list[p]])))
                    min_indxs = np.where(list_len_nghbrs_of_nghbrs == np.min(list_len_nghbrs_of_nghbrs))[0]
                    chromosome[gene] = neighbors_list[min_indxs[np.random.randint(0, len(min_indxs))]]
                    remover(chromosome[gene], dic)

        return chromosome

    neighbours_dict = dictionary(first_chromosome, second_chromosome)
    ch1 = Chromosome(child_maker(child1, neighbours_dict), 0)
    ch2 = Chromosome(child_maker(child2, neighbours_dict), 0)
    return ch1, ch2


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


def boltzmann_population_selection(parents, children, n, parameters={'T': 1}):
    """
    :param parameters: dictionary of parameters that key = parameter name and value = parameter value
    :param parents: list of Parents of current Generation, List
    :param children: list of new children of current Generation, List
    :param n: Number of remaining population, Integer
    :return: list of remained Chromosomes
    """
    t = float(parameters['T'])
    population = parents + children
    fitness_arr = np.array([np.e(x.fitness/t) for x in population])
    fitness_arr /= np.sum(fitness_arr)
    return stochastic_universal_selection(population, fitness_arr, n)


def q_tornoment_based_population_selection(parents, children, n, parameters=None):
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
    return q_tournament_selection(population, fitness_arr, n)

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
