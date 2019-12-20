import random
import numpy as np
from chromosome import Chromosome
import warnings

'''
-------------------------------------
Selection Algorithms, (SA)
-------------------------------------
input: parameters (dictionary of algorithm parameterss key: parameters name, value: parameters value)
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


def neighbour_based_mutation(chromosome, parameters=None):
    gene = chromosome.genotype.copy()
    begin_indx, end_indx = np.random.randint(len(gene)), np.random.randint(len(gene))
    if begin_indx <= end_indx:
        s = gene[begin_indx:end_indx]
        s = s[::-1]
        gene[begin_indx:end_indx] = s
    else:
        c = len(gene) - (abs(begin_indx - end_indx))
        # print(c)
        for i in range(c):
            print(end_indx, begin_indx)
            gene[end_indx] = chromosome.genotype[begin_indx]
            gene[begin_indx] = chromosome.genotype[end_indx]
            end_indx = (end_indx - 1) % len(gene)
            begin_indx = (begin_indx + 1) % len(gene)
            # print('new b and e =', begin_indx, end_indx)

    chromosome.genotype = gene

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


def neighbour_based_Cross_Over(parent1, parent2, parameter = None):
    """
    :param parent1: First parent chromosome,     Gene, np.array with shape = (1,len(parent))
    :param parent2: Second parent chromosome, Gene, np.array with shape = (1,len(parent))
    :return: return two chromosome for each children, Chromosome
    """

    import copy
    gen1, gen2 = parent1.genotype, parent2.genotype
    child1, child2 = np.zeros(len(parent1.genotype)), np.zeros(len(parent1.genotype))

    # find neighbours of a gene
    def neighbour(arr_1_dim, index):
        if index == len(arr_1_dim) - 1:
            return arr_1_dim[index - 1], arr_1_dim[0]
        if index == 0:
            return arr_1_dim[-1], arr_1_dim[index + 1]
        return arr_1_dim[index - 1], arr_1_dim[index + 1]

    def has_superior_neighbour(list):
        for i in range(len(list)):
            temp = list.copy()
            temp.remove(list[i])
            for j in range(len(temp)):
                if list[i] in temp:
                    return True, list[i]
        else:
            return False, 'ho ha ha'

    def remover(item, dic):
        for i in range(len(dic)):
            if item in dic[i]:
                dic[i].remove(item)
                if item in dic[i]:
                    dic[i].remove(item)

    def string_maker(gen_arr1dim, dict):

        dic = copy.deepcopy(dict)
        gen_arr1dim[0] = np.random.randint(0, len(gen_arr1dim))
        # print('orginal dic',dic)
        # print('orgin gen =',gen_arr1dim)
        remover(gen_arr1dim[0], dic)

        empty_counter = 0
        counter = 1
        for block in range(1, len(gen_arr1dim)):

            counter += 1
            if counter == 8:
                index = list(range(-empty_counter, len(gen_arr1dim) - 1 - empty_counter))
                tt = list(range(len(gen_arr1dim)))
                for i in range(len(index)):
                    tt.remove(gen_arr1dim[index[i]])
                if not len(tt) == 1:
                    print('error')
                gen_arr1dim[-empty_counter - 1] = tt[0]
                continue

            if not empty_counter == 0:
                block = block - empty_counter

            list_nghrs = dic[gen_arr1dim[block - 1]]

            # print('current gen =',gen_arr1dim)
            # print('current dic =',dic)
            # print('in block{}  list nghbrs = {}'.format(block,list_nghrs))

            tf, _ = has_superior_neighbour(list_nghrs)
            if tf:
                # print('\n',list_nghrs)
                # print(gen_arr1dim,'\n')
                gen_arr1dim[block] = _
                remover(gen_arr1dim[block], dic)
                continue

            if len(list_nghrs) == 0:
                # print('ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss')
                # print('empty counter = ',empty_counter)
                # print('block number =',block)
                index = list(range(-empty_counter, block + empty_counter))
                # print('index zero nghbrs=',index)
                tt = list(range(len(gen_arr1dim)))
                for i in range(len(index)):
                    tt.remove(gen_arr1dim[index[i]])
                # print('tt zero nghbrs= ',tt)
                # print('sexy gen =',gen_arr1dim)
                gen_arr1dim[block] = np.random.choice(tt)
                remover(gen_arr1dim[block], dic)
                continue

            list_len_nghbrs_of_nghbrs = []
            for p in range(len(list_nghrs)):
                list_len_nghbrs_of_nghbrs.append(len(set(dic[list_nghrs[p]])))
            if 0 in list_len_nghbrs_of_nghbrs:
                empty_counter += 1
                gen_arr1dim[-empty_counter] = list_nghrs[list_len_nghbrs_of_nghbrs.index(0)]
                remover(list_nghrs[list_len_nghbrs_of_nghbrs.index(0)], dic)
                continue

            # print('gen = ',gen_arr1dim)
            # print('list len neghbrs',list_len_nghbrs_of_nghbrs,'\n')

            c = 0
            x_list = []
            x_list.append(list_nghrs[list_len_nghbrs_of_nghbrs.index(min(list_len_nghbrs_of_nghbrs))])
            for t in range(list_len_nghbrs_of_nghbrs.index(min(list_len_nghbrs_of_nghbrs)) + 1,
                           len(list_len_nghbrs_of_nghbrs)):
                if list_len_nghbrs_of_nghbrs[t] == min(list_len_nghbrs_of_nghbrs):
                    x_list.append(list_nghrs[t])
                    c += 1
            same_min_list = x_list

            if c == 0:
                # print('\n',list_nghrs)
                # print(gen_arr1dim,'\n')
                gen_arr1dim[block] = list_nghrs[list_len_nghbrs_of_nghbrs.index(min(list_len_nghbrs_of_nghbrs))]
                remover(gen_arr1dim[block], dic)
                continue

            else:
                # print('\n',list_nghrs)
                # print(gen_arr1dim,'\n')
                # print('pekhh',same_min_list)
                gen_arr1dim[block] = list_nghrs[list_nghrs.index(same_min_list[np.random.randint(len(same_min_list))])]
                remover(gen_arr1dim[block], dic)

        return gen_arr1dim

    # make a dictionary of the neighbours
    def dictionary(gen1, gen2):
        dic = {}
        for i in range(len(gen1)):
            dic[i] = []
        for i in range(len(gen1)):
            a, b = neighbour(gen1, i)
            dic[gen1[i]].append(a)
            dic[gen1[i]].append(b)
            a, b = neighbour(gen2, i)
            dic[gen2[i]].append(a)
            dic[gen2[i]].append(b)
        return dic

    neighbours_dict = dictionary(gen1, gen2)
    ch1 = Chromosome(string_maker(child1, neighbours_dict), 0)
    ch2 = Chromosome(string_maker(child2, neighbours_dict), 0)

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
