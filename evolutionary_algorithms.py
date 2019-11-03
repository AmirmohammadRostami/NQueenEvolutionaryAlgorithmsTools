import numpy as np
from chromosome import Chromosome
import random
import datetime
import pickle

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


best_chromosome_fitness_in_total = -1
best_phenotype = [1]


class EvolutionaryAlgorithm:
    def __init__(self,
                 max_generation=100,
                 n=8,
                 n_parent=40,
                 m=160,
                 y=80,
                 mutation=default_mutation,
                 cross_over=default_cross_over,
                 parent_selection=default_parent_selection,
                 remaining_population_selection=default_population_selection,
                 evaluator=default_evaluator,
                 random_gene_generator=default_random_gene_generator,
                 stop_condition=default_stop_condition):
        '''

        :param max_generation: Max number of generation, Integer
        :param n: Number of Queens, maybe power of two!, Integer
        :param n_parent: Number of Parent, Integer
        :param m: Mu (number of population), number of population, Integer
        :param y: Lambda (number of childs), number of children, Integer
        :param mutation: Mutation algorithm, Function
        :param cross_over: Cross over algorithm, Function
        :param parent_selection: Selection algorithm for parents, Function
        :param remaining_population_selection: Selection algorithm for remaining population, Function
        :param evaluator: Evaluator algorithm for each chromosome, Function
        :param random_gene_generator: Random algorithm for initial population, Function
        :param stop_condition: Stop condition function, Function
        '''
        self._max_generation = max_generation
        self._generation_counter = 0
        self._cross_over = cross_over
        self._population = []
        self._m = m
        self._n = n
        self._n_parent = n_parent
        self._y = y
        self._mutation = mutation
        self._remaining_population_selection = remaining_population_selection
        self._parent_selection = parent_selection
        self._random_gene_generator = random_gene_generator
        self._evaluator = evaluator
        self._stop_condition = stop_condition
        self._log = []

    def run(self,
            variance_per_generation=[],
            avg_per_generation=[],
            best_chromosome=[1],
            log=False,
            save_log=True,
            save_log_path='./log_files/'):
        file_name = str(datetime.datetime.now())
        print('EA algorithms Running . . . ')
        self._initial_population()
        self._generation_counter = 1
        self._log.append(self._save_current_log(avg_per_generation, variance_per_generation, best_chromosome))
        if log:
            print(self._log[-1])
        while not self._stop_condition(self._generation_counter, self._max_generation):
            self._generation_counter += 1
            print(self._generation_counter)
            parents = self._parent_selection(self._population, self._n_parent)
            childs = self._new_childs(parents)
            self._population = self._remaining_population_selection(self._population, childs, self._m)
            self._log.append(self._save_current_log(avg_per_generation, variance_per_generation, best_chromosome))
            if log:
                print(self._log[-1])

        file_name += '.pickle'
        print('yes')
        if save_log:
            with open(save_log_path + file_name, 'wb') as file:
                pickle.dump(self._log, file)
            print('log file successfully saved!')

    def _save_current_log(self, avg_fitness_per_generation, variance_per_generation, best_chromosome):
        fittness = []
        best_phenotype_index = 0
        for i in range(1, len(self._population)):
            if self._population[i].fitness > self._population[best_phenotype_index].fitness:
                best_phenotype_index = i
            fittness.append(self._population[i].fitness)
        var_fitness = np.var(fittness)
        avg_fitness = np.average(fittness)
        avg_fitness_per_generation.append(avg_fitness)
        variance_per_generation.append(var_fitness)
        global best_chromosome_fitness_in_total, best_phenotype
        if self._population[best_phenotype_index].fitness > best_chromosome_fitness_in_total:
            best_chromosome_fitness_in_total = self._population[best_phenotype_index].fitness
            best_phenotype = self._population[best_phenotype_index].get_phenotype()
        best_chromosome[-1] = best_phenotype
        return {'generation': self._generation_counter,
                'avg_fitness': avg_fitness,
                'var_fitness': var_fitness,
                'best_phenotype': best_chromosome,
                'best_genotype': self._population[best_phenotype_index].genotype.tolist(),
                'best_fitness': self._population[best_phenotype_index].fitness,
                }

    def _new_childs(self, parents):
        childs = []
        random.shuffle(parents)
        for i in range(0, len(parents) - 1, 2):
            chromosome1, chromosome2 = self._cross_over(parents[i], parents[i + 1])
            chromosome1.fitness = self._evaluator(chromosome1)
            chromosome2.fitness = self._evaluator(chromosome2)
            childs += [chromosome1, chromosome2]
            if len(childs) >= self._y:
                break
        return childs[:self._y]

    def _best_gen(self):
        best = self._population[0]
        for i in range(1, len(self._population)):
            if self._population[i].fitness > best.fitness:
                best = self._population[i]
        return best

    def _initial_population(self):
        for i in range(self._m):
            random_gene = self._random_gene_generator(self._n)
            chromosome = Chromosome(random_gene, 0)
            chromosome.fitness = self._evaluator(chromosome)
            self._population.append(chromosome)


if __name__ == '__main__':
    ga = EvolutionaryAlgorithm()
    ga.run()
