import numpy as np
from chromosome import Chromosome
import random
import pickle
from evolutionary_algorithms_functions import *

best_chromosome_fitness_in_total = -1
best_phenotype = [1]


class EvolutionaryAlgorithm:
    def __init__(self,
                 max_generation=50,
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
        :param y: Lambda (number of children), number of children, Integer
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
            name,
            variance_per_generation=[],
            avg_per_generation=[],
            best_chromosome=[1],
            log=False,
            save_log=True,
            save_log_path='./log_files/'):
        file_name = name
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
            children = self._new_children(parents)
            self._population = self._remaining_population_selection(self._population, children, self._m)
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

    def _new_children(self, parents):
        children = []
        random.shuffle(parents)
        for i in range(0, len(parents) - 1, 2):
            chromosome1, chromosome2 = self._cross_over(parents[i], parents[i + 1])
            chromosome1.fitness = self._evaluator(chromosome1)
            chromosome2.fitness = self._evaluator(chromosome2)
            children += [chromosome1, chromosome2]
            if len(children) >= self._y:
                break
        return children[:self._y]

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
