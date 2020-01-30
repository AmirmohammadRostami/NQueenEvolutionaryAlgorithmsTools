import numpy as np
from chromosome import Chromosome
import random
import pickle
import os

best_chromosome_fitness_in_total = -1
best_phenotype = [1]


class EvolutionaryAlgorithm:
    def __init__(self,
                 mutation,
                 cross_over,
                 parent_selection,
                 remaining_population_selection,
                 evaluator,
                 gene_generator,
                 stop_condition,
                 max_generation=200,
                 n=8,
                 m=160,
                 y=80, ):
        '''
        :param max_generation: Max number of generation, Integer
        :param n: Number of Queens, maybe power of two!, Integer
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
        global best_chromosome_fitness_in_total
        global best_phenotype
        self._max_generation = max_generation
        self._generation_counter = 0
        self._population = []
        self._m = m
        self._n = n
        self._y = y
        self._n_parent = y
        self._cross_over = cross_over[0]
        self._cross_over_params = cross_over[1]
        self._mutation = mutation[0]
        self._mutation_params = mutation[1]
        self._remaining_population_selection = remaining_population_selection[0]
        self._remaining_population_selection_params = remaining_population_selection[1]
        self._parent_selection = parent_selection[0]
        self._parent_selection_params = parent_selection[1]
        self._random_gene_generator = gene_generator
        self._evaluator = evaluator
        self._stop_condition = stop_condition
        self._log = []
        best_chromosome_fitness_in_total = -1
        best_phenotype = [1]

    def run(self,
            name,
            variance_per_generation=[],
            avg_per_generation=[],
            best_chromosome=[1],
            verbose=False,
            save_log=True,
            save_log_path='/Users/miladbohlouli/Documents/evolutionary_algorithms_tools_for_n_queen/log_files/'):
        file_name = name
        if verbose:
            print('EA algorithms Running . . . ')
        self._initial_population()
        self._generation_counter = 1
        self._log.append(self._save_current_log(avg_per_generation, variance_per_generation, best_chromosome))
        if verbose:
            print(self._log[-1])
        while not self._stop_condition(self._generation_counter, self._max_generation):
            self._generation_counter += 1
            if verbose:
                print(self._generation_counter)
            parents = self._parent_selection(self._population, self._n_parent, self._parent_selection_params)
            children = self._new_children(parents)
            if type(self._population) != list:
                self._population = self._population.tolist()
            self._population = self._remaining_population_selection(self._population, children, self._m,
                                                                    self._remaining_population_selection_params)
            self._log.append(self._save_current_log(avg_per_generation, variance_per_generation, best_chromosome))
            if verbose:
                print(self._log[-1])
        file_name += '.pickle'
        print('yes')
        if save_log:
            with open(save_log_path + file_name, 'wb') as file:
                pickle.dump(self._log, file)
            print('log file successfully saved!')

    def _save_current_log(self, avg_fitness_per_generation, variance_per_generation, best_chromosome):
        fitness = []
        best_phenotype_index = 0
        for i in range(1, len(self._population)):
            if self._population[i].fitness > self._population[best_phenotype_index].fitness:
                best_phenotype_index = i
            fitness.append(self._population[i].fitness)
        var_fitness = np.var(fitness)
        avg_fitness = np.average(fitness)
        avg_fitness_per_generation.append(avg_fitness)
        variance_per_generation.append(var_fitness)
        global best_chromosome_fitness_in_total, best_phenotype
        if self._population[best_phenotype_index].fitness >= best_chromosome_fitness_in_total:
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
            chromosome1, chromosome2 = self._cross_over(parents[i], parents[i + 1], self._cross_over_params)
            chromosome1 = self._mutation(chromosome1, self._mutation_params)
            chromosome2 = self._mutation(chromosome2, self._mutation_params)
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
