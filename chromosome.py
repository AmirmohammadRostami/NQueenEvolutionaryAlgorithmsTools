import numpy as np


class Chromosome:
    def __init__(self, genotype, fitness):
        self.fitness = fitness
        self.genotype = genotype

    def get_phenotype(self):
        phenotype = np.zeros([len(self.genotype), len(self.genotype)]) + 0.05
        for i in range(len(self.genotype)):
            for j in range(0, len(self.genotype), 2):
                phenotype[i][(i + j) % len(self.genotype)] += 0.15
        for i in range(len(self.genotype)):
            con = True
            for j in range(len(self.genotype)):
                if i != j:
                    if self.genotype[i] == self.genotype[j] or abs(self.genotype[i] - self.genotype[j]) == abs(i - j):
                        phenotype[i][int(self.genotype[i])] = 0.6
                        con = False
                        break
            if con:
                phenotype[i][int(self.genotype[i])] = 1.0
        return phenotype.tolist()
