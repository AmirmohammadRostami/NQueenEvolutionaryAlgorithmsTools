import numpy as np

## milad!

class Chromosome:
    def __init__(self, genotype, fitness):
        self.fitness = fitness
        self.genotype = genotype

    def get_phenotype(self):
        phenotype = np.zeros([len(self.genotype), len(self.genotype)]) +0.2
        for i in range(len(self.genotype)):
            for j in range(0,len(self.genotype),2):
                phenotype[i][(i+j)%len(self.genotype)] += 0.4
        for i in range(len(self.genotype)):
            phenotype[i][int(self.genotype[i])] =1
        return phenotype.tolist()
