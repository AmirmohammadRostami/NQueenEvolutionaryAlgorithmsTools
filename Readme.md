
# Evolutionary Algorithm for N queen problem
In this project evolutionary algorithm with various methods e.g. mutation, cross-over have been implemented. The list of the algorithms that are implemented as reported as below:




# Table of contents
0. [*n queen problem*](#n-queen-problem)
1. [*evolutionary_algorithms.py*](#evolutionaryalgorithm-class-evolutionary_algorithmspy)
2. [*evolutionary_algorithms_functions.py*](#evolution-algorithm-functions-evolutionary_algorithms_functionspy)
3. [*chromosome.py*](#chromosome-class-chromosomepy)
4. [*main.py*](#main-mainpy)
5. [*summary*](#summary)

## N queen problem
N queen problem was invented by chess composer, Max Bezzel, in 1848. Since then many mathematicians and computer scientists have been interested on this problem. In 1972, Edsgar Dijkstra proposed a method based on depth-first-backtracking to solve the challenge.<br/>
The whole idea pf the challenge is *to place n queens on a nxn chessboard in a way that not any couple of queens threaten each other*. Figure 1 shows the case where n is 4. It can be seen in this figure that none of the queens threaten each other.<br/><br/>
<!-- <img src="images/N_Queen_Problem.jpg" alt="5 queen problem" style="width:20vw;margin:0 10vw"/> -->
![5 queen problem](./images/N_Queen_Problem.jpg =100x100)
<center> Figure 1</center><br/>

Evolutionary algorithms are one of the solutions to solve this challenge. These algorithms are inspired from the evolutionary instinct of the being in the universe. In order to solve a challenge using this approach, the challenge has to be formulated in a specific manner. In general a evolutionary algorithm consists of the following stages which have been summarized in Figure 2:
1. to be continued...
2. to be continued...
3. to be continued...
[have to insert figure 2]

In order to formulate any problem to be solved using evolutionary algorithms, the following steps have to be followed:
1. We should define a chromosome-based representation for the possible solutions, which in this specific problem it could be estimated with a list of numbers from 1 to n, with size of n where each of the elements show the ith queen's position in the chessboard e.g. the chromosome of figure 1 could be defined as [3, 1, 4, 2].
2. A fitness function should be defined showing the merit of a possible solution. In this challenge the fitness functions could be (1/the number of the queens threatening each other).
3. The initial population should be generated which consists of random chromosomes.


## EvolutionaryAlgorithm class (*evolutionary_algorithms.py*)

### list of the attributes of the class
|Attribute name|type|Initial value|description|
|-|:-:|:-:|:-:|
|_max_generation|integer|200||
|_generation_counter|integer|0||
|_population|list|[]|the list containing the whole population|
|_m|integer|160|initial number of population|
|_n|integer|8|number of the queens|
|_y|integer|80||
|_ n_parent|integer|80||
|_cross_over|function|default_cross_over||
|_cross_over_params|dictionary|{'prob': float(parents_prob)}|a dictionary containing the parameters of the _cross_over function|
|_mutation|function|default_mutation||
|_mutation_params|dictionary|{'prob': float(mutation_prob)}|a dictionary containing the parameters of the _mutation function|
|_remaining_population_selection|function|default_population_selection||
|_remaining_population_selection_params|dictionary|None|a dictionary containing the parameters of the _remaining_population_selection function|
|_parent_selection|function|default_parent_selection||
|_parent_selection_params|dictionary|None|a dictionary containing the parameters of the _parent_selection function|
|_random_gene_generator|function|permutation_random_gene_generator||
|_evaluator|function|default_evaluator|a function which calculates the fitness|
|_stop_condition|function|default_stop_condition|responsible for checking the stop condition(returns True if met)|
|_log|list|[]|To be saved for restoring|


### list of the methods of the class
|function name|parameters|returns|description|
|:-----------:|:--------:|:-----:|:---------:|
|[*__ init __*](#__-init-__) |max_generation=200 <br/>n = 8 <br/> m = 160 <br/> number of population <br/> y = 80 <br/> mutation <br/> cross_over <br/> parent_selection <br/> remaining_population_selection <br/> evaluator <br/>  random_gene_generator <br/> stop_condition |void| Constructor method for evolutionary algorithms class
|[*run*](#run)|name <br/> variance_per_generation=[] <br/> avg_per_generation=[] <br/> best_chromosome=[1] <br/> verbose=False <br/> save_log=True <br/> save_log_path|void|The main method where the evolutionary algorithm is called|
|[*_save_current_log*](#_-save_current_log)|avg_fitness_per_generation <br/> variance_per_generation <br/> best_chromosome|dictionary|Method used for saving the recent run's log|
|[_new_children](#_-new_children)|parents|list|Takes a list of parents and generates a list of children with size of y|
|[_best_gen](#_-best_gen)|-|Chromosome|Returns the best chromosome according to fitness function in the population|
|[_initial_population](#_-initial_population)|-|void|Generates the initial population |


### __ init __

```python
def __init__(mutation,
             cross_over,
             parent_selection,
             remaining_population_selection,
             evaluator,
             gene_generator,
             stop_condition,
             max_generation=200,
             n=8,
             m=160,
             y=80)
```

**max_generation (Integer)**: Defines the maximum number of the generations, <br/>
**n (Integer)**: Number of the queens, maybe power of 2<br/>
**m (Integer)**: Shows the number of the population<br/>
**y (Integer)**: Lambda (number of children), number of children, <br/>
**mutation (Function)**: Mutation algorithm<br/>
**cross_over (Function)**: Cross over algorithm<br/>
**parent_selection (Function)**: Selection algorithm for parents<br/>
**remaining_population_selection (Function)**: Selection algorithm for remaining population<br/>
**evaluator (Function)**: Evaluator algorithm for each chromosome<br/>
**random_gene_generator (Function)**: Random algorithm for initial population <br/>
**stop_condition (Function)**: Stop condition function<br/>

### run
```python
def run(self,
          name,
          variance_per_generation=[],
          avg_per_generation=[],
          best_chromosome=[1],
          verbose=False,
          save_log=True,
          save_log_path='./log_files/'):
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
```

### _ save_current_log
```python
def _save_current_log(self,
                      avg_fitness_per_generation,
                      variance_per_generation,
                      best_chromosome):
```

#### _ new_children
```python
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
```


#### _ best_gen
```python
def _best_gen(self):
    best = self._population[0]
    for i in range(1, len(self._population)):
        if self._population[i].fitness > best.fitness:
            best = self._population[i]
    return best
```


#### _ initial_population

```python
def _initial_population(self):
      for i in range(self._m):
          random_gene = self._random_gene_generator(self._n)
          chromosome = Chromosome(random_gene, 0)
          chromosome.fitness = self._evaluator(chromosome)
          self._population.append(chromosome)
```

## Evolution algorithm functions  (*evolutionary_algorithms_functions.py*)
Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

## Chromosome class (*chromosome.py*)
Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

## main (*main.py*)
Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.



## Summary
Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

### Cross over methods
- [x] default cross over
- [x] multi point cross over
- [ ] other ideas

### Mutation methods
- [x] default method
- [x] swap mutation
- [ ] other ideas



<!-- # Todo list:
- [ ] The whole idea behind evolutionary algorithm will be explained
- [ ] General structure of the main code will be explained
- [ ] Consequently the prototype of the methods will be discussed and a short description for the given methods
- [ ] The summary of the implemented methods will be added to the end
- [ ] Any other useful changes are appreciated. -->







<!--
> This is simply to emphasize a paragraph

______

|title1|title2|Title3|
|-----|:------:|-------:|
|This is case one |This |asdlasjd|
|*asdajsdhk*|asdasd|asdasd|
|asdajsdhk|`asdasd`|asdasd|
|asdajsdhk|asdasd|asdasd|

[This is the link](https:www.google.com)

```python
import numpy as np


``` -->
