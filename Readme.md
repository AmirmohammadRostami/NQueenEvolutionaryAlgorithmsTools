
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
![5 queen problem](./images/N_Queen_Problem.jpg)
<center> Figure 1</center><br/>

Evolutionary algorithms are one of the solutions to solve this challenge. These algorithms are inspired from the evolutionary instinct of the being in the universe. In order to solve a challenge using this approach, the challenge has to be formulated in a specific manner. In this regard the following steps have to be followed:
1. We should define a chromosome-based representation for the possible solutions, which in this specific problem it could be estimated with a list of numbers from 1 to n, with size of n where each of the elements shows the ith queens positiono in the chessboard e.g. the chromosome of figure 1 could be defined as [3, 1, 4, 2].
2. A fitness function should be defined showing the merit of a possible solution. In this challenge the fitness functions could be (1/the number of the queens threatening each other).
3. The initial population should will be generated which consists of random chromosomes.
to be continued...



## EvolutionaryAlgorithm class (*evolutionary_algorithms.py*)

|function name|parameters|returns|description|
|:-----------:|:--------:|:-----:|:---------:|
|[*__ init __*](#__-init-__) |max_generation=200 <br/>n = 8 <br/> m = 160 <br/> number of population <br/> y = 80 <br/> mutation <br/> cross_over <br/> parent_selection <br/> remaining_population_selection <br/> evaluator <br/>  random_gene_generator <br/> stop_condition |void| Constructor method for evolutionary algorithms class
|[*run*](#run)|name <br/> variance_per_generation=[] <br/> avg_per_generation=[] <br/> best_chromosome=[1] <br/> verbose=False <br/> save_log=True <br/> save_log_path|void||
|[*_save_current_log*](#_-save_current_log)|avg_fitness_per_generation <br/> variance_per_generation <br/> best_chromosome|dictionary||
|[_new_children](#_-new_children)|parents|list||
|[_best_gen](#_-best_gen)|-|NA|Chromosome|
|[_initial_population](#_-initial_population)|-|void||


#### __ init __

max_generation (Integer): Defines the maximum number of the generations, <br/>
n (Integer): Number of the queens, maybe power of 2<br/>
m (Integer): Shows the number of the population<br/>
y Integer: Lambda (number of children), number of children, <br/>
mutation: Mutation algorithm, Function<br/>
cross_over: Cross over algorithm, Function<br/>
parent_selection: Selection algorithm for parents, Function<br/>
remaining_population_selection: Selection algorithm for remaining population, Function<br/>
evaluator: Evaluator algorithm for each chromosome, Function<br/>
random_gene_generator: Random algorithm for initial population, Function<br/>
stop_condition: Stop condition function, Function<br/>


#### run
Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

#### _ save_current_log
Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

#### _ new_children
Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

#### _ best_gen
Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

#### _ initial_population
Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.


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
