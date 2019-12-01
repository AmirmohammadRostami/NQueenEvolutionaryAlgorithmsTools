
# Evolutionary Algorithm for N queen problem
> The Readme file is under construction. It is estimated that the readme will be finished due to 1 a.m. 30 November 2019. <p style="color: red"> Due to the current climate disorders that have encountered in Tehran, the deadline of the completion has been postponed to tomorrow. Thanks for your patience. <h2 style="color:green">Special thanks to Amir Rostami for his patience and collaboration</h2> </p>

In this project evolutionary algorithm with various methods e.g. mutation, cross-over have been implemented. The list of the algorithms that are implemented as reported as below:

### Cross over methods
- [x] default cross over
- [x] multi point cross over
- [ ] other ideas

### Mutation methods
- [x] default method
- [x] swap mutation
- [ ] other ideas

# Table of contents
1. [*evolutionary_algorithms.py*](#evolutionary_algorithmspy)
2. [*evolutionary_algorithms_functions.py*](#evolutionary_algorithms_functionspy)
3. [*chromosome.py*](#chromosomepy)
4. [*main.py*](#mainpy)


## *evolutionary_algorithms.py*

|function name|parameters|returns|description|
|:-----------:|:--------:|:-----:|:---------:|
|[*__ init __*](#__-init-__) |max_generation <br/>n <br/> m <br/> number of population <br/> y <br/> mutation <br/> cross_over <br/> parent_selection <br/> remaining_population_selection <br/> evaluator <br/>  random_gene_generator <br/> stop_condition |NA| Constructor method for evolutionary algorithms class
|[*run*](#run)|name <br/> variance_per_generation=[] <br/> avg_per_generation=[] <br/> best_chromosome=[1] <br/> verbose=False <br/> save_log=True <br/> save_log_path|NA||
|[*_save_current_log*](#_-save_current_log)|avg_fitness_per_generation <br/> variance_per_generation <br/> best_chromosome|NA||
|[_new_children](#_-new_children)|parents|NA||
|[_best_gen](#_-best_gen)|-|NA||
|[_initial_population](#_-initial_population)|-|NA||


#### __ init __
max_generation: Max number of generation, Integer<br/>
n: Number of Queens, maybe power of two!, Integer<br/>
m: Mu (number of population), number of population, Integer<br/>
y: Lambda (number of children), number of children, Integer<br/>
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

## chromosome class (*chromosome.py*)
Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

## main (*main.py*)
Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.


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
