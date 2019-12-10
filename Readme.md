
# Evolutionary Algorithm for N queen problem
In this project, the evolutionary algorithm with various approaches has been implemented and applied on the n queen problem. In the Introduction section the well-known n queen problem and the general overview of evolutionary algorithms will be discussed. Afterwards, in the implementation section we will go into the details of evolutionary algorithm's implementation. A quick installation guide and the procedure of running and analyzing the results have been described in section 3. Eventually, if you want to contribute in the project, see the final part of section 3 which covers the rules you have to follow. For any further information, feel free to contact us using the information given at section 4.

# Table of contents
<ol>

  <li><b>Introduction</b>
  <ol>
    <li> [*N queen problem*](#n-queen-problem)</li>
    <li> [Evolutionary algorithms](#evolutionary-algorithms)</li>
  </ol>
  </li>

  <li><b>Implementation</b>
  <ol>
    <li>[*evolutionary_algorithms.py*](#evolutionaryalgorithm-class-evolutionary_algorithmspy)</li>
    <li>[*evolutionary_algorithms_functions.py*](#evolution-algorithm-functions-evolutionary_algorithms_functionspy)</li>
    <li>[*chromosome.py*](#chromosome-class-chromosomepy)</li>
  </ol>
  </li>
  <li><b>Quick tour</b>
  <ol>
    <li>[*Installation guide*](#installation-guide)</li>
    <li>[*Graphical User Interface*](#graphical-user-interface)</li>
    <li>[*A few words with contributors*](#a-few-words-with-contributors)</li>
  </ol>
  </li>
  <li><b>[contact us](#contact-us)</b>
  </li>
</ol>

# 1. Introduction

## N queen problem
N queen problem was invented by chess composer, Max Bezzel, in 1848. Since then many mathematicians and computer scientists have been interested on this problem. In 1972, Edsgar Dijkstra proposed a method based on depth-first-backtracking to solve the challenge.<br/>
The whole idea pf the challenge is *to place n queens on a nxn chessboard in a way that not any couple of queens threaten each other*. Figure 1 shows the case where n is 4. It can be seen in this figure that none of the queens threaten each other.<br/><br/>

<p width="auto" align="center">
<img align="center" src="./images/N_Queen_Problem.jpg" alt="5 queen problem" width=200px>
</p>

## Evolutionary algorithms
Evolutionary algorithms are one of the solutions to solve this challenge. These algorithms are inspired from the evolutionary instinct of the being in the universe. In order to solve a challenge using this approach, the challenge has to be formulated in a specific manner. In general a evolutionary algorithm consists of the following stages which have been summarized in Figure 2:
1. define the representation
2. Initiate the population
3. parent selection
4. Children generation
5. children evaluation
6. next generation selection
7. check the stop condition, if not met repeat from step 3

<p align="center" width="200px">
<img src="./images/flowchart.png" alt="flowchart" width=200px>
</p>

In order to formulate any problem to be solved using evolutionary algorithms, the following steps have to be followed:
1. We should define a chromosome-based representation for the possible solutions, which in this specific problem it could be estimated with a list of numbers from 1 to n, with size of n where each of the elements show the ith queen's position in the chessboard e.g. the chromosome of figure 1 could be defined as [3, 1, 4, 2].
2. A fitness function should be defined showing the merit of a possible solution. In this challenge the fitness functions could be (1/the number of the queens threatening each other).
3. The initial population should be generated which consists of random chromosomes.


# 2. Implementation

## EvolutionaryAlgorithm class (*evolutionary_algorithms.py*)
This is the main class which handles the evolution process.

### list of the attributes of the class
|Attribute name|type|Initial value|description|
|-|:-:|:-:|:-:|
|_max_generation|integer|200|Maximum number of steps that the evolution will be progressed|
|_generation_counter|integer|0|The current step of evolution|
|_population|list|[]|the list containing the whole population|
|_m|integer|160|initial number of population|
|_n|integer|8|number of the queens|
|_y|integer|80|The number of the children that are selected from generated children|
|_ n_parent|integer|80|number of the parents that should be selected from the population|
|_cross_over|function|default_cross_over|The function used to cross over two chromosomes|
|_cross_over_params|dictionary|{'prob': float(parents_prob)}|a dictionary containing the parameters of the _cross_over function|
|_mutation|function|default_mutation|The mutation approach which will be applied on a chromosome|
|_mutation_params|dictionary|{'prob': float(mutation_prob)}|a dictionary containing the parameters of the _mutation function|
|_remaining_population_selection|function|default_population_selection|Approach used for selecting the next population among the new children and the current population|
|_remaining_population_selection_params|dictionary|None|a dictionary containing the parameters of the _remaining_population_selection function|
|_parent_selection|function|default_parent_selection|functions used for selecting a subset from the parents|
|_parent_selection_params|dictionary|None|a dictionary containing the parameters of the _parent_selection function|
|_random_gene_generator|function|permutation_random_gene_generator|Function used for generating the initial population|
|_evaluator|function|default_evaluator|a function which calculates the fitness|
|_stop_condition|function|default_stop_condition|responsible for checking the stop condition(returns True if met)|
|_log|list|[]|To be saved for restoring|


### list of the methods of the class
|function name|parameters|returns|description|
|:-----------:|:--------:|:-----:|:---------:|
|[*__ init __*](#__-init-__) |max_generation=200 <br/>n = 8 <br/> m = 160 <br/> number of population <br/> y = 80 <br/> mutation <br/> cross_over <br/> parent_selection <br/> remaining_population_selection <br/> evaluator <br/>  random_gene_generator <br/> stop_condition |void| Constructor method for evolutionary algorithms class
|[*run*](#run)|name <br/> variance_per_generation=[] <br/> avg_per_generation=[] <br/> best_chromosome=[1] <br/> verbose=False <br/> save_log=True <br/> save_log_path|void|The main method where the evolutionary algorithm is called|
|[*_save_current_log*](#_-save_current_log)|avg_fitness_per_generation <br/> variance_per_generation <br/> best_chromosome|dictionary|Method used for saving the recent run's log|
|[*_new_children*](#_-new_children)|parents|list|Takes a list of parents and generates a list of children with size of y|
|[*_best_gen*](#_-best_gen)|-|Chromosome|Returns the best chromosome according to fitness function in the population|
|[*_initial_population*](#_-initial_population)|-|void|Generates the initial population |


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
**returns ()**:<br/>


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
```

**name (string)**: the name which the log file will be saved with.<br/>
**variance_per_generation (list)**: A list of the fitnesses for each of the generations.<br/>
**avg_per_generation (list)**: A list of the averages for fitnesses of each generation (every generation consists of many solutions which each has a fitness).<br/>
**best_chromosome (list)**: A list containing best phenotypes of the population.<br/>
**verbose (boolean)**: If True the log will also be printed.<br/>
**save_log (boolean)**: If True the log will be saved otherwise not.<br/>
**save_log_path (string)**: Defines the path in which the log will be saved.<br/>
**returns ()**:<br/>

The whole process of running and finding the best solution is done in the above method. It can be seen that in the first part of the function, the initial population is called which has been described later. Using a while loop, which iterates until the stop_condition has been met, the the whole process goes on. In this loop a subset of the parents are chosen, then the children are generated from the selected parents. Finally the new population is selected among the current population and the new generated children. It should also be noted that the log of the whole operation is saved at the end of the function.


### _ save_current_log
```python
def _save_current_log(self,
                      avg_fitness_per_generation,
                      variance_per_generation,
                      best_chromosome):
```
**avg_fitness_per_generation (float)**: the global variable containing the average fitness values for chromosomes on a generation<br/>
**variance_per_generation (float)**: the global variable containing the variance of fitness values for chromosomes on a generation<br/>
**best_chromosome (list)**: A list containing the phenotype of the best chromosome on all of the generations<br/>

All the evaluation metrics are calculated in the above method. At the first step the phenotype with the fitness is found from the population, and meanwhile the for loop, all the fitness values of each chromosome are extracted and stored in a list. The variance and the average fitness has been calculated from the above list which specify the average and variance values for this generation (you should remind that the above method is called once on each iteration of the common while loop discussed in run section), this means that all of the evaluation metrics are calculated per generation and appended to lists (*avg_fitness_per_generation, variance_per_generation*) to be depicted on the results plot, then the best chromosome fitness of this generation (*self._population[best_phenotype_index].fitness*) is compared to the previous generations (*best_chromosome_fitness_in_total*). Finally a dictionary with the below keys is returned:
- generation (integer): the number of the generation
- avg_fitness (float): the average fitness of the current generation
- var_fitness (float): the variance fitness of the current generation
- best_phenotype (list): phenotype for the best chromosome in all of the generations
- best_genotype (list): genotype for the best chromosome in the current population
- best_fitness (float): best fitness value for the current population


#### _ new_children
```python
    def _new_children(self, parents):
```
**parents (list)**: list of the parents that have been selected from the population <br/>

This function is the main kernel of the evolutionary algorithm since the cross over and the mutation operations are done in this function. At first the parents have been shuffled. Using a for loop which iterates over the shuffled parents, the children are generated. On each iteration of this loop, at first two chromosomes are generated by combining two parents(cross over), then the mutation operation is done on each of the generated children individually, afterwards the fitness values of each of the generated children are calculated. The generated children are appended to a list. Eventually the children list contains all the generated children, the first *y* number of the children are returned as the selected new children.


#### _ best_gen
```python
def _best_gen(self):
```
In the above function the best chromosome in the current population is found according to their fitness values.

#### _ initial_population

```python
def _initial_population(self):
```
The population attribute of the EvolutionaryAlgorithm class is initiated in this function based on the gene generation approach (_ random_gene_generator). It can be seen that m samples are generated with size of n, where m shown the number of the initial population and n defines the number of queens.

## Evolution algorithm functions  (*evolutionary_algorithms_functions.py*)

|function name|parameters|returns|description|
|:-:|:-:|:-:|:-:|
|[*warning_data_type_check_selection_algorithms*](#warning_data_type_check_selection_algorithms)|items, probs|||
|[*roulette_wheel_selection*](#roulette_wheel_selection)|items, probs, n|||
|[*stochastic_universal_selection*](#stochastic_universal_selection)|items, probs, n|||
|[*default_random_gene_generator*](#default_random_gene_generator)|number_of_queen, parameters=None|||
|[*permutation_random_gene_generator*](#permutation_random_gene_generator)|number_of_queen, parameters=None|||
|[*default_evaluator*](#default_evaluator)|chromosome, parameters=None|||
|[*default_mutation*](#default_mutation)|chromosome, parameters={'prob': 0.05}|||
|[*random_swap_mutation*](#random_swap_mutation)|chromosome, parameters={'prob': 0.05}|||
|[*default_cross_over*](#default_cross_over)|parent1, parent2, parameters={'prob': 0.4}|||
|[*multi_points_crossover*](#multi_points_crossover)|parent1, parent2, parameters={'prob': 0.4, 'points_count': 'middle'}|||
|[*default_parent_selection*](#default_parent_selection)|population, n, parameter=None|||
|[*default_population_selection*](#default_population_selection])|parents, children, n, parameters=None|||
|[*fitness_based_population_selection*](#fitness_based_population_selection)|parents, children, n, parameters=None|||
|[*default_stop_condition*](#default_stop_condition)|generation, max_generation, parameters=None|||

### warning_data_type_check_selection_algorithms
```python
def warning_data_type_check_selection_algorithms(items, probs):
```
**param items (np.array or list)**: Items that want to choose from them, np.array or list <br/>
**param probs (np.array or list)**: Probabilities of each item<br/>
**returns (np.array)**: fixed items and probs<br/>
The probs is a list of probabilities for the items, in this function the probs are checked to be in the correct format. These features include:
- checking if the items and the probs have the same size
- convert the items and the probs to ndarray format
- check if the probabilities are positive
- Normalize the probs values in order to have a sum of 1

### roulette_wheel_selection
```python
def roulette_wheel_selection(items, probs, n):
```
**items (np.array or list)**:  Items that want to choose from them<br/>
**probs (np.array or list)**:  Probabilities of each item<br/>
**n (Integer)**: number of selected item(s)<br/>
**return (np.array)**: array of selected Items<br/>
The main goal of this method is to select n items from a list with specified probabilities. In this method a random list is generated with values in range [0, 1]. The cumulative probability of the probs parameter is calculated afterwards. Using a for loop which iterates over the generated random values, each time the lowest index where the cumulative sum is higher than the generated random value is chosen as an item to return. Eventually a list of the selected indexes is returned (It should be mentioned that the list may contain repetitive values).

### stochastic_universal_selection
```python
def stochastic_universal_selection(items, probs, n):
```
**items (np.array or list)**:  Items that want to choose from them<br/>
**probs (np.array or list)**:  Probabilities of each item<br/>
**n (Integer)**: number of selected item(s)<br/>
**return (np.array)**: array of selected Items<br/>
In this function the well-known SUS algorithm has been implemented. In this selection approach, at first, the probs and the items are shuffled with the same manner. Then n (number of the desired selections) numbers will be generated which are linearly selected from [0, 1-(1/n)] and are summed with a bias value which is selected randomly from U(0, (1/n))(uniform distribution). This results in a list of float values which could vary in [0, 1]from the cumulative probability is calculated from the probs parameter, afterwards the cumulative probabilities will be compared with the final generated values. To conduct this operation, a for loop is applied on the generated values where on each iteration one value is chosen from the list and the cumulative probabilities are compared with the selected value. This has been implemented by comparing the probabilities consequently till we reach a probability compared to the selected value (Because both of the generated values and the cumulative probabilities are incremental, there is no need to reset the comparison on each iteration of the outer loop). <br/>
For a deeper understanding, read the below numerical example:
Suppose n is 5, the generated list (which is named as index_of_choose in implementation) is generated as [0.3, 0.4, 0.5, 0.8, 0.9] and the probs parameter is a list of [0.1, 0.2, 0.05, 0.01, 0.05, 0.04, 0.2, 0.06, 0.1, 0.1] (remind that n is not supposedly equal with the size of the items list):
cum_sum = [0.1, 0.3, 0.35, 0.45, 0.5, 0.54, 0.74, 0.8, 0.9, 1]
An iteration is done over the generated values, which has been summarized in the below table:<br/>

|outer loop iteration number|seleted index from index_of_choice |items_pointer before the inner while loop|items_pointer after the inner while loop|selected_items|
|:-:|:-:|:-:|:-:|:-:|
|1|0.3|0|1|items[1]|
|2|0.4|1|3|+ items[3]|
|3|0.5|3|4|+ items[4]|
|4|0.8|4|7|+ items[7]|
|5|0.9|7|8|+ items[8]|


### default_random_gene_generator
```python
def default_random_gene_generator(number_of_queen, parameters=None):
```
**number_of_queen (integer)**: Number of Queen <br/>
**parameters (dictionary)**: dictionary of parameters that key = parameter name and value = parameter value <br/>
**returns (np.array)**: ndarray with length of number_of_queen for each row<br/>

This is the default random gene generation method which returns a list of n values in range of [0, n]. You should notice that the numbers inside a list(gene) are not necessarily unique.

### permutation_random_gene_generator
```python
def permutation_random_gene_generator(number_of_queen, parameters=None):
```
**number_of_queen (integer)**: Number of Queen <br/>
**parameters (dictionary)**: dictionary of parameters that key = parameter name and value = parameter value <br/>
**returns (np.array)**: ndarray with length of number_of_queen for each row<br/>

Another method used for gene generation. In this method a list of n numbers from 1 to n are generated, then the generated list is shuffled. The main difference of this method compared top the default_random_gene_generator is the uniqueness of the generated values.

### default_evaluator
```python
def default_evaluator(chromosome, parameters=None):
```
**chromosome (Chromosome)**: The specified chromosome to calculate the fitness for<br/>
**parameters (dictionary)**: dictionary of parameters that key = parameter name and value = parameter value<br/>
**returns (float)**: fitness of that chromosome which is a value in range [0, 1]<br/>

In this function the fitness value of the given chromosome is calculated. As discussed before the fitness value should specify the amount of the similarity of the chromosome to the desired output. In n queen problem this could be defined as the reverse of the number of the threats between the queens (1 / number of threats). As high the number of the threats is, the lower the fitness will be, and the value of the fitness converges to infinite when the threats converge to zero.

### default_mutation
```python
def default_mutation(chromosome, parameters={'prob': 0.05}):
```
**chromosome (Chromosome)**: the chromosome that the mutation will be applied on<br/>
**parameters (dictionary)**: dictionary of parameters that key = parameter name and value = parameter value<br/>
**return (Chromosome)**: The mutated chromosome<br/>

One of the fundamental stages in evolutionary algorithms is mutation, which tries to manipulate the given chromosome in a specific manner. This function is the default mutation algorithm which changes some of the genes of the chromosome with probability of prob (defined in the parameters dictionary with initial value of 0.5). As higher the value of the probability, the more chance of changing the genes. Eventually the manipulated chromosome will be returned.

> Author: mohammad Tavakkoli, will be completed

### random_swap_mutation
```python
def random_swap_mutation(chromosome, parameters={'prob': 0.05}):
```
**chromosome (Chromosome)**: the chromosome that the mutation will be applied on<br/>
**parameters (dictionary)**: dictionary of parameters that key = parameter name and value = parameter value<br/>
**return (Chromosome)**: The mutated chromosome<br/>


### default_cross_over
```python
def default_cross_over(parent1, parent2, parameters={'prob': 0.4}):
```
**parameters (dictionary)**: dictionary of parameters that key = parameter name and value = parameter value<br/>
**parent1 (Chromosome)**: First parent chromosome, Gene, np.array with len [n^2,1]<br/>
**parent2 (Chromosome)**: Second parent chromosome, Gene, np.array with len [n^2,1]<br/>
**returns (Chromosome, Chromosome)**: return two chromosome for each children, Chromosome<br/>

Similar to mutation, cross over is the other fundamental stage in evolutionary algorithms, which tries to combine two chromosomes named as parents in order to generate two children in a specific manner. The above function is a single point cross over, which tries to combine the given chromosomes from the middle point with probability of prob (which is specified in the parameters dictionary with initial value of 0.4). For more understanding read the next numerical example:<br/>
suppose the number of queens is 4, <br/>
parent1: [1, 2, 3, 4]<br/>
parent2: [4, 3, 2, 1]<br/>
With a probability of probe, the cross over operation will be applied between the parents (shown as below), otherwise the stated parents will be returned without any changes:<br/>
chromosome1: [4, 3, 3, 4]<br/>
chromosome2: [1, 2, 2, 1]<br/>



### multi_points_crossover
```python
def multi_points_crossover(parent1, parent2, parameters={'prob': 0.4, 'points_count': 'middle'}):
```
**parameters (dictionary)**: dictionary of parameters that key = parameter name and value = parameter value<br/>
**parent1 (Chromosome)**: First parent chromosome, Gene, np.array with len [n^2,1]<br/>
**parent2 (Chromosome)**: Second parent chromosome, Gene, np.array with len [n^2,1]<br/>
**returns (Chromosome, Chromosome)**: return two chromosome for each children, Chromosome<br/>

> Author: mohammad Tavakkoli, will be completed

### default_parent_selection
```python
def default_parent_selection(population, n, parameter=None):
```
**parameter (dictionary)**: dictionary of parameters that key = parameter name and value = parameter value<br/>
**population (list)**: list of current population Chromosomes<br/>
**n (integer)**: Number of Parents that should be chosen, the value should be less or equal to the length of population<br/>
**return (list)**: list of selected Parents<br/>

In order to generate new children, a subset of the parents should be chosen to be mutated and cross-overed (which could also be the whole population). In this function n number of the given population will be chosen and returned to be used in the next stages.



### default_population_selection
```python
def default_population_selection(parents, children, n, parameters=None):
```
**parameters (dictionary)**: dictionary of parameters that key = parameter name and value = parameter value<br/>
**parents (list)**: list of Parents of current Generation<br/>
**children (list)**: list of new children of current Generation<br/>
**n (integer)**: Number of remaining population<br/>
**returns (list)**: list of remained Chromosomes with size of n<br/>

After generating new children from the selected parents, the next population has to be selected from the parents and the new children. The default approach to select the new generation is implemented in the above function which chooses n chromosomes randomly from the list of parents concatenated with children. The returned list will always have a size of n which technically is the size of specified population.

### fitness_based_population_selection
```python
def fitness_based_population_selection(parents, children, n, parameters=None):
```
**parameters (dictionary)**: dictionary of parameters that key = parameter name and value = parameter value<br/>
**parents (list)**: list of Parents of current Generation<br/>
**children (list)**: list of new children of current Generation<br/>
**n (integer)**: Number of remaining population<br/>
**returns (list)**: list of remained Chromosomes with size of n<br/>

As discussed in default_population_selection part, population selection is to select n chromosomes among the parents and children to be used as the next generation. In this approach a list of chromosomes with length of n will be returned containing the selected chromosomes. The main idea behind this approach is the [roulette wheel selection](#roulette_wheel_selection) which has been discussed before. Using this approach chromosomes with higher fitness values have higher probabilities to be chosen. This idea is similar to the evolution of the live beings in the nature, where animals with higher abilities have a higher chance of survival.

### default_stop_condition
```python
def default_stop_condition(generation, max_generation, parameters=None):
```
**parameters (dictionary)**: dictionary of parameters that key = parameter name and value = parameter value<br/>
**generation (integer)**: The step of current generation<br/>
**max_generation (integer)**: The maximum number of generations that the algorithm may continue<br/>
**returns (Boolean)**: True if the condition has reached otherwise False<br/>

The evolution process has to be stopped at one generation. The above function breaks the evolution process when the evolution has been done max_generation times.

## Chromosome class (*chromosome.py*)

### List of the attributes of the class
|Attribute name|type|initial value|description|
|:-:|:-:|:-:|:-:|
|fitness|float|None|The fitness of this chromosome|
|genotype|list|None|A list containing n (number of queens) in range [1, n]|

### list of the methods of the class
|function name|parameters|returns|descriptions|
|:-:|:-:|:-:|:-:|
|[*__ init__*](#__-init__)|genotype<br/>fitness|void|The constructor function of the Chromosome class|
|[*get_phenotype*](#get_phenotype)|void|list|returns the phenotype of the chromosome|

### __init__
```python
def __init__(self, genotype, fitness):
```
**genotype (list)**: A list containing of n integers in range [1, n]<br/>
**fitness (float)**: fitness of the specified chromosome<br/>


### get_phenotype
```python
def get_phenotype(self):
```
**returns (list)**: Returns a 2d array with integer values which specify the phenotype of the Chromosome

In order to convert the genotype to phenotype ??????


# 3. Quick Tour

### Installation guide

### Graphical User Interface

### A few words with the contributors


# 4. Contact Us
Feel free to contact us for any further information using the following channels:

### Amirmohhammad rostami:
- email: [*email@gmail.com*](#emailto:email@gmail.com)
- linkdin: [*linkdin_id*](#linkdin_link)

### Milad Bohlouli:
- email: [*miladbohlouli@gmail.com*](#emailto:miladbohlouli@gmail.com)
- linkdin: [*milad_bohlouli*](#https://www.linkedin.com/in/milad-bohlouli-536011163)
- homepage: [*ceit.aut.ac.ir/~bohlouli*](#https://ceit.aut.ac.ir/~bohlouli/index.html)

<!-- ## Cross over methods
- [x] default cross over
- [x] multi point cross over
- [ ] other ideas

## Mutation methods
- [x] default method
- [x] swap mutation
- [ ] other ideas -->



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
