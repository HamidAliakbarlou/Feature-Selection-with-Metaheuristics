## Feature Selection with Metaheuristics (Genetic Algorithm, SA, VNS)

#### Coding from scratch for feature selection process (Python implementation of the Genetic Algorithm to select the best features for machine learning):


Feature selection is a critical stage in machine learning, enhancing both computational efficiency and predictive accuracy by retaining only the most pertinent predictors. Feature selection is an NP-hard problem, and there is no algorithm that can be used to solve it in polynomial time. Given the number of variables, we can find the number of variables(n) to the power of 2 for our feature combinations.

Alongside widely used supervised feature selection approaches such as filter and wrapper methods, global search methods like Genetic Algorithm emerge as potent techniques for this purpose.

In our project, we introduce modern metaheuristic methods designed to identify viable solutions in feature selection tasks using a Support Vector Machine model. This includes Simulated Annealing, Variable Neighborhood Search, and Genetic Algorithm, offering a diverse array of approaches to optimize feature selection processes.

## Dataset
The dataset we used for this project is the German Credit Data, which categorises loan applicants
as good or bad credit risks based on a set of attributes. There are 1000 observations in this
dataset, with 21 variables (7 numerical and 13 categorical), and no missing values. The target
variable Y is a binary variable with two levels: good credit and bad credit. If a loan applicant is
labelled as 0, it indicates that he or she has "bad credit" based on the 21 variables associated, or
vice versa. Each variable is described in detail in the appendix (Table 1: Description of the
dataset).
