# import necessary libraries
from sklearn.datasets import load_boston
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pickle
import time

Xtrain = pd.read_csv('Xtrain.csv',header=0)
ytrain = pd.read_csv('ytrain.csv',header=0)
Xvalid = pd.read_csv('Xvalid.csv',header=0)
yvalid = pd.read_csv('yvalid.csv',header=0)
Xtest = pd.read_csv('Xtest.csv',header=0)
ytest = pd.read_csv('ytest.csv',header=0)
ytrain,yvalid,ytest =  ytrain["GoodCredit"],yvalid["GoodCredit"],ytest["GoodCredit"]

Xtrain.head()

"""# GA"""

#Scaling features to a range
min_max_scaler = preprocessing.MinMaxScaler()
Xtrain = pd.DataFrame(min_max_scaler.fit_transform(Xtrain))
Xvalid = pd.DataFrame(min_max_scaler.fit_transform(Xvalid))
Xtest = pd.DataFrame(min_max_scaler.fit_transform(Xtest))

from sklearn.svm import SVC
svm =  SVC(gamma='auto')

"""# Define fitness and comparision"""

# fitness and comparision
def accuracy(feature):
    """
    Return:accuray on the validation set given the current solution using Logitsic regession classifier
    Args: feature is the set of features. For this particular task, it is our solution set
    """
    if sum(feature) == 0:
        # all zeros means Xtrain_tmp would be empty
        return -1 # a bad value!
    else:
        # evaluate accuray based on the selected features
        features = [column for (column, binary_value) in zip(Xtrain.columns, feature) if binary_value] 
        Xtrain_tmp = Xtrain[features]
        Xvalid_tmp = Xvalid[features]
        # fit and score
        svm.fit(Xtrain_tmp, ytrain)
        # print(f"svm-Cof-size",svm.coef_[0].shape)
        return svm.score(Xvalid_tmp, yvalid)

"""# Genetic Algorithm"""

#GA Step
# We will create an initial population of N arrays, each with a random number of 0s and 1s
def init(n):
    '''
    Return:  n_var-dimension binary array --individual
    Args: n is the initiated variables size
    '''
    return [random.randrange(2) for _ in range(n)]
    #[1,0,1] n=3

def selection(pop, fitness, size):
    """
    Args:   
        pop: is population size
        fitness: the fitness of each individual
        size: individuals will be chosen for crossover"
    Return: 
        random select "size" individuals among "pop" individuals
        eg: random select "2" individuals for cross over among "100" individuals
    """
    candidates = random.sample(list(range(len(pop))), size)
    winner = max(candidates, key=lambda c: fitness[c])
    return pop[winner]

# we have used uniform crossover, other option one point and multiple point crossovers
def uniform_crossover(i1, i2):# swap elements in individual 1 and 2 based on their comparasion with 0.5
    """
    Args:   
        i1: individual one selected in the selection step 
        i2: individual two selected in the selection step 
    Return: 
        c1: offspring 1
    """
    c1 = []
    for i in range(len(i1)): #i1: [1,0,1,1] i2: [1,0,0,0]
        if random.random() < 0.5:#[1,0,1,0]
            c1.append(i1[i])
        else:
            c1.append(i2[i])
    return np.array(c1)

def mutation(x,n_var,pmut):#[1,0,1,1,0,1,1,0....]
    """
    Args:   
        x: offspring to mutate
        n_var: total numbers of variables
        pmut: random number decide when to mutate
    Return: 
        x: mutated offspring 
    """
    mutation_point = random.randint(0, n_var)# features vector which position to mutate
    if random.random() < pmut:# when to mutate
        x[mutation_point] = 0
    else:
        x[mutation_point] = 1
    return x

init_sol = [1]*(Xtrain.shape[1])
init_acc = accuracy(init_sol)
print(f'init_acc_full_set',init_acc)
# GA for minimisation
def GA(f, init, crossover, select, popsize, ngens,n_var,pmut):
    """
    Args:   
        f: accuracy function
        init: function to initiate the initial solution
        select: selection function
        crossover: uniform crossover function
        popsize: population size
        ngens: number of generation 
        n_var: number of the vairables selected
        pmut: probability 
    Return: 
        fitness[bestidx]: the accuracy for the best individual 
        pop[bestidx]: the best individual 
        history: all the accuracy for each generations
    """
    history = []
    # make initial population, evaluate fitness, print stats
    pop = [init() for _ in range(popsize)]
    fitness = [f(x) for x in pop]
    history.append(stats(0, fitness))
    for gen in range(1, ngens):
        # make an empty new population
        newpop = []
        # elitism : directly select the best candidate to next population as it is
        bestidx = max(range(popsize), key=lambda i: fitness[i])
        best = pop[bestidx]
        newpop.append(best)
        while len(newpop) < popsize:
            # select, crossover, mutation
            i1 = select(pop, fitness)
            i2 = select(pop, fitness)
            c1 = crossover(i1, i2)
            c1 = mutation(c1,n_var,pmut)   
            # add the new individuals to the population
            newpop.append(c1)
        pop = newpop
        fitness = [f(x) for x in pop]
        # print(max(fitness))
        history.append(stats(gen, fitness))
    bestidx = np.argmax(fitness)
    return fitness[bestidx], pop[bestidx], history

def stats(gen, fitness):
    '''
    return the generation number and the number of individuals which have been evaluated
    '''
    return gen, (gen+1) * len(fitness), np.min(fitness), np.mean(fitness), np.median(fitness), np.max(fitness), np.std(fitness)

########################################################################
#  FEATURE SELECTION USING GENERIC ALGORITHM ON SPAM CREDIT DATASET  #
########################################################################
n = Xtrain.shape[1]
f = accuracy
popsize = 1000
ngens = 100
n_var = Xtrain.shape[1]-1
tsize = 2
pmut = 0.8
bestf, best, h = GA(f,
                    lambda: init(n),
                    uniform_crossover,
                    lambda pop, fitness: selection(pop, fitness, tsize),
                    popsize,
                    ngens,
                    n_var,pmut
)

# history format : gen, (gen+1) * len(fitness), np.min(fitness), np.mean(fitness), np.median(fitness), np.max(fitness), np.std(fitness)
h = np.array(h) 

print("bestf : ",bestf)
print("best indipendent features selected by genetic algorithm : ",best)

features = [column for (column, binary_value) in zip(Xtest.columns, best) if binary_value] 
print(len(features))
svm.fit(Xtrain[features], ytrain)
result = svm.score(Xtest[features], ytest)
print(result)
