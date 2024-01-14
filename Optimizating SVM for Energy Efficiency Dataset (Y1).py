"""
Created on Fri Dec 29 16:51:12 2023

@author: Haddany Abdelkhalek & Rachid Amin
"""

import warnings
import numpy as np
import pandas as pd
import random as rd
import joblib
import os
#import time
from sklearn import preprocessing
from time import monotonic

'''
from sklearn.model_selection import KFold
from sklearn import svm
'''
import SVM_HParam_Opt_Functions as svm_hp_opt

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    svr_model = joblib.load('model.joblib')

#==========================================================  Data Preprocessing ======================================================


# Loading the data, shuffling and preprocessing it
data = pd.read_excel("ENB2012_data.xlsx")
data = data.sample(frac=1)

test = data.info()

# original data
x_org_data = pd.DataFrame(data,columns=["X1","X2","X3","X4",
                                        "X5","X6","X7","X8"])
y = pd.DataFrame(data,columns=["Y1"]).values

x_with_dummies = pd.get_dummies(x_org_data,columns=["X6","X8"])
var_prep = preprocessing.MinMaxScaler()

x = var_prep.fit_transform(x_with_dummies)

data_count = len(x)
print()
print("number of obsrvations:",data_count)
#==========================================================  Data Preprocessing End ===================================================

# hyperparameters (user inputted parameters)
prob_crsvr = 1 # probablity of crossover
prob_mutation = 0.3 # probablity of mutation
population = 20 # population number
generations = 5 # generation number

kfold = 3

# x and y decision variables' encoding
# 12 genes for x and 12 genes for y (arbitrary number)  24 genes in total
x_y_string = np.array([0,1,0,0,0,1,0,0,1,0,0,1,
                       0,1,1,1,0,0,1,0,1,1,1,0]) # initial solution


# create an empty array to put initial population
pool_of_solutions = np.empty((0,len(x_y_string)))


# create an empty array to store a solution from each generation
# for each generation, we want to save the best solution in that generation
# to compare with the convergence of the algorithm
best_of_a_generation = np.empty((0,len(x_y_string)+1))

mutant_data = []

# shuffle the elements in the initial solution (vector)
# shuffle n times, where n is the no. of the desired population
for i in range(population):
    rd.shuffle(x_y_string)
    pool_of_solutions = np.vstack((pool_of_solutions,x_y_string))


# so now, pool_of_solutions, has n (population) chromosomes

gen = 1 # we start at generation no.1 (tracking purposes)


iteration = int(population/2)


with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    for i in range(generations): # do it n (generation) times
        
        # an empty array for saving the new generation
        # at the beginning of each generation, the array should be empty
        # so that you put all the solutions created in a certain generation
        new_population = np.empty((0,len(x_y_string)))
        
        # an empty array for saving the new generation plus its obj func val
        new_population_with_obj_val = np.empty((0,len(x_y_string)+1))
        
        # an empty array for saving the best solution (chromosome)
        # for each generation
        sorted_best = np.empty((0,len(x_y_string)+1))
        
        print()
        print()
        print("--> Generation: #", gen) # tracking purposes
        
        
        family = 1 # we start at family no.1 (tracking purposes)
        
        
        
        for j in range(iteration): # iteration = new population/2 because each gives 2 parents
            
            print()
            print("--> Family: #", family)
            

            start_time = monotonic()
            print(f"Run time 1 {monotonic() - start_time} seconds")

            # selecting 2 parents using tournament selection
            parent_1 = svm_hp_opt.find_parents_ts(pool_of_solutions,
                                                  x=x,y=y)[0]
            parent_2 = svm_hp_opt.find_parents_ts(pool_of_solutions,
                                                  x=x,y=y)[1]
            print(f"Run time 2 {monotonic() - start_time} seconds")
            
            # crossover the 2 parents to get 2 children
            child_1 = svm_hp_opt.crossover(parent_1,parent_2,
                                   prob_crsvr=prob_crsvr)[0]
            child_2 = svm_hp_opt.crossover(parent_1,parent_2,
                                   prob_crsvr=prob_crsvr)[1]
            print(f"Run time 3 {monotonic() - start_time} seconds")
            
            # mutating the 2 children to get 2 mutated children
            mutated_child_1 = svm_hp_opt.mutation(child_1,child_2,
                                          prob_mutation=prob_mutation)[0]
            mutated_child_2 = svm_hp_opt.mutation(child_1,child_2,
                                          prob_mutation=prob_mutation)[1]
            print(f"Run time 4 {monotonic() - start_time} seconds")
            # Predict the objective value for mutated_child_1
            predicted_value_1 = svr_model.predict(pd.DataFrame(mutated_child_1.reshape(1, -1)))[0]
            predicted_value_2 = svr_model.predict(pd.DataFrame(mutated_child_2.reshape(1, -1)))[0]
            print(f"Run time 5 {monotonic() - start_time} seconds")
    
            if predicted_value_2 <= 0.1 or predicted_value_1 <= 0.1:        
            
                # getting the obj val (fitness value) for the 2 mutated children
                obj_val_mutated_child_1 = svm_hp_opt.objective_value(x=x,y=y,
                                                                     chromosome=mutated_child_1,
                                                                     kfold=kfold)[2]
                obj_val_mutated_child_2 = svm_hp_opt.objective_value(x=x,y=y,
                                                                     chromosome=mutated_child_2,
                                                                     kfold=kfold)[2]
                
                
                # for each mutated child, put its obj val next to it
                mutant_1_with_obj_val = np.hstack((obj_val_mutated_child_1,
                                                       mutated_child_1)) # lines 132 and 140
                
                mutant_2_with_obj_val = np.hstack((obj_val_mutated_child_2,
                                                       mutated_child_2)) # lines 134 and 143
                
                
                
                #----------------------------------------------------------------- data colllection ------------------------------------------------------
                #mutant_data.append(mutant_1_with_obj_val)
                #mutant_data.append(mutant_2_with_obj_val)
                #---------------------------------------------------------------------------------------------------------------------------
                
                # we need to create the new population for the next generation
                # so for each family, we get 2 solutions
                # we keep on adding them till we are done with all the families in one generation
                # by the end of each generation, we should have the same number as the initial population
                # so this keeps on growing and growing
                # when it's a new generation, this array empties and we start the stacking process
                new_population = np.vstack((new_population,
                                            mutated_child_1,
                                            mutated_child_2))
                
                
                new_population_with_obj_val = np.vstack((new_population_with_obj_val,
                                                         mutant_1_with_obj_val,
                                                         mutant_2_with_obj_val))
                print(f"Run time 6 {monotonic() - start_time} seconds")
            else:
                print("--> Family: #", family," Skipped") 
            
            # after getting 2 mutated children (solutions), we get another 2, and so on
            # then we go to the next generation and start over
            # since we ended up with 2 solutions, we move on to the next possible solutions
            family = family+1
            
              
        
        # we replace the initial (before) population with the new one (current generation)
        # this new pool of solutions becomes the starting population of the next generation
        pool_of_solutions = new_population
        
        iteration = int(len(pool_of_solutions)/2)
        # for each generation
        # we want to find the best solution in that generation
        # so we sort them based on index [0], which is the obj val
        sorted_best = np.array(sorted(new_population_with_obj_val,
                                                   key=lambda x:x[0]))
        
        
        # since we sorted them from best to worst
        # the best in that generation would be the first solution in the array
        # so index [0] of the "sorted_best" array
        best_of_a_generation = np.vstack((best_of_a_generation,
                                          sorted_best[0]))
        
        if iteration < 2:
            break
        # increase the counter of generations (tracking purposes)
        gen = gen+1       


#---------------------------------------------------------  Store collection data into csv file  ------------------------------------------------------------------
#columns = ["Obj Value"] + [f"Gene_{i+1}" for i in range(len(x_y_string))]
#mutant_df = pd.DataFrame(mutant_data, columns=columns)

#mutant_df.to_csv("GA_data.csv", index=False)
#---------------------------------------------------------------------------------------------------------------------------


# for our very last generation, we have the last population
# for this array of last population (convergence), there is a best solution
# so we sort them from best to worst
sorted_last_population = np.array(sorted(new_population_with_obj_val,
                                         key=lambda x:x[0]))

sorted_best_of_a_generation = np.array(sorted(best_of_a_generation,
                                         key=lambda x:x[0]))

sorted_last_population[:,0] = 1-(sorted_last_population[:,0]) # get accuracy instead of error
sorted_best_of_a_generation[:,0] = 1-(sorted_best_of_a_generation[:,0])

# since we sorted them from best to worst
# the best would be the first solution in the array
# so index [0] of the "sorted_last_population" array
best_string_convergence = sorted_last_population[0]

best_string_overall = sorted_best_of_a_generation[0]


print("------------------------------")
print()
print("Final Solution (Convergence):",best_string_convergence[1:]) # final solution entire chromosome
print("Encoded C (Convergence):",best_string_convergence[1:14]) # final solution x chromosome
print("Encoded Gamma (Convergence):",best_string_convergence[14:]) # final solution y chromosome
print()
print("Final Solution (Best):",best_string_overall[1:]) # final solution entire chromosome
print("Encoded C (Best):",best_string_overall[1:14]) # final solution x chromosome
print("Encoded Gamma (Best):",best_string_overall[14:]) # final solution y chromosome

# to decode the x and y chromosomes to their real values
final_solution_convergence = svm_hp_opt.objective_value(x=x,y=y,
                                                        chromosome=best_string_convergence[1:],
                                                        kfold=kfold)

final_solution_overall = svm_hp_opt.objective_value(x=x,y=y,
                                                    chromosome=best_string_overall[1:],
                                                    kfold=kfold)

# the "svm_hp_opt.objective_value" function returns 3 things -->
# [0] is the x value
# [1] is the y value
# [2] is the obj val for the chromosome (avg. error)
print()
print("Decoded C (Convergence):",round(final_solution_convergence[0],5)) # real value of x
print("Decoded Gamma (Convergence):",round(final_solution_convergence[1],5)) # real value of y
print("Obj Value - Convergence:",round(1-(final_solution_convergence[2]),5)) # obj val of final chromosome
print()
print("Decoded C (Best):",round(final_solution_overall[0],5)) # real value of x
print("Decoded Gamma (Best):",round(final_solution_overall[1],5)) # real value of y
print("Obj Value - Best in Generations:",round(1-(final_solution_overall[2]),5)) # obj val of final chromosome
print()
print("------------------------------")





