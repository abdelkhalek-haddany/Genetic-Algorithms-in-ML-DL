
# Optimize Deep Learning and Machine Learning Hyperparameters using Machine Learning

## Overview

This project focuses on optimizing hyperparameters for machine learning models using a genetic algorithm. The genetic algorithm employs techniques such as crossover, mutation, and tournament selection to evolve a population of hyperparameter configurations. The objective is to find the optimal hyperparameters for a Support Vector Machine (SVM) model applied to cancer detection.

## Project Structure

- **SVM_HParam_Opt_Functions.py**: Python module containing functions for objective value calculation, parent selection, crossover, and mutation.
- **Optimizer_Main.py**: The main script that implements the genetic algorithm for hyperparameter optimization.
- **ENB2012_data.xlsx**: Dataset used for cancer detection. Contains features (X1 to X8) and target variable (Y1).
- **model.joblib**: Pre-trained SVM model used to predict objective values.
- **README.md**: Project documentation providing an overview, usage, and results.

## Dependencies

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Joblib

Install the dependencies using:

```bash
pip install numpy pandas scikit-learn joblib
```

## Usage

1. **Optimizer_Main.py**: Execute this script to run the genetic algorithm for hyperparameter optimization.

```bash
python Optimizer_Main.py
```

2. Adjust hyperparameters such as population size, generations, crossover probability, and mutation probability in the script as needed.

## Dataset

The project uses the "ENB2012_data.xlsx" dataset for cancer detection. It includes features X1 to X8 and the target variable Y1.

## Results

The genetic algorithm evolves a population over multiple generations, aiming to find the best hyperparameters for an SVM model. The final solution represents the convergence of the algorithm, while the best solution overall is the best-performing configuration found across all generations.

### Convergence Results:

- **Decoded C (Convergence):** [Real value of x]
- **Decoded Gamma (Convergence):** [Real value of y]
- **Obj Value - Convergence:** [Objective value of final chromosome]

### Best Overall Results:

- **Decoded C (Best):** [Real value of x]
- **Decoded Gamma (Best):** [Real value of y]
- **Obj Value - Best in Generations:** [Objective value of best chromosome]

## Acknowledgments

This project was created by Haddany Abdelkhalek.

