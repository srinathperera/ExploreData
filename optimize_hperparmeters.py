import optuna

# 1. Define the cost function you want to minimize
def objective(trial):
    # Define the range for your hyperparameter
    # This could be 'suggest_float', 'suggest_int', or 'suggest_categorical'
    hyper_parameter = trial.suggest_float("x", -10, 10)
    
    # Your cost function: cost = f(hyper_parameter)
    # Example: a simple parabola (x - 2)^2 + 5 (min should be at x=2, cost=5)
    cost = (hyper_parameter - 2)**2 + 5
    
    return cost

# 2. Create a 'study' object
# direction="minimize" because we want to find the lowest cost
study = optuna.create_study(direction="minimize")

# 3. Run the optimization
study.optimize(objective, n_trials=100)

# 4. Print results
print(f"Best Hyperparameter: {study.best_params}")
print(f"Minimum Cost: {study.best_value}")