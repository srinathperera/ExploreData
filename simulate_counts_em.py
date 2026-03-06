


# p_train: array of shape (10,) - prevalence in training data - data use to train the forecasting model 
# scores: array of shape (5000, 10) - model's soft output for each point - This is a 2D matrix of shape (5000, 10). The Format: Instead of just the class label (e.g., "Class 4"), you need the probability distribution for each point. For a neural network, this is usually the output of the Softmax layer.
#       Example for one data point: [0.01, 0.02, 0.05, 0.02, 0.80, ...] (where 0.80 is the model's confidence that this point is Class 4).
# prev_guess: array of shape (10,) - start with [0.1, 0.1, ...]

for iteration in range(max_iter):
    # E-Step: Re-weight scores by (current_guess / train_prevalence)
    weights = prev_guess / p_train
    updated_scores = scores * weights
    # Normalize each row to sum to 1
    updated_scores /= updated_scores.sum(axis=1, keepdims=True)
    
    # M-Step: New guess is the average of updated scores
    new_guess = updated_scores.mean(axis=0)
    
    # Check for convergence
    if np.linalg.norm(new_guess - prev_guess) < 1e-6:
        break
    prev_guess = new_guess