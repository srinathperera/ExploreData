import numpy as np

def simulate_angel_portfolio(n_investments=20, n_simulations=100000):
    # Probability weights based on your table
    # 70% Loss, 24% Break Even, 5% Significant Win, 1% Home Run
    probs = [0.70, 0.24, 0.05, 0.01]
    outcome_labels = ['loss', 'breakeven', 'win', 'homerun']
    
    portfolio_outcomes = []
    
    for _ in range(n_simulations):
        # Pick outcomes for 20 investments
        choices = np.random.choice(outcome_labels, size=n_investments, p=probs)
        
        returns = []
        for choice in choices:
            if choice == 'loss':
                returns.append(0)
            elif choice == 'breakeven':
                # Uniformly distributed between 1x and 3x
                returns.append(np.random.uniform(1, 3))
            elif choice == 'win':
                # Uniformly distributed between 10x and 30x
                returns.append(np.random.uniform(10, 30))
            elif choice == 'homerun':
                # Uniformly distributed between 50x and 100x
                returns.append(np.random.uniform(50, 100))
        
        # Portfolio return is the average return across all 20 investments
        portfolio_outcomes.append(np.mean(returns))
        
    return np.array(portfolio_outcomes)

# Run simulation

results = simulate_angel_portfolio(n_investments=8, n_simulations=100000)

# Calculate Percentiles
print(f"P20 (Bottom 20%): {np.percentile(results, 20):.2f}x")
print(f"P50 (Median): {np.percentile(results, 50):.2f}x")
print(f"P75 (Top 25%): {np.percentile(results, 75):.2f}x")
print(f"P90 (Top 10%): {np.percentile(results, 90):.2f}x")
print(f"Average (Mean): {np.mean(results):.2f}x")