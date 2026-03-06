from scipy.stats import beta
import random
import numpy as np

#use baysian updates with beta distribution to estimate the error rate of the model
class ErrorEstimation:
    def __init__(self, error_count, clean_count):
        self.alpha_post = error_count
        self.beta_post = clean_count
        
    def update_observations(self, obs):
        for error, clean in obs:
            self.alpha_post = self.alpha_post + error
            self.beta_post = self.beta_post + clean
    def get_mean(self):
        return self.alpha_post / (self.alpha_post + self.beta_post)
    def get_lcb(self):
        return beta.ppf(0.05, self.alpha_post, self.beta_post)
    def get_ucb(self):
        return beta.ppf(0.95, self.alpha_post, self.beta_post)
    def get_cb(self, confidence):
        return beta.ppf(confidence, self.alpha_post, self.beta_post)


# return 1 mean M 
def route(error_estimation, cf, cd):
    if cf + 2*cd > 2:
        error  = error_estimation.get_ucb()
    else: 
        error = error_estimation.get_lcb()
    ai_expected_cost = error * cf + cd*(1-error)
    human_expected_cost = 1
    return 1 if ai_expected_cost <= human_expected_cost else 0


def run_simulation(cf, cd):
    cluster_obs = [[6,34], [7,33], [8,32], [9,31], [10,30], [11,29], [12,28], [13,27], [14,26], [15,25], [16,24], [17,23], [18,22], [19,21], [20,20]]
    error_estimations = [ErrorEstimation(e, c) for [e,c] in cluster_obs]
    prompt_clusters = [random.randint(0, len(cluster_obs)-1) for _ in range(100)]
    cf = 10
    cd = 0.1

    model_count = 0
    for prompt_cluster in prompt_clusters:
        error_estimation = error_estimations[prompt_cluster]
        decision = route(error_estimation, cf, cd)
        model_count += decision
        
    print(f"Model count: {model_count}")
    print(f"Model percentage: {model_count/len(prompt_clusters)}")


#TODO  do the same idea for uncertinity threashold estimation

def plot_scores(algo_fn):
    import numpy as np
    import matplotlib.pyplot as plt

    # 1. Define the parameter ranges
    errors = np.linspace(0, 1, 100)        # X-axis: Error values from 0 to 10
    cf_values = np.linspace(1, 100, 9)      # Lines: 5 different Cf values for each plot
    lambdas = np.linspace(0.1, 1.0, 5)      # Subplots: 9 different lambda values

    # 2. Setup the figure and axes (3x3 grid)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)
    fig.suptitle('Parameter Sweep: $score\_ai = error \cdot C_f \cdot \lambda$', fontsize=16)

    # 3. Perform the sweep
    for i, cf in enumerate(cf_values):
        ax = axes.flat[i]
        for _, lam in enumerate(lambdas):
            # Calculate the score
            #score_ai = errors * cf * lam
            #score_ai = cf * (1 - np.exp(-lam * errors))
           
            
            # Plot each Cf line
            ax.plot(errors, score_ai, label=f'lama={cf:.0f}')
        
        ax.set_title(f'cf = {cf:.0f}$')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Labeling only the outer edges for cleanliness
        if i >= 6: ax.set_xlabel('Error')
        if i % 3 == 0: ax.set_ylabel('Score AI')
        if i == 0: ax.legend(fontsize='small')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


est = ErrorEstimation(30, 70)
#one suggested by chatgpt 
def algo0(error, cf, cd, lam):
    score_ai = cf * (lam * error)/(1 + lam * error)
    score_human = 1.0
    return score_ai, score_human

#using lambda to scale the percentile - are we coule countung when multiplying by lambda?
def algo6(error, cf, cd, lam):
    adj_cf = cf * (1 - 0.2 + 0.4*lam)
    cb = max(0.5, 1/adj_cf)
    error = est.get_cb(cb)
    score_ai = error * cf 
    score_human = 1.0
    return score_ai, score_human

#using lambda to scale the error
def algo7(error, cf, cd, lam):
    error = est.get_cb(0.7)
    score_ai = error * cf * (1.2- lam*0.4)
    score_human = 1.0
    return score_ai, score_human

#using lambda to scale the error + normal distribution
def algo8(error, cf, cd, lam):
    error = est.get_cb(0.7)
    score_ai = error * cf * (1.2- 0.4 * np.random.normal(lam, 1))
    score_human = 1.0
    return score_ai, score_human

#uniroute style algo with error + cost * lambda
def algo9(error, cf, cd, lam):
    error = est.get_cb(0.7)
    score_ai = error  + (error*cf + (1-error)*cd)*lam
    score_human = 1.0*lam
    return score_ai, score_human

# do not use lambda, will tell if we are overfitting 
def algo10(error, cf, cd, lam):
    error = est.get_cb(0.7)
    score_ai = error*cf  + cd
    score_human = 1.0
    return score_ai, score_human


