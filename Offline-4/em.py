import numpy as np
from scipy.stats import poisson

# Load dataset
data = np.loadtxt("data/em_data.txt")

# Initialize parameters
np.random.seed(0)
lambda_1, lambda_2 = np.random.uniform(1, 5, 2)  # Initial guesses for means
pi_1, pi_2 = 0.5, 0.5  # Initial guess for proportions

# Convergence parameters
tolerance = 1e-6
max_iterations = 100
log_likelihoods = []

def em_algorithm(data, lambda_1, lambda_2, pi_1, pi_2, tolerance, max_iterations):
    n = len(data)

    for iteration in range(max_iterations):
        # E-Step: Calculate responsibilities (posterior probabilities)
        # Using Poisson PMF for each type (family planning vs no family planning)
        prob_family_planning = pi_1 * poisson.pmf(data, lambda_1)
        prob_no_family_planning = pi_2 * poisson.pmf(data, lambda_2)
        
        # Normalize to get probabilities
        responsibilities_1 = prob_family_planning / (prob_family_planning + prob_no_family_planning)
        responsibilities_2 = 1 - responsibilities_1

        # M-Step: Update parameters using weighted averages
        lambda_1_new = np.sum(responsibilities_1 * data) / np.sum(responsibilities_1)
        lambda_2_new = np.sum(responsibilities_2 * data) / np.sum(responsibilities_2)
        pi_1_new = np.mean(responsibilities_1)
        pi_2_new = 1 - pi_1_new

        # Check for convergence
        if abs(lambda_1_new - lambda_1) < tolerance and abs(lambda_2_new - lambda_2) < tolerance:
            break

        # Update parameters
        lambda_1, lambda_2 = lambda_1_new, lambda_2_new
        pi_1, pi_2 = pi_1_new, pi_2_new

        # (Optional) Log-likelihood for monitoring
        log_likelihood = np.sum(np.log(pi_1 * poisson.pmf(data, lambda_1) + pi_2 * poisson.pmf(data, lambda_2)))
        log_likelihoods.append(log_likelihood)
        print(f"Iteration {iteration + 1}, Log Likelihood: {log_likelihood}")

    return lambda_1, lambda_2, pi_1, pi_2

# Run the EM algorithm
lambda_1, lambda_2, pi_1, pi_2 = em_algorithm(data, lambda_1, lambda_2, pi_1, pi_2, tolerance, max_iterations)

# Output the estimated parameters
print("Estimated Parameters:")
print(f"Mean number of children (with family planning): {lambda_1}")
print(f"Mean number of children (without family planning): {lambda_2}")
print(f"Proportion of families with family planning: {pi_1}")
print(f"Proportion of families without family planning: {pi_2}")
