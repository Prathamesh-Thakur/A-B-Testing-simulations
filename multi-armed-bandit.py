# Library imports
import numpy as np

# Simulated true conversion rates ['A', 'B', 'C']
true_rates = [0.04, 0.09, 0.01]

# Initial alpha and beta values for all three groups ['A', 'B', 'C']
alpha_vals = [1, 1, 1]
beta_vals = [1, 1, 1]

# Total visitor count for each group
total_counts = [0, 0, 0]

# Number of simulations (or visitors)
visitors = 10000

# Random number generator instance
rng = np.random.default_rng()

for _ in range(visitors):
    # Values to be compared (Monte Carlo Simulation)
    A_value = rng.beta(a = alpha_vals[0], b = beta_vals[0], size = 1)
    B_value = rng.beta(a = alpha_vals[1], b = beta_vals[1], size = 1)
    C_value = rng.beta(a = alpha_vals[2], b = beta_vals[2], size = 1)

    # Find the winning variation
    winner = np.argmax(np.array([A_value, B_value, C_value]))

    # Update total count for the winner
    total_counts[winner] += 1

    # Simulating chance of conversion
    conversion = rng.binomial(n = 1, p = true_rates[winner])
    
    # Check if convered or not, and update alpha and beta values accordingly
    if conversion == 1:
        alpha_vals[winner] += 1
    else:
        beta_vals[winner] += 1

print("Total visitors for A:", total_counts[0])
print("Total visitors for B:", total_counts[1])
print("Total visitors for C:", total_counts[2])
