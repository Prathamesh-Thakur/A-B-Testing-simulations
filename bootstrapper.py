# Library imports
import numpy as np

# Random number generator
rng = np.random.default_rng()

# Sample size
sample_size = 5000

# Generate random samples with outliers
group_A = rng.pareto(a = 2, size = sample_size)
group_B = rng.pareto(a = 2, size = sample_size)

# Number of simulations for bootstrapping
tot_simulations = 10000

# Difference list
median_differences = []

# Bootstrapping
for _ in range(tot_simulations):
    # Generate random samples with replacement
    random_sample_A = rng.choice(group_A, size = sample_size, replace = True)
    random_sample_B = rng.choice(group_B, size = sample_size, replace = True)

    # Calculate medians
    median_A = np.median(random_sample_A)
    median_B = np.median(random_sample_B)

    # Append difference
    median_differences.append(median_B - median_A)

# Compute 2.5th and 97.5th percentile
lower_percentile = np.percentile(median_differences, 2.5)
higher_percentile = np.percentile(median_differences, 97.5)

# Check if zero is between the interval 
if lower_percentile < 0 and higher_percentile > 0:
    print("We cannot say for sure!")

elif higher_percentile < 0:
    print("Certified loser!")

else:
    print("Certified winner!")
