# Library imports
import numpy as np
import scipy.stats as sst

# Alpha value
alpha = 0.05

# Cumulative conversion values
control_conv_tot = 430
variant_conv_tot = 450

# Cumulative total values
control_tot = 10000
variant_tot = 10000

# Prior values for Bayesian testing
control_alpha = 1
control_beta = 1

variant_alpha = 1
variant_beta = 1


def run_z_test():
    # Calculate mean conversion rates for both groups
    control_mean = control_conv_tot / control_tot
    variant_mean = variant_conv_tot / variant_tot

    # Pooled conversion probability
    pooled_prob = (control_conv_tot + variant_conv_tot) / (control_tot + variant_tot)

    # Standard error calculation
    standard_error = np.sqrt(pooled_prob * (1 - pooled_prob) * ((1 / control_tot) + (1 / variant_tot)))

    # Calculate z value
    z = (control_mean - variant_mean) / standard_error

    # Get critical z value
    p_value = 2 * sst.norm.sf(abs(z))

    print(f"Calculated p value: {p_value:.4f}")

    # Reject or accept the null hypothesis
    if p_value < alpha:
        print("Congrats! You have a winner!")

    else:
        print("Nope! Please try again.")


def run_bayesian_test():
    global control_alpha, control_beta, variant_alpha, variant_beta

    # Update alpha and beta values for both groups
    control_alpha += control_conv_tot
    control_beta += (control_tot - control_conv_tot)

    variant_alpha += variant_conv_tot
    variant_beta += (variant_tot - variant_conv_tot)

    # Random number generator
    rng = np.random.default_rng()
    
    # Sample size to compare
    sample_size = 100000

    # Values to be compared (Monte Carlo Simulation)
    control_values = rng.beta(a = control_alpha, b = control_beta, size = sample_size)
    variant_values = rng.beta(a = variant_alpha, b = variant_beta, size = sample_size)

    # Calculate how many times control had a smaller value than variant
    wins = (variant_values > control_values).sum()

    # Probability of variant winnnig
    probability = (wins / sample_size) * 100

    print(f"There is an {probability}% probability that Version B is better than Version A.")

if __name__ == "__main__":
    run_z_test()
    run_bayesian_test()
