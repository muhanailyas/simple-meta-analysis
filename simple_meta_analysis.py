import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Define the data (mean difference and standard error)
# Example data: Mean differences (effect sizes) and standard errors from 5 studies
data = {
    'study': ['Study 1', 'Study 2', 'Study 3', 'Study 4', 'Study 5'],
    'mean_diff': [0.2, 0.4, 0.1, 0.3, 0.25],  # Mean difference from each study
    'se': [0.1, 0.2, 0.15, 0.1, 0.12]  # Standard errors for each study
}

# Step 2: Create a DataFrame
df = pd.DataFrame(data)

# Step 3: Calculate the weight for each study (1 / variance)
df['variance'] = df['se'] ** 2  # Variance is the square of the standard error
df['weight'] = 1 / df['variance']  # Weight is the inverse of variance

# Step 4: Calculate the weighted mean difference (effect size)
weighted_mean_diff = np.sum(df['weight'] * df['mean_diff']) / np.sum(df['weight'])

# Step 5: Calculate the random effects model using DerSimonian-Laird method
# The Q statistic tests for heterogeneity (how much the studies differ)
Q = np.sum(df['weight'] * (df['mean_diff'] - weighted_mean_diff) ** 2)

# Step 6: Calculate the between-study variance (tau squared)
tau_squared = (Q - (len(df) - 1)) / np.sum(df['weight'])

# Step 7: Pooled effect estimate
pooled_effect = np.sum(df['weight'] * df['mean_diff']) / np.sum(df['weight'])
pooled_se = np.sqrt(1 / np.sum(df['weight']))  # Standard error for the pooled effect estimate

# Print the results
print(f"Pooled Effect Estimate: {pooled_effect:.2f}")
print(f"Pooled Standard Error: {pooled_se:.2f}")

# Step 8: Plot the Forest Plot
fig, ax = plt.subplots(figsize=(8, 5))

# Plot individual study mean differences and their 95% confidence intervals
for i, row in df.iterrows():
    ax.plot([row['mean_diff'] - 1.96 * row['se'], row['mean_diff'] + 1.96 * row['se']], [i, i], color='black')
    ax.plot(row['mean_diff'], i, 'bo')  # Blue dot for the mean difference

# Plot the pooled effect estimate and its 95% CI
ax.plot([pooled_effect - 1.96 * pooled_se, pooled_effect + 1.96 * pooled_se], [len(df), len(df)], color='red', linewidth=2)
ax.plot(pooled_effect, len(df), 'ro')  # Red dot for the pooled effect

ax.set_yticks(np.arange(len(df) + 1))
ax.set_yticklabels(df['study'].tolist() + ['Pooled Effect'])
ax.set_xlabel('Mean Difference')
ax.set_title('Forest Plot of Meta-Analysis')

plt.grid(True)
plt.show()
