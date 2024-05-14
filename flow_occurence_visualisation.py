import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
from sklearn.linear_model import LinearRegression
import matplotlib.ticker as ticker
import seaborn as sns



# Load the SciencePlots style
plt.style.use(['science'])

# Load experimental_data_old from a CSV file
df = pd.read_csv('experimental_data/flow-occurrence.csv')
df.columns = ['Cone Angle (deg)', 'Angular Velocity (rad/s) of Flow Occurrence']

# Prepare the experimental_data_old for the plots
grouped_data = df.groupby('Cone Angle (deg)')['Angular Velocity (rad/s) of Flow Occurrence'].apply(list)

# Convert grouped experimental_data_old into a list of lists for plotting
data_to_plot = [group for _, group in grouped_data.items()]

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Violin plot
parts = axs[0].violinplot(data_to_plot, showmeans=False, showmedians=True)
axs[0].set_xlabel('Cone Angle, $\Delta$ (°)')
axs[0].set_ylabel('Angular Velocity (rad/s) of Flow Occurrence')
axs[0].set_xticks([y + 1 for y in range(len(grouped_data))])
axs[0].set_xticklabels([f'{angle}°' for angle in grouped_data.keys()])

# Add individual points to the violin plot with uniform jitter
for i, (angle, data) in enumerate(grouped_data.items(), start=1):
    y = data
    x = np.random.uniform(i - 0.3, i + 0.3, size=len(data))
    axs[0].scatter(x, y, alpha=0.5, color='red', s=5)

# Box plot
axs[1].boxplot(data_to_plot)
axs[1].set_xlabel('Cone Angle, $\Delta$ (°)')
axs[1].set_ylabel('Angular Velocity (rad/s) at completed Flow Transportation')
axs[1].set_xticks([y + 1 for y in range(len(grouped_data))])
axs[1].set_xticklabels([f'{angle}°' for angle in grouped_data.keys()])

# # Adding scatter points on the box plot, different colour per group, slightly transparent
# colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(grouped_data)))  # Generates a color map
# for i, (angle, experimental_data_old) in enumerate(grouped_data.items(), start=1):
#     y = experimental_data_old
#     x = np.random.normal(i, 0.075, size=len(experimental_data_old))  # Adjust jittering here if needed
#     axs[1].scatter(x, y, alpha=0.5, color='red', edgecolors='grey', s=7)  # Adjust alpha for transparency

# Calculate and plot the medians to connect with a straight line
medians = [np.median(data) for data in data_to_plot]
axs[1].plot(range(1, len(medians) + 1), medians, marker='x', linestyle='--', linewidth=1, color='black')

# Perform linear regression and display equation
x = np.array(range(1, len(medians) + 1)).reshape(-1, 1)
y = np.array(medians)
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
axs[1].text(0.05, 0.95, f'$\omega_f$ = {model.coef_[0]:.3f}$\Delta$ + {model.intercept_:.3f}\nR² = {r_sq:.3f}',
            transform=axs[1].transAxes, va='top', fontsize=12,
            bbox=dict(boxstyle='round', edgecolor='black', facecolor='white', alpha=0.9090))

# Adding horizontal grid lines for better readability
for ax in axs:
    ax.yaxis.grid(True)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=5))  # Adjust base for finer grid



# Save the figure as a high-resolution PNG file
plt.savefig('boxplot_1.png', dpi=1000)
plt.close()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
from sklearn.linear_model import LinearRegression
import matplotlib.ticker as ticker

# Load the SciencePlots style
plt.style.use(['science'])

# Load experimental_data_old from a CSV file
df = pd.read_csv('experimental_data/flow-occurrence.csv')
df.columns = ['Cone Angle (deg)', 'Angular Velocity (rad/s) of Flow Occurrence']

# Prepare the experimental_data_old for the plots
grouped_data = df.groupby('Cone Angle (deg)')['Angular Velocity (rad/s) of Flow Occurrence'].apply(list)

# Convert grouped experimental_data_old into a list of lists for plotting
data_to_plot = [group for _, group in grouped_data.items()]

fig, ax = plt.subplots(figsize=(6, 6))

# Box plot
ax.boxplot(data_to_plot)
ax.set_xlabel('Cone Angle, $\Delta$ (°)')
ax.set_ylabel('Angular Velocity (rad/s) at completed Flow Transportation. $\omega_f$')
ax.set_xticks([y + 1 for y in range(len(grouped_data))])
ax.set_xticklabels([f'{angle}°' for angle in grouped_data.keys()])

def simple_beeswarm(y, nbins=None):
    """
    Returns x coordinates for the points in ``y``, so that plotting ``x`` and
    ``y`` results in a bee swarm plot.
    """
    y = np.asarray(y)
    if nbins is None:
        nbins = len(y) // 6

    # Get upper bounds of bins
    x = np.zeros(len(y))
    ylo = np.min(y)
    yhi = np.max(y)
    dy = (yhi - ylo) / nbins
    ybins = np.linspace(ylo + dy, yhi - dy, nbins - 1)

    # Divide indices into bins
    i = np.arange(len(y))
    ibs = [0] * nbins
    ybs = [0] * nbins
    nmax = 0
    for j, ybin in enumerate(ybins):
        f = y <= ybin
        ibs[j], ybs[j] = i[f], y[f]
        nmax = max(nmax, len(ibs[j]))
        f = ~f
        i, y = i[f], y[f]
    ibs[-1], ybs[-1] = i, y
    nmax = max(nmax, len(ibs[-1]))

    # Assign x indices
    dx = 0.3 / (nmax // 2)
    for i, y in zip(ibs, ybs):
        if len(i) > 1:
            j = len(i) % 2
            i = i[np.argsort(y)]
            a = i[j::2]
            b = i[j+1::2]
            x[a] = (0.5 + j / 3 + np.arange(len(b))) * dx
            x[b] = (0.5 + j / 3 + np.arange(len(b))) * -dx

    return x

# Adding scatter points on the box plot, different colour per group, slightly transparent
colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(grouped_data)))  # Generates a color map
for i, (angle, data) in enumerate(grouped_data.items(), start=1):
    y = data
    x = simple_beeswarm(y) + i  # Adjust x coordinates for each group
    ax.scatter(x, y, alpha=0.75, color=colors[i-1], edgecolors='black', s=9)  # Adjust alpha for transparency


# Calculate and plot the medians to connect with a straight line
medians = [np.median(data) for data in data_to_plot]
ax.plot(range(1, len(medians) + 1), medians, marker='x', linestyle='--', linewidth=1, color='black')

# Perform linear regression and display equation
x = np.array(range(1, len(medians) + 1)).reshape(-1, 1)
y = np.array(medians)
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
ax.text(0.05, 0.95, f'$\omega_f$ = {model.coef_[0]:.3f}$\Delta$ + {model.intercept_:.3f}\nR² = {r_sq:.3f}',
            transform=ax.transAxes, va='top', fontsize=12,
            bbox=dict(boxstyle='round', edgecolor='black', facecolor='white', alpha=0.9090))

# Adding horizontal grid lines for better readability
ax.yaxis.grid(True)
ax.yaxis.set_major_locator(ticker.MultipleLocator(base=5))  # Adjust base for finer grid

# Save the figure as a high-resolution PNG file
plt.savefig('boxplot.png', dpi=1000)

# Display the plot
# plt.show()