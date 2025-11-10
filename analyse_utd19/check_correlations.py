import csv
import matplotlib.pyplot as plt
import os
import numpy as np

correlations = {}
with open("sampled_correlations.csv", "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        det_id = row.get("detector_id")
        if det_id:
            correlations[det_id] = {
                "correlation": float(row['correlation']) if row['correlation'] not in ['nan', 'N/A'] else np.nan,
                "correlation_ma": float(row['correlation_ma']) if row['correlation_ma'] not in ['nan', 'N/A'] else np.nan
            }

print(f"Loaded correlations for {len(correlations)} detector IDs.")

# Plot histogram of correlation and a plot of correlation_ma values and save to correlations_plot.png file and write the number of NaN values in the title
def plot_correlation_histogram_and_ma(correlations: dict, out_path: str):
    correlation_values = [v["correlation"] for v in correlations.values() if not np.isnan(v["correlation"])]
    correlation_ma_values = [v["correlation_ma"] for v in correlations.values() if not np.isnan(v["correlation_ma"])]

    correlation_nan_count = sum(1 for v in correlations.values() if np.isnan(v["correlation"]))
    correlation_ma_nan_count = sum(1 for v in correlations.values() if np.isnan(v["correlation_ma"]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Histogram of correlation values
    ax1.hist(correlation_values, bins=100, color='blue', alpha=0.7)
    ax1.set_title(f"Histogram of Correlation Values (NaN count: {correlation_nan_count})")
    ax1.set_xlabel("Correlation")
    ax1.set_ylabel("Frequency")

    # Plot of correlation_ma values
    ax2.hist(correlation_ma_values, bins=100, color='green', alpha=0.7)
    ax2.set_title(f"Histogram of Moving Average Correlation Values (NaN count: {correlation_ma_nan_count})")
    ax2.set_xlabel("Moving Average Correlation")
    ax2.set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

plot_correlation_histogram_and_ma(correlations, "correlations_plot.png")

negative_corr_count = sum(1 for v in correlations.values() if not np.isnan(v["correlation"]) and v["correlation"] < 0)
negative_corr_ma_count = sum(1 for v in correlations.values() if not np.isnan(v["correlation_ma"]) and v["correlation_ma"] < 0)

print(f"Number of detector IDs with negative correlation: {negative_corr_count}")
print(f"Number of detector IDs with negative moving average correlation: {negative_corr_ma_count}")