import csv, matplotlib.pyplot as plt, os
import numpy as np
import math
import pandas as pd
from datetime import timedelta
from tqdm import tqdm
import argparse
from sample import sample_utd_by_city

# Load all detector IDs
def load_all_detector_ids(csv_path: str):
    detector_ids = set()

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            det_id = row.get("detid")
            if det_id:
                detector_ids.add(det_id)

    return list(detector_ids)

# Load UTD data for a given detector ID
def load_utd_data(csv_path: str, loaded_data, detector_id: str):
    if loaded_data:
        data = []
        for row in loaded_data:
            if row.get("detid") == detector_id:
                data.append(row)
        return data

    data = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        # i = 0
        for row in reader:
            if row.get("detid") == detector_id:
                data.append(row)
            if len(data) > 0 and row.get("detid") != detector_id:
                # Assuming data is sorted by detid, we can break early
                break

            # if i % 1_000_000 == 0:
            #     print(f"Processed {i:,} rows...")
            # i += 1

    return data

def plot_flow_occ_over_time(data: list[dict], out_path: str):
    """
    Plot 'flow' and 'occ' over time (one above the other)
    from a list of dict records and save to a file.

    Parameters
    ----------
    data : list of dict
        Each element must have keys: 'day', 'interval', 'flow', 'occ'
    out_path : str
        File path to save the generated plot (e.g. 'plot.png')
    """
    if not data:
        raise ValueError("Empty dataset provided")

    # --- Convert to DataFrame ---
    df = pd.DataFrame(data)

    # Ensure numeric values
    df["flow"] = pd.to_numeric(df["flow"], errors="coerce")
    df["occ"] = pd.to_numeric(df["occ"], errors="coerce")
    df["interval"] = pd.to_numeric(df["interval"], errors="coerce")

    # Build datetime column (assuming 'interval' is seconds from midnight)
    df["datetime"] = pd.to_datetime(df["day"]) + pd.to_timedelta(df["interval"], unit="s")
    df = df.sort_values("datetime")

    # --- Create the figure ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"Flow and Occupancy Over Time ({df['city'].iloc[0] if 'city' in df.columns else ''})")

    # Plot flow
    ax1.plot(df["datetime"], df["flow"], label="Flow", color="tab:blue")
    ax1.set_ylabel("Flow")
    ax1.grid(True)
    ax1.legend()

    # Plot occupancy
    ax2.plot(df["datetime"], df["occ"], label="Occupancy", color="tab:orange")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Occupancy")
    ax2.grid(True)
    ax2.legend()

    # Improve layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out_path}")

def compute_correlation(df):
    """
    Compute correlation between flow and occupancy
    at data points where occupancy > 66th percentile,
    and also for their moving averages.
    """
    # Compute correlation between flow and occupancy at the data points where occupancy > 66th_percentile
    valid_mask = df["occ"] > df["occ"].quantile(0.66)
    if valid_mask.sum() > 0.2 * valid_mask.shape[0]:
        correlation = df.loc[valid_mask, ["flow", "occ"]].corr().iloc[0, 1]
        correlation = f"{correlation:.3f}"
    else:
        correlation = "N/A"

    # Compute correlation between flow and occupancy for the moving averages at the data points where occupancy MA > 66th_percentile
    valid_ma_mask = df["occ_ma"] > df["occ"].quantile(0.66)
    if valid_ma_mask.sum() > 0.2 * valid_ma_mask.shape[0]:
        correlation_ma = df.loc[valid_ma_mask, ["flow_ma", "occ_ma"]].corr().iloc[0, 1]
        correlation_ma = f"{correlation_ma:.3f}"
    else:
        correlation_ma = "N/A"

    return correlation, correlation_ma

def plot_flow_occ_over_time_with_ma(data: list[dict], out_path: str, N: int, plot: bool = True):
    """
    Plot 'flow', 'occ', and their ratio over time (stacked subplots)
    from a list of dict records and save to a file.

    In each subplot, plot both the original values and
    an N-point moving average.

    Parameters
    ----------
    data : list of dict
        Each element must have keys: 'day', 'interval', 'flow', 'occ'
    out_path : str
        File path to save the generated plot (e.g. 'plot.png')
    N : int
        Window size for the moving average (in number of points).
    """
    if not data:
        raise ValueError("Empty dataset provided")

    if N <= 0:
        raise ValueError("N (moving average window) must be a positive integer")

    # --- Convert to DataFrame ---
    df = pd.DataFrame(data)

    # Ensure numeric values
    df["flow"] = pd.to_numeric(df["flow"], errors="coerce")
    df["occ"] = pd.to_numeric(df["occ"], errors="coerce")
    df["interval"] = pd.to_numeric(df["interval"], errors="coerce")

    # Build datetime column (assuming 'interval' is seconds from midnight)
    df["datetime"] = pd.to_datetime(df["day"]) + pd.to_timedelta(df["interval"], unit="s")
    df = df.sort_values("datetime")

    # --- Compute moving averages for flow and occ ---
    df["flow_ma"] = df["flow"].rolling(window=N, min_periods=1).mean()
    df["occ_ma"] = df["occ"].rolling(window=N, min_periods=1).mean()

    # --- Compute flow/occupancy ratio and its moving average ---
    # Avoid division by zero: where occ <= 0, set ratio to NaN
    df["ratio"] = np.where(df["occ"] > 0, df["flow"] / df["occ"], np.nan)
    df["ratio_ma"] = df["ratio"].rolling(window=N, min_periods=1).mean()

    correlation, correlation_ma = compute_correlation(df)

    if plot:
        # --- Create the figure with 3 subplots ---
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        city_str = df["city"].iloc[0] if "city" in df.columns and not df["city"].isna().all() else ""
        title_suffix = f" ({city_str})" if city_str else ""

        fig.suptitle(f"Flow, Occupancy and Flow/Occ Ratio Over Time{title_suffix} - Correlation {correlation}; MA Correlation {correlation_ma}")

        # Plot flow (original + moving average)
        ax1.plot(df["datetime"], df["flow"], label="Flow (original)")
        ax1.plot(df["datetime"], df["flow_ma"], label=f"Flow (MA, N={N})", linestyle="--")
        ax1.set_ylabel("Flow")
        ax1.grid(True)
        ax1.legend()

        # Plot occupancy (original + moving average)
        ax2.plot(df["datetime"], df["occ"], label="Occupancy (original)")
        ax2.plot(df["datetime"], df["occ_ma"], label=f"Occupancy (MA, N={N})", linestyle="--")
        # plot the 66th percentile and mean occupancy as a horizontal line
        occ_66th = df["occ"].quantile(0.66)
        ax2.axhline(occ_66th, color="gray", linestyle=":", label=f"66th Percentile Occ = {occ_66th:.2f}")
        avg_occ = df["occ"].mean()
        ax2.axhline(avg_occ, color="blue", linestyle=":", label=f"Average Occ = {avg_occ:.2f}")
        ax2.set_ylabel("Occupancy")
        ax2.grid(True)
        ax2.legend()

        # Plot ratio (original + moving average)
        ax3.plot(df["datetime"], df["ratio"], label="Flow/Occ ratio (original)")
        ax3.plot(df["datetime"], df["ratio_ma"], label=f"Flow/Occ ratio (MA, N={N})", linestyle="--")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Flow / Occ")
        ax3.grid(True)
        ax3.legend()

        # Improve layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

    return correlation, correlation_ma

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate correlations between flow and occupancy for UTD data.")
    parser.add_argument("--utd_path", type=str, default="../data/traffic_data/utd19_u.csv", help="Path to the UTD CSV file.")
    parser.add_argument("--sampled_path", type=str, default="utd_samples/sampled_utd19.csv", help="Path to the sampled UTD CSV file.")
    parser.add_argument("--cities", type=str, default="graz,munich", help="Comma-separated list of cities to filter by. If not provided, all cities are included.")
    parser.add_argument("--output_dir", type=str, default="sampled_correlations", help="Path to save the correlations CSV file.")
    parser.add_argument("--should_plot", type=bool, default=False, help="Whether to generate and save plots for each detector ID.")

    utd_path = parser.parse_args().utd_path
    sampled_path = parser.parse_args().sampled_path
    cities_str = parser.parse_args().cities
    if cities_str:
        cities = [city.strip() for city in cities_str.split(",")]
    else:
        cities = None
    output_dir = parser.parse_args().output_dir
    should_plot = parser.parse_args().should_plot

    return utd_path, sampled_path, cities, output_dir, should_plot


if __name__ == "__main__":
    utd_path, sampled_path, cities, output_dir, should_plot = parse_args()
    print("UTD path:", utd_path)
    print("Sampled path:", sampled_path)
    print("Cities:", cities if cities else "All cities")
    print("Output directory:", output_dir)
    print("Should plot:", should_plot)

    # if sampling with these parameters has not been done, perform it now
    if not os.path.exists(sampled_path):
        sample_utd_by_city(cities=cities, utd19_path=utd_path, sampled_path=sampled_path)

    detector_ids = load_all_detector_ids(sampled_path)
    detector_ids.sort()

    csv_data = []
    with open(sampled_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_data.append(row)

    correlations = {}

    os.makedirs(output_dir, exist_ok=True)

    if should_plot:
        os.makedirs(f"{output_dir}/correlation_plots", exist_ok=True)

    for det_id in tqdm(detector_ids):
        det_data = load_utd_data(sampled_path, csv_data, det_id)

        # print("=====================================")
        # print(f"Loaded {len(det_data)} rows for detector ID {det_id}")
        # print("Sample data:", det_data[:2])
        correlation, correlation_ma = plot_flow_occ_over_time_with_ma(det_data, f"{output_dir}/correlation_plots/{det_id}.png", N=3, plot=should_plot)
        # print("Plot saved for detector ID", det_id)
        # print(f"Correlation (occ > 66th percentile): {correlation}")
        # print(f"MA Correlation (occ MA > 66th percentile): {correlation_ma}")
        
        correlations[det_id] = {
            "correlation": correlation,
            "correlation_ma": correlation_ma
        }

    # Save correlations to a CSV file
    with open(f"{output_dir}/correlations.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["detector_id", "correlation", "correlation_ma"])
        for det_id, corr_data in correlations.items():
            writer.writerow([det_id, corr_data["correlation"], corr_data["correlation_ma"]])

    with open(f"{output_dir}/cities_used.txt", "w") as f:
        if cities is None:
            f.write("All cities were used.\n")
        else:
            f.write("Cities used for sampling:\n")
            for city in cities:
                f.write(f"- {city}\n")