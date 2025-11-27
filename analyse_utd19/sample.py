import csv, os

test_cities = ['graz', 'munich']

def sample_utd_by_city(cities=test_cities, utd19_path="../data/traffic_data/utd19_u.csv", sampled_path="utd_samples/sampled_utd19.csv"):
    """
    Sample UTD data for specified cities and save to 'sampled_utd19.csv'.
    Args:
        cities (list): List of city names to filter data by. If None, all data is included.
        utd19_path (str): Path to the input UTD CSV file.
        sampled_path (str): Path to the output sampled CSV file.
    """
    # create the directory for sampled data if it doesn't exist
    os.makedirs(os.path.dirname(sampled_path), exist_ok=True)

    with open(utd19_path, "r", newline="") as f, open(sampled_path, "w", newline="") as out_f:
        reader = csv.DictReader(f)
        writer = csv.DictWriter(out_f, fieldnames=reader.fieldnames)
        writer.writeheader()
        for i, row in enumerate(reader):
            city = row.get("city")
            if cities is None or city in cities:
                writer.writerow(row)
            
            if i % 1_000_000 == 0:
                print(f"Processed {i:,} rows...")