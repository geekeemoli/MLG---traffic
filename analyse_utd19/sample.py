import csv

cities = ['graz', 'munich']

with open("../../utd19_u.csv", "r", newline="") as f, open("sampled_utd19.csv", "w", newline="") as out_f:
    reader = csv.DictReader(f)
    writer = csv.DictWriter(out_f, fieldnames=reader.fieldnames)
    writer.writeheader()
    for i, row in enumerate(reader):
        city = row.get("city")
        if city in cities:
            writer.writerow(row)
        
        if i % 1_000_000 == 0:
            print(f"Processed {i:,} rows...")
