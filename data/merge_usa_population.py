"""
Extract population data for Los Angeles from USA CSV files.
Filters rows within Los Angeles bounding box and merges into one file.
Outputs in Lat, Lon, Population format.
"""

import os
import csv
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POP_DATA_DIR = os.path.join(SCRIPT_DIR, "population_data")

LA_SOUTH = 33
LA_NORTH = 35
LA_WEST = -119
LA_EAST = -117

print(f"Los Angeles bounding box:")
print(f"  Lat: {LA_SOUTH} to {LA_NORTH}")
print(f"  Lon: {LA_WEST} to {LA_EAST}")

usa_files = sorted([f for f in os.listdir(POP_DATA_DIR) 
                    if 'usa' in f.lower() and f.endswith('.csv') and 'merged' not in f.lower() and 'los_angeles' not in f.lower()])

print(f"\nFound {len(usa_files)} USA population files:")
for f in usa_files:
    print(f"  - {f}")

data = defaultdict(float)
location_counts = defaultdict(int)
total_rows_processed = 0
total_la_rows = 0

for filename in usa_files:
    filepath = os.path.join(POP_DATA_DIR, filename)
    file_rows = 0
    la_rows = 0
    
    print(f"\nProcessing {filename}...")
    
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        fields = reader.fieldnames
        fields_lower = [f.lower() for f in fields]
        
        # Detect column names
        lon_field = lat_field = pop_field = None
        for i, f in enumerate(fields_lower):
            if f in ('lon', 'long', 'longitude', 'x'):
                lon_field = fields[i]
            elif f in ('lat', 'latitude', 'y'):
                lat_field = fields[i]
            elif f in ('population', 'pop', 'value', 'count'):
                pop_field = fields[i]
        
        if not all([lon_field, lat_field, pop_field]):
            print(f"  Warning: Could not detect columns, using positional")
            lon_field, lat_field, pop_field = fields[:3]
        
        print(f"  Columns: lon={lon_field}, lat={lat_field}, pop={pop_field}")
        
        for row in reader:
            file_rows += 1
            try:
                lat_str = row[lat_field].strip()
                lon_str = row[lon_field].strip()
                lat = float(lat_str)
                lon = float(lon_str)
                pop = float(row[pop_field])
                
                # Check if inside LA bounding box
                if LA_SOUTH <= lat <= LA_NORTH and LA_WEST <= lon <= LA_EAST:
                    key = (lat_str, lon_str)
                    data[key] += pop
                    location_counts[key] += 1
                    la_rows += 1
                    
            except Exception:
                continue
    
    total_rows_processed += file_rows
    total_la_rows += la_rows
    print(f"  Processed {file_rows} rows, found {la_rows} in LA bbox")

print(f"\n{'='*50}")
print(f"Total rows processed: {total_rows_processed}")
print(f"Total rows in LA bbox: {total_la_rows}")
print(f"Unique LA locations: {len(data)}")

duplicates = {k: v for k, v in location_counts.items() if v > 1}
print(f"Duplicate locations (summed): {len(duplicates)}")

output_path = os.path.join(POP_DATA_DIR, "population_usa_los_angeles.csv")
with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Lat', 'Lon', 'Population'])
    
    for (lat, lon), pop in data.items():
        writer.writerow([lat, lon, pop])

print(f"\nLA population file saved to: {output_path}")
print(f"Total rows written: {len(data)}")
