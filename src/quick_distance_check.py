"""
Quick distance check without heavy dependencies
"""
import csv
import math

# Read CSV manually
points = []
with open('../data/raw/ground_truth/Training_Points_CSV.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        points.append({
            'id': int(row['id']),
            'label': int(row['label']),
            'x': float(row['x']),
            'y': float(row['y'])
        })

print(f"Total points: {len(points)}")

# Calculate all pairwise distances (sample)
distances = []
min_dist = float('inf')
closest_pair = None

for i in range(len(points)):
    for j in range(i+1, len(points)):
        dx = points[i]['x'] - points[j]['x']
        dy = points[i]['y'] - points[j]['y']
        dist = math.sqrt(dx*dx + dy*dy)
        distances.append(dist)

        if dist < min_dist:
            min_dist = dist
            closest_pair = (i, j, dist)

# Statistics
distances.sort()
n = len(distances)

print(f"\nDistance statistics:")
print(f"Min: {distances[0]:.2f}m")
print(f"Max: {distances[-1]:.2f}m")
print(f"Median: {distances[n//2]:.2f}m")
print(f"Mean: {sum(distances)/len(distances):.2f}m")

# Count close pairs
within_30 = sum(1 for d in distances if d < 30)
within_50 = sum(1 for d in distances if d < 50)
within_100 = sum(1 for d in distances if d < 100)

print(f"\nPairs within 30m: {within_30}")
print(f"Pairs within 50m: {within_50}")
print(f"Pairs within 100m: {within_100}")

print(f"\nClosest pair: Point {closest_pair[0]+1} and Point {closest_pair[1]+1}, distance: {closest_pair[2]:.2f}m")
