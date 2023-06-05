import csv
import random

output_file = "output.csv"

# Generate data for the CSV file
data = []
for i in range(30):
    if i < 20:
        reward = random.uniform(50, 100)
    else:
        reward = 1000
    steps = 250 * i
    data.append((steps, reward))

# Write data to CSV file
with open(output_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Steps", "Reward"])  # Write header
    writer.writerows(data)  # Write data rows

