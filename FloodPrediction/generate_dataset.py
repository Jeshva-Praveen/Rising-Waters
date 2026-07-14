import pandas as pd
import random
import os

data = []

for i in range(1000):
    rainfall = random.randint(20, 250)
    cloud = random.randint(5, 80)
    seasonal = random.randint(20, 220)
    temperature = random.randint(20, 40)
    humidity = random.randint(30, 95)

    if rainfall > 150 and humidity > 70 and seasonal > 120:
        flood = 1
    else:
        flood = 0

    data.append([
        rainfall,
        cloud,
        seasonal,
        temperature,
        humidity,
        flood
    ])

df = pd.DataFrame(data, columns=[
    "Rainfall",
    "CloudVisibility",
    "SeasonalRainfall",
    "Temperature",
    "Humidity",
    "Flood"
])

os.makedirs("dataset", exist_ok=True)

df.to_csv("dataset/flood.csv", index=False)

print("Dataset created successfully!")
print(df.head())