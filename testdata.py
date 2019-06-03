from pathlib import Path
from dataset import CustomDataset

path = Path("/mnt/mldata/data/cars/car_data")
data = CustomDataset(path)
for i in range(5):
    img = data[i]
    print(img[1])

