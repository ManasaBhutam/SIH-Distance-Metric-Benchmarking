import numpy as np

file_path = "data/processed/distance_results.npz"

data = np.load(file_path)

print("Available arrays:", data.files)

for name in data.files:
    arr = data[name]

    print("\nDistance:", name)
    print("Shape:", arr.shape)
    print("Dimensions:", arr.ndim)
    print("Data type:", arr.dtype)