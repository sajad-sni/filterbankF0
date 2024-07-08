import numpy as np

# Example data
X = np.random.rand(37, 421)  # Example matrix of shape (37, 421)
f = np.random.randint(0, 421, size=37)  # Example vector of indices

# Use NumPy indexing to extract elements from X
extracted_elements = X[np.arange(len(f)), f]

# Print the shape of the extracted elements (should be (37,))
print("Shape of extracted elements:", extracted_elements.shape)