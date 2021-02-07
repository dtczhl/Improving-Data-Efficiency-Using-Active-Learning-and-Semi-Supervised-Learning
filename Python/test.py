import numpy as np

proba = np.array([[0.1 , 0.85, 0.05],
                  [0.6 , 0.3 , 0.1 ],
                  [0.39, 0.61, 0.0 ]])

print(proba)

uncertainty = np.argmax(1 - proba.max(axis=1))

print(uncertainty)