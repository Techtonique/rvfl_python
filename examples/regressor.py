from rvfl import RVFLRegressor
import numpy as np
from time import time 

X= np.random.normal(0, 1, (100, 10))
y=np.random.normal(0, 1, (100,))

model = RVFLRegressor(n_nodes=200, alpha=1e-2)
start = time()
model.fit(X, y)
print("Training time:", time()-start)

print(model.predict(X))