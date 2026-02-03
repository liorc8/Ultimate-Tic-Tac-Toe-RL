import numpy as np

data = np.load("self_play_data.npz")
states = data["states"]
pis = data["pis"]
zs = data["zs"]

print("states:", states.shape, states.dtype)
print("pis:", pis.shape, pis.dtype)
print("zs:", zs.shape, zs.dtype)

print("pi sum min/max:", pis.sum(axis=1).min(), pis.sum(axis=1).max())
print("z unique:", sorted(set(zs.tolist())))
