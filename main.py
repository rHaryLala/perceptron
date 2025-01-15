import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Entrées
entree_entrainement = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])

# Sorties
sortie_entrainement = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)

# Vecteur de poids aléatoire
poids_synaptique = 2 * np.random.random((3, 1)) - 1
print(f"Poids synaptiques aléatoires : \n {poids_synaptique}")


for i in range(1):
    couche_entree = entree_entrainement
    sorties = sigmoid(np.dot(couche_entree, poids_synaptique))

print(f"Sortie après entrainement : \n {sorties}")