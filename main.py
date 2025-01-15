import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)



# Entrées
entree_entrainement = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])

# Sorties
sortie_entrainement = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)

# Vecteur de poids aléatoire
poids_synaptique = 2 * np.random.random((3, 1)) - 1
print(f"Poids synaptiques aléatoires : \n {poids_synaptique}")

print()

for i in range(10000):
    # Définition de la couche d'entrée
    couche_entree = entree_entrainement

    # Normalisation du produit des entrées par les poids synaptiques
    sorties = sigmoid(np.dot(couche_entree, poids_synaptique))

    #erreur commise
    erreur = sortie_entrainement - sorties

    # ajustements
    ajustements = erreur * sigmoid_derivative(sorties)

    # Mise à jour des poids
    poids_synaptique += np.dot(couche_entree.T, ajustements)

print(f"Poids synaptiques après entrainement : \n {poids_synaptique}")

print(f"Sorties après entrainement : \n {sorties}")