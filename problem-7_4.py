"""
Solution to Problem 6.4 of Pattern Recognition (Stork, Hart and Duda)
---------------------------------------------------------------------
Bernoulii Restricted Boltzmann Machine Model
---------------------------------------------------------------------

"""


from sklearn.neural_network import BernoulliRBM
import numpy as np

input_vector = np.array([[1, 1, 1, 1, 1, 1, 0], [0, 1, 1, 0, 0, 0, 0], [1, 1, 0, 1, 1, 0, 1], [1, 1, 1, 1, 0, 0, 1], [0, 1, 1, 0, 0, 1, 1], [1, 0, 1, 1, 0, 1, 1], [1, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 1, 1]])

output_vector=np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 0, 0, 0, 1, 0],[0, 0, 0, 0, 0, 0, 0, 1, 0, 0],[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

#Create a Boltzmann Network
model = BernoulliRBM(n_components=10)
model.fit(input_vector,output_vector)


