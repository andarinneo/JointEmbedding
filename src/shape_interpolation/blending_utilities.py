import os
import sys
import shutil
import fileinput
import numpy as np

from global_variables import *
from utilities_caffe import *
from scipy.spatial import distance_matrix


def blend_2_inputs(coordinates1, manifold1, coordinates2, manifold2):
    n_elements1 = manifold1.shape[0]
    n_elements2 = manifold2.shape[0]

    assert n_elements1 == n_elements2

    sorted_distances1_np = np.asarray([math.sqrt(sum((coordinates1 - manifold_element) ** 2)) for idx, manifold_element in enumerate(manifold1)])
    sorted_distances2_np = np.asarray([math.sqrt(sum((coordinates2 - manifold_element) ** 2)) for idx, manifold_element in enumerate(manifold2)])

    blended_distances_np = sorted_distances1_np + sorted_distances2_np

    sorted_indexes = blended_distances_np.argsort()

    return sorted_indexes



def blend_3_inputs(coordinates1, manifold1, coordinates2, manifold2, coordinates3, manifold3):
    n_elements1 = manifold1.shape[0]
    n_elements2 = manifold2.shape[0]
    n_elements3 = manifold3.shape[0]

    assert n_elements1 == n_elements2
    assert n_elements2 == n_elements3

    sorted_distances1_np = np.asarray([math.sqrt(sum((coordinates1 - manifold_element) ** 2)) for idx, manifold_element in enumerate(manifold1)])
    sorted_distances2_np = np.asarray([math.sqrt(sum((coordinates2 - manifold_element) ** 2)) for idx, manifold_element in enumerate(manifold2)])
    sorted_distances3_np = np.asarray([math.sqrt(sum((coordinates3 - manifold_element) ** 2)) for idx, manifold_element in enumerate(manifold3)])

    blended_distances_np = sorted_distances1_np + sorted_distances2_np + sorted_distances3_np

    sorted_indexes = blended_distances_np.argsort()

    return sorted_indexes

