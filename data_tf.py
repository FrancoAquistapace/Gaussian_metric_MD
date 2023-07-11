# Copyright 2023 Franco Aquistapace
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

# ----- Here we'll define the data related functions of the project ------
# We'll use a functional programming approach for now, we can develop classes 
# later if we need them.

# Import modules
import tensorflow as tf
import ovito
from tools_tf import *


# ------ I/O functions --------

# Define function to read files given a path
def read_file(path):
    '''
    Params:
        path : str
            Path of an ovito readable file containing
            at least a set of atomic positions.
    Output:
        Returns an ovito Pipeline object built from 
        the file located at path. This is a wrapper 
        around ovito's io.import_file() function.
    '''
    return ovito.io.import_file(path)


# ------ Data functions --------

# Define a function that returns the neighbour list
# for a given path
def neighbors_from_file(path, N, deltas=True):
    '''
    Params:
        path : str
            Path of an ovito readable file containing
            at least a set of atomic positions.
        N : int
            Number of neighbors to find for each atom.
        deltas : bool (optional)
            When True, returns an array of shape 
            (M, N, 3) with the delta vectors of each 
            atom's neighbors, where M is the amount of 
            atoms in the system. If False, returns an 
            array of shape (M, N) containing the indices
            of the neighbors of each atom.
    Output:
        Returns an array containing information about the
        N nearest neighbors of each atom in the given path.
    '''
    pipeline = read_file(path)
    computed = pipeline.compute()
    finder = ovito.data.NearestNeighborFinder(N, computed)
    neighbors = finder.find_all()
    if deltas:
        return neighbors[1]
    return neighbors[0]