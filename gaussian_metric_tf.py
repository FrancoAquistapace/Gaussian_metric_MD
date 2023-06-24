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

# ----- Here we'll define the functions related to our metric, with TF --------

# Import modules
import tensorflow as tf
from tools_tf import *

# ----- General utility functions -----------------

# Define function to get cubic domain that contains
# two configurations
def get_cubic_domain(C1, C2, step_size, margin_size):
    '''
    Params:
        C1, C2 : tf.Tensor
            Arrays containing atomic configurations
            C1 and C2, respectively.
        step_size : int or float
            Size in A of the side of a cubic volume, which
            is used to discretise the cubic domain.
        margin_size : int or float
            Size in A of the margins added in each direction
            to the cubic integration domain.
        
    Output:
        Returns a cubic space that contains both C1 and C2, 
        discretised as cubes of side step_size and represented
        by their center vector. This space is extended in each 
        dimension an amount 2 * margin_size. 
    '''
    # Build joint configuration, to help get cubic domain
    C_joint = tf.concat([C1,C2], 0)
    # Determine max and min values of the cubic domain
    x_max, y_max, z_max = tf.reduce_max(C_joint, axis=0).numpy() + margin_size
    x_min, y_min, z_min = tf.reduce_min(C_joint, axis=0).numpy() - margin_size
    
    # Build domain as a set of points
    n1 = int((x_max - x_min)/step_size) + 1
    n2 = int((y_max - y_min)/step_size) + 1
    n3 = int((z_max - z_min)/step_size) + 1
    dom = gen_sc_lattice(n1, n2, n3, step_size)
    # Displace domain to its correct origin
    dom = displace(dom, tf.constant([x_min, y_min, z_min]))

    return dom