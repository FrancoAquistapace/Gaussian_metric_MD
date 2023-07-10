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