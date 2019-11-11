import numpy as np
from slabInit import Initialize

def CGBeadMappingMain(filename, resname, x_points, y_points, z_points):
    
    atoms, CGbeads, spacings = Initialize(filename, resname, x_points, y_points, z_points)
    
    """ 
    The idea: 
    0. Create AllCGBead_AtomIndexes
    1. Pick an atom (atoms[atom_number])
    2. Calculate distances between atom and all CG bead centers
    3. Do np.argsort on the distances, pick the first number from np.argsort(distances) - that will
    be the index of the closest CG bead (CGbeads[CGbead_index])!
    5. Start building CGBead_AtomIndexes: AllCGbead_AtomIndexes[CGbead_index].append(atom_number)
    ???
    """
    
    AllCGBead_AtomIndexes = []
    
    # Build a matrix that contains len(CGbeads) repeats of the atomic coordinates
    atoms_matrix = np.repeat(atoms, len(CGbeads), axis=0)
    
    # Reshape the matrix, so that it has len(atoms) elements, each containing len(CGbeads) repeats
    # of the coordinates of atoms[atom_number]
    atoms_matrix = np.reshape(atoms_matrix, (len(atoms), len(CGbeads), 3))
    
    # Build a matrix that contains len(atoms) repeats of all CGbeads coordinates
    CGbeads_matrix = np.repeat([CGbeads], len(atoms), axis=0)
    
    # Calculate all distances
    AllDistances = np.linalg.norm(atoms_matrix - CGbeads_matrix, axis=2)
    print("Distances between every atom and all CG beads have been calculated.\n")




    
    return(AllDistances)    
