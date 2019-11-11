import numpy as np
import matplotlib.pyplot as plt
from slabInit import Initialize

def CGBeadMappingMain(filename, resname, x_points, y_points, z_points):
    
    atoms, CGbeads, spacings = Initialize(filename, resname, x_points, y_points, z_points)
    
    # Creating a list of atom indeces that are assigned to a certain CG bead
    AllCGBead_AtomIndexes = []
    for i in range(len(CGbeads)):
        AllCGBead_AtomIndexes.append([])
    
    # Build a matrix that contains len(CGbeads) repeats of the atomic coordinates
    atoms_matrix = np.repeat(atoms, len(CGbeads), axis=0)
    
    # Reshape the matrix, so that it has len(atoms) elements, each containing len(CGbeads) repeats
    # of the coordinates of atoms[atom_number]
    atoms_matrix = np.reshape(atoms_matrix, (len(atoms), len(CGbeads), 3))
    
    # Build a matrix that contains len(atoms) repeats of all CGbeads coordinates
    CGbeads_matrix = np.repeat([CGbeads], len(atoms), axis=0)
    
    # Calculate all distances
    print("Calculating distances between every atom and all CG beads...")
    AllDistances = np.linalg.norm(atoms_matrix - CGbeads_matrix, axis=2)
    
    # Getting global (or flat) indices of N (N = len(atoms)) shortest distances
    print("Sorting distances...")
    closestAtoms = np.argsort(AllDistances, axis=None)[:len(atoms)]

    # We can get the atom index and CG bead index from the global (flat) index of the distance
    # By modulo operation we get the atom index, by floor division we get the CG bead index,
    # that are separated by the calculated distance
    atomIndices = closestAtoms % len(atoms)
    CGbeadIndices = closestAtoms // len(atoms)

    for i in range(len(atoms)):
        AllCGBead_AtomIndexes[CGbeadIndices[i]].append(atomIndices[i])

    return AllCGBead_AtomIndexes

def CGbeadHistogram(AllCGbead_AtomIndexes, Nbins):
    atoms_in_CGbead = []
    for CGbead in AllCGbead_AtomIndexes:
        atoms_in_CGbead.append(len(CGbead))
    atoms_in_CGbead = np.array(atoms_in_CGbead)
    fig = plt.figure(figsize=(12,7))
    plt.ylabel('Count', size=20)
    plt.xlabel('Number of atoms per CG bead', size=20)
    plt.grid(b=True, which='major' , linestyle='--', color= 'grey', linewidth=1, zorder=1)
    plt.grid(b=True, which='minor', linestyle='--', color= 'grey', linewidth=1, zorder=1)
    plt.hist(atoms_in_CGbead, bins = Nbins)
    plt.show()
    
    return 