import numpy as np
import matplotlib.pyplot as plt

def CGBeadMappingMain(atoms, CGbeads, spacings):
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
    #print(closestAtoms)
    #print(np.take(AllDistances, closestAtoms, axis=None))

    # We can get the atom index and CG bead index from the global (flat) index of the distance
    # By modulo operation we get the CG bead index, by floor division we get the atom index,
    # that are separated by the calculated distance
    atomIndices = closestAtoms // len(CGbeads)
    CGbeadIndices = closestAtoms % len(CGbeads)

    # Calculating a half of grid cell diagonal from spacings to compare it with
    # the distances between CG bead and assigned atoms
    # We take half of the diagonal because the CG bead is located in the center
    gridCelldiag = np.sqrt(spacings[0]**2 + spacings[1]**2 + spacings[2]**2) / 2

    print("Assigning atoms to CG beads...")
   
    for i in range(len(atoms)):
        AllCGBead_AtomIndexes[CGbeadIndices[i]].append(atomIndices[i])

    print("Checking the total number of atoms")
    atomCheckSum = 0
    for CGbead_index in range(len(AllCGBead_AtomIndexes)):
        atomCheckSum += len(AllCGBead_AtomIndexes[CGbead_index])

    print(atomCheckSum)


    print("Fixing atom distribution...")
    warningCounter = 0
    atomsThreshold = 4

    for CGbead_index in range(len(AllCGBead_AtomIndexes)):
        NatomsInBead = len(AllCGBead_AtomIndexes[CGbead_index])
        
        if NatomsInBead < atomsThreshold:
            print("Fixing atom distribution for CG bead", CGbead_index, ":", AllCGBead_AtomIndexes[CGbead_index])
            print("Number of atoms in the CG bead", CGbead_index, "is", NatomsInBead)
            CGbeadAtomDistances = np.argsort(AllDistances.T[CGbead_index])
            #print(AllCGBead_AtomIndexes[CGbead_index])
            #print(CGbeadAtomDistances[0:6])
            #print(CGbeadAtomDistances[NatomsInBead])
"""
        while NatomsInBead < atomsThreshold:
            for CGbead in AllCGBead_AtomIndexes:
                if CGbeadAtomDistances[NatomsInBead] in CGbead:
                    print("Picking CG bead", CGbead)
                    CGbead.remove(CGbeadAtomDistances[NatomsInBead])
                    print("Removing atom", CGbeadAtomDistances[NatomsInBead], "from CG bead", CGbead)

            AllCGBead_AtomIndexes[CGbead_index].append(CGbeadAtomDistances[NatomsInBead])
            print("Adding atom", CGbeadAtomDistances[NatomsInBead], "to CG bead", CGbead_index, ":", AllCGBead_AtomIndexes[CGbead_index])
            NatomsInBead += 1
            print("Number of atoms in the CG bead", CGbead_index, "now is", NatomsInBead, "\n")
"""

        while NatomsInBead < atomsThreshold:
            for CGbead in AllCGBead_AtomIndexes:
                    
                try:
                    CGbead.remove(CGbeadAtomDistances[NatomsInBead])
                    print("Picking CG bead", CGbead)
                    print("Removing atom", CGbeadAtomDistances[NatomsInBead], "from CG bead", CGbead)
                except ValueError:
                    print("Cannot find atom", CGbeadAtomDistances[NatomsInBead],)

            AllCGBead_AtomIndexes[CGbead_index].append(CGbeadAtomDistances[NatomsInBead])
            print("Adding atom", CGbeadAtomDistances[NatomsInBead], "to CG bead", CGbead_index, ":", AllCGBead_AtomIndexes[CGbead_index])
            NatomsInBead += 1
            print("Number of atoms in the CG bead", CGbead_index, "now is", NatomsInBead, "\n")
        
        
        
        
      


    print("Checking the distances between CG beads and assigned atoms...")
    for i in range(len(atoms)):
        if AllDistances[atomIndices[i]][CGbeadIndices[i]] > gridCelldiag:
            warningCounter += 1
            print("Distance between atom", atomIndices[i], "and CG bead", CGbeadIndices[i], 
            "is larger than half of the grid cell diagonal.")

    if warningCounter == 0:
        print("All distances between CG beads and assigned atoms are within half of the grid cell diagonal.")
    else:
        print(warningCounter, "distances are larger than half of the grid cell diagonal.")


    print("Checking the total number of atoms")
    atomCheckSum = 0
    for CGbead_index in range(len(AllCGBead_AtomIndexes)):
        atomCheckSum += len(AllCGBead_AtomIndexes[CGbead_index])

    print(atomCheckSum)

    #assert atomCheckSum == len(atoms), "Total number of atoms does not match the original number!"
    

    return AllCGBead_AtomIndexes

# Find CG beads that belong to the surface and first layers and bulk

def FindSlabLayers(CGbeads, AllCGBead_AtomIndexes):
    # Generate an array which contains unique Z-coordinate values
    values_z = np.unique(CGbeads.T[2])

    # Surface CG beads have either minimum or maximum values of Z-coordinate 
    surface_z = [values_z[0], values_z[len(values_z)-1]]
    # First subsurface CG beads have next minimum or next maximum values of Z-coordinate
    first_layer_z = [values_z[1], values_z[len(values_z)-2]]

    # Set coordinate tolerance value for comparing floats
    tol = 1e-5

    # Construct an array of surface CG beads indexes
    CGbeads_surface_indexes = []
    for CGbead_index in range(len(CGbeads)):
        if surface_z[0] - tol <= CGbeads[CGbead_index][2] <= surface_z[0] + tol:
            CGbeads_surface_indexes.append(CGbead_index)
        elif surface_z[1] - tol <= CGbeads[CGbead_index][2] <= surface_z[1] + tol:
            CGbeads_surface_indexes.append(CGbead_index)

    CGbeads_surface_indexes = np.array(CGbeads_surface_indexes)

    # Construct an array of first subsurface CG beads indexes
    CGbeads_first_layer_indexes = []
    for CGbead_index in range(len(CGbeads)):
        if first_layer_z[0] - tol <= CGbeads[CGbead_index][2] <= first_layer_z[0] + tol:
            CGbeads_first_layer_indexes.append(CGbead_index)
        elif first_layer_z[1] - tol <= CGbeads[CGbead_index][2] <= first_layer_z[1] + tol:
            CGbeads_first_layer_indexes.append(CGbead_index)
            
    CGbeads_first_layer_indexes = np.array(CGbeads_first_layer_indexes)
            
    # Construct arrays of surface and first subsurface from the main array
    CGbeads_Surface = np.take(AllCGBead_AtomIndexes, CGbeads_surface_indexes)
    CGbeads_FirstLayer = np.take(AllCGBead_AtomIndexes, CGbeads_first_layer_indexes)
    
    return CGbeads_Surface, CGbeads_FirstLayer


def CGbeadHistogram(AllCGbead_AtomIndexes, label, Nbins):
    atoms_in_CGbead = []
    for CGbead in AllCGbead_AtomIndexes:
        atoms_in_CGbead.append(len(CGbead))
    atoms_in_CGbead = np.array(atoms_in_CGbead)
    fig = plt.figure(figsize=(12,7))
    plt.title(label)
    plt.ylabel('Count', size=20)
    plt.xlabel('Number of atoms per CG bead', size=20)
    plt.grid(b=True, which='major' , linestyle='--', color= 'grey', linewidth=1, zorder=1)
    plt.grid(b=True, which='minor', linestyle='--', color= 'grey', linewidth=1, zorder=1)
    plt.hist(atoms_in_CGbead, bins = Nbins)
    plt.show()
    
    return 