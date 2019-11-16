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

    # Remove unnecessary objects
    del(atoms_matrix)
    del(CGbeads_matrix)

    # Doing an indirect sort on AllDistances matrix along the axis=1
    # Indices of the closest CG beads to every atom appear in the beginning of every row

    print("Sorting distances...")
    AllDistancesSorted = np.argsort(AllDistances, axis=1)

    # The list of closest beads to every atom
    closestCGBeads = np.copy(AllDistancesSorted.T[0])

    # Remove unnecessary objects
    del(AllDistancesSorted)
    
    # Check that the number of atoms and CG beads match
    assert len(closestCGBeads) == len(atoms), "Number of atoms do not match!"
    assert len(np.unique(closestCGBeads)) == len(CGbeads), "Number of CG beads do not match!"

    # Assign atoms to CG beads
    for atom_index in range(len(closestCGBeads)):
        AllCGBead_AtomIndexes[closestCGBeads[atom_index]].append(atom_index)

    # Check the assignements
    checkAssignedAtoms(AllCGBead_AtomIndexes, atoms, spacings, AllDistances, closestCGBeads)
  
    # Fix the atom distribution
    print("Fixing atom distribution...")
    atomsThreshold = len(atoms) // len(CGbeads)
    print("Setting number of atoms per CG bead threshold to", atomsThreshold)
    numIterations = 15
    for i in range(numIterations):
        print("\nFixing iteration number", i + 1, "...")
        fixAtomDistribution(atomsThreshold, AllCGBead_AtomIndexes, AllDistances, CGbeads, spacings)
    

    # Check the assignements again after the fixing
    checkAssignedAtoms(AllCGBead_AtomIndexes, atoms, spacings, AllDistances, closestCGBeads)
    
    return AllCGBead_AtomIndexes

def fixAtomDistribution(atomsThreshold, AllCGBead_AtomIndexes, AllDistances, CGbeads, spacings):
    # Fixing the atom distribution
    
    smallBeads = 0
    Nfixes = 0

    for CGbead_index1 in range(len(AllCGBead_AtomIndexes)):
        NatomsInBead = len(AllCGBead_AtomIndexes[CGbead_index1])

        if NatomsInBead < atomsThreshold:
            CGBeadAtomDistances = np.argsort(AllDistances.T[CGbead_index1])
            smallBeads += 1

            counter = 0
            while NatomsInBead < atomsThreshold:
                #print("Fixing atom distribution for CG bead", CGbead_index1, ":", AllCGBead_AtomIndexes[CGbead_index1])
                #print("Number of atoms in the CG bead", CGbead_index1, "is", NatomsInBead)
                counter += 1

                if counter > 10:
                    #print("Cannot add more atoms to the CG bead " + str(CGbead_index1) + ". Going to the next one.")
                    break

                for CGbead_index2 in range(len(AllCGBead_AtomIndexes)):
                    if (#isNeighbor(CGbeads[CGbead_index1], CGbeads[CGbead_index2], spacings) and#
                        NatomsInBead < len(AllCGBead_AtomIndexes[CGbead_index2]) and 
                        CGBeadAtomDistances[NatomsInBead] in AllCGBead_AtomIndexes[CGbead_index2]):

                        Nfixes += 1

                        #print("Picking CG bead", CGbead_index2, ":", AllCGBead_AtomIndexes[CGbead_index2])
                        #print("Removing atom", CGBeadAtomDistances[NatomsInBead])

                        AllCGBead_AtomIndexes[CGbead_index2].remove(CGBeadAtomDistances[NatomsInBead])
                        AllCGBead_AtomIndexes[CGbead_index1].append(CGBeadAtomDistances[NatomsInBead])
                        
                        NatomsInBead += 1
                    
                    elif (#isNeighbor(CGbeads[CGbead_index1], CGbeads[CGbead_index2], spacings) and#
                        NatomsInBead < len(AllCGBead_AtomIndexes[CGbead_index2]) and
                        CGBeadAtomDistances[NatomsInBead + 1] in AllCGBead_AtomIndexes[CGbead_index2]):

                        Nfixes += 1

                        #print("Picking CG bead", CGbead_index2, ":", AllCGBead_AtomIndexes[CGbead_index2])
                        #print("Removing atom", CGBeadAtomDistances[NatomsInBead + 1])

                        AllCGBead_AtomIndexes[CGbead_index2].remove(CGBeadAtomDistances[NatomsInBead + 1])
                        AllCGBead_AtomIndexes[CGbead_index1].append(CGBeadAtomDistances[NatomsInBead + 1])
                        
                        NatomsInBead += 1
                    
                    elif (#isNeighbor(CGbeads[CGbead_index1], CGbeads[CGbead_index2], spacings) and#
                        NatomsInBead < len(AllCGBead_AtomIndexes[CGbead_index2]) and
                        CGBeadAtomDistances[NatomsInBead + 2] in AllCGBead_AtomIndexes[CGbead_index2]):

                        Nfixes += 1

                        #print("Picking CG bead", CGbead_index2, ":", AllCGBead_AtomIndexes[CGbead_index2])
                        #print("Removing atom", CGBeadAtomDistances[NatomsInBead + 2])

                        AllCGBead_AtomIndexes[CGbead_index2].remove(CGBeadAtomDistances[NatomsInBead + 2])
                        AllCGBead_AtomIndexes[CGbead_index1].append(CGBeadAtomDistances[NatomsInBead + 2])
                        
                        NatomsInBead += 1
                    
                    elif (#isNeighbor(CGbeads[CGbead_index1], CGbeads[CGbead_index2], spacings) and#
                        NatomsInBead < len(AllCGBead_AtomIndexes[CGbead_index2]) and 
                        CGBeadAtomDistances[NatomsInBead + 3] in AllCGBead_AtomIndexes[CGbead_index2]):

                        Nfixes += 1

                        #print("Picking CG bead", CGbead_index2, ":", AllCGBead_AtomIndexes[CGbead_index2])
                        #print("Removing atom", CGBeadAtomDistances[NatomsInBead + 3])

                        AllCGBead_AtomIndexes[CGbead_index2].remove(CGBeadAtomDistances[NatomsInBead + 3])
                        AllCGBead_AtomIndexes[CGbead_index1].append(CGBeadAtomDistances[NatomsInBead + 3])
                        
                        NatomsInBead += 1
                    
                    elif (#isNeighbor(CGbeads[CGbead_index1], CGbeads[CGbead_index2], spacings) and#
                        NatomsInBead < len(AllCGBead_AtomIndexes[CGbead_index2]) and
                        CGBeadAtomDistances[NatomsInBead + 4] in AllCGBead_AtomIndexes[CGbead_index2]):

                        Nfixes += 1

                        #print("Picking CG bead", CGbead_index2, ":", AllCGBead_AtomIndexes[CGbead_index2])
                        #print("Removing atom", CGBeadAtomDistances[NatomsInBead + 4])

                        AllCGBead_AtomIndexes[CGbead_index2].remove(CGBeadAtomDistances[NatomsInBead + 4])
                        AllCGBead_AtomIndexes[CGbead_index1].append(CGBeadAtomDistances[NatomsInBead + 4])
                        
                        NatomsInBead += 1
                    
                    elif (#isNeighbor(CGbeads[CGbead_index1], CGbeads[CGbead_index2], spacings) and#
                        NatomsInBead < len(AllCGBead_AtomIndexes[CGbead_index2]) and
                        CGBeadAtomDistances[NatomsInBead + 5] in AllCGBead_AtomIndexes[CGbead_index2]):

                        Nfixes += 1

                        #print("Picking CG bead", CGbead_index2, ":", AllCGBead_AtomIndexes[CGbead_index2])
                        #print("Removing atom", CGBeadAtomDistances[NatomsInBead + 5])

                        AllCGBead_AtomIndexes[CGbead_index2].remove(CGBeadAtomDistances[NatomsInBead + 5])
                        AllCGBead_AtomIndexes[CGbead_index1].append(CGBeadAtomDistances[NatomsInBead + 5])
                        
                        NatomsInBead += 1
                    
                    elif (#isNeighbor(CGbeads[CGbead_index1], CGbeads[CGbead_index2], spacings) and#
                        NatomsInBead < len(AllCGBead_AtomIndexes[CGbead_index2]) and
                        CGBeadAtomDistances[NatomsInBead + 6] in AllCGBead_AtomIndexes[CGbead_index2]):

                        Nfixes += 1

                        #print("Picking CG bead", CGbead_index2, ":", AllCGBead_AtomIndexes[CGbead_index2])
                        #print("Removing atom", CGBeadAtomDistances[NatomsInBead + 6])

                        AllCGBead_AtomIndexes[CGbead_index2].remove(CGBeadAtomDistances[NatomsInBead + 6])
                        AllCGBead_AtomIndexes[CGbead_index1].append(CGBeadAtomDistances[NatomsInBead + 6])
                        
                        NatomsInBead += 1
                    
                    elif (#isNeighbor(CGbeads[CGbead_index1], CGbeads[CGbead_index2], spacings) and#
                        NatomsInBead < len(AllCGBead_AtomIndexes[CGbead_index2]) and
                        CGBeadAtomDistances[NatomsInBead + 7] in AllCGBead_AtomIndexes[CGbead_index2]):

                        Nfixes += 1

                        #print("Picking CG bead", CGbead_index2, ":", AllCGBead_AtomIndexes[CGbead_index2])
                        #print("Removing atom", CGBeadAtomDistances[NatomsInBead + 6])

                        AllCGBead_AtomIndexes[CGbead_index2].remove(CGBeadAtomDistances[NatomsInBead + 7])
                        AllCGBead_AtomIndexes[CGbead_index1].append(CGBeadAtomDistances[NatomsInBead + 7])
                        
                        NatomsInBead += 1
                    """
                    """
    print("\nNumber of beads with less atoms than the threshold value:", smallBeads)
    print("Number of performed fixes:", Nfixes)

    return AllCGBead_AtomIndexes

def checkAssignedAtoms(AllCGBead_AtomIndexes, atoms, spacings, AllDistances, closestCGBeads):
    # Check the number of atoms in the AllCGBead_AtomIndexes list
    Natoms = 0
    for CGbead in AllCGBead_AtomIndexes:
        Natoms += len(CGbead)

    assert Natoms == len(atoms), "Number of atoms do not match!"

    # Check if all indexes are unique
    AllAtomIndexes = []
    for CGbead in AllCGBead_AtomIndexes:
        if isinstance(CGbead, list):
            for atom in CGbead:
                AllAtomIndexes.append(atom)
        else:
            AllAtomIndexes.append(CGbead)
    
    assert len(AllAtomIndexes) == len(set(AllAtomIndexes)), "There are duplicate atoms!"
  

    # Calculating a grid cell diagonal from spacings to compare it with
    # the distances between CG bead and assigned atoms
    gridCelldiag = np.sqrt(spacings[0]**2 + spacings[1]**2 + spacings[2]**2)

    warningCounter = 0
    for CGbead_index in range(len(AllCGBead_AtomIndexes)):
        for atom_index in AllCGBead_AtomIndexes[CGbead_index]:
            if AllDistances[atom_index][CGbead_index] > gridCelldiag:
                warningCounter += 1
                print("\nDistance between atom", atom_index, "and CG bead", CGbead_index, 
                "is larger than the grid cell diagonal.")

    if warningCounter == 0:
        print("All distances between CG beads and assigned atoms are within the grid cell diagonal.")
    else:
        print("\n", warningCounter, "distances are larger than half of the grid cell diagonal.")

    """
    warningCounter = 0
    for atom_index in range(len(closestCGBeads)):
        if AllDistances[atom_index][closestCGBeads[atom_index]] > gridCelldiag:
            warningCounter += 1
            print("Distance between atom", atom_index, "and CG bead", closestCGBeads[atom_index], 
            "is larger than half of the grid cell diagonal.")

    if warningCounter == 0:
        print("All distances between CG beads and assigned atoms are within half of the grid cell diagonal.")
    else:
        print(warningCounter, "distances are larger than half of the grid cell diagonal.")
    """
    return


# Function that determines whether the two CG beads are neighbors

def isNeighbor(CGbead1, CGbead2, spacings):

    Neighbor = True

    # Set coordinate tolerance value for comparing floats
    tol = 1e-5

    assert len(CGbead1) == len(CGbead2) == len(spacings) == 3, "CG bead and spacings vector sizes do not match."

    distanceVector = np.abs(np.subtract(CGbead2, CGbead1))

    equality = 0

    for i in range(len(distanceVector)):
        if np.abs(distanceVector[i]) < tol:
            equality += 1

    #assert equality < 3, "The two CG beads are the same!"

    if equality == 3:
        Neighbor = False

    for i in range(len(distanceVector)):
        if (distanceVector[i] - spacings[i]) > tol:
            Neighbor = False

    return Neighbor


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