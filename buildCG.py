import numpy as np

A_to_nm = 0.1

def readCGSlab(conf_file, header_rows = 2):
    """
    Opens and reads input configuration file with the 
    CG slab geometry, returns list of lines
    with bead coordinates and the box vectors (nm)
    """
    lines = []
    try:
        with open(conf_file, 'r') as file:
            for row in file:
                lines.append(row.split())
    except IOError:
        print(f'Could not read the file {conf_file}!')

    # Save the total number of beads
    try:
        NBeads = int(lines[0][0])
    except:
        print('Could not read the number of beads!')

    # Save the box vectors
    try:
        box = [float(lines[1][3]) * A_to_nm, float(lines[1][4]) * A_to_nm, float(lines[1][5]) * A_to_nm]
    except:
        print('Could not read the box vectors!') 

    # Remove header
    assert len(lines) != 0, 'Could not read bead coordinates!'

    del(lines[0:header_rows])

    assert len(lines) == NBeads, 'Number of lines in the file does not match the total number of beads!'

    return lines, box

def getBeadTypes(lines):
    """
    Reads CG bead names, removes numbers from the names and 
    save unique CG bead types in the beadTypes list
    """
    beadNames = []
    for line in lines:
        beadNames.append(''.join([i for i in line[0] if not i.isdigit()]))

    beadTypes = list(set(beadNames))
    
    assert len(beadTypes) != 0, 'Could not read bead types!'

    return beadTypes


def beadCoordinates(lines, beadType):
    """
    Returns the coordinates (nm) of all beads
    of the input type in a numpy array
    """
    assert type(beadType) == str, 'Could not read bead type!'

    beadCoordinates = []

    for line in lines:
        if beadType in line[0]:
             beadCoordinates.append([float(line[1]), float(line[2]), float(line[3])])

    beadCoordinates = np.array(beadCoordinates) * A_to_nm

    assert len(beadCoordinates) != 0, 'Could not read bead coordinates!'

    assert len(beadCoordinates.T) == 3, 'Could not read X, Y or Z bead coordinates!'

    return beadCoordinates


def getBeadSeparation(beadCoordinates, axis=2):
    """
    Returns average separation (nm) along one axis
    between two groups of the same bead type
    in the opposite planes of the slab

    For the surfacemost beads gives the slab width
    """

    assert type(beadCoordinates) == np.ndarray, 'Could not read bead coordinates!'

    # Calculate the mean coordinate along one axis
    meanCoordinate = np.mean(beadCoordinates, axis=0)[axis]

    # Divides bead coordinates into two groups:
    # 1. coordinate < mean coordinate
    # 2. coordinate > mean coordinate
    minCoordinates = beadCoordinates[beadCoordinates.T[axis] < meanCoordinate]
    maxCoordinates = beadCoordinates[beadCoordinates.T[axis] > meanCoordinate]

    assert len(minCoordinates) + len(maxCoordinates) == len(beadCoordinates), 'Number of beads with min and max coordinate does not add up!'

    BeadSeparation = np.mean(maxCoordinates.T[axis]) - np.mean(minCoordinates.T[axis])

    assert BeadSeparation > 0, 'Could not calculate the average separation!'

    return BeadSeparation


def getBeadTypeSeparation(beadCoordinates1, beadCoordinates2, axis=2):
    """
    Returns average separation between two
    bead types along one direction using the
    following algorithm:

    Slab schematics:
    BeadType1 --- BeadType2 --- --- BeadType2 --- BeadType1

    Separation between two bead types:
    abs(getBeadSeparation(beadCoordinates1) - getBeadSeparation(beadCoordinates2)) / 2
    """

    assert type(beadCoordinates1) == np.ndarray, 'Could not read bead 1 coordinates!'
    assert type(beadCoordinates2) == np.ndarray, 'Could not read bead 2 coordinates!'

    separation = (getBeadSeparation(beadCoordinates1, axis=2) - getBeadSeparation(beadCoordinates2, axis=2)) / 2
    print(f'Bead interplanar separation = {round(separation, 3)} nm')

    return separation

def getBeadsPerArea(beadCoordinates, box, axis=2):
    """Calculates the average number of beads per area per plane.
    Input axis is normal to the slab 
    """
    dimensions = [0, 1, 2]
    dimensions.remove(axis)

    assert len(dimensions) == 2, 'Could not select the plane!'

    # Beads per area per plane
    beadsPerArea = len(beadCoordinates) / (box[dimensions[0]] * box[dimensions[1]]) / 2
    print(f'Bead surface density = {round(beadsPerArea, 3)} beads/nm^2')

    return beadsPerArea

def fibonacciLattice(N, r, r0 = [0, 0, 0], canonical=True):
    """Evenly distributes N points on a sphere of radius r with the center
    located at r0. Use canonical or offset Fibonacci lattice 
    (offset increases the maximum distance between the points,
    good when N > 1000)"""
    goldenRatio = (1 + 5**0.5)/2
    i = np.arange(0, N)

    if canonical:
        theta = 2 * np.pi * i / goldenRatio
        phi = np.arccos(1 - 2 * (i + 0.5) / N)

    else:
        if N >= 600000:
            epsilon = 214
        elif N >= 400000:
            epsilon = 75
        elif N >= 11000:
            epsilon = 27
        elif N >= 890:
            epsilon = 10
        elif N >= 177:
            epsilon = 3.33
        elif N >= 24:
            epsilon = 1.33
        else:
            epsilon = 0.33
        
        theta = 2 * np.pi * i / goldenRatio
        phi = np.arccos(1 - 2 * (i + epsilon)/(N - 1 + 2*epsilon))
    
    x = r * (np.cos(theta) * np.sin(phi)) + r0[0]
    y = r * (np.sin(theta) * np.sin(phi)) + r0[1]
    z = r * np.cos(phi) + r0[2]

    return x,y,z

def getCGSphereCoordinates(beadType1='TiA', beadType2='TiB', outerRadius=2, 
            originalSlabGeometry='last_frame.xmol', pbc_box=[10, 10, 10], canonicalFibonacci=True):
    """Get the coordinates of CG spherical nanoparticle using the Fibonacci lattice algorithm"""
    # Read the configuration of the original slab
    CGslabFile, box = readCGSlab(originalSlabGeometry)

    # Read the coordinates
    coordinates1 = beadCoordinates(CGslabFile, beadType1)
    coordinates2 = beadCoordinates(CGslabFile, beadType2)

    # Calculate separation between two bead types
    separation = getBeadTypeSeparation(coordinates1, coordinates2)

    # Calculate beads per area
    beadsPerArea = getBeadsPerArea(coordinates1, box, axis=2)

    # Calculate the number of points to be distributed on each subshell
    innerRadius = outerRadius - separation
    outerArea = 4 * np.pi * outerRadius**2
    innerArea = 4 * np.pi * innerRadius**2

    outerN = int(round(outerArea * beadsPerArea))
    innerN = int(round(innerArea * beadsPerArea))

    print(f'Building a spherical CG nanoparticle:')
    print(f'Outer radius = {outerRadius:.2f} nm; Outer area = {outerArea:.2f} nm^2; Outer N = {outerN}')
    print(f'Inner radius = {innerRadius:.2f} nm; Inner area = {innerArea:.2f} nm^2; Inner N = {innerN}')

    # Get the coordinates of the CG beads on each subshell:
    r0 = [pbc_box[0]/2, pbc_box[1]/2, pbc_box[2]/2]
    outerX, outerY, outerZ = fibonacciLattice(outerN, outerRadius, r0, canonical=canonicalFibonacci)
    innerX, innerY, innerZ = fibonacciLattice(innerN, innerRadius, r0, canonical=canonicalFibonacci)

    outerShell = np.array([outerX.T, outerY.T, outerZ.T]).T
    innerShell = np.array([innerX.T, innerY.T, innerZ.T]).T

    NP = np.concatenate((outerShell, innerShell))

    # Generate list of atomnames
    atomnames = []
    for bead_index in range(len(NP)):
        if bead_index < len(outerShell):
            atomnames.append(beadType1)
        else:
            atomnames.append(beadType2)

    return NP, pbc_box, atomnames
    
         
def getCGSlabCoordinates(beadType1='TiA', beadType2='TiB', latticePoints=32, width=3, originalSlabGeometry='last_frame.xmol'):
    """Get the CG bead coordinates and the corresponding
    PBC box vector"""

    # Read the configuration of the original slab
    CGslabFile, box = readCGSlab(originalSlabGeometry)

    # Read the coordinates
    coordinates1 = beadCoordinates(CGslabFile, beadType1)
    coordinates2 = beadCoordinates(CGslabFile, beadType2)

    # Calculate separation between two bead types
    separation = getBeadTypeSeparation(coordinates1, coordinates2)

    # Calculate beads per area
    beadsPerArea = getBeadsPerArea(coordinates1, box, axis=2)

    # Calculate the slab length and bead spacing
    slabLength = np.sqrt(latticePoints ** 2 / beadsPerArea) # nm

    beadSpacing = slabLength / latticePoints # nm     

    # Initialize a grid
    xx, yy, zz = np.meshgrid(range(latticePoints), range(latticePoints), range(2))
    grid = np.array([xx, yy, zz]).T.reshape(2 * latticePoints**2, 3)

    # Initialize slab planes coordinates and join them in one array
    planeLower = np.array([grid.T[0] * beadSpacing, grid.T[1] * beadSpacing, grid.T[2] * separation]).T
    planeUpper = np.array([grid.T[0] * beadSpacing, grid.T[1] * beadSpacing, (grid.T[2] * separation) + width]).T

    # Generate list of atomnames
    tol = 1e-5
    atomnames = []
    for Z in planeLower.T[2]:
        if Z == 0:
            atomnames.append(beadType1)
        else:
            atomnames.append(beadType2)

    for Z in planeUpper.T[2]:
        if Z == width:
            atomnames.append(beadType2)
        else:
            atomnames.append(beadType1)

    atom_indices = []
    beadType1_counter = 0
    beadType2_counter = 0
    for atom in atomnames:
        if atom == beadType1:
            beadType1_counter += 1
            atom_indices.append(beadType1_counter)
        else:
            beadType2_counter += 1
            atom_indices.append(beadType2_counter)



    slab = np.concatenate((planeLower, planeUpper))
    

    # Write the PBC box vector
    pbc_box = [np.amax(slab.T[0]) + beadSpacing, 
               np.amax(slab.T[1]) + beadSpacing, 
               np.amax(slab.T[2]) + 1] # add 1 nm of vacuum to Z and 1 * beadSpacing to X and Y
                                       # so that the beads do not overlap

    assert slab.shape == (4 * latticePoints**2, 3), 'Number of beads does not match the input values!'

    return slab, pbc_box, atomnames, atom_indices


def writeCGSlabGRO(CGbeads, resname, atomnames, pbc_box):
    """Writes the GRO file"""
    CGbeads = np.around(CGbeads, decimals=3)

    assert len(atomnames) == len(CGbeads), 'List of atomnames and the coordinates are incompatible!'
    
    filename = resname + "-" + str(len(CGbeads)) + ".gro"
    file_header = "resname: " + resname + "; " + str(len(CGbeads))+ " " + "CG beads" + "\n" + str(len(CGbeads))
    
    gro_file = open(filename, 'w+')
    gro_file.write(file_header + '\n')
    
    for CGbead_index in range(len(CGbeads)):
        gro_file.write(format(resname).rjust(9))
        #gro_file.write(format(atomnames[CGbead_index] + str(atom_indices[CGbead_index])).rjust(6))
        gro_file.write(format(atomnames[CGbead_index]).rjust(6))
        gro_file.write(format(str(CGbead_index+1)).rjust(5))
        for axis_index in range(len(CGbeads[CGbead_index])):
            gro_file.write(format(CGbeads[CGbead_index][axis_index], '#.3f').rjust(8))
        gro_file.write("\n")
    
    for axis_index in range(len(pbc_box)):
        gro_file.write(format(float(pbc_box[axis_index]), '#.5f').rjust(10))
    gro_file.write("\n")
    gro_file.close()
    print(filename, "saved.")
    return

def writeCGSlabPDB(CGbeads, resname, atomnames, pbc_box):
    """Writes the PDB file"""
    CGbeads = np.around((CGbeads * 10), decimals=3)

    assert len(atomnames) == len(CGbeads), 'List of atomnames and the coordinates are incompatible!'

    filename = resname + "-" + str(len(CGbeads)) + ".pdb"
    file_header = "REMARK    resname: " + resname + "; " + str(len(CGbeads)) + " " + "CG beads" + "\n"
    pbc_record = "CRYST1".rjust(6)
    for axis_index in range(len(pbc_box)):
        pbc_record += format(float(pbc_box[axis_index]) * 10, '#.3f').rjust(9)
    pbc_record += "  90.00  90.00  90.00 P 1           1"

    
    with open(filename, 'w') as file:
        file.write(file_header)
        file.write(pbc_record + '\n')
        file.write("MODEL   0" + '\n')
        for CGbead_index in range(len(CGbeads)):
            file.write("ATOM  ".rjust(6))
            file.write(format(str(CGbead_index+1)).rjust(5))
            file.write(format(atomnames[CGbead_index]).rjust(5))
            file.write(format(resname).rjust(4))
            file.write(format("A").rjust(2))
            file.write(format("1").rjust(4))
            file.write(format("").rjust(5))
            for axis_index in range(len(CGbeads[CGbead_index])):
                file.write(format(CGbeads[CGbead_index][axis_index], '#.3f').rjust(7))
                file.write(format("").rjust(1))
            file.write(format('1.00 0.00').rjust(12))
            file.write(format(atomnames[CGbead_index]).rjust(12))
            file.write("\n")
        file.write("TER")
        file.write(format(str(len(CGbeads) + 1).rjust(9)))
        file.write(format(resname).rjust(10))
        file.write(format("1\n").rjust(4))
        file.write('ENDMDL\n')
        file.write('END\n')
        print(filename, "saved.")


# Execute the script:
#slab, pbc_box, atomnames, atom_indices = getCGSlabCoordinates(beadType1='TiA', beadType2='TiB', latticePoints=32, width=3, originalSlabGeometry='last_frame.xmol')

#writeCGSlabGRO(slab, 'a101', atomnames, pbc_box)
#writeCGSlabPDB(slab, 'a', atomnames, pbc_box)

NP, pbc_box, atomnames = getCGSphereCoordinates(beadType1='TiA', beadType2='TiB', outerRadius=2, 
originalSlabGeometry='last_frame.xmol', pbc_box=[14.444, 14.444, 14.444], canonicalFibonacci=True)
writeCGSlabPDB(NP, 'a', atomnames, pbc_box)