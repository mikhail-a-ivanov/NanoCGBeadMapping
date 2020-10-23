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

    return beadsPerArea

def buildCGSlab(beadType1='TiA', beadType2='TiB', latticePoints=32, width=3, originalSlabGeometry='anatase-101/last_frame.xmol'):

    # Read the configuration of the original slab
    CGslabFile, box = readCGSlab(originalSlabGeometry)

    # Read the coordinates
    coordinates1 = beadCoordinates(CGslabFile, beadType1)
    coordinates2 = beadCoordinates(CGslabFile, beadType2)

    # Calculate separation between two bead types
    separation = getBeadTypeSeparation(coordinates1, coordinates2)

    # Calculate beads per area
    beadsPerArea1 = getBeadsPerArea(coordinates1, box, axis=2)
    beadsPerArea2 = getBeadsPerArea(coordinates2, box, axis=2)

    # Calculate the slab length
    slabLength1 = np.sqrt(latticePoints ** 2 / beadsPerArea1) # nm
    slabLength2 = np.sqrt(latticePoints ** 2 / beadsPerArea2) # nm    

    # Initialize a grid
    xLower, yLower, zLower = np.meshgrid(range(latticePoints), range(latticePoints), range(2))
    gridLower = np.array([xLower, yLower, zLower]).T.reshape(2 * latticePoints**2, 3)

    planeLower = np.array([gridLower.T[0], gridLower.T[1], gridLower.T[2] * separation]).T



    xUpper, yUpper, zUpper = np.meshgrid(range(latticePoints), range(latticePoints), range(2))
    gridUpper = np.array([xUpper, yUpper, zUpper]).T.reshape(2 * latticePoints**2, 3)

    planeUpper = np.array([gridUpper.T[0], gridUpper.T[1], (gridUpper.T[2] * separation) + width]).T
    
    

    return






CGslabFile, box = readCGSlab('anatase-101/last_frame.xmol')
print(box)

coordinates1 = beadCoordinates(CGslabFile, 'TiA')
coordinates2 = beadCoordinates(CGslabFile, 'TiB')

width_TiA = getBeadSeparation(coordinates1, axis=2)
print(width_TiA)

width_TiB = getBeadSeparation(coordinates2, axis=2)
print(width_TiB)

separation = getBeadTypeSeparation(coordinates1, coordinates2)
print(separation)

beadsPerArea1 = getBeadsPerArea(coordinates1, box, axis=2)
print(beadsPerArea1)

beadsPerArea2 = getBeadsPerArea(coordinates2, box, axis=2)
print(beadsPerArea2)
