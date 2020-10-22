import numpy as np

def readCGSlab(mmol_file, header_rows = 2):
    # Open the mmol file with the CG slab geometry
    lines = []
    try:
        with open(mmol_file, 'r') as file:
            for row in file:
                lines.append(row.split())
    except IOError:
        print(f'Could not read the file {mmol_file}!')

    # Save the total number of beads
    try:
        NBeads = int(lines[0][0])
    except:
        print('Could not read the number of beads!')

    # Remove header
    assert len(lines) != 0, 'Could not read bead coordinates!'

    del(lines[0:header_rows])

    assert len(lines) != 0, 'Could not read bead coordinates!'

    return lines, NBeads


def getBeadTypes(lines):
    # Read CG bead names, remove numbers and 
    # save unique CG bead types in the beadTypes list
    beadNames = []
    for line in lines:
        beadNames.append(''.join([i for i in line[0] if not i.isdigit()]))

    beadTypes = list(set(beadNames))
    
    assert len(beadTypes) != 0, 'Could not read bead types!'

    return beadTypes


def beadCoordinates(lines, beadType):
    # Return coordinates of all beads
    # of the input type in a numpy array

    beadCoordinates = []

    for line in lines:
        if beadType in line[0]:
             beadCoordinates.append([float(line[1]), float(line[2]), float(line[3])])

    beadCoordinates = np.array(beadCoordinates)

    assert len(beadCoordinates) != 0, 'Could not read bead coordinates!'

    assert len(beadCoordinates.T) == 3, 'Could not read X, Y or Z bead coordinates!'

    return beadType, beadCoordinates


def averageMaxSeparation(beadCoordinates, axis=2):
    
    meanCoordinate = np.mean(beadCoordinates, axis=0)[axis]

    minCoordinates = beadCoordinates[beadCoordinates.T[axis] < meanCoordinate]
    maxCoordinates = beadCoordinates[beadCoordinates.T[axis] > meanCoordinate]

    assert len(minCoordinates) + len(maxCoordinates) == len(beadCoordinates), 'Number of beads with min and max coordinate does not add up!'

    return minCoordinates, maxCoordinates






CGslabFile, NBeads = readCGSlab('anatase-101/a101.CG.mmol')
beadTypes = getBeadTypes(CGslabFile)
print(NBeads)
print(beadTypes)
beadType, beadCoordinates = beadCoordinates(CGslabFile, beadTypes[0])
print(beadType)
print(beadCoordinates)
print(beadCoordinates.shape)

z_min, z_max = averageMaxSeparation(beadCoordinates, axis=2)
#print(z_min)
#print(z_max)