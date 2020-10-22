import numpy as np
from slabCGanalyze import beadCoordinates, getBeadSeparation, getBeadTypeSeparation, getBeadsPerArea

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


def buildCGSlab(beadType1='TiA', beadType2='TiB', slabLenght=32, width=3, originalSlabGeometry='anatase-101/last_frame.xmol'):

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

    




    return


def writeCGSlabGRO(CGbeads, filename, resname, atomname):
    CGbeads = np.around(CGbeads, decimals=3)
    
    with open(filename, 'r') as file:
        lines = file.read().splitlines()
        pbc = lines[-1].split()
    
    
    filename = resname + "_" + str(len(CGbeads)) + "CGbeads.gro"
    file_header = "resname: " + resname + "; " + str(len(CGbeads))+ " " + "CG beads" + "\n" + str(len(CGbeads))
    
    gro_file = open(filename, 'w+')
    gro_file.write(file_header + '\n')
    
    for CGbead_index in range(len(CGbeads)):
        gro_file.write(format(resname).rjust(9))
        gro_file.write(format(atomname + str(CGbead_index+1)).rjust(6))
        gro_file.write(format(str(CGbead_index+1)).rjust(5))
        for axis_index in range(len(CGbeads[CGbead_index])):
            gro_file.write(format(CGbeads[CGbead_index][axis_index], '#.3f').rjust(8))
        gro_file.write("\n")
    
    for axis_index in range(len(pbc)):
        gro_file.write(format(float(pbc[axis_index]), '#.5f').rjust(10))
    gro_file.write("\n")
    gro_file.close()
    print(filename, "saved.")
    return

# This function reads the gro file, looks for all the strings with that contain the name of the slab residue
# and save the data to the output in the format: ['Atom name and atom number', 'atom number', 'X', 'Y', 'Z']
# Note that gro file should not contain velocities!

def readGRO(filename, resname):
    
    lines = []
    slab_gmx = [] 
    file = open(filename)
    
    row_normal_width = 6 # resname atomname atomnumber X Y Z
    row_reduced_width = 5 # resname atomname-atomnumber X Y Z (happens when atomname and atomnumber are too long)
    
    for line in file:
        lines.append(line.split())
    file.close()
    
    # go through the rows in lines list (correspond to rows in the input file)
    # check the number of elements in row
    # save only X Y Z coordinates
    for row in range(len(lines)):
        if resname in lines[row][0]:
            if len(lines[row]) == row_normal_width:
                slab_gmx.append(lines[row][3:6])
            elif len(lines[row]) == row_reduced_width:
                slab_gmx.append(lines[row][2:5])
            else:
                print(lines[row])
                print("Number of row elements error")
    
    # convert all strings to floats
    for atom in slab_gmx:
        for coordinate in range(len(atom)):
            atom[coordinate] = float(atom[coordinate])
           
    return(np.array(slab_gmx))
    
    