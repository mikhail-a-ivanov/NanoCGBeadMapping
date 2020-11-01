import numpy as np

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
    
    