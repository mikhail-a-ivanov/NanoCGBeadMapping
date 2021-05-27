def readXMOL(filename):
    """Reads xmol files"""
    xmol = []
    with open(filename) as file:
        lines = file.readlines()
        for line in lines[2:]:
            xmol.append(line.split())
    
    return xmol

def readMMOL(filename):
    """Reads mmol files"""
    mmol = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines[2:]:
            mmol.append(line.split())
    return mmol

def writeMCMfile(filename, xmol):
    """Writes mcm files"""
    with open(filename, 'w') as file:
        file.write(str(len(xmol)) + '\n')
        file.write('#\n')
        for line_index in range(len(xmol)):
            file.write(str(xmol[line_index][0]) + ' ' + str(xmol[line_index][1]) + 
                      ' ' + str(xmol[line_index][2]) + ' ' + str(xmol[line_index][3]) + 
                      ' ' + str(mass) + ' ')
            if str(xmol[line_index][0]) == 'TiA':
                file.write(str(TiA_charge) + ' 1 ' + 'TiA\n')
            elif str(xmol[line_index][0]) == 'TiB':
                file.write(str(TiB_charge) + ' 2 ' + 'TiB\n')
        file.write('0\n')
        file.write('0  Order=1-2-3\n')

def mmol2gro(mmol_filename, filename, resname):
    """Converts mmol to gro"""
    mmol = readMMOL(mmol_filename)

    with open(filename, 'w') as file:
        file.write(filename + '\n')
        file.write(str(len(mmol)) + '\n')
        for atom_index in range(len(mmol)):
            file.write(format(resname).rjust(9))
            file.write(format(str(mmol[atom_index][0])).rjust(6))
            file.write(format(str(atom_index + 1)).rjust(5))
            for axis_index in range(len(mmol[atom_index][1:4])):
                file.write(format((float(mmol[atom_index][1:4][axis_index]) * 0.1), '#.3f').rjust(8))
            file.write("\n") 
        file.write(format('10.00000').rjust(10))
        file.write(format('10.00000').rjust(10))
        file.write(format('10.00000\n').rjust(10))

def mmol2pdb(mmol_filename, filename, resname):
    """Writes mmol to pdb"""
    mmol = readMMOL(mmol_filename)

    file_header = "REMARK    " + filename + '\n'
    pbc_record = "CRYST1"
    pbc_box = [100., 100., 100.]
    for axis_index in range(len(pbc_box)):
        pbc_record += format(float(pbc_box[axis_index]), '#.3f').rjust(9)
    pbc_record += "  90.00  90.00  90.00 P 1           1"

    with open(filename, 'w') as file:
        file.write(file_header)
        file.write(pbc_record + '\n')
        file.write("MODEL   0" + '\n')
        for atom_index in range(len(mmol)):
            file.write("ATOM  ".rjust(6))
            file.write(format(str(atom_index+1)).rjust(5))
            file.write(format(mmol[atom_index][0]).rjust(5))
            file.write(format(resname).rjust(4))
            file.write(format("A").rjust(2))
            file.write(format("1").rjust(4))
            file.write(format("").rjust(5))
            for axis_index in range(len(mmol[atom_index][1:4])):
                file.write(format((float(mmol[atom_index][1:4][axis_index])), '#.3f').rjust(7))
                file.write(format("").rjust(1))
            file.write(format('1.00 0.00').rjust(12))
            file.write(format(format(mmol[atom_index][0]).rjust(12)))
            file.write("\n")
        file.write("TER")
        file.write(format(str(len(mmol) + 1).rjust(9)))
        file.write(format(resname).rjust(10))
        file.write(format("1\n").rjust(4))
        file.write('ENDMDL\n')
        file.write('END\n')
        print(filename, "saved.")

def readPDBdata(filename):
    """Reads data from pdb"""
    pdb_data = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines[5:-1]:
            pdb_data.append(line.split())

    xyz = []
    atomnames = []
    for line in pdb_data:
        xyz.append(np.array([float(line[6]), float(line[7]), float(line[8])]))
        atomnames.append(line[-1])
    xyz = np.array(xyz)
    
    return xyz, atomnames