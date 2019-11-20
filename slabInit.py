import numpy as np
from IO import readGRO

def SlabGeometry(atoms):
    # Get the extreme slab coordinates and save them to the corresponding X, Y, Z lists:

    X = []
    X.append(np.amin(atoms.T[0]))
    X.append(np.amax(atoms.T[0]))
    
    Y = []
    Y.append(np.amin(atoms.T[1]))
    Y.append(np.amax(atoms.T[1]))
    
    Z = []
    Z.append(np.amin(atoms.T[2]))
    Z.append(np.amax(atoms.T[2]))
    
    # Note that the coordinates are in nm
    
    return(X, Y, Z)

def GenerateCGbeadCenters(atoms, x_points, y_points, z_points):
    
    # Get slab extreme coordinates [[x_min, x_max],[y_min, y_max],[z_min, z_max]]
    slab_geometry = SlabGeometry(atoms)
    
    # Calculate x, y and z spacings
    x_spacing = (slab_geometry[0][1] - slab_geometry[0][0]) / x_points
    y_spacing = (slab_geometry[1][1] - slab_geometry[1][0]) / y_points
    z_spacing = (slab_geometry[2][1] - slab_geometry[2][0]) / z_points
    
    spacings = np.array([x_spacing, y_spacing, z_spacing])
    
    # Generate grid points
    grid = []
    for x in range(x_points):
        for y in range(y_points):
            for z in range(z_points):
                grid.append(np.array([float(x),float(y),float(z)]))
    
    # Transform it to a numpy array
    grid = np.array(grid)
    
    # Convert the grid to the original coordinate system...
    # ...and move each point by half of spacing
    # We want them to become CG bead centers after all!
    grid.T[0] *= x_spacing
    grid.T[0] += (slab_geometry[0][0] + x_spacing/2)
    
    grid.T[1] *= y_spacing
    grid.T[1] += (slab_geometry[1][0] + y_spacing/2)
    
    grid.T[2] *= z_spacing
    grid.T[2] += (slab_geometry[2][0] + z_spacing/2)
    
    
    return(grid, spacings)

def Initialize(filename, resname, x_points, y_points, z_points):
    
    # Call ReadGRO function for atomic coordinates
    print("Initializing the input...\n")
    
    atoms = readGRO(filename, resname)
    
    print("Reading", filename, "...")
    print(resname, "residue is selected for the bead mapping...")
    print("Total number of atoms in the slab:", len(atoms), "\n")
    
    # Call GenerateCGbeadCenters for CG bead centers and grid spacings
    CGbeads, spacings = GenerateCGbeadCenters(atoms, x_points, y_points, z_points)
   
    print("Using", x_points,"x", y_points,"x",z_points, "grid for CG centers...")
    print("Using", np.around(spacings, decimals=4), "as a grid spacing vector...")

    print(str(len(CGbeads)), "CG centers will be generated...")
    print("Average number of atoms per CG bead is:", round(float(len(atoms))/float(len(CGbeads)), 4))
    print("")
    
    return(atoms, CGbeads, spacings)