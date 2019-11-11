import numpy as np
from timeit import default_timer as timer
from IO import writeGRO
from slabInit import Initialize
from beadMapping import CGBeadMappingMain, FindSlabLayers, CGbeadHistogram

start = timer()
atoms, CGbeads, spacings = Initialize('anatase-101-POPE.gro', '1H151', 16, 16, 9)
AllCGBead_AtomIndexes = CGBeadMappingMain(atoms, CGbeads, spacings)
CGbeads_Surface, CGbeads_FirstLayer = FindSlabLayers(CGbeads, AllCGBead_AtomIndexes)
end = timer()

print(len(CGbeads_FirstLayer), len(CGbeads_Surface), len(AllCGBead_AtomIndexes))

CGbeadHistogram(AllCGBead_AtomIndexes, "All atoms", 100)
CGbeadHistogram(CGbeads_Surface, "Surface atoms", 100)
CGbeadHistogram(CGbeads_FirstLayer, "First layer atoms", 100)

print('Bead mapping took',round(end - start, 4), ' seconds.')



