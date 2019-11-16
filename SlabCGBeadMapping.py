import numpy as np
from timeit import default_timer as timer
from IO import writeGRO
from slabInit import Initialize
from beadMapping import CGBeadMappingMain, FindSlabLayers, CGbeadHistogram

start = timer()
atoms, CGbeads, spacings = Initialize('anatase-101-POPE.gro', '1H151', 16, 16, 9, 0.)
#atoms, CGbeads, spacings = Initialize('rutile-101-DMPC.gro', '1H234', 18, 18, 9, 0.)

AllCGBead_AtomIndexes = CGBeadMappingMain(atoms, CGbeads, spacings)
CGbeads_Surface, CGbeads_FirstLayer = FindSlabLayers(CGbeads, AllCGBead_AtomIndexes)
end = timer()

CGbeadHistogram(AllCGBead_AtomIndexes, "All atoms", 100)
CGbeadHistogram(CGbeads_Surface, "Surface atoms", 100)
CGbeadHistogram(CGbeads_FirstLayer, "First layer atoms", 100)

print('\nBead mapping took',round(end - start, 4), ' seconds.')



