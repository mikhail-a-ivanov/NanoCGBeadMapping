import numpy as np
from timeit import default_timer as timer
from IO import writeGRO
from slabInit import Initialize
from beadMapping import CGBeadMappingMain, FindSlabLayers, CGbeadHistogram, WriteCGBeadMapping

start = timer()
atoms, CGbeads, spacings = Initialize('anatase-101-POPE.gro', '1H151', 16, 16, 9)
#atoms, CGbeads, spacings = Initialize('rutile-101-DMPC.gro', '1H234', 18, 18, 9)

AllCGBead_AtomIndexes = CGBeadMappingMain(atoms, CGbeads, spacings, 20, 5)
CGbeads_Surface, CGbeads_FirstLayer = FindSlabLayers(CGbeads, AllCGBead_AtomIndexes)
end = timer()

CGbeadHistogram(CGbeads_Surface, "anatase-101-SurfaceAtoms-16x16x9-i20-n5", 100)
CGbeadHistogram(CGbeads_FirstLayer, "anatase-101-FirstLayer-16x16x9-i20-n5", 100)
CGbeadHistogram(AllCGBead_AtomIndexes, "anatase-101-AllAtoms-16x16x9-i20-n5", 100)
#CGbeadHistogram(CGbeads_Surface, "rutile-101-SurfaceAtoms", 100)
#CGbeadHistogram(AllCGBead_AtomIndexes, "rutile-101-AllAtoms", 100)


WriteCGBeadMapping('anatase-101-CG-surface.dat', CGbeads_Surface, 'TiA')
WriteCGBeadMapping('anatase-101-CG-firstlayer.dat', CGbeads_FirstLayer, 'TiB')
#WriteCGBeadMapping('rutile-101-CG-surface.dat', CGbeads_Surface, 'Ti')


print('\nBead mapping took',round(end - start, 4), ' seconds.')



