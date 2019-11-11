import numpy as np
from timeit import default_timer as timer
from IO import writeGRO
from beadMapping import CGBeadMappingMain, CGbeadHistogram

start = timer()
AllCGBead_AtomIndexes = CGBeadMappingMain('anatase-101-POPE.gro', '1H151', 12, 12, 9)
end = timer()

CGbeadHistogram(AllCGBead_AtomIndexes, 100)

print('Bead mapping took',round(end - start, 4), ' seconds.')



