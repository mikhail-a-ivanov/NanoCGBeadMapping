import numpy as np
from timeit import default_timer as timer
from IO import writeGRO

from beadMapping import CGBeadMappingMain

start = timer()
AllDistances = CGBeadMappingMain('anatase-101-POPE.gro', '1H151', 16, 16, 9)

closest_atoms = np.argsort(AllDistances, axis=None)[:13910]

print(closest_atoms // 13910)
print(closest_atoms % 13910)


end = timer()

print('Bead mapping took',round(end - start, 4), ' seconds.')



