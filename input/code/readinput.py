import numpy as np
from astropy.io import ascii

data = ascii.read('bands.txt',guess=False)

a = data['bands'].data
print a
