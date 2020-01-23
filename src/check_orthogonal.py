import numpy as np 

from scipy.spatial.distance import cosine
import sys


inSigUS = np.load('data/scream.npy')
inSigCS = np.load('data/footstep.npy')
inSigContext = np.load('data/clock.npy')

print(cosine(inSigUS, inSigContext))

print(cosine(inSigUS, inSigCS))

print(cosine(inSigCS, inSigContext))

try:
	x = np.load(sys.argv[1])
	y = np.load(sys.argv[2])
	print('{} and {}: {}'.format(sys.argv[1], sys.argv[2], cosine(x,y)))
except:
	print("no arguments")