import numpy as np 
import sys

print('python preprocess.py {} {}'.format(sys.argv[1], sys.argv[2]))

footstepArr = np.load('{}'.format(sys.argv[1]))

sampled = [ footstepArr[i] for i in range(500, 2500, 50)]
sampled = np.array(sampled)

sampled = sampled.reshape(-1)

print(sampled.shape)


np.save('data/{}'.format(sys.argv[2]), sampled)