from model import Model
import numpy as np 
from scipy.stats import ortho_group
import matplotlib.pyplot as plt
import copy
from scipy.spatial.distance import cosine

m = Model(nodePerStimuli=160)
stimuli = ortho_group.rvs(160)
Ntrials = 20

UseRandom = False

frequency = (20, 8000)
frequency2 = (20, 8000)

if UseRandom:
	inSigUS = stimuli[0]
	inSigCS = stimuli[1]
	inSigContext = stimuli[2]
	inSigNoUS = stimuli[7]
	inSigUnpairedCS = stimuli[3]
	inSigUnpairedContext = stimuli[4]

else:
	inSigUS = np.load('data/scream2_{}_{}.npy'.format(frequency2[0], frequency2[1]))
	#inSigUS = np.load('data/scream2_20_20000.npy')
	inSigCS = np.load('data/footstep_{}_{}.npy'.format(frequency2[0], frequency2[1]))
	inSigContext = np.load('data/clock_{}_{}.npy'.format(frequency2[0], frequency2[1]))

	inSigNoUS = np.load('data/white_{}_{}.npy'.format(frequency[0], frequency[1]))
	inSigUnpairedCS = np.load('data/footstep2_{}_{}.npy'.format(frequency[0], frequency[1]))
	inSigUnpairedContext = np.load('data/blowwind_{}_{}.npy'.format(frequency[0], frequency[1]))


print('US')
print(cosine(inSigUS, inSigContext))
print(cosine(inSigUS, inSigCS))
print(cosine(inSigUS, inSigNoUS))
print(cosine(inSigUS, inSigUnpairedCS))
print(cosine(inSigUS, inSigUnpairedContext))

print('CS')
print(cosine(inSigCS, inSigContext))
print(cosine(inSigCS, inSigNoUS))
print(cosine(inSigCS, inSigUnpairedCS))
print(cosine(inSigCS, inSigUnpairedContext))

print('UnpairedCS')
#print(cosine(inSigUnpairedCS, inSigUnpairedContext))
print(cosine(inSigUnpairedCS, inSigUnpairedContext))


def fearAcq():
	global Ntrials, inSigUS, inSigContext, inSigNoUS, inSigUnpairedCS, inSigUnpairedContext
	global m
	fearAcqCS = []
	fearAcqUnpairedCS = []
	fearAcqContext = []
	for t in range(Ntrials):
		
		m.forward(inSigUS, inSigCS, inSigContext, USpresent=1)
		fearAcqCS.append( m.evaluate(inSigNoUS, inSigCS, inSigUnpairedContext) )
		fearAcqUnpairedCS.append( m.evaluate(inSigNoUS, inSigUnpairedCS, inSigUnpairedContext) )
		fearAcqContext.append( m.evaluate(inSigNoUS, inSigUnpairedCS, inSigContext))

	return np.array(fearAcqCS), np.array(fearAcqUnpairedCS), np.array(fearAcqContext)


fearAcqCS, fearAcqUnpairedCS, fearAcqContext = fearAcq()

plt.plot(range(Ntrials), fearAcqCS, marker='o', label='CS+')
plt.plot(range(Ntrials), fearAcqUnpairedCS , marker='o', label='CS-' )
plt.plot(range(Ntrials), fearAcqContext , marker='o', label='context' )
plt.legend()
plt.xlabel('Number of Trials')
plt.ylabel('Fear Response')
plt.title('Fear Acquisition')
plt.savefig('fearAcq.png')
plt.clf()



fearExt = []
fearExtDiffContext = []

m1 = copy.copy(m)
m2 = copy.copy(m)


Ntrials2 = 1500

for t in range(Ntrials2):
	
	m1.forward(inSigNoUS, inSigCS, inSigContext, USpresent=0)
	fearExt.append( m1.evaluate(inSigNoUS, inSigCS, inSigContext))

	m2.forward(inSigNoUS, inSigCS, inSigUnpairedContext, USpresent=0)
	fearExtDiffContext.append( m2.evaluate(inSigNoUS, inSigCS, inSigUnpairedContext))

#print(fearExt)
plt.plot(range(Ntrials2), fearExt, marker='o', label='same context')
plt.xlabel('Number of Trials')
plt.ylabel('Fear Response')
plt.title('Fear Extinction')
plt.savefig('fearExt.png')


