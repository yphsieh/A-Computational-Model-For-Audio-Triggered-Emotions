import numpy as np 

def logistic(x):
	return 1/(1+np.exp((-1)*x))

class Hippocampus:
	def __init__(self, inNodeNum, outNodeNum, learnRate):
		#The weights are initialized to random values in the range [0, 0.6].
		#Add Gaussian noise with zero mean and standard deviation of 0.025
		self.learnRate = learnRate
		self.weights = np.random.random((outNodeNum, inNodeNum))*0.6
		self.activation = np.zeros(outNodeNum)

	def forward(self, inSignal):
		'''
		Get the activation and update weights.
		Args:
			inSignal {np.array}: shape [inNodeNum], 1-dimensional
		'''
		self.activation = logistic(np.matmul(self.weights+np.random.normal(loc=0, scale=0.025, size=(self.weights.shape)), inSignal.reshape(-1,1)))

		self.weights = self.weights + self.learnRate* np.matmul(self.activation.reshape(-1,1), inSignal.reshape(1,-1))
		return self.activation

	def evaluate(self, inSignal):
		return logistic(np.matmul(self.weights+np.random.normal(loc=0, scale=0.025, size=(self.weights.shape)), inSignal.reshape(-1,1)))

class BLA:
	def __init__(self, inNodeNum, outNodeNum, learnRate, gamma=0.99):
		#The weights are initialized to random values in the range [0, 0.6].
		#Add Gaussian noise with zero mean and standard deviation of 0.025
		self.learnRate = learnRate
		self.weights = np.random.random((outNodeNum, inNodeNum))*0.6 
		self.activation = np.zeros(outNodeNum)
		self.predictWeights = np.random.random((inNodeNum))*0.6 
		self.formerPredict = 0
		self.discountFactor = gamma

	def forward(self, inSignal, USpresent):
		'''
		Get the activation and update weights.
		Args:
			inSignal {np.array}: shape [inNodeNum], 1-dimensional
			USpresent {bool}: true if US is presented
		'''
		self.activation = logistic(np.matmul(self.weights+np.random.normal(loc=0, scale=0.025, size=(self.weights.shape)), inSignal.reshape(-1,1)))
		P = np.matmul((self.predictWeights+np.random.normal(loc=0, scale=0.025, size=(self.predictWeights.shape))).reshape(1,-1), inSignal.reshape(-1,1))
		TD = int(USpresent) + self.discountFactor*P - self.formerPredict
		self.formerPredict = P

		self.weights = self.weights + self.learnRate * TD * np.matmul(self.activation.reshape(-1,1), inSignal.reshape(1,-1))
		self.predictWeights = self.predictWeights + self.learnRate * TD * inSignal

		return self.activation

	def evaluate(self, inSignal):
		return logistic(np.matmul(self.weights+np.random.normal(loc=0, scale=0.025, size=(self.weights.shape)), inSignal.reshape(-1,1)))
		
class VMPFC:
	def __init__(self, inNodeNum, outNodeNum, learnRate, gamma=0.99):
		#The weights are initialized to random values in the range [0, 0.6].
		#Add Gaussian noise with zero mean and standard deviation of 0.025
		self.learnRate = learnRate
		self.weights = np.random.random((outNodeNum, inNodeNum))*0.6 
		self.activation = np.zeros(outNodeNum)
		self.predictWeights = np.random.random((inNodeNum))*0.6 
		self.formerPredict = 0
		self.discountFactor = gamma

	def forward(self, inSignal, USpresent):
		'''
		Get the activation and update weights.
		Args:
			inSignal {np.array}: shape [inNodeNum], 1-dimensional
			USpresent {bool}: true if US is presented
		'''
		self.activation = logistic(np.matmul(self.weights+np.random.normal(loc=0, scale=0.025, size=(self.weights.shape)), inSignal.reshape(-1,1)))
		P = np.matmul((self.predictWeights+np.random.normal(loc=0, scale=0.025, size=(self.predictWeights.shape))).reshape(1,-1), inSignal.reshape(-1,1))
		TD = int(USpresent) + self.discountFactor*P - self.formerPredict
		self.formerPredict = P

		self.weights = self.weights - self.learnRate * TD * np.matmul(self.activation.reshape(-1,1), inSignal.reshape(1,-1))
		self.predictWeights = self.predictWeights - self.learnRate * TD * inSignal
		
		return self.activation

	def evaluate(self, inSignal):
		return logistic(np.matmul(self.weights+np.random.normal(loc=0, scale=0.025, size=(self.weights.shape)), inSignal.reshape(-1,1)))

class Model:
	def __init__(self, nodePerStimuli, stimuliNum=3, learnRate=0.01):
		'''
		Args:
			stimuliNum {int}: one context, one US, one CS
		'''
		self.nodePerStimuli = nodePerStimuli
		self.stimuliNum = stimuliNum
		self.learnRate = learnRate

		inNodeNumHipp = nodePerStimuli * stimuliNum
		outNodeNumHipp = 20
		inNodeNumBLA = outNodeNumHipp + 2*nodePerStimuli
		outNodeNumBLA = 40
		self.hippocampus = Hippocampus(inNodeNum=inNodeNumHipp, outNodeNum=outNodeNumHipp, learnRate=learnRate)
		self.bLA = BLA(inNodeNum=inNodeNumBLA, outNodeNum=outNodeNumBLA, learnRate=learnRate, gamma=0.99)
		self.vmPFC = VMPFC(inNodeNum=inNodeNumBLA, outNodeNum=outNodeNumBLA, learnRate=learnRate, gamma=0.99)

	def forward(self, inSigUS, inSigCS, inSigContext, USpresent):
		inputHipp = np.concatenate((inSigUS, inSigCS, inSigContext))
		activeHipp = self.hippocampus.forward(inputHipp)
		inputBLA = np.concatenate((inSigUS, inSigCS, activeHipp.reshape(-1)))
		activeBLA = self.bLA.forward(inputBLA, USpresent)
		activeVMPFC = self.vmPFC.forward(inputBLA, USpresent)
		fearResponse = np.sum(activeBLA) - np.sum(activeVMPFC)
		return fearResponse

	def evaluate(self, inSigUS, inSigCS, inSigContext):
		inputHipp = np.concatenate((inSigUS, inSigCS, inSigContext))
		activeHipp = self.hippocampus.evaluate(inputHipp)

		inputBLA = np.concatenate((inSigUS, inSigCS, activeHipp.reshape(-1)))
		activeBLA = self.bLA.evaluate(inputBLA)
		activeVMPFC = self.vmPFC.evaluate(inputBLA)
		fearResponse = np.sum(activeBLA) - np.sum(activeVMPFC)
		return fearResponse

	def __copy__(self):
		newModel = Model(nodePerStimuli=self.nodePerStimuli, stimuliNum=self.stimuliNum, learnRate=self.learnRate)

		newModel.hippocampus.weights = self.hippocampus.weights.copy()
		
		newModel.bLA.weights = self.bLA.weights.copy()
		newModel.bLA.predictWeights = self.bLA.predictWeights.copy()

		newModel.vmPFC.weights = self.vmPFC.weights.copy()
		newModel.vmPFC.predictWeights = self.vmPFC.predictWeights.copy()

		return newModel


if __name__ == '__main__':
	m = Model(3)
	inSigUS = np.array([0,1,0])
	inSigCS = np.array([1,0,0])
	inSigContext = np.array([0,0,1])
	response = m.forward(inSigUS, inSigCS, inSigContext, USpresent=1)

	print(response)