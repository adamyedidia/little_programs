import sys
import pickle
from scipy.special import softmax
from math import log, sqrt
import numpy as np
import matplotlib.pyplot as p
import time

t = time.time()

totalNumExamples = 1000
#n = 100
#totalNumExamples = 2


def randomPermutationMat(n):
	i = np.identity(n)
#	rr = range(n)
	np.random.shuffle(i)
	return i



numExamples = 100

assert totalNumExamples % numExamples == 0

numBatches = int(totalNumExamples / numExamples)

#numTeacherLayers = 2
numLayers = 2

#inertiaFraction = (numBatches - 1)/numBatches

def softPlus(x):
	return np.logaddexp(0, x)	

def softPlusPrime(x):
	return softmax([x, 0])[0]

def relu(x):
	return x*(x>=0)

def reluPrime(x):
	return 1*(x>=0)

def softSign(x):
	return x/(abs(x) + 1)

def softSignPrime(x):
	return 1/(1 + abs(x))**2



#f, fPrime = relu, reluPrime
#f, fPrime = softPlus, softPlusPrime


f, fPrime = softSign, softSignPrime


def evaluateNetwork(ws, X, showLayers=False):
	numLayers = len(ws)
	layerOutput = X


	for i in range(numLayers):
		if showLayers:
			p.matshow(matrixF(WsDotXs(ws[i], layerOutput)))
			p.colorbar()
			p.show()			


		if i < numLayers-1:
#			layerOutput = splitAndBiasX(np.vectorize(f)(WsDotXs(ws[i], layerOutput)))
			layerOutput = splitAndBiasX(matrixF(WsDotXs(ws[i], layerOutput)))
		else:
#			layerOutput = np.vectorize(f)(WsDotXs(ws[i], layerOutput))
			layerOutput = matrixF(WsDotXs(ws[i], layerOutput))



	return layerOutput

def augmentWithBias(X):
#	print(X)
#	print(np.ones((1, numExamples)))

	biasAugmentedX = np.concatenate([X, np.ones((1, X.shape[1]))], axis=0)
	return biasAugmentedX

def augmentWithZeros(X):
	biasAugmentedX = np.concatenate([X, np.zeros((1, numExamples))], axis=0)
	return biasAugmentedX


def diminishWithBias(W):
	biasDiminishedW = W[:,:-1]
	return biasDiminishedW

def selectSubset(overallX, overallT, index, numExamples):
	return overallX[:, index*numExamples:(index+1)*numExamples], \
		overallT[:, index*numExamples:(index+1)*numExamples]

def selectRandomSubset(overallX, overallT, numExamples):
	indices = np.random.choice(overallX.shape[1], numExamples, replace=False)


#	print("hi")

#	print(indices)
#	for i in range(numExamples):
#		print(np.transpose(overallT[:, indices])[i])
#		showExample(overallX[:, indices], i)

#	overallX[:, indices]

	return overallX[:, indices], overallT[:, indices]

def showExample(X, i):
	p.matshow(np.reshape(np.transpose(X)[i], (7,7)))
	p.show()

def repeatT(T, numRepeats):
	return np.repeat(T, numRepeats, axis=0)

#X = np.array([[1,2,3], [4,5,6], [7,8,9]])
#T = X
#T = np.array([[4,5,6], [7,8,9], [1,2,3]])

#print(selectRandomSubset(X, T, 2))
#sys.exit()


def batchArrayAlongZeroAxis(arr, batchSize):

    listOfBigFrames = []
    frameSum = np.zeros(arr[0].shape)
    numFrames = len(arr)

    for i in range(numFrames):
        frameSum += arr[i]

        if i % batchSize == batchSize - 1:
            listOfBigFrames.append(frameSum / batchSize)
            frameSum = np.zeros(arr[0].shape)

    if numFrames % batchSize != 0:
        listOfBigFrames.append(frameSum / (numFrames % batchSize))

#    print len(listOfBigFrames)
    return np.array(listOfBigFrames)

def batchArrayAlongAxis(arr, axis, batchSize):
    rearrangedArr = np.swapaxes(arr, 0, axis)
    batchedArray = batchArrayAlongZeroAxis(rearrangedArr, batchSize)
    return(np.swapaxes(batchedArray, 0, axis))

def vote(finalZ):
#	print(batchArrayAlongAxis(finalZ, 0, 5).shape)

	return np.argmax(batchArrayAlongAxis(finalZ, 0, 5), axis=0) 



overallX, overallT = pickle.load(open("little_mnist_one_minus_one.p", "rb"))

overallX = augmentWithBias(overallX)
overallT = repeatT(overallT, 5)

#p.matshow(repeatT(overallT, 5))
#p.show()	

#p.plot(vote(repeatT(overallT, 5)))
#p.show()	



#print(X.shape)
#print(T.shape)

#X = np.random.normal(size=(n, numExamples))
#X = np.array([[1,-2]])
#X = np.identity(n)
#print(X.shape)

#teacherWs = [np.random.normal(0,1,size=(n,n+1)) for _ in range(numTeacherLayers)]

#print(teacherWs)

#T = evaluateNetwork(teacherWs, augmentWithBias(X))
#T = np.array([[1, 2]])
#T = np.array([[1, 2], [3, 4]])
#T = np.array([[1]])

#T = np.vectorize(softPlus)(np.random.normal(size=(n, numExamples)))
#T = np.vectorize(relu)(np.random.normal(size=(n, numExamples)))
#T = X


#print(X)
#print(T)

#layerOutput = X
#for i in range(numTeacherLayers):
#	layerOutput = np.dot(teacherWs[i], layerOutput)

#T = layerOutput




#T = np.array([[1]])
#T = np.random.exponential(size=(n, numExamples))


#print(T)


#X = T
#X = -T


inputSize = 50
middleSize = 50
outputSize = 50 

cohortSize = 10
numCohorts = 5


randomPerm = randomPermutationMat(numCohorts*cohortSize)
#randomPerm = np.identity(numCohorts*cohortSize)
#randomPerm = np.random.normal(0,1,size=(numCohorts*cohortSize, numCohorts*cohortSize))

#p.matshow(randomPerm)
#p.show()

#inputSize = 784
#middleSize = 784
#outputSize = 10

#layerSizes = [2,2]
layerSizes = [inputSize, middleSize, outputSize]
#layerSizes = [inputSize, middleSize, middleSize, outputSize]

#W = np.ones((n,n))

#                                     Looks like typo but it's not
#Ws = [np.random.normal(0,1,size=(layerSizes[i+1],layerSizes[i]+1)) for i in range(numLayers)]
Ws = [[np.random.normal(0,1,size=(cohortSize,cohortSize+1)) for _ in range(numCohorts)] for i in range(numLayers)]

#print(len(Ws[0]))

#Ws = [np.zeros((n, n+1)) for _ in range(numLayers)]
#Ws = [np.array([-1/3, 4/3]), np.array([1, 0])]

#Ys = [np.zeros((n, numExamples)) for _ in range(numLayers)]
Ys = [[np.random.normal(0,1,size=(layerSizes[i+1], numExamples)) for i in range(numLayers)] for _ in range(numBatches)]
lamdas = [[np.zeros((layerSizes[i+1], numExamples)) for i in range(numLayers)] for _ in range(numBatches)]

#Ys = [np.array([[-4,3]]), np.array([[1,2]])]
#Ys = [np.array([[1,-1]]), np.array([[-1,1]])]


#Zs = [np.ones((n, numExamples)) for _ in range(numLayers)]
Zs = [[np.random.normal(0,1,size=(layerSizes[i+1],numExamples)) for i in range(numLayers)] for _ in range(numBatches)]
mus = [[np.zeros((layerSizes[i+1], numExamples)) for i in range(numLayers)] for _ in range(numBatches)]

#Zs = [np.array([[1,2]]), np.array([[1,2]])]
#Y = T
#Z = Y


numIter = 2000
#rhalpha = 1/2
#rhalpha = 1/((numExamples*n)**numLayers)
#rhalpha = 1/(numExamples*n)
#rhalpha = 1
#rho = 1/n
#alpha = 1/n**2
rho = 1
alpha = 1/inputSize
#rhalpha = 1e-3
#LR = 1/n

#bigWeightPenalty = 0
gamma = 10


def softLog(x):
	if x <= 0:
		return -100
	else:
		return log(x)

def splitAndBiasX(X):
	Xs = []

	for cohortIndex in range(numCohorts):
		Xs.append(augmentWithBias(X[cohortIndex*cohortSize:(cohortIndex+1)*cohortSize,:]))

	return Xs

def splitX(X):
	Xs = []

	for cohortIndex in range(numCohorts):
		Xs.append(X[cohortIndex*cohortSize:(cohortIndex+1)*cohortSize,:])

	return Xs	

def WsDotXs(Ws, Xs):#, permute=False):
#	print(len(Ws))
##	print([W.shape for W in Ws])
#	print(np.array(Ws).shape)
#	print([np.dot(W, X).shape for W, X in zip(Ws, Xs)])

#	if permute:
#		return np.dot(randomPerm, np.concatenate([np.dot(W, X) for W, X in zip(Ws, Xs)], axis=0))
#	else:
#		print([np.dot(W, X).shape for W, X in zip(Ws, Xs)])
	return np.concatenate([np.dot(W, X) for W, X in zip(Ws, Xs)], axis=0)

def lamdaStep(prevLamda, prevY, prevWs, Xs):
#	print(prevY.shape)
#	print(prevW.shape)
#	print(X.shape)
#	return prevLamda + alpha*(prevY - np.dot(prevW, X))
	return prevLamda + alpha*(prevY - WsDotXs(prevWs, Xs))


def muStep(prevMu, prevZ, fOfPrevY):
	return prevMu + alpha*(prevZ - fOfPrevY)

def finalZStep(T, mu, fOfPrevY):
#	print("last Z", T - mu - rhalpha*(prevZ - fOfPrevY))

#	return T - mu - rho*(prevZ - fOfPrevY)
	if False:
		print("")
		print("-mu/rho", -mu/rho)
		print("fOfPrevY", fOfPrevY)
		print("T", T)


	return (-mu/rho + fOfPrevY + T)/2
#	return -mu/rho + fOfPrevY + T - prevZ

def ZStepNotLast(muThisLayer, fOfPrevYThisLayer, WsNextLayer, lamdaNextLayer, YNextLayer, prevZsThisLayer):
#	print("uh-oh!")
#	print("not last Z", -muThisLayer + fOfPrevYThisLayer + np.dot(np.transpose(WNextLayer), lamdaNextLayer) + \
#		rhalpha*np.dot(np.transpose(WNextLayer), YNextLayer - np.dot(WNextLayer, prevZThisLayer)))
	if False:
		print("")

		print("mu term", -muThisLayer/rho)
		print("fOfPrevY", fOfPrevYThisLayer)
		print("W transpose lambda", np.dot(np.transpose(WNextLayer), lamdaNextLayer)/rho)
		print("WT(Y-WZ) term", np.dot(np.transpose(WNextLayer), YNextLayer - np.dot(WNextLayer, prevZThisLayer)))
		print("overall", -muThisLayer/rho + fOfPrevYThisLayer + np.dot(np.transpose(diminishWithBias(WNextLayer)), lamdaNextLayer)/rho + \
		np.dot(np.transpose(diminishWithBias(WNextLayer)), YNextLayer - np.dot(WNextLayer, prevZThisLayer)))

		print("")

	diminishedTransposedWs = [np.transpose(diminishWithBias(W)) for W in WsNextLayer]

#np.dot(np.transpose(diminishWithBias(WNextLayer)), lamdaNextLayer)

#	return -muThisLayer/rho + fOfPrevYThisLayer + WsDotXs(diminishedTransposedWs, splitX(lamdaNextLayer))/rho + \
#		WsDotXs(diminishedTransposedWs, splitX(YNextLayer - WsDotXs(WsNextLayer, prevZsThisLayer)))

#	splitX(np.dot(np.transpose(randomPerm), lamdaNextLayer))

	return -muThisLayer/rho + fOfPrevYThisLayer + WsDotXs(diminishedTransposedWs, splitX(lamdaNextLayer))/rho + \
		WsDotXs(diminishedTransposedWs, splitX(YNextLayer - WsDotXs(WsNextLayer, prevZsThisLayer)))


#	wLamdaPlusYMat = np.dot(np.transpose(WNextLayer), np.transpose(augmentWithBias(np.transpose(lamdaNextLayer/rho + YNextLayer - np.dot(WNextLayer, prevZThisLayer)))))


#	return -muThisLayer/rho + fOfPrevYThisLayer + wLamdaPlusYMat

#	return fOfPrevYThisLayer + rhalpha*np.dot(np.transpose(diminishWithBias(WNextLayer)), lamdaNextLayer) + \
#		rhalpha*np.dot(np.transpose(diminishWithBias(WNextLayer)), YNextLayer - np.dot(WNextLayer, prevZThisLayer))


#	return -muThisLayer + fOfPrevYThisLayer + np.dot(np.transpose(WNextLayer), lamdaNextLayer) + \
#		rhalpha*np.dot(np.transpose(WNextLayer), YNextLayer - np.dot(WNextLayer, prevZThisLayer))

def ZStepNotLastInvertStyle(muThisLayer, fOfPrevYThisLayer, WNextLayer, lamdaNextLayer, YNextLayer, prevZThisLayer):
	matToInv = np.dot(np.transpose(diminishWithBias(WNextLayer)), WNextLayer) + np.transpose(augmentWithZeros(np.identity(n)))
#	invMat = np.dot(np.linalg.inv(np.dot(np.transpose(matToInv), matToInv)), np.transpose(diminishWithBias(matToInv)))
	invMat = np.dot(np.transpose(diminishWithBias(matToInv)), np.linalg.inv(np.dot(matToInv, np.transpose(matToInv))))
#	invMat = np.dot(np.linalg.inv(np.dot(np.transpose(matToInv), matToInv)), np.transpose(matToInv))


	wLamdaPlusYMat = np.dot(np.transpose(diminishWithBias(WNextLayer)), lamdaNextLayer/rho + YNextLayer)


	print(invMat.shape)
	print(muThisLayer.shape)
	print(wLamdaPlusYMat.shape)
	print(fOfPrevYThisLayer.shape)

	return np.dot(invMat, -muThisLayer/rho + fOfPrevYThisLayer + wLamdaPlusYMat)


def YStep(prevWs, Xs, mu, fPrimeOfPrevY, lamda, Z, fOfPrevY):
	if False:
		print("")
		print("Y exploration")
		print("WX term", np.dot(prevW, X))
		print("mu fprime term", mu*fPrimeOfPrevY/rho)
		print("lamda term", -lamda/rho)
		print("fPrime term", fPrimeOfPrevY*(Z - fOfPrevY))
		print("fPrimeOfPrevY", fPrimeOfPrevY)
		print("")

	return WsDotXs(prevWs, Xs) + mu*fPrimeOfPrevY/rho - lamda/rho + fPrimeOfPrevY*np.dot(np.transpose(randomPerm), (Z - fOfPrevY))



def WStepPerfect(lamda, prevW, Y, X):

#	print(X)

#	print(X)

#	if n >= numExamples:
	if X.shape[0] >= X.shape[1]:
		gramX = np.dot(np.transpose(X), X)
	else:
		gramX = np.dot(X, np.transpose(X))

#	print(X)

	frobeniusGramX = np.sum(gramX*gramX)

#	print(frobeniusGramX)

#	print(gramX + \
#			bigWeightPenalty*frobeniusGramX*np.identity(numExamples))

#	if n >= numExamples:
	if X.shape[0] >= X.shape[1]:
		pseudoInv = np.dot(np.linalg.inv(gramX + \
			bigWeightPenalty*np.identity(numExamples)), np.transpose(X))
	else:
#		print(np.dot(X, np.transpose(X)) + \
#			bigWeightPenalty*np.identity(n))
		pseudoInv = np.dot(np.transpose(X), np.linalg.inv(gramX + \
			bigWeightPenalty*np.identity(gramX.shape[0])))

#	print("lamda plus Y", lamda + Y)
#	print("pseudoInv", pseudoInv)


	return inertiaFraction*prevW + (1-inertiaFraction)*np.dot((lamda/rho + Y), pseudoInv)	



def WStepNew(lamda, prevW, Y, X):
#	print(np.array(X).shape)
	inputLayerSize = X.shape[0] # this is like n+1

	firstMat = gamma*prevW + np.dot((lamda/rho + Y), np.transpose(X))
	secondMat = np.linalg.inv(np.dot(X, np.transpose(X)) + gamma*np.identity(inputLayerSize))

	return np.dot(firstMat, secondMat)

def jointWStepNew(lamda, prevWs, Y, Xs):
	lamdas = splitX(lamda)
	Ys = splitX(Y)

	return [WStepNew(lamda, prevW, Y, X) for lamda, prevW, Y, X in zip(lamdas, prevWs, Ys, Xs)]

#def rearrange(fullZ, cohortSize=cohortSize, numCohorts=numCohorts, numExamples=numExamples):
#	fullZ = np.reshape(fullZ, (cohortSize, numCohorts, numExamples))
#	fullZ = np.swapaxes(fullZ, 0, 1)
#	fullZ = np.reshape(fullZ, (cohortSize*numCohorts, numExamples))
#	return fullZ

#def rearrangeMaker():
#	return 

if __name__ == "__main__":
	errors = []
	logErrors = []

	percent = 0

	lamda0s = []
	mu0s = []
	W0s = []
	Y0s = []
	Z0s = []
	lamda1s = []
	mu1s = []
	W1s = []
	Y1s = []
	Z1s = []

	W2s = []

	batchIndex = 0

	for i in range(numIter):
		if i/numIter > percent/100 and (i-1)/numIter <= percent/100:
			print(i, "/", numIter)
			percent += 1

		X, T = selectSubset(overallX, overallT, batchIndex, numExamples)	
	#	X, T = selectRandomSubset(overallX, overallT, numExamples)	

#		biasAugmentedX = augmentWithBias(X)
		splitAndBiasedXs = splitAndBiasX(X)

		batchIndex = (batchIndex + 1) % numBatches

		prevLamdas = lamdas[:]
		prevMus = mus[:]
		prevZs = Zs[:]
		prevYs = Ys[:]
		prevWs = [W[:] for W in Ws]

	#	print(W)

#		fOfPrevYs = [[np.vectorize(f)(prevY) for prevY in batchPrevYs] for batchPrevYs in prevYs]
#		fPrimeOfPrevYs = [[np.vectorize(fPrime)(prevY) for prevY in batchPrevYs] for batchPrevYs in prevYs]

		matrixF = lambda Y: np.dot(randomPerm, np.vectorize(f)(Y))
#		matrixFprime = lambda Y: np.dot(randomPerm, np.vectorize(fPrime)(Y))

#		matrixF = np.vectorize(f)
#		matrixFprime = np.vectorize(fPrime)

		fOfPrevYs = [[matrixF(prevY) for prevY in batchPrevYs] for batchPrevYs in prevYs]
		fPrimeOfPrevYs = [[np.vectorize(fPrime)(prevY) for prevY in batchPrevYs] for batchPrevYs in prevYs]


		for layer in range(numLayers-1, -1, -1):

			splitAndBiasedPrevZs = splitAndBiasX(prevZs[batchIndex][layer-1])


	#		print(layer)

			# Lambda step
			if layer == 0:
				lamdas[batchIndex][layer] = lamdaStep(prevLamdas[batchIndex][layer], prevYs[batchIndex][layer], prevWs[layer], splitAndBiasedXs)
			else:
#				print(np.array(prevWs[layer]).shape)

				lamdas[batchIndex][layer] = lamdaStep(prevLamdas[batchIndex][layer], prevYs[batchIndex][layer], \
					prevWs[layer], splitAndBiasedPrevZs)

	#		print("lamda",layer, lamdas[layer])

			if layer == 0:
				lamda0s.append(lamdas[batchIndex][layer][0][0])
			if layer == 1:
				lamda1s.append(lamdas[batchIndex][layer][0][0])


			# Mu step
	#		print(layer)
	#		print(len(mus))
	#		print(len(prevMus))
	#		print(len(prevZs))
	#		print(len(fOfPrevYs))
			mus[batchIndex][layer] = muStep(prevMus[batchIndex][layer], prevZs[batchIndex][layer], fOfPrevYs[batchIndex][layer])

	#		print("mu", layer, mus[layer])

			if layer == 0:
				mu0s.append([mus[batchIndex][layer]][0][0][0])
			if layer == 1:
				mu1s.append([mus[batchIndex][layer]][0][0][0])


			# Z step
			if layer == numLayers - 1:
				Zs[batchIndex][layer] = finalZStep(T, mus[batchIndex][layer], fOfPrevYs[batchIndex][layer])
			else:
				Zs[batchIndex][layer] = ZStepNotLast(mus[batchIndex][layer], fOfPrevYs[batchIndex][layer], \
					Ws[layer+1], lamdas[batchIndex][layer+1], Ys[batchIndex][layer+1], splitAndBiasX(prevZs[batchIndex][layer]))

	#		print("Y",layer, Ys[layer])
	#		print("Z",layer, Zs[layer])

			if layer == 0:
				Z0s.append([Zs[batchIndex][layer]][0][0][0])
			if layer == 1:
				Z1s.append([Zs[batchIndex][layer]][0][0][0])



			# Y step
			if layer == 0:
				Ys[batchIndex][layer] = YStep(prevWs[layer], splitAndBiasedXs, mus[batchIndex][layer], fPrimeOfPrevYs[batchIndex][layer], \
					lamdas[batchIndex][layer], Zs[batchIndex][layer], fOfPrevYs[batchIndex][layer])
			else:
				Ys[batchIndex][layer] = YStep(prevWs[layer], splitAndBiasX(prevZs[batchIndex][layer-1]), mus[batchIndex][layer], fPrimeOfPrevYs[batchIndex][layer], \
					lamdas[batchIndex][layer], Zs[batchIndex][layer], fOfPrevYs[batchIndex][layer])			

	#		print("Y",layer, Ys[layer])

			if layer == 0:
				Y0s.append([Ys[batchIndex][layer]][0][0][0])
			if layer == 1:
				Y1s.append([Ys[batchIndex][layer]][0][0][0])


			# W step
			if layer == 0:
				Ws[layer] = jointWStepNew(lamdas[batchIndex][layer], prevWs[layer], Ys[batchIndex][layer], splitAndBiasedXs)
			else:
				Ws[layer] = jointWStepNew(lamdas[batchIndex][layer], prevWs[layer], Ys[batchIndex][layer], splitAndBiasX(prevZs[batchIndex][layer-1]))

	#		print("W",layer, Ws[layer])

			if layer == 0:
				W0s.append([Ws[layer]][0][0][0])
			if layer == 1:
				W1s.append([Ws[layer]][0][0][0])
			if layer == 2:
				W2s.append([Ws[layer]][0][0][0])

	#	mu = muStep(prevMu, prevZ, fOfPrevY)
	#	Z = ZStep(T, mu, prevZ, fOfPrevY)
	#	Y = YStep(prevW, X, mu, fPrimeOfPrevY, lamda, Z, fOfPrevY)	
	#	W = WStepPerfect(lamda, prevW, Y, X)

#		if i % 200 == 0:
		if False:
			output = evaluateNetwork(Ws, splitAndBiasedXs, showLayers=True)
			p.matshow(T)
			p.colorbar()
			p.show()

			p.matshow(T - output)
			p.title("output - T")
			p.colorbar()
			p.show()
		else:
			output = evaluateNetwork(Ws, splitAndBiasedXs, showLayers=False)


	#	print("output", output)
	#	print("T", T)

		diffMat = T - output

#		print(diffMat)

	#	print(T)
	#	print(output)

		error = np.trace(np.dot(np.transpose(diffMat), diffMat))

		errors.append(error)
		logErrors.append(softLog(error))

	numCorrect = 0


	p.matshow(X)
	p.colorbar()
	p.show()				
	output = evaluateNetwork(Ws, splitAndBiasedXs, showLayers=True)

#	p.matshow(output)
#	p.colorbar()
#	p.show()

	for batchIndex in range(numBatches):
		X, T = selectSubset(overallX, overallT, batchIndex, numExamples)	
	#	X = augmentWithBias(X)
	#	X, T = selectRandomSubset(overallX, overallT, numExamples)	
		voteT = vote(T)

		splitAndBiasedXs = splitAndBiasX(X)

		output = evaluateNetwork(Ws, splitAndBiasedXs)
		voteOutput = vote(output)

#		print(output)

		for i in range(numExamples):
		#	print(np.transpose(output)[i])
		#	print(np.argmax(np.transpose(output)[i]))

		#	print(np.transpose(T)[i])
		#	print(np.argmax(np.transpose(T)[i]))

		#	print("System's guess:", np.argmax(np.transpose(output)[i]), \
		#		"Correct:", np.argmax(np.transpose(T)[i]))

#			print(voteOutput[i], voteT[i])

			if voteOutput[i] == voteT[i]:
				numCorrect += 1

	accuracy = numCorrect / totalNumExamples

	print("Training Accuracy:", accuracy)

	overallXtest, overallTtest = pickle.load(open("little_mnist_one_minus_one_test.p", "rb"))

	overallTtest = repeatT(overallTtest, 5)
	overallXtest = augmentWithBias(overallXtest)

	#testOutput = evaluateNetwork(Ws, augmentWithBias(Xtest))

	numCorrect = 0
	for batchIndex in range(numBatches):
	#	X, Ttest = selectRandomSubset(Xtest, Ttest, numExamples)	
		X, Ttest = selectSubset(overallXtest, overallTtest, batchIndex, numExamples)	
	#	X = augmentWithBias(X)

		splitAndBiasedXs = splitAndBiasX(X)

		testOutput = evaluateNetwork(Ws, splitAndBiasedXs)

		voteTtest = vote(Ttest)
		voteTestOutput = vote(testOutput)

	#	print(batchIndex)
	#	print("a", testOutput.shape)
	#	print("b", Ttest.shape)


		for i in range(numExamples):
	#		print("System's guess:", np.argmax(np.transpose(testOutput)[i]), \
	#			"Correct:", np.argmax(np.transpose(Ttest)[i]))

	#		p.matshow(np.reshape(np.transpose(X)[i], (7,7)))
	#		p.show()



			if voteTestOutput[i] == voteTtest[i]:
				numCorrect += 1

	accuracy = numCorrect / totalNumExamples
	print("Test Accuracy:", accuracy)


	#	p.matshow(np.reshape(np.transpose(X)[i], (7,7)))
	#	p.show()



	#print("teacher weights", teacherWs)
	#print("student weights", Ws)

	#print("teacher output", evaluateNetwork(teacherWs, X))
	#print("student output", evaluateNetwork(Ws, X))
	#print(T)
	#print(output)
	#print(diffMat)

	#print(Ys)

	p.plot(lamda0s, label='lamda0s')
	p.plot(mu0s, label='mu0s')
	p.plot(W0s, label='W0s')
	p.plot(Y0s, label='Y0s')
	p.plot(Z0s, label='Z0s')
	p.plot(lamda1s, label='lamda1s')
	p.plot(mu1s, label='mu1s')
	p.plot(W1s, label='W1s')
	p.plot(Y1s, label='Y1s')
	p.plot(Z1s, label='Z1s')
	#p.plot(W2s, label="W2s")

	p.legend()
	p.show()

	print(time.time() - t)
	p.plot(logErrors)
	p.ylabel("Log error")
	p.xlabel("Iteration")
	p.show()	

