from scipy.special import softmax
from math import log, sqrt
import numpy as np
import matplotlib.pyplot as p
import time

t = time.time()

n = 2
numExamples = 2
numTeacherLayers = 1
numLayers = 1

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

def evaluateNetwork(ws, X):
	numLayers = len(ws)
	layerOutput = X
	for i in range(numLayers):
		if i < numLayers-1:
			layerOutput = augmentWithBias(np.vectorize(f)(np.dot(ws[i], layerOutput)))
		else:
			layerOutput = np.vectorize(f)(np.dot(ws[i], layerOutput))

	return layerOutput

def augmentWithBias(X):
#	print(X)
#	print(np.ones((1, numExamples)))

	biasAugmentedX = np.concatenate([X, np.ones((1, numExamples))], axis=0)
	return biasAugmentedX

def augmentWithZeros(X):
	biasAugmentedX = np.concatenate([X, np.zeros((1, numExamples))], axis=0)
	return biasAugmentedX


def diminishWithBias(W):
	biasDiminishedW = W[:,:-1]
	return biasDiminishedW


#X = np.random.normal(size=(n, numExamples))
X = np.array([[0,0], [0,1]])

#X = np.array([[1,-2]])
#X = np.identity(n)
#print(X.shape)

teacherWs = [np.random.normal(0,1,size=(n,n+1)) for _ in range(numTeacherLayers)]

#print(teacherWs)

#T = evaluateNetwork(teacherWs, augmentWithBias(X))
T = np.array([[1,1],[1,1]])
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



#W = np.ones((n,n))
Ws = [np.random.normal(0,1,size=(n,n+1)) for _ in range(numLayers)]
#Ws = [np.zeros((n, n+1)) for _ in range(numLayers)]
#Ws = [np.array([-1/3, 4/3]), np.array([1, 0])]

#Ys = [np.zeros((n, numExamples)) for _ in range(numLayers)]
Ys = [np.random.normal(0,1,size=(n, numExamples)) for _ in range(numLayers)]
#Ys = [np.array([[-4,3]]), np.array([[1,2]])]
#Ys = [np.array([[1,-1]]), np.array([[-1,1]])]


#Zs = [np.ones((n, numExamples)) for _ in range(numLayers)]
Zs = [np.random.normal(0,1,size=(n,numExamples)) for _ in range(numLayers)]
#Zs = [np.array([[1,2]]), np.array([[1,2]])]
#Y = T
#Z = Y

lamdas = [np.zeros((n, numExamples)) for _ in range(numLayers)]
mus = [np.zeros((n, numExamples)) for _ in range(numLayers)]

numIter = 1000
#rhalpha = 1/2
#rhalpha = 1/((numExamples*n)**numLayers)
#rhalpha = 1/(numExamples*n)
#rhalpha = 1
#rho = 1/n
#alpha = 1/n**2
rho = 1
alpha = 1/n
#rhalpha = 1e-3
#LR = 1/n

bigWeightPenalty = 0


def softLog(x):
	if x <= 0:
		return -100
	else:
		return log(x)

def lamdaStep(prevLamda, prevY, prevW, X):
#	print(prevY.shape)
#	print(prevW.shape)
#	print(X.shape)

	return prevLamda + alpha*(prevY - np.dot(prevW, X))

def muStep(prevMu, prevZ, fOfPrevY):
	return prevMu + alpha*(prevZ - fOfPrevY)

def finalZStep(T, mu, prevZ, fOfPrevY):
#	print("last Z", T - mu - rhalpha*(prevZ - fOfPrevY))

#	return T - mu - rho*(prevZ - fOfPrevY)
	if False:
		print("")
		print("-mu/rho", -mu/rho)
		print("fOfPrevY", fOfPrevY)
		print("T", T)
		print("-prevZ", -prevZ)

	return (-mu/rho + fOfPrevY + T)/2
#	return -mu/rho + fOfPrevY + T - prevZ

def ZStepNotLast(muThisLayer, fOfPrevYThisLayer, WNextLayer, lamdaNextLayer, YNextLayer, prevZThisLayer):
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

	return -muThisLayer/rho + fOfPrevYThisLayer + np.dot(np.transpose(diminishWithBias(WNextLayer)), lamdaNextLayer)/rho + \
		np.dot(np.transpose(diminishWithBias(WNextLayer)), YNextLayer - np.dot(WNextLayer, prevZThisLayer))

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


def YStep(prevW, X, mu, fPrimeOfPrevY, lamda, Z, fOfPrevY):
	if False:
		print("")
		print("Y exploration")
		print("WX term", np.dot(prevW, X))
		print("mu fprime term", mu*fPrimeOfPrevY/rho)
		print("lamda term", -lamda/rho)
		print("fPrime term", fPrimeOfPrevY*(Z - fOfPrevY))
		print("fPrimeOfPrevY", fPrimeOfPrevY)
		print("")

	return np.dot(prevW, X) + mu*fPrimeOfPrevY/rho - lamda/rho + fPrimeOfPrevY*(Z - fOfPrevY)



def WStepPerfect(lamda, prevW, Y, X):

#	print(X)

#	print(X)

	if n >= numExamples:
		gramX = np.dot(np.transpose(X), X)
	else:
		gramX = np.dot(X, np.transpose(X))

#	print(X)

	frobeniusGramX = np.sum(gramX*gramX)

#	print(frobeniusGramX)

#	print(gramX + \
#			bigWeightPenalty*frobeniusGramX*np.identity(numExamples))

	if n >= numExamples:
		pseudoInv = np.dot(np.linalg.inv(gramX + \
			bigWeightPenalty*np.identity(numExamples)), np.transpose(X))
	else:
#		print(np.dot(X, np.transpose(X)) + \
#			bigWeightPenalty*np.identity(n))
		pseudoInv = np.dot(np.transpose(X), np.linalg.inv(gramX + \
			bigWeightPenalty*np.identity(n+1)))

#	print("lamda plus Y", lamda + Y)
#	print("pseudoInv", pseudoInv)


	return np.dot((lamda/rho + Y), pseudoInv)	

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

biasAugmentedX = augmentWithBias(X)

for i in range(numIter):
	if i/numIter > percent/100 and (i-1)/numIter <= percent/100:
		print(i, "/", numIter)
		percent += 1

	prevLamdas = lamdas[:]
	prevMus = mus[:]
	prevZs = Zs[:]
	prevYs = Ys[:]
	prevWs = Ws[:]

#	print(W)

	fOfPrevYs = [np.vectorize(f)(prevY) for prevY in prevYs]
	fPrimeOfPrevYs = [np.vectorize(fPrime)(prevY) for prevY in prevYs]

	for layer in range(numLayers-1, -1, -1):

#		print(layer)

		# Lambda step
		if layer == 0:
			lamdas[layer] = lamdaStep(prevLamdas[layer], prevYs[layer], prevWs[layer], biasAugmentedX)
		else:
			lamdas[layer] = lamdaStep(prevLamdas[layer], prevYs[layer], \
				prevWs[layer], augmentWithBias(prevZs[layer-1]))

#		print("lamda",layer, lamdas[layer])

		if layer == 0:
			lamda0s.append(lamdas[layer][0][0])
		if layer == 1:
			lamda1s.append([lamdas[layer]][0][0][0])


		# Mu step
#		print(layer)
#		print(len(mus))
#		print(len(prevMus))
#		print(len(prevZs))
#		print(len(fOfPrevYs))
		mus[layer] = muStep(prevMus[layer], prevZs[layer], fOfPrevYs[layer])

#		print("mu", layer, mus[layer])

		if layer == 0:
			mu0s.append([mus[layer]][0][0][0])
		if layer == 1:
			mu1s.append([mus[layer]][0][0][0])


		# Z step
		if layer == numLayers - 1:
			Zs[layer] = finalZStep(T, mus[layer], prevZs[layer], fOfPrevYs[layer])
		else:
			Zs[layer] = ZStepNotLast(mus[layer], fOfPrevYs[layer], \
				Ws[layer+1], lamdas[layer+1], Ys[layer+1], augmentWithBias(prevZs[layer]))

#		print("Y",layer, Ys[layer])
#		print("Z",layer, Zs[layer])

		if layer == 0:
			Z0s.append([Zs[layer]][0][0][0])
		if layer == 1:
			Z1s.append([Zs[layer]][0][0][0])



		# Y step
		if layer == 0:
			Ys[layer] = YStep(prevWs[layer], biasAugmentedX, mus[layer], fPrimeOfPrevYs[layer], \
				lamdas[layer], Zs[layer], fOfPrevYs[layer])
		else:
			Ys[layer] = YStep(prevWs[layer], augmentWithBias(prevZs[layer-1]), mus[layer], fPrimeOfPrevYs[layer], \
				lamdas[layer], Zs[layer], fOfPrevYs[layer])			

#		print("Y",layer, Ys[layer])

		if layer == 0:
			Y0s.append([Ys[layer]][0][0][0])
		if layer == 1:
			Y1s.append([Ys[layer]][0][0][0])


		# W step
		if layer == 0:
			Ws[layer] = WStepPerfect(lamdas[layer], prevWs[layer], Ys[layer], biasAugmentedX)
		else:
			Ws[layer] = WStepPerfect(lamdas[layer], prevWs[layer], Ys[layer], augmentWithBias(prevZs[layer-1]))

#		print("W",layer, Ws[layer])

		if layer == 0:
			W0s.append([Ws[layer]][0][0][0])
		if layer == 1:
			W1s.append([Ws[layer]][0][0][0])


#	mu = muStep(prevMu, prevZ, fOfPrevY)
#	Z = ZStep(T, mu, prevZ, fOfPrevY)
#	Y = YStep(prevW, X, mu, fPrimeOfPrevY, lamda, Z, fOfPrevY)	
#	W = WStepPerfect(lamda, prevW, Y, X)

	
	output = evaluateNetwork(Ws, biasAugmentedX)
	
#	print("output", output)
#	print("T", T)

	diffMat = T - output

#	print(T)
#	print(output)

	error = np.trace(np.dot(np.transpose(diffMat), diffMat))/n**2

	errors.append(error)
	logErrors.append(softLog(error))

#print("teacher weights", teacherWs)
#print("student weights", Ws)

#print("teacher output", evaluateNetwork(teacherWs, X))
#print("student output", evaluateNetwork(Ws, X))
#print(T)
#print(output)
print(diffMat)

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

p.legend()
p.show()

print(time.time() - t)
p.plot(logErrors)
p.ylabel("Log error")
p.xlabel("Iteration")
p.show()	

