from scipy.special import softmax
from math import log
import numpy as np
import matplotlib.pyplot as p
import time

t = time.time()

n = 50
numExamples = int(n)


def relu(x):
	return x*(x>=0)

#T = np.array([[1]])
#T = np.random.exponential(size=(n, numExamples))
#T = np.vectorize(relu)(np.random.normal(size=(n, numExamples)))
#T = np.identity(n)
T = np.vectorize(relu)(np.random.normal(size=(n, numExamples)))

print(T[0][0])

#print(T)

#sX = np.zeros((n, numExamples))

#X = np.transpose(np.array([[0,0],[1,1]]))

X = np.random.normal(size=(n, numExamples))
#X = T
#X = -T

#print(X[0][0])


#W = np.ones((n,n))
#W = np.random.normal(0,0.01,size=(n,n))
W = np.zeros((n, n+1))

Y = np.zeros((n, numExamples))
Z = np.zeros((n, numExamples))
#Y = T
#Z = Y

lamda = np.zeros((n, numExamples))
mu = np.zeros((n, numExamples))

numIter = 100

#rhalpha = 1
rho = 1
alpha = 1/n
#rhalpha = 1
#LR = 1/n

def softLog(x):
	if x <= 0:
		return -100
	else:
		return log(x)

def softPlus(x):
	return np.logaddexp(0, x)	

def softPlusPrime(x):
	return softmax([x, 0])[0]

def reluPrime(x):
	return 1*(x>=0)

def lamdaStep(prevLamda, prevY, prevW, X):
	return prevLamda + alpha*(prevY - np.dot(prevW, X))

def muStep(prevMu, prevZ, fOfPrevY):
	return prevMu + alpha*(prevZ - fOfPrevY)

def ZStep(T, mu, prevZ, fOfPrevY):
	return T - mu - rhalpha*(prevZ - fOfPrevY)

def ZStepAlternate(T, mu, prevZ, fOfPrevY):
	return -mu + fOfPrevY + rhalpha*(T - prevZ)

def ZStepAlternate2(T, mu, prevZ, fOfPrevY):
	return (-mu + rho*fOfPrevY + rho*(T - prevZ))/rho

def ZStepAlternate3(T, mu, prevZ, fOfPrevY):
	return (-mu/rho + fOfPrevY + T)/2

def YStep(prevW, X, mu, fPrimeOfPrevY, lamda, Z, fOfPrevY):
	return np.dot(prevW, X) + mu*fPrimeOfPrevY - lamda + rhalpha*fPrimeOfPrevY*(Z - fOfPrevY)

def YStepAlternate(prevW, X, mu, fPrimeOfPrevY, lamda, Z, fOfPrevY):
	return (rho*np.dot(prevW, X) + mu*fPrimeOfPrevY - lamda + rho*fPrimeOfPrevY*(Z - fOfPrevY))/rho


def WStepPerfect(lamda, prevW, Y, X):

#	print(X)

#	print(np.dot(np.transpose(X), X))
#	print(np.dot(X, np.transpose(X)))

	if n >= numExamples:
		pseudoInv = np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X))
	else:
		pseudoInv = np.dot(np.transpose(X), np.linalg.inv(np.dot(X, np.transpose(X))))

	return np.dot((lamda + Y), pseudoInv)	

def WStepAlternate(lamda, prevW, Y, X):

#	print(X)

#	print(np.dot(np.transpose(X), X))
#	print(np.dot(X, np.transpose(X)))

	if n >= numExamples:
		pseudoInv = np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X))
	else:
		pseudoInv = np.dot(np.transpose(X), np.linalg.inv(np.dot(X, np.transpose(X))))

	return np.dot((lamda/rho + Y), pseudoInv)	

errors = []
logErrors = []

f, fPrime = relu, reluPrime
#f, fPrime = softPlus, softPlusPrime

percent = 0


biasAugmentedX = np.concatenate([X, np.ones((1, numExamples))], axis=0)
#biasAugmentedX = X


for i in range(numIter):
	if i/numIter > percent/100 and (i-1)/numIter <= percent/100:
#		print(i, "/", numIter)
		percent += 1

	prevLamda = lamda
	prevMu = mu
	prevZ = Z
	prevY = Y
	prevW = W

#	print(W)

	fOfPrevY = np.vectorize(f)(prevY)
	fPrimeOfPrevY = np.vectorize(fPrime)(prevY)

	lamda = lamdaStep(prevLamda, prevY, prevW, biasAugmentedX)
	mu = muStep(prevMu, prevZ, fOfPrevY)
#	Z = ZStep(T, mu, prevZ, fOfPrevY)
#	Z = ZStepAlternate(T, mu, prevZ, fOfPrevY)
#	Y = YStep(prevW, biasAugmentedX, mu, fPrimeOfPrevY, lamda, Z, fOfPrevY)	
#	W = WStepPerfect(lamda, prevW, Y, biasAugmentedX)

	Z = ZStepAlternate3(T, mu, prevZ, fOfPrevY)
	Y = YStepAlternate(prevW, biasAugmentedX, mu, fPrimeOfPrevY, lamda, Z, fOfPrevY)	
	W = WStepAlternate(lamda, prevW, Y, biasAugmentedX)


	diffMat = T - f(np.dot(W, biasAugmentedX))

	error = np.trace(np.dot(np.transpose(diffMat), diffMat))/n

	errors.append(error)
	logErrors.append(softLog(error))

print(time.time() - t)
p.plot(logErrors)
p.ylabel("Log error")
p.xlabel("Iteration")
p.show()	

