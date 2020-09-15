from math import log
import numpy as np
import matplotlib.pyplot as p
import time

t = time.time()

def softLog(x):
	if x <= 0:
		return -100
	else:
		return log(x)

n = 500

#T = np.array([4,4,7])
#X = np.array([2,8,1])


#T = np.transpose(np.array([[4,1,3], [1, 10, 4], [4,2,-5],[3,0,-1]]))
#X = np.transpose(np.array([[2,3,-1], [4, 5, 8]]))
numExamples = int(n)
#numExamples = 4
#T = np.random.normal(size=(n, numExamples))
#T = np.array([[1]])
# = np.ones((n, numExamples))
#X = T


T = np.random.normal(size=(n, numExamples))
X = np.random.normal(size=(n, numExamples))




W = np.ones((n,n))
#W = np.identity(n) + np.random.normal(0, 1e-5, size=(n,n))
#W = np.zeros((n,n))
Y = np.zeros((n, numExamples))
#Y = T
lamda = np.zeros((n, numExamples))

numIter = 10

rhalpha = 1/n
LR = 1/n

def lamdaStep(prevLamda, prevY, prevW, X):
	return prevLamda + rhalpha*(prevY - np.dot(prevW, X))

def yStep(T, lamda, prevY, prevW, X):
	return (T - lamda - rhalpha*(prevY - np.dot(prevW, X)))
#	return T
#	return prevY + np.dot(prevW, X) - lamda + T

def WStepOld(lamda, prevW, Y, X):
	update = np.outer(lamda, X) + np.outer(np.dot(prevW, X) - Y, X)

	return prevW - update/np.dot(X, X)

def WStep(lamda, prevW, Y, X):
#	print(np.dot(lamda, np.transpose(X)))
#	print(np.dot(np.dot(prevW, X) - Y, np.transpose(X)))

	update = -np.dot(lamda, np.transpose(X)) + np.dot(np.dot(prevW, X) - Y, np.transpose(X))

	return prevW - LR*update/np.trace(np.dot(np.transpose(X), X))

def WStepPerfect(lamda, prevW, Y, X):

	if n > numExamples:
		pseudoInv = np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X))
	else:
		pseudoInv = np.dot(np.transpose(X), np.linalg.inv(np.dot(X, np.transpose(X))))

	return np.dot((lamda + Y), pseudoInv)

errors = []
logErrors = []

for i in range(numIter):
#	print("W", W)
#	print("lamda", lamda)
#	print("Y", Y)


	prevW = W
	prevY = Y
	prevLamda = lamda

	lamda = lamdaStep(prevLamda, Y, W, X)
#	print(lamda)
	Y = yStep(T, lamda, prevY, W, X)
	W = WStepPerfect(lamda, prevW, Y, X)

#	print("lamda", lamda)
#	print("Y", Y)


#	print("lamda", np.transpose(lamda))
#	print("Y", np.transpose(Y))
#	W = np.outer(1./x, lamda + y)/n

#	print(lamda+y)
#	print(np.transpose(1./x))



#	print(np.transpose(np.dot(W, X)))

	diffMat = T - np.dot(W, X)
#	print("diffMat", diffMat)

	error = np.trace(np.dot(np.transpose(diffMat), diffMat))

	errors.append(error)
	logErrors.append(softLog(error))



print(time.time() - t)
p.plot(logErrors)
p.ylabel("Log error")
p.xlabel("Iteration")
p.show()
