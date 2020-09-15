import numpy as np

n = 3
numIter = 10

#T = np.transpose(np.array([[4,1,3], [1, 10, 4]]))
#X = np.transpose(np.array([[2,3,-1], [4, 5, 8]]))

#T = np.array([4,4])
#X = np.array([2,1])

#listOfTs = [np.array([5,1])]
#listOfXs = [np.array([-3,4])]

numExamples = 2

#W = np.random.normal(size=(n, n))
#y = np.random.normal(size=n)
#lamda = np.random.normal(size=n)

W = np.ones((n,n))
#W = np.identity(n)
#W = np.zeros((n,n))
Y = np.zeros((n, numExamples))
#Y = T
lamda = np.zeros((n, numExamples))

def lamdaStep(prevLamda, prevY, prevW, X):
	return prevLamda + (Y - np.dot(W, X))

def lamdaStepOld(prevLamda, prevY, prevW, listOfXs):
	lamdaTotal = np.zeros(n)
	for x in listOfXs:
		lamdaTotal += prevLamda + (prevY - np.dot(prevW, x))

	return lamdaTotal/numExamples

def yStep(T, lamda, prevY, prevW, X):
	return T - lamda - (prevY - np.dot(prevW, X))

def yStepOld(listOfTs, lamda, prevY, prevW, listOfXs):
	yTotal = np.zeros(n)
	for j in range(numExamples):
		t = listOfTs[j]
		x = listOfXs[j]

		yTotal += t - lamda - (prevY - np.dot(prevW, x))

	return yTotal/numExamples

def WStep(lamda, Y, X):
	pseudoInv = np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X))
#	print()
	return np.dot((lamda + Y), pseudoInv)

def WStepOld(lamda, y, listOfXs):
	WTotal = np.zeros((n,n))
	for j in range(numExamples):
		x = listOfXs[j]

		WTotal += np.outer(lamda + y, 1./x)/n

	return WTotal/numExamples

def WStepNew(lamda, prevW, Y, X):
	update = np.outer(lamda, )

for i in range(numIter):
	prevW = W
	prevY = Y
	prevLamda = lamda

	W = WStep(lamda, Y, X)
	lamda = lamdaStep(prevLamda, prevY, W, X)
	Y = yStep(T, lamda, prevY, W, X)
#	W = np.outer(1./x, lamda + y)/n

#	print(lamda+y)
#	print(np.transpose(1./x))


	print("W", W)
	print("lamda", lamda)
	print("Y", Y)

	print(np.dot(W, X))