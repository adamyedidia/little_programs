import numpy as np

n = 3
numIter = 10

t = np.array([2,5,-1])
x = np.array([-1,2,4])

#W = np.random.normal(size=(n, n))
#y = np.random.normal(size=n)
#lamda = np.random.normal(size=n)

#W = np.zeros((n,n))
#W = np.identity(n)
W = np.zeros((n,n))
y = 2*np.zeros(n)
lamda = np.zeros(n)

for i in range(numIter):
	prevW = W.copy()
	prevY = y.copy()
	prevLamda = lamda.copy()

	lamda = prevLamda + (prevY - np.dot(prevW, x))
	y = t - lamda - (prevY - np.dot(prevW, x))
	W = np.outer(lamda + y, 1./x)/n
#	W = np.outer(1./x, lamda + y)

#	print(lamda+y)
#	print(np.transpose(1./x))

#	print("lamda", lamda)
#	print("y", y)
#	print("W", W)


	print(np.dot(W, x))