import pickle
import matplotlib.pyplot as p
import numpy as np
#import mnist_loader
import sys
import h5py

startFromScratch = False
convertFromPython27 = False
loadFromPickleFile = True


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


def batchAndDifferentiate(arr, listOfResponses):
    dim  = len(arr.shape)

    assert dim == len(listOfResponses)

    # batch things
#    print "Batching..."
    for i in range(dim):
        arr = batchArrayAlongAxis(arr, i, listOfResponses[i][0])

#    viewFrame(arr, 1e0, False)

    # take gradients
#    print "Differentiating..."
    for i in range(dim - 1, -1, -1):
        if listOfResponses[i][1]:
            arr = np.gradient(arr, axis=i)

#            viewFrame(arr, 3e2, True)

#    arr = blur2DImage(arr, 5)

#    viewFrame(arr, 3e2, True)

    return arr

#print(training_data)

if startFromScratch:
	training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

	pickle.dump((training_data, validation_data, test_data), open("big_mnist.p", "wb"))
	sys.exit()

if convertFromPython27:
	data = pickle.load(open("big_mnist.p", "rb"), encoding="latin1")
	pickle.dump(data, open("big_mnist.p", "wb"))

if loadFromPickleFile:
	training_data, validation_data, test_data = pickle.load(open("big_mnist.p", "rb"))

	numDataPoints = len(training_data)

	X = []
	T = []

	batchedTrainingData = []
	batchedTestData = []




	for i in range(numDataPoints):
		x = np.reshape(training_data[i][0], (28, 28))
#		batchedX = batchAndDifferentiate(x, [(4, False), (4, False)])
		batchedX = x

	#	p.matshow(x)
	#	p.show()
	#	p.matshow(batchedX)
	#	p.show()
		t = [2*(i[0]-0.5) for i in training_data[i][1]]
#		print(t)

		flattenedBatchedX = batchedX.flatten()

		batchedTrainingData.append((np.reshape(flattenedBatchedX, (len(flattenedBatchedX), 1)), training_data[i][1]))
	
		X.append(batchedX.flatten())
		T.append(t)

#	f = h5py.File("little_mnist.hdf5", "w")
	f = h5py.File("big_mnist.hdf5", "w")
	f['x'] = np.transpose(np.array(X))
	f['t'] = np.transpose(np.array(T))


#	pickle.dump((np.transpose(np.array(X)), np.transpose(np.array(T))), open("little_mnist_one_minus_one.p", "wb"))
#	pickle.dump((np.transpose(np.array(X)), np.transpose(np.array(T))), open("little_mnist_one_minus_one.p", "wb"))


	X = []
	T = []


	for i in range(len(test_data)):
		x = np.reshape(test_data[i][0], (28, 28))
#		batchedX = batchAndDifferentiate(x, [(4, False), (4, False)])
		batchedX = x

	#	p.matshow(x)
	#	p.show()
	#	p.matshow(batchedX)
	#	p.show()
#		print(test_data[i][1])

#		nielsenT = [0]*10
#		nielsenT[test_data[i][1]] = 1

		flattenedBatchedX = batchedX.flatten()


		batchedTestData.append((np.reshape(flattenedBatchedX, (len(flattenedBatchedX), 1)), test_data[i][1]))

		t = [-1]*10
		t[test_data[i][1]] = 1

		#t = [2*(i[0]-0.5) for i in test_data[i][1]]
#		print(t)

#		print(t)

		X.append(batchedX.flatten())
		T.append(t)

	f['x_test'] = np.transpose(np.array(X))
	f['t_test'] = np.transpose(np.array(T))

#	pickle.dump((batchedTrainingData, None, batchedTestData), open("big_batched_mnist_for_comparison.p", "wb"), protocol=2)

#	pickle.dump((np.transpose(np.array(X)), np.transpose(np.array(T))), open("big_mnist_one_minus_one_test.p", "wb"))

