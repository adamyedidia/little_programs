# L-8 MCS 507 Fri 14 Sep 2012 : permanent.py

"""
Given an n-by-n matrix A, the permanent of the matrix is
the sum of the products A[i,s[i]], for i = 1,2,..,n,
where the sum runs over all permutations s of (1,2,..,n).
The expansion formula for the permanent above is very similar
to the row expansion formula for the determinant, except for
the sign changes, which are absent in the permanent expansion.
The Python function permanent below takes on input
a matrix and that returns the permanent of the matrix.
An auxiliary recursive function uses the expansion formula
above to compute the permanent.
"""

import numpy as np
import itertools
def per(mtx, column, selected, prod, output=False):
    """
    Row expansion for the permanent of matrix mtx.
    The counter column is the current column, 
    selected is a list of indices of selected rows,
    and prod accumulates the current product.
    """
    if column == mtx.shape[1]:
        if output:
            print(selected, prod)
        return prod
    else:
        result = 0
        for row in range(mtx.shape[0]):
            if not row in selected:
                result = result \
                + per(mtx, column+1, selected+[row], prod*mtx[row,column])
        return result

def permanent(mat):
    """
    Returns the permanent of the matrix mat.
    """
    return per(mat, 0, [], 1)

def makeRowStochastic(mat):
    newMat = []
    for row in mat:
        newMat.append(row/sum(row))
    return np.array(newMat)

def makeColumnStochastic(mat):
    return np.transpose(makeRowStochastic(np.transpose(mat)))

def generateRandomP(n):
#    mat = np.random.uniform(size=(n, n))
#    mat = np.random.exponential(size=(n,n))
    mat = np.random.lognormal(sigma=3, size=(n,n))
#    print(mat)

    for _ in range(50):
        mat = makeRowStochastic(mat)
        mat = makeColumnStochastic(mat)

    return mat

def permuteFromList(mat, l):
    returnMat = []

    for i in l:
        returnMat.append(mat[i])

    return np.array(returnMat)

def allPermutations(n):
    return list(itertools.permutations(range(n)))

def allPermutationMats(n):
    mat = np.identity(n)
    allPerms = allPermutations(n)
    return [permuteFromList(mat, perm) for perm in allPerms]

def main():
    """
    Test on the permanent.
    """
#    dim = input('give the dimension : ')
 #   rmt = np.random.random_integers(0, 1, size=(dim, dim))
 #   print('a random 0/1-matrix :\n', rmt)
 #   print('permanent :', permanent(rmt))

    n = 3
#    A = np.identity(n)
#    A = np.array([[1,2,3],[4,5,6],[7,8,9]])

    a = 0.1
    A = np.array([[5,0,5],[0,1,0],[5,0,5]])
#    A = np.array([[1,1,1,1],[1,a,a,a],[1,a,a,a],[1,a,a,a]])
    print(permanent(A)/2**(n/2.))

    gamma = -1/2

    bestP = np.identity(n)
    maxProd = 0
    for _ in range(1000):

        P = generateRandomP(n)

        prod = 1
        for i in range(n):
            for j in range(n):
                prod *= (A[i][j]/P[i][j])**P[i][j] * (1/(1-P[i][j]))**(gamma*(1-P[i][j]))

#        print(np.sum(P, axis=1))
 #       print(np.sum(P, axis=0))

 #       print(prod)
        if prod > maxProd:
            bestP = P

#        print(P)

        maxProd = max(prod, maxProd)

    allPermMats = allPermutationMats(n)
    for P in allPermMats:
#    for P in []:
        prod = 1
        for i in range(n):
            for j in range(n):
                prod *= (A[i][j]/P[i][j])**P[i][j] * (1/(1-P[i][j]))**(gamma*(1-P[i][j]))

#        print(np.sum(P, axis=1))
 #       print(np.sum(P, axis=0))

        if prod > maxProd:
            bestP = P
 #       print(prod)
#        print(prod)
        maxProd = max(prod, maxProd)


    print(bestP)
    print(maxProd)

if __name__ == "__main__":
    main()

