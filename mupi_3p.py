from __future__ import division
from math import sqrt

# prevPis is an array of size k-1
def findPk(k, prevPis):
    sumPrevPis = sum(prevPis)
    oneMinusEtc = 1-sumPrevPis

    winSquares = sum([i*i for i in prevPis])
    loseOneMinusEtc = sum([i*oneMinusEtc for i in prevPis])

    loseBothBelow = 0
    for i in range(k):
        for j in range(i+1, k):
            loseBothBelow += prevPis[i] * prevPis[j]

    a = 2
    b = -3 + 3*sumPrevPis
    c = winSquares + oneMinusEtc*oneMinusEtc - \
        loseOneMinusEtc - loseBothBelow

#    print winSquares, oneMinusEtc*oneMinusEtc, loseOneMinusEtc, loseBothBelow


#    print "sum", sumPrevPis
#
#    print "k", k+1


#    print "a", a
#    print "b", b
#    print "c", c

#    print "b2", b*b
#    print "4ac", 4*a*c


    result = (-b - sqrt(b*b-4*a*c))/(2*a)

    print "result", result

    assert result >= 0
    assert result < 1

    return (-b - sqrt(b*b-4*a*c))/(2*a)

prevPis = []
for k in range(100):
    prevPis.append(findPk(k, prevPis))

print prevPis
