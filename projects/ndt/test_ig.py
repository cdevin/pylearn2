import numpy as np


def entropy(data):

    n_c = np.zeros(10)
    for item in data:
        n_c[item] += 1

    entr = n_c / len(data)
    entr = entr * np.log(entr)
    entr[np.isnan(entr)] = 0
    return -entr.sum()




def ig(data, right, left):

    H = entropy(data)
    H_r = entropy(right)
    H_l = entropy(left)
    return H -(len(right)/float(len(data)) * H_r + len(left)/float(len(data)) * H_l)



data = xrange(10)
#right = []
#left = []
#for item in data:
    #right = [ite]

#data = [1, 1, 1, 2, 2, 2, 3, 3, 3,4]
##data = xrange(10)
#right = [1,1,2,3,4]
#left = [1,2,2,3,3]
##data = np.ones(10)
#print entropy(data), entropy(right), entropy(left)
#print ig(data, right, left)


#right = [1,1,1,2,2]
#left=[2,3,3,3,4]
#print '/n'
#print entropy(data), entropy(right), entropy(left)
#print ig(data, right, left)


#right = [1,1,1]
#left=[2,2,2,3,3,3,4]
#print '/n'
#print entropy(data), entropy(right), entropy(left)
#print ig(data, right, left)

right = [0]
left=[1,2,3,4,5,6,7,8,9]
print '/n'
print entropy(data), entropy(right), entropy(left)
print ig(data, right, left)

right = [0,1]
left=[2,3,4,5,6,7,8,9]
print '/n'
print entropy(data), entropy(right), entropy(left)
print ig(data, right, left)

right = [0,1,2]
left=[3,4,5,6,7,8,9]
print '/n'
print entropy(data), entropy(right), entropy(left)
print ig(data, right, left)

right = [0,1,2,3]
left=[4,5,6,7,8,9]
print '/n'
print entropy(data), entropy(right), entropy(left)
print ig(data, right, left)

right = [0,1,2,3,4]
left=[5,6,7,8,9]
print '/n'
print entropy(data), entropy(right), entropy(left)
print ig(data, right, left)

right = [0,1,2,3,4,5]
left=[6,7,8,9]
print '/n'
print entropy(data), entropy(right), entropy(left)
print ig(data, right, left)

# conclusion, if the classes are balanced you get maximum information gaine by splitting in half
