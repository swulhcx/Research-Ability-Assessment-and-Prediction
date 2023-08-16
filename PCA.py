import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rc('font', family='Times New Roman')

def mean(arr):
    return sum(arr) / len(arr)

def std_var(arr):
    miu = mean(arr)
    var = sum((ai-miu)**2 for ai in arr)/(len(arr)-1)
    return var**0.5

data = pd.read_csv('original-new.csv')
data['x1'] = 0.4*data['Q1'] + 0.3*data['Q2'] + 0.2*data['Q3_4'] + 0.1*data['Other']
data['x2'] = data['Patent']
data['x3'] = data['Competition_award']
data['x4'] = data['Innovation_projects']


# Normalization
idxs = ['x1', 'x2', 'x3', 'x4']
for idx in idxs:
    miu = mean(data[idx])
    sigma = std_var((data[idx]))
    data[idx] = (data[idx] - miu) / sigma


# Correlation matrix
R = np.array([[0 for i in range(4)] for j in range(4)]).astype(np.float)
for i in range(4):
    for j in range(4):
        R[i][j] = sum(data[idxs[i]][k] * data[idxs[j]][k] for k in range(39)) / 38
print(R)


# Eigen
eigvals, eigvecs = np.linalg.eig(R)
eigvecs = eigvecs.transpose()
eig = [[eigvals[i], eigvecs[i]] for i in range(4)]
eig.sort(reverse = True)
eigvals, eigvecs = zip(*eig)
# print(eigvals)
# print(eigvecs)
# for i in range(4):
#     print(np.dot(R,eigvecs[i]))
#     print(eigvals[i]*eigvecs[i])

alpha = [eigvals[i]/sum(eigvals) for i in range(4)]
print('Information contribution rate:')
print(alpha)
print('Cumulative contribution rate:')
print([sum(alpha[:i]) for i in range(1,5)])

score = []
for i in range(39):
    item = np.array([data[idx][i] for idx in idxs])
    y1 = np.dot(eigvecs[0],item)
    y2 = np.dot(eigvecs[1],item)
    y3 = np.dot(eigvecs[2],item)
    Z = alpha[0]*y1 + alpha[1]*y2 + alpha[2]*y3
    score.append(Z)

# plt.hist(score, bins=12)
# plt.xlabel('Research Ability')
# plt.ylabel('Frequency')
# plt.savefig('distribution.png', dpi=600)
for item in score:
    print(item)