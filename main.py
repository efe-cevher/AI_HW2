import numpy as np

data = np.load('data1.npy') 

probs = []

print(data)

for line in data:
    total = len(line)
    true_count = 0
    for val in line:
        if val:
            true_count+=1
    probs.append(true_count/total)

print(probs)

x = probs[3] * (1 - probs[6])

print(x)

data_traversed = []

for i in range(len(data[0])):
    values = []
    for j in range(len(data)):
        values.append(data[j][i])

    data_traversed.append(values)
        
pool = []


for line in data_traversed:
    if line[0] and line[2]:
        pool.append(line)

positive = 0
for line in pool:
    if line[3]:
        positive +=1

total1 = len(pool)
print(total1)
res1 = positive/total1

positive = 0
for line in pool:
    if line[3] and not line[6]:
        positive +=1

total1 = len(pool)
res2 = positive/total1



print(res1)
print(res2)
print(res1*res2)


