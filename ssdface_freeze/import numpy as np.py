import numpy as np


num = [3,5,6,9]
split_param = []
split_param.append(len(num[0]))
for layer_num in num[1:-2]:
    split_param.append(layer_num + split_param[-1])
print(split_param)


to_split = np.range(23)
split = np.split(to_split, split_param)
print(split)