# import numpy as np
index = range(1520)
num_priors = 380


for index in range(1515,1520):
    i = index % 4
    c = index // 4
    d = (index // 4 )% num_priors
    pi = d * 4
    vi = pi + num_priors * 4

    print('index = {}, i = {}, c = {}, d = {}, pi = {}, vi {}'.format(index, i, c, d, pi, vi))

