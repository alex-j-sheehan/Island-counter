from islands import Ocean
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, product
from sympy import cos, simplify, poly, expand, symbols
import pandas as pd


WIDTH = 3
HEIGHT = 3

width = WIDTH
length = HEIGHT

def create_every_ocean(length=2, width=2):
    """
    Sum and count islands for all possible oceans of size length and width
    I wouldn't use this with sizes greater than something, otherwise bad stuff will happen to your computer
    """

    cells = length * width
    elements = 2 ** cells
    ocean_object = Ocean(2, 2, .1)
    if (length * width) > 20:
        print('uh oh....')
        print('gonna have %d elements... x3' % elements)
        print('maybe hit ctrl+c before this thing explodes')
    all_the_islands = []
    all_the_landmass = []

    for bi_order in product([0, 1], repeat=cells):
        ocean = np.array(bi_order).reshape((length, width))
        all_the_landmass.append(sum(bi_order))
        ocean_object.set_ocean(ocean, length, width)
        all_the_islands.append(ocean_object.count_distinct_islands())

    return all_the_landmass, all_the_islands


def simplify_probability_distribution(landmass, island_count, max_landmass, verbose=True, progprint=True):
    p = symbols('p')
    flag = 0
    size = len(landmass)
    for i in range(size):
        lm, ic = landmass[i], island_count[i]
        wc = max_landmass - lm
        if ic!=0:
            if verbose:
                print('Landmass: %d out of %d' % (lm, lm + wc))
                print('Island Count: %d' % (ic))
                print((p ** wc) * (1 - p) ** lm)
                print(poly(expand(ic * ((p ** wc) * (1 - p) ** lm))))
            if not flag:
                term = poly(expand(ic * ((p ** wc) * (1 - p) ** lm)))
                flag = 1
            else:
                term += poly(expand(ic * ((p ** wc) * (1 - p) ** lm)))
        elif verbose:
            print('no islands...')
        if progprint and not i%100:
            print('%d done out of %d'%(i, size))

    return term


def simplify_probability_distribution_no_ic(landmass, island_count, max_landmass, verbose=True, progprint=True):
    p = symbols('p')
    flag = 0
    size = len(landmass)
    for i in range(size):
        lm, ic = landmass[i], island_count[i]
        wc = max_landmass - lm
        if ic!=0:
            if verbose:
                print('Landmass: %d out of %d' % (lm, lm + wc))
                print('Island Count: %d' % (ic))
                print((p ** wc) * (1 - p) ** lm)
                print(poly(expand(((p ** wc) * (1 - p) ** lm))))
            if not flag:
                term = poly(expand(((p ** wc) * (1 - p) ** lm)))
                flag = 1
            else:
                term += poly(expand(((p ** wc) * (1 - p) ** lm)))
        elif verbose:
            print('no islands...')
        if progprint and not i%100:
            print('%d done out of %d'%(i, size))

    return term


#all_the_landmass, all_the_islands = create_every_ocean(3, 3)
#print('got landmasses')
#func = simplify_probability_distribution(all_the_landmass, all_the_islands, 3 * 3, verbose=False)

funcs = []
for w1 in range(1,6):
    holder = {}
    for h1 in range(1,4):
        all_the_landmass, all_the_islands = create_every_ocean(w1, h1)
        print('got landmasses')
        func = simplify_probability_distribution(all_the_landmass, all_the_islands, w1 * h1, verbose=False)
        holder[h1] = func
    funcs.append(holder)

df = pd.DataFrame.from_records(funcs)
df.index = range(1,6)

x = np.arange(0,1,.01)
for i in range(1, 4):
    plt.plot(x, [df[i][i].eval(x_) for x_ in x])
    plt.show()
    plt.clf()

plt.plot(x, [func.eval(x_) for x_ in x])
plt.show()
plt.clf()



all_the_landmass, all_the_islands = create_every_ocean(3, 3)
print('got landmasses')
func = simplify_probability_distribution(all_the_landmass, all_the_islands, 3 * 3, verbose=True)

