from islands import *
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, product
from sympy import cos, simplify, poly, expand, symbols


WIDTH = 4
HEIGHT = 4

width = WIDTH
length = HEIGHT

def create_every_ocean(length=2, width=2):
    """
    Sum and count islands for all possible oceans of size length and width
    I wouldn't use this with sizes greater than something, otherwise bad stuff will happen to your computer
    """
    cells = length*width
    elements = 2 ** cells

    if (length*width) > 20:
        print('uh oh....')
        print('gonna have %d elements... x3'%elements)
        print('maybe hit ctrl+c before this thing explodes')
    all_the_islands = []
    all_the_landmass = []

    for bi_order in product([0, 1], repeat=cells):
        ocean = []
        holder = []
        for ind in range(cells):
            if not (ind)%width:
                if len(holder):
                    ocean.append(holder)
                holder = [bi_order[ind]]
            else:
                holder.append(bi_order[ind])
        ocean.append(holder)
        all_the_landmass.append(sum(bi_order))
        all_the_islands.append(count_distinct_islands(ocean))
    return all_the_landmass, all_the_islands


def simplify_probability_distribution(landmass, island_count, max_landmass):
    p = symbols('p')
    flag = 0
    for i in range(len(landmass)):
        lm, ic = landmass[i], island_count[i]
        wc = max_landmass - lm
        #print(ic)
        #print(lm)
        #print(wc)
        #print('-'*20)
        if ic!=0:
            if not flag:
                term = poly(expand(ic * ((p ** wc) * (1 - p) ** lm)))
                flag = 1
            else:
                term += poly(expand(ic * ((p ** wc) * (1 - p) ** lm)))
            #print(term)
        else:
            print('no islands...')
        if not i%100:
            pass
            #print('%d done out of %d'%(i, len(landmass)))
    return term


funky = []
for w in range(2,9):
    all_the_landmass, all_the_islands = create_every_ocean(w, HEIGHT)
    print('got landmasses')

    func = simplify_probability_distribution(all_the_landmass, all_the_islands, w * HEIGHT)
    funky.append(func)
    print(func)

    print(len(all_the_landmass))
    print(w * HEIGHT)
    print('width: %d \n height: %d' % (w, HEIGHT))



