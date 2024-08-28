from islands import *
import numpy as np
import matplotlib.pyplot as plt


WIDTH = 2
HEIGHT = 2
STEPSIZE = .02
SIMULATIONS = 400


binned_samps = []
bins = []

fstep = .01
while fstep < 1:
    samps = []
    for i in range(SIMULATIONS):
        ocean = create_ocean(length=HEIGHT, width=WIDTH, frequency=fstep)
        #print_ocean(ocean)
        Icount = count_distinct_islands(ocean)
        samps.append(Icount)
    bins.append(fstep)
    binned_samps.append(np.mean(samps))
    fstep+=STEPSIZE


plt.plot(bins, binned_samps, alpha=.8, color='blue')
plt.plot(bins, [island_count_expectation_value_1d(WIDTH, b) for b in bins], alpha=.5, color='red')
plt.title('Width: %d'%WIDTH)
plt.legend(['Simulation', 'Expected Value'])
plt.show()
plt.clf()


