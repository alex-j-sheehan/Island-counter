from islands import Ocean, island_count_expectation_value_1d, island_count_expectation_value_2d
import numpy as np
import matplotlib.pyplot as plt


WIDTH = 10
HEIGHT = 2
STEPSIZE = .02
SIMULATIONS = 100


binned_samps = []
bins = []

fstep = .01
while fstep < 1:
    samps = []
    for i in range(SIMULATIONS):
        ocean = Ocean(length=HEIGHT, width=WIDTH, frequency=fstep)
        #print_ocean(ocean)
        Icount = ocean.count_distinct_islands()
        samps.append(Icount)
    bins.append(fstep)
    binned_samps.append(np.mean(samps))
    fstep+=STEPSIZE


plt.plot(bins, binned_samps, alpha=.8, color='blue')
plt.plot(bins, [island_count_expectation_value_2d(WIDTH, b) for b in bins], alpha=.5, color='red')
plt.title('Width: %d'%WIDTH)
plt.legend(['Simulation', 'Expected Value'])
plt.show()
plt.clf()


