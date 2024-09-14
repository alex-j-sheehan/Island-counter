import time
import pyqtgraph.examples
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtWidgets, mkQApp, QtCore
from scipy import signal


window_dim = 512

kernel = np.ones((3, 3), dtype=np.int16)
kernel[1, 1] = 0

window = QtWidgets.QMainWindow()

gr_wid = pg.GraphicsLayoutWidget(show=True)
window.setCentralWidget(gr_wid)
window.setWindowTitle('pyqtgraph example: Correlation matrix display')
window.resize(600, 500)
window.show()

def acorn(corrMatrix, placex=128, placey=128):
    adder = np.zeros((3,7))
    adder[2,1] = 1
    adder[0,1] = 1
    adder[2,0] = 1
    adder[1,3] = 1
    adder[2,4:7] = 1
    corrMatrix[placex:placex+3, placey:placey+7] = adder
    return corrMatrix

def Rpentomino(corrMatrix, placex=128, placey=128):
    adder = np.zeros((3, 3))
    adder[0, 2] = 1
    adder[:, 1] = 1
    adder[1, 0] = 1
    corrMatrix[placex:placex + 3, placey:placey + 3] = adder
    return corrMatrix

def Diehard(corrMatrix, placex=128, placey=128):
    adder = np.zeros((3, 8))
    adder[1, 0:2] = 1
    adder[2, 1] = 1
    adder[2, 5:8] = 1
    adder[0, 6] = 1
    corrMatrix[placex:placex + 3, placey:placey + 8] = adder
    return corrMatrix

def pentaDecatholon(corrMatrix, placex=128, placey=128):
    adder = np.ones((8, 3))
    adder[1, 1] = 0
    adder[6, 1] = 0
    corrMatrix[placex:placex + 8, placey:placey + 3] = adder
    return corrMatrix

def gosperGliderGun(corrMatrix, placex=128, placey=128):
    adder = np.zeros((9, 37))
    adder[4:6, 0:2] = 1
    adder[4:7, 10] = 1
    adder[7, 11] = 1
    adder[3, 11] = 1
    adder[2, 12:14] = 1
    adder[8, 12:14] = 1
    adder[5, 14] = 1
    adder[3, 15] = 1
    adder[7, 15] = 1
    adder[4:7, 16] = 1
    adder[5, 17] = 1
    adder[2:5, 20] = 1
    adder[2:5, 21] = 1
    adder[5, 22] = 1
    adder[1, 22] = 1
    adder[0:2, 24] = 1
    adder[5:7, 24] = 1
    adder[2:4, 34:36] = 1
    #print(adder[:,15:])
    corrMatrix[placex:placex + 9, placey:placey + 37] = adder
    return corrMatrix


corrMatrix = np.zeros((window_dim, window_dim), dtype=np.int16)
#corrMatrix = np.zeros((window_dim, window_dim), dtype=np.int16)
#corrMatrix[window_dim//4:window_dim*3//4, window_dim//4:window_dim*3//4] += 1#np.random.randint(0, 2, (window_dim//2, window_dim//2), dtype=np.int16)

print(corrMatrix)

corrMatrix = acorn(corrMatrix, 200, 200)
#corrMatrix = Rpentomino(corrMatrix, 200, 200)
#corrMatrix = gosperGliderGun(corrMatrix, 20, 20)

pg.setConfigOption('imageAxisOrder', 'row-major')  # Switch default order to Row-major

correlogram = pg.ImageItem()
# create transform to center the corner element on the origin, for any assigned image:
tr = QtGui.QTransform().translate(-0.5, -0.5)
correlogram.setTransform(tr)
correlogram.setImage(corrMatrix)

plotItem = gr_wid.addPlot()  # add PlotItem to the main GraphicsLayoutWidget
plotItem.invertY(True)  # orient y axis to run top-to-bottom
plotItem.setDefaultPadding(0.0)  # plot without padding data range
plotItem.addItem(correlogram)  # display correlogram

# show full frame, label tick marks at top and left sides, with some extra space for labels:
plotItem.showAxes(True, showValues=(True, True, False, False), size=20)

# define major tick marks and labels:
plotItem.getAxis('bottom').setHeight(10)  # include some additional space at bottom of figure


mkQApp("Conways Game of Life")

def update():
    global correlogram, corrMatrix
    #print(index % z.shape[0])
    #if np.sum(corrMatrix) != 0:
    neighbors = signal.convolve(corrMatrix, kernel, mode='same')
    #else:
    #    neighbors = np.zeros(shape=(window_dim,window_dim), dtype=np.int16)

    for i in range(1, corrMatrix.shape[0]-1):
        for j in range(1, corrMatrix.shape[1]-1):
             if not corrMatrix[i,j]:
                 if neighbors[i, j]==3:
                     corrMatrix[i,j] = 1
             else:
                 if neighbors[i, j]>3 or neighbors[i, j]<2:
                     corrMatrix[i, j] = 0
    correlogram.setImage(corrMatrix)



timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(30)
## Start Qt event loop
if __name__ == '__main__':
    pg.exec()









