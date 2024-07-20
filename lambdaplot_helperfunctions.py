
import numpy as np

import matplotlib.pyplot as plt

from analyse_DCT import *


def lambda_grid(ax,N, N_minor_ticks = 1, alpha = .1):

    locs_ = ax.get_xticks()

    spacing = (locs_[1]-locs_[0])/N_minor_ticks

    found = 0
    for jj in range(len(locs_)-1,0-1,-1):
         if locs_[jj] <= N and found !=1:
              last_tick = locs_[jj]
              found = 1

    last_tick = locs_[-1]

    for y0 in locs_:
            for y1 in np.arange(N_minor_ticks):
                x0 = y0+y1*spacing

                if 0<=x0<=N: 
                    if y1 == 0:
                        ax.plot( [x0,(x0+N)/2],[0,N-x0] , "-", color = 'black', alpha=alpha)
                    else:
                        ax.plot( [x0,(x0+N)/2],[0,N-x0] , "-", color = 'black', alpha=alpha/2)
                if 0<= last_tick-x0 <= N:
                    if y1 == 0:
                        ax.plot( [(last_tick-x0)/2,last_tick-x0],[last_tick-x0,0] , "-", color = 'black', alpha=alpha)
                    else:
                        ax.plot( [(last_tick-x0)/2,last_tick-x0],[last_tick-x0,0] , "-", color = 'black', alpha=alpha/2)

         
