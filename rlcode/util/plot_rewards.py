import numpy as np
import matplotlib.pyplot as plt
import math


tnormal, hnormal, anormal=[60,80], [40,60], [0,400]
trange, hrange, arange =120,100,1000
tscale, hscale, ascale= -1,-1,-.15

def compute_reward(rList, cNum, rScale):
    if rList[0] <= cNum <= rList[1]:
        rDist = 10
    else:     
        rNear = min(rList, key=lambda x:abs(x-cNum))
        rDist = abs(rNear - cNum) * rScale
    return rDist 

tv = [ compute_reward(tnormal, v, tscale) for v in range(-trange, trange)]
hv = [ compute_reward(hnormal, v, hscale) for v in range(-hrange, hrange)]
av = [ compute_reward(anormal, v, ascale) for v in range(-arange, arange)]
       
tx= np.linspace(-trange, trange, num=trange * 2)
hx= np.linspace(-hrange, hrange, num=hrange * 2)
ax= np.linspace(-arange, arange, num=arange * 2)

fig, (ax1, ax2, ax3) = plt.subplots(3,1)



ax1.plot(tx,tv)
ax1.set_title('T: normal range: {} - {} scale: {}'.format(tnormal[0], tnormal[1],tscale))
ax1.set_ylabel('t-rewards')

ax2.plot(hx, hv)
ax2.set_title('H: normal range: {} - {} scale: {}'.format(hnormal[0], hnormal[1],hscale))
ax2.set_ylabel('h-rewards')

ax3.plot(ax, av)
ax3.set_title('A: normal range: {} - {} scale: {}'.format(anormal[0], anormal[1],ascale))
ax3.set_ylabel('a-rewards')

fig.tight_layout()
plt.show()
