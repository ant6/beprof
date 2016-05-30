import random
import urllib.request as ul
from beprof import profile, functions
import numpy as np

# todo: temp file from raw.git
url = "https://raw.githubusercontent.com/grzanka/beprof/feature/examples/examples/data1/1.dat"
# load some data and transpose to receive x=col_0, y1=col_9
data = np.loadtxt(ul.urlopen(url), dtype=np.float, delimiter=' ', usecols=(0, 9)).T
x, y1 = data[0], data[1]
# noisy parabola
y2 = [0.01 * random.uniform(-1., 1.) + 1 - i**2 for i in x]

one = profile.Profile(np.stack((x, y1), axis=1))
two = profile.Profile(np.stack((x, y2), axis=1))
# normalize profiles
one.normalize(dt=x[-1])  # from -x[-1] to x[-1]
two.normalize(1000)  # from -1000 to 1000
# smooth profiles
one.smooth(window=3)
two.smooth()
three = functions.subtract(one, two, 0.0)


# todo: remove the following debug section
from matplotlib import use
use("Qt5Agg")  # backend quick-fix
import matplotlib.pyplot as plt
a, = plt.plot(one.x, one.y, label="one")
b, = plt.plot(two.x, two.y, label="two")
c, = plt.plot(three.x, three.y, label="one - two")
plt.legend(handles=[a, b, c])
plt.show()
