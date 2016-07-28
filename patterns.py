import numpy as np
import matplotlib.pyplot as plt

def run():
    words = ['go', 'back', 'right', 'left', 'stop']

    for w in words:
        f = open(w + '.bin', 'rb')
        data = f.readlines()

        plt.figure()
        plt.title(w)
        for i in range(np.shape(data)[0]):
            data2 = map(float, data[i].split())
            plt.plot(np.arange(0, len(data2)/36., 1/36.), data2)

        f.close()

    plt.show()

run()