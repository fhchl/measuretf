
import measuretf as mtf
import numpy as np
import matplotlib.pyplot as plt
import measuretf.signals

class Test:
    def test_coherence_from_averages(self):
        snr = 1
        x = mtf.signals.white_noise(1, 1000, tfade=None, flim=None, noise='MLS')

        x /= x.std()
        print(x.std()**2)
        y = np.random.normal(scale=1/snr, size=(1000, *x.shape)) + x[None]

        plt.plot(x)
        plt.figure()
        plt.plot(mtf.coherence_from_averages(x[None], y))
        plt.show()
        assert False
