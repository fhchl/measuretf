import measuretf as mtf
import sounddevice as sd
import matplotlib.pyplot as plt

# Print the available audio devices together with their id
print(sd.query_devices())

# specify the device you want to use (None for default)
device = None

# generate sweep
T_sweep = 5
samplerate = int(48e3)
tfade = 0 # do not fade in or out
sweep = mtf.exponential_sweep(T_sweep, samplerate, tfade=tfade)

# measure the impulse response
gain = 0.3
impulse_response = mtf.measure_single_output_impulse_response(
    gain * sweep, samplerate, out_ch=1, in_ch=1
)

# transpose to make one graph with 96000 samples and not 96000 plots of one sample
plt.plot(impulse_response.T)
plt.show()
