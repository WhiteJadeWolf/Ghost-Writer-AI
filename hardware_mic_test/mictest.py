import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# audio stream parameters
SAMPLE_RATE = 44100 # samples per second
WINDOW_SIZE = 2000 # number of audio points to show on screen at once
plot_data = np.zeros((WINDOW_SIZE, 1))

# setting up plot window
fig, ax = plt.subplots()
line = ax.plot(plot_data)[0]
ax.set_ylim([-0.5, 0.5]) # scale for audio volume range
ax.set_title("Live Audio Waveform")
ax.axis('off') # hide axes for cleaner look

# this function gets called automatically every time the mic hears a sound chunk
def audio_callback(indata, frames, time, status):
    global plot_data
    plot_data = np.roll(plot_data, -frames, axis=0) # shift old data out
    plot_data[-frames:] = indata # slide new data in at the end for the animation
    
# this function updates the graph animation frame by frame
def update_plot(frame):
    line.set_ydata(plot_data[:, 0]) # update the line with the latest audio data
    return line, # returning line object to let mplib know which objects were modified so as to update the plot efficiently

# starting the live mono-audio microphone stream
stream = sd.InputStream(channels=1, samplerate=SAMPLE_RATE, device=1, callback=audio_callback)
with stream:
    ani = FuncAnimation(fig, update_plot, interval=30, blit=True, cache_frame_data=False) # start the animation loop
    plt.show()