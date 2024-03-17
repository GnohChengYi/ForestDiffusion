from ForestDiffusion import ForestDiffusionModel

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pickle

keyframe_path = 'XFC4101/Preparation/keyframes.pkl'
with open(keyframe_path, 'rb') as file:
    keyframes = pickle.load(file)
print('original:', keyframes)

X = np.asarray(keyframes)
forest_model = ForestDiffusionModel(X, n_t=50, duplicate_K=100, bin_indexes=[], cat_indexes=[], int_indexes=[], diffusion_type='vp', n_jobs=-1)
keyframes_generated = forest_model.generate(batch_size=len(keyframes))
keyframes_generated = sorted(keyframes_generated, key=lambda x: x[0])
print(keyframes_generated)

def animate_keyframes(keyframes, width=500, height=500, fps=10):
    fig, ax = plt.subplots()
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.set_axis_off()

    ball, = ax.plot([], [], 'ro')

    def update(frame):
        i, x, y = frame
        ball.set_data((x,), (y,))
        return ball,

    anim = animation.FuncAnimation(fig, update, frames=keyframes, interval=1000/fps, blit=True)
    plt.show()


width = 150
height = 150
animate_keyframes(keyframes_generated, width=width, height=height)