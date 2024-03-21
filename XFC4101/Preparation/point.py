import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle


def generate_keyframes(num_frames=100, initial_position=(0, 0), velocity=(1, 1)):
    keyframes = []
    x = initial_position[0]
    y = initial_position[1]

    for i in range(num_frames):
        keyframes.append((i, x, y))
        x += velocity[0]
        y += velocity[1]

    return keyframes


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

    output_file = 'animation.mp4'
    anim.save(output_file, writer='ffmpeg')
    print(f'Animation saved to {output_file}')


num_frames = 100
initial_position = (0, 0)
velocity = (1, 1)
keyframes = generate_keyframes(num_frames, initial_position=initial_position, velocity=velocity)


width = 150
height = 150
animate_keyframes(keyframes, width=width, height=height)


keyframe_path = 'XFC4101/Preparation/keyframes.pkl'
with open(keyframe_path, 'wb') as file:
    pickle.dump(keyframes, file)

with open(keyframe_path, 'rb') as file:
    loaded_keyframes = pickle.load(file)
    
print(loaded_keyframes)