#%%

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation

import os
project_path = os.path.dirname(__file__)
animation_folder = project_path + '\\' + 'animations'

if not os.path.exists(animation_folder):
    os.mkdir(animation_folder)


def discrete_laplacian(M):
    return (np.roll(M, -1, axis=1) +
            np.roll(M,  1, axis=1) +
            np.roll(M, -1, axis=0) +
            np.roll(M,  1, axis=0) -
            4*M)

def update_A(A, B, Da, f, delta_t):
    timestep = Da*discrete_laplacian(A) - A*B**2 + f*(1 - A)
    
    return A + timestep*delta_t
    
def update_B(A, B, Db, f, k, delta_t):
    timestep = Db*discrete_laplacian(B) + A*B**2 - (k + f)*B
    
    return B + timestep*delta_t
    
def grey_scott(A, B, Da, Db, f, k, delta_t):
    A = update_A(A, B, Da, f, delta_t)
    B = update_B(A, B, Db, f, k, delta_t)
    return A, B
    

def initial_configuration_square_middle(N, random_influence=0.2):    
    # We start with a configuration where on every grid cell 
    # there's a lot of chemical A, so the concentration is high
    A = (1-random_influence) * np.ones((N,N)) + random_influence * np.random.random((N,N))
    
    # Let's assume there's only a bit of B everywhere
    B = random_influence * np.random.random((N,N))
    
    # Now let's add a disturbance in the center
    middle = N//2
    r = int(N/10.0)
    
    middle_left  = middle - r
    middle_right = middle + r
    
    A[middle_left:middle_right, middle_left:middle_right] = 0.50
    B[middle_left:middle_right, middle_left:middle_right] = 0.25
    
    return A, B

def initial_configuration_several_squares(N, N_squares, random_influence=0.2):    
    # We start with a configuration where on every grid cell 
    # there's a lot of chemical A, so the concentration is high
    A = (1-random_influence) * np.ones((N,N)) + random_influence * np.random.random((N,N))
    
    # Let's assume there's only a bit of B everywhere
    B = random_influence * np.random.random((N,N))
    
    
    
    
    # # Now let's add a disturbance in the center
    # middle = N//4
    # r = int(N/10.0)
    
    # middle_left = middle - r
    # middle_right = middle +r
    
    # A[middle_left:middle_right, middle_left:middle_right] = 0.50
    # B[middle_left:middle_right, middle_left:middle_right] = 0.25
    
    return A, B


#### TODO
# 1. Implement Grey-Scott in C
# 2. Save A and B data and run again!
# 3. 


animation_name = 'test'
saved_animation_folder = animation_folder + '\\' + animation_name

if not os.path.exists(saved_animation_folder):
    os.mkdir(saved_animation_folder)
    


## Parameters for simulation!
# update in time
delta_t = 1.0

# Diffusion coefficients
Da = 0.16
Db = 0.08

# define feed/kill rates
f = 0.060
k = 0.062

# Grid size
grid_size = 200
x = np.arange(-grid_size/2, grid_size/2 + 1, 1)
y = x

X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

# simulation steps
N = 100
only_save_every = 10
print_every     = 1000

A, B = initial_configuration_square_middle(grid_size)

plt.figure(1)
plt.imshow(A)

#%%

animation_images = []
fig = plt.figure()
timetrace = []

for t in range(N):
    
    A, B = grey_scott(A, B, Da, Db, f, k, delta_t)
    
    if t % only_save_every == 0:
        image = plt.imshow(A, animated=True, origin='lower')
        animation_images.append([image])

    if t % print_every == 0:
        print(t)
        

ani = animation.ArtistAnimation(fig, animation_images, interval=4, blit=True,
                                repeat_delay=1000)


#%%

ani.save('first_reaction_jox.gif')

#%%

plt.figure(figsize=[6, 6])
x = np.arange(0, 100, 0.00001)
y = np.sin(np.pi*x/8)

plt.plot(y)
plt.axis('off')
plt.gca().set_position([0, 0, 1, 1])

plt.savefig('test.svg' , format='svg')

#%%
