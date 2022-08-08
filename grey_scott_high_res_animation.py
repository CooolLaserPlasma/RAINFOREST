#%%

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp2d

import matplotlib.animation as animation

import os
project_path = os.path.dirname(__file__)
animation_folder = project_path + '\\' + 'animations'

if not os.path.exists(animation_folder):
    os.mkdir(animation_folder)


animation_file = animation_folder + '\\' + 'highres_animation_part_3.mp4'
matrix_file_A = animation_folder + '\\' + 'A_array_3'
matrix_file_B = animation_folder + '\\' + 'B_array_3'

matrix_loadfile_A = animation_folder + '\\' + 'A_array_2.npy'
matrix_loadfile_B = animation_folder + '\\' + 'B_array_2.npy'

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
    

def initial_configuration_square_middle(Nx, Ny, random_influence=0.2):    
    # We start with a configuration where on every grid cell 
    # there's a lot of chemical A, so the concentration is high
    A = (1-random_influence) * np.ones((Nx,Ny)) + random_influence * np.random.random((Nx,Ny))
    
    # Let's assume there's only a bit of B everywhere
    B = random_influence * np.random.random((Nx,Ny))
    
    # Now let's add a disturbance in the center
    middle_x = Nx//2
    middle_y = Ny//2
    r = int(Nx/20.0)
    
    middle_left_x  = middle_x - r
    middle_right_x = middle_x + r
    
    middle_left_y  = middle_y - r
    middle_right_y = middle_y + r
    # A[middle_left-100:middle_right-80, middle_left-100:middle_right-80] = 0.50
    # B[middle_left-100:middle_right-80, middle_left-100:middle_right-80] = 0.25
    
    # A[middle_left+80:middle_right+100, middle_left+80:middle_right+100] = 0.50
    # B[middle_left+80:middle_right+100, middle_left+80:middle_right+100] = 0.25
    
    A[middle_left_x:middle_right_x, middle_left_y:middle_right_y] = 0.50
    B[middle_left_x:middle_right_x, middle_left_y:middle_right_y] = 0.25
    
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
    # middle_right = middle + r
    
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
grid_size_x = 200
grid_size_y = 112

# simulation steps
N = 5000
only_save_every = 10
print_every     = 100

A, B = initial_configuration_square_middle(grid_size_x, grid_size_y)

# plt.figure(1)
# plt.imshow(A)

#%%

A = np.load(matrix_loadfile_A)
B = np.load(matrix_loadfile_B)

x_coarse = np.arange(0, grid_size_x, 1)
y_coarse = np.arange(0, grid_size_y, 1)

new_resolution = 2000
x_fine = np.linspace(0, grid_size_x, new_resolution)
y_fine = np.linspace(0, grid_size_y, new_resolution)

threshold = 0.7

animation_images = []
fig = plt.figure()
timetrace = []

for t in range(N):
    
    A, B = grey_scott(A, B, Da, Db, f, k, delta_t)
    
    if t % only_save_every == 0:
        A_interp_fun = interp2d(y_coarse, x_coarse, A, kind='cubic')
        A_interp = A_interp_fun(y_fine, x_fine)
        
        A_binary = np.zeros(A_interp.shape)
        A_binary[A_interp > threshold] = 0
        A_binary[A_interp <= threshold] = 1
        
        image = plt.imshow(A_binary, extent =[y_fine.min(),
                                     y_fine.max(),
                                     x_fine.min(),
                                     x_fine.max()], animated=True, origin='lower', cmap='gray')
        plt.axis('off')
        animation_images.append([image])

    if t % print_every == 0:
        print(t)
        
np.save(matrix_file_A, A)
np.save(matrix_file_B, B)
        

ani = animation.ArtistAnimation(fig, animation_images, 
                                interval=1, 
                                blit=True,
                                repeat_delay=1000)

FFwriter = animation.FFMpegFileWriter()

ani.save(animation_file, writer=FFwriter, dpi=200)

# plt.figure(1)
# plt.imshow(A, animated=True, origin='lower')

# plt.axis('off')
# plt.savefig('large_middle_square_50000.svg' , format='svg')

# plt.figure(2)
# plt.imshow(binary, animated=True, origin='lower')

# plt.axis('off')
# plt.savefig('large_middle_square_50000_binary.svg' , format='svg')


#%%

# ani.save('first_reaction_jox.gif')

#%%

plt.figure(figsize=[6, 6])
x = np.arange(0, 100, 0.00001)
y = np.sin(np.pi*x/8)

plt.plot(y)
plt.axis('off')
plt.gca().set_position([0, 0, 1, 1])

plt.savefig('test.svg' , format='svg')

#%%
