import numpy             as np
import matplotlib.pyplot as plt

from hilbert import decode

import matplotlib.animation as animation

import os
project_path = os.path.dirname(__file__)
animation_folder = project_path + '\\' + 'animations'

if not os.path.exists(animation_folder):
    os.mkdir(animation_folder)
    
animation_file = animation_folder + '\\' + '4x4_two_step_linewidth0_5_hilbert_animation.mp4'


def generate_locs(num_dims, num_bits):

    # The maximum Hilbert integer.
    max_h = 2**(num_bits*num_dims)

    # Generate a sequence of Hilbert integers.
    hilberts = np.arange(max_h)

    # Compute the 2-dimensional locations.
    ints = decode(hilberts, num_dims, num_bits)
    floats = np.array(ints, dtype=float)
    
    return floats


def rotate_curve(coords, theta, offset, flip):
    
    fix_offset = coords + (-offset)
    
    theta_rad = theta*np.pi/180
    rotation_matrix = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                                [np.sin(theta_rad), np.cos(theta_rad)]])
    
    
    rotated_coords = (rotation_matrix @ fix_offset.T).T
    
    if flip:
        flip_matrix = np.array([[1, 0],
                                [0, -1]])
        rotated_coords = (flip_matrix @ rotated_coords.T).T
        
    return rotated_coords + offset


#%%


num_dims = 2
num_bits = 5
coords = generate_locs(num_dims, num_bits)

offset = 32

animation_images = []
fig = plt.figure()

linewidth = 0.5

x_animate = np.array([coords[0, 0], coords[1, 0]])
y_animate = np.array([coords[0, 1], coords[1, 1]])



for i in range(1,len(coords[:, 0]) - 1):
    image, = plt.plot(x_animate, y_animate, 'black')
    
    plt.axis('equal')
    plt.axis('off')
    
    animation_images.append([image])
    
    x_animate = np.append(x_animate, coords[i + 1, 0])
    y_animate = np.append(y_animate, coords[i + 1, 1])
    
ani = animation.ArtistAnimation(fig, animation_images, 
                                interval=20, 
                                blit=True,
                                repeat_delay=1000)

#%%

num_dims = 2
num_bits = 5

offset_rotation = 15.5

coords_OG = generate_locs(num_dims, num_bits) 

## FIRST ANI
coords_00 = rotate_curve(coords_OG, 90, offset_rotation, True) 

coords_10 = rotate_curve(coords_OG, -90, offset_rotation, False) 
coords_10[:, 0] = coords_10[:, 0] + 32

coords_01 = rotate_curve(coords_OG, 180, offset_rotation, True)
coords_01[:, 1] = coords_01[:, 1] + 32

coords_11 = generate_locs(num_dims, num_bits) 
coords_11[:, 0] = coords_11[:, 0] + 32
coords_11[:, 1] = coords_11[:, 1] + 32

## SECOND ANI
# coords_2_1 = generate_locs(num_dims, num_bits) 
# coords_2_1[:, 0] = coords_2_1[:, 0] + 32
# coords_2_1[:, 1] = coords_2_1[:, 1] 

coords_20 = rotate_curve(coords_OG, -90, offset_rotation, False) 
coords_20[:, 0] = coords_20[:, 0] + 32*2

coords_21 = generate_locs(num_dims, num_bits) 
coords_21[:, 0] = coords_21[:, 0] + 32*2
coords_21[:, 1] = coords_21[:, 1] + 32

coords_2_1 = rotate_curve(coords_OG, 0, offset_rotation, False)  
coords_2_1[:, 0] = coords_2_1[:, 0] + 32*2
coords_2_1[:, 1] = coords_2_1[:, 1] - 32

coords_22 = generate_locs(num_dims, num_bits) 
coords_22[:, 0] = coords_22[:, 0] + 32*2
coords_22[:, 1] = coords_22[:, 1] + 32*2

coords_1_1 = rotate_curve(coords_OG, 0, offset_rotation, False) 
coords_1_1[:, 0] = coords_1_1[:, 0] + 32
coords_1_1[:, 1] = coords_1_1[:, 1] - 32

coords_12 = generate_locs(num_dims, num_bits) 
coords_12[:, 0] = coords_12[:, 0] + 32
coords_12[:, 1] = coords_12[:, 1] + 32*2

coords_0_1 = rotate_curve(coords_OG, 90, offset_rotation, False)  
coords_0_1[:, 0] = coords_0_1[:, 0]
coords_0_1[:, 1] = coords_0_1[:, 1] - 32

coords_02 = rotate_curve(coords_OG, 180, offset_rotation, True) 
coords_02[:, 0] = coords_02[:, 0]
coords_02[:, 1] = coords_02[:, 1] + 32*2

coords__1__1 = rotate_curve(coords_OG, 90, offset_rotation, False) 
coords__1__1[:, 0] = coords__1__1[:, 0] - 32
coords__1__1[:, 1] = coords__1__1[:, 1] - 32

coords__1_0 = rotate_curve(coords_OG, 90, offset_rotation, True) 
coords__1_0[:, 0] = coords__1_0[:, 0] - 32
coords__1_0[:, 1] = coords__1_0[:, 1]

coords__1_1 = rotate_curve(coords_OG, 180, offset_rotation, False) 
coords__1_1[:, 0] = coords__1_1[:, 0] - 32
coords__1_1[:, 1] = coords__1_1[:, 1] + 32

coords__1_2 = rotate_curve(coords_OG, 180, offset_rotation, True) 
coords__1_2[:, 0] = coords__1_2[:, 0] - 32
coords__1_2[:, 1] = coords__1_2[:, 1] + 32*2


# plt.figure(1)
# plt.plot(coords_00[:, 0], coords_00[:, 1])
# plt.plot(coords_10[:, 0], coords_10[:, 1])
# plt.plot(coords_01[:, 0], coords_01[:, 1])
# plt.plot(coords_11[:, 0], coords_11[:, 1])
# plt.plot(coords_20[:, 0], coords_20[:, 1])
# plt.plot(coords_21[:, 0], coords_21[:, 1])
# plt.plot(coords_2_1[:, 0], coords_2_1[:, 1])
# plt.plot(coords_22[:, 0], coords_22[:, 1])
# plt.plot(coords_1_1[:, 0], coords_1_1[:, 1])
# plt.plot(coords_12[:, 0], coords_12[:, 1])
# plt.plot(coords_0_1[:, 0], coords_0_1[:, 1])
# plt.plot(coords_02[:, 0], coords_02[:, 1])
# plt.plot(coords__1__1[:, 0], coords__1__1[:, 1])
# plt.plot(coords__1_0[:, 0], coords__1_0[:, 1])
# plt.plot(coords__1_1[:, 0], coords__1_1[:, 1])
# plt.plot(coords__1_2[:, 0], coords__1_2[:, 1])


# # # plt.axis('equal')
# # plt.axis('off')
# plt.xlim([-32, 96])
# plt.ylim([-32, 96])

animation_images = []
fig = plt.figure()

linewidth = 0.5

## FIRST ANIMATION PART
x_animate_00 = np.array([coords_00[0, 0], coords_00[1, 0]])
y_animate_00 = np.array([coords_00[0, 1], coords_00[1, 1]])

x_animate_10 = np.array([coords_10[0, 0], coords_10[1, 0]])
y_animate_10 = np.array([coords_10[0, 1], coords_10[1, 1]])

x_animate_01 = np.array([coords_01[0, 0], coords_01[1, 0]])
y_animate_01 = np.array([coords_01[0, 1], coords_01[1, 1]])

x_animate_11 = np.array([coords_11[0, 0], coords_11[1, 0]])
y_animate_11 = np.array([coords_11[0, 1], coords_11[1, 1]])

## SECOND ANIMATION PART

x_animate_20 = np.array([coords_20[0, 0], coords_20[1, 0]])
y_animate_20 = np.array([coords_20[0, 1], coords_20[1, 1]])

x_animate_21 = np.array([coords_21[0, 0], coords_21[1, 0]])
y_animate_21 = np.array([coords_21[0, 1], coords_21[1, 1]])

x_animate_2_1 = np.array([coords_2_1[0, 0], coords_2_1[1, 0]])
y_animate_2_1 = np.array([coords_2_1[0, 1], coords_2_1[1, 1]])

x_animate_22 = np.array([coords_22[0, 0], coords_22[1, 0]])
y_animate_22 = np.array([coords_22[0, 1], coords_22[1, 1]])

x_animate_1_1 = np.array([coords_1_1[0, 0], coords_1_1[1, 0]])
y_animate_1_1 = np.array([coords_1_1[0, 1], coords_1_1[1, 1]])

x_animate_12 = np.array([coords_12[0, 0], coords_12[1, 0]])
y_animate_12 = np.array([coords_12[0, 1], coords_12[1, 1]])

x_animate_0_1 = np.array([coords_0_1[0, 0], coords_0_1[1, 0]])
y_animate_0_1 = np.array([coords_0_1[0, 1], coords_0_1[1, 1]])

x_animate_02 = np.array([coords_02[0, 0], coords_02[1, 0]])
y_animate_02 = np.array([coords_02[0, 1], coords_02[1, 1]])

x_animate__1__1 = np.array([coords__1__1[0, 0], coords__1__1[1, 0]])
y_animate__1__1 = np.array([coords__1__1[0, 1], coords__1__1[1, 1]])

x_animate__1_0 = np.array([coords__1_0[0, 0], coords__1_0[1, 0]])
y_animate__1_0 = np.array([coords__1_0[0, 1], coords__1_0[1, 1]])

x_animate__1_1 = np.array([coords__1_1[0, 0], coords__1_1[1, 0]])
y_animate__1_1 = np.array([coords__1_1[0, 1], coords__1_1[1, 1]])

x_animate__1_2 = np.array([coords__1_2[0, 0], coords__1_2[1, 0]])
y_animate__1_2 = np.array([coords__1_2[0, 1], coords__1_2[1, 1]])

linewidth = 0.5

for i in range(1,len(coords_00[:, 0]) - 1):
        
    image_00, = plt.plot(x_animate_00, y_animate_00, 'black', linewidth=linewidth)
    image_10, = plt.plot(x_animate_10, y_animate_10, 'black', linewidth=linewidth)
    image_01, = plt.plot(x_animate_01, y_animate_01, 'black', linewidth=linewidth)
    image_11, = plt.plot(x_animate_11, y_animate_11, 'black', linewidth=linewidth)
    
    plt.xlim([-32, 96])
    plt.ylim([-32, 96])
    plt.axis('off')
    
    animation_images.append([image_00, image_10, image_01, image_11])
    
    x_animate_00 = np.append(x_animate_00, coords_00[i + 1, 0])
    y_animate_00 = np.append(y_animate_00, coords_00[i + 1, 1])
    
    x_animate_10 = np.append(x_animate_10, coords_10[i + 1, 0])
    y_animate_10 = np.append(y_animate_10, coords_10[i + 1, 1])
    
    x_animate_01 = np.append(x_animate_01, coords_01[i + 1, 0])
    y_animate_01 = np.append(y_animate_01, coords_01[i + 1, 1])
    
    x_animate_11 = np.append(x_animate_11, coords_11[i + 1, 0])
    y_animate_11 = np.append(y_animate_11, coords_11[i + 1, 1])
    
for i in range(1, len(coords_00[:, 0]) - 1):
        
    image_00, = plt.plot(x_animate_00, y_animate_00, 'black', linewidth=linewidth)
    image_10, = plt.plot(x_animate_10, y_animate_10, 'black', linewidth=linewidth)
    image_01, = plt.plot(x_animate_01, y_animate_01, 'black', linewidth=linewidth)
    image_11, = plt.plot(x_animate_11, y_animate_11, 'black', linewidth=linewidth)

    image_20,    = plt.plot(x_animate_20, y_animate_20, 'black', linewidth=linewidth)
    image_21,    = plt.plot(x_animate_21, y_animate_21, 'black', linewidth=linewidth)
    image_2_1,   = plt.plot(x_animate_2_1, y_animate_2_1, 'black', linewidth=linewidth)
    image_22,    = plt.plot(x_animate_22, y_animate_22, 'black', linewidth=linewidth)
    
    image_1_1,   = plt.plot(x_animate_1_1, y_animate_1_1, 'black', linewidth=linewidth)
    image_12,    = plt.plot(x_animate_12, y_animate_12, 'black', linewidth=linewidth)
    image_0_1,   = plt.plot(x_animate_0_1, y_animate_0_1, 'black', linewidth=linewidth)
    image_02,    = plt.plot(x_animate_02, y_animate_02, 'black', linewidth=linewidth)
    
    image__1__1, = plt.plot(x_animate__1__1, y_animate__1__1, 'black', linewidth=linewidth)
    image__1_0,  = plt.plot(x_animate__1_0, y_animate__1_0, 'black', linewidth=linewidth)
    image__1_1,  = plt.plot(x_animate__1_1, y_animate__1_1, 'black', linewidth=linewidth)
    image__1_2,  = plt.plot(x_animate__1_2, y_animate__1_2, 'black', linewidth=linewidth)
    
    
    plt.xlim([-32, 96])
    plt.ylim([-32, 96])
    plt.axis('off')
    
    animation_images.append([image_00, image_10, image_01, image_11, 
                             image_20, image_21, image_2_1, image_22,
                             image_1_1, image_12, image_0_1, image_02,
                             image__1__1, image__1_0, image__1_1, image__1_2])
    
    x_animate_20 = np.append(x_animate_20, coords_20[i + 1, 0])
    y_animate_20 = np.append(y_animate_20, coords_20[i + 1, 1])
    
    x_animate_21 = np.append(x_animate_21, coords_21[i + 1, 0])
    y_animate_21 = np.append(y_animate_21, coords_21[i + 1, 1])
    
    x_animate_2_1 = np.append(x_animate_2_1, coords_2_1[i + 1, 0])
    y_animate_2_1 = np.append(y_animate_2_1, coords_2_1[i + 1, 1])
    
    x_animate_22 = np.append(x_animate_22, coords_22[i + 1, 0])
    y_animate_22 = np.append(y_animate_22, coords_22[i + 1, 1])
    
    
    
    x_animate_1_1 = np.append(x_animate_1_1, coords_1_1[i + 1, 0])
    y_animate_1_1 = np.append(y_animate_1_1, coords_1_1[i + 1, 1])
    
    x_animate_12 = np.append(x_animate_12, coords_12[i + 1, 0])
    y_animate_12 = np.append(y_animate_12, coords_12[i + 1, 1])
    
    x_animate_0_1 = np.append(x_animate_0_1, coords_0_1[i + 1, 0])
    y_animate_0_1 = np.append(y_animate_0_1, coords_0_1[i + 1, 1])
    
    x_animate_02 = np.append(x_animate_02, coords_02[i + 1, 0])
    y_animate_02 = np.append(y_animate_02, coords_02[i + 1, 1])
    
    
    
    x_animate__1__1 = np.append(x_animate__1__1, coords__1__1[i + 1, 0])
    y_animate__1__1 = np.append(y_animate__1__1, coords__1__1[i + 1, 1])
    
    x_animate__1_0 = np.append(x_animate__1_0, coords__1_0[i + 1, 0])
    y_animate__1_0 = np.append(y_animate__1_0, coords__1_0[i + 1, 1])
    
    x_animate__1_1 = np.append(x_animate__1_1, coords__1_1[i + 1, 0])
    y_animate__1_1 = np.append(y_animate__1_1, coords__1_1[i + 1, 1])
    
    x_animate__1_2 = np.append(x_animate__1_2, coords__1_2[i + 1, 0])
    y_animate__1_2 = np.append(y_animate__1_2, coords__1_2[i + 1, 1])

    
for i in range(1, 50):
        
    image_00, = plt.plot(x_animate_00, y_animate_00, 'black', linewidth=linewidth)
    image_10, = plt.plot(x_animate_10, y_animate_10, 'black', linewidth=linewidth)
    image_01, = plt.plot(x_animate_01, y_animate_01, 'black', linewidth=linewidth)
    image_11, = plt.plot(x_animate_11, y_animate_11, 'black', linewidth=linewidth)

    image_20,    = plt.plot(x_animate_20, y_animate_20, 'black', linewidth=linewidth)
    image_21,    = plt.plot(x_animate_21, y_animate_21, 'black', linewidth=linewidth)
    image_2_1,   = plt.plot(x_animate_2_1, y_animate_2_1, 'black', linewidth=linewidth)
    image_22,    = plt.plot(x_animate_22, y_animate_22, 'black', linewidth=linewidth)
    
    image_1_1,   = plt.plot(x_animate_1_1, y_animate_1_1, 'black', linewidth=linewidth)
    image_12,    = plt.plot(x_animate_12, y_animate_12, 'black', linewidth=linewidth)
    image_0_1,   = plt.plot(x_animate_0_1, y_animate_0_1, 'black', linewidth=linewidth)
    image_02,    = plt.plot(x_animate_02, y_animate_02, 'black', linewidth=linewidth)
    
    image__1__1, = plt.plot(x_animate__1__1, y_animate__1__1, 'black', linewidth=linewidth)
    image__1_0,  = plt.plot(x_animate__1_0, y_animate__1_0, 'black', linewidth=linewidth)
    image__1_1,  = plt.plot(x_animate__1_1, y_animate__1_1, 'black', linewidth=linewidth)
    image__1_2,  = plt.plot(x_animate__1_2, y_animate__1_2, 'black', linewidth=linewidth)
    
    
    plt.xlim([-32, 96])
    plt.ylim([-32, 96])
    plt.axis('off')
    
    animation_images.append([image_00, image_10, image_01, image_11, 
                             image_20, image_21, image_2_1, image_22,
                             image_1_1, image_12, image_0_1, image_02,
                             image__1__1, image__1_0, image__1_1, image__1_2])

    
ani = animation.ArtistAnimation(fig, animation_images, 
                                interval=1, 
                                blit=True,
                                repeat_delay=1000)

FFwriter = animation.FFMpegFileWriter()

ani.save(animation_file, writer=FFwriter, dpi=200)


# for i in range(len(x_offset)):
#     plt.plot(locs_3[:, 0] + x_offset[i], locs_3[:, 1] + y_offset[i], 'black', linewidth=linewidth)

# plt.savefig('smallestest_hilbert.svg' , format='svg', dpi=2400)

