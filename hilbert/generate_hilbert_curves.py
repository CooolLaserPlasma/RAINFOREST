import numpy             as np
import matplotlib.pyplot as plt

from hilbert import decode


def generate_locs(num_dims, num_bits):

    # The maximum Hilbert integer.
    max_h = 2**(num_bits*num_dims)

    # Generate a sequence of Hilbert integers.
    hilberts = np.arange(max_h)

    # Compute the 2-dimensional locations.
    return decode(hilberts, num_dims, num_bits)

# num_dims = 2
# num_bits = 2
# locs_2 = generate_locs(num_dims, num_bits)
    
num_dims = 2
num_bits = 5
locs_3 = generate_locs(num_dims, num_bits)

offset = 32

fig = plt.figure(1)

linewidth = 0.5

x_offset = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4])*32
y_offset = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4])*32

for i in range(len(x_offset)):
    plt.plot(locs_3[:, 0] + x_offset[i], locs_3[:, 1] + y_offset[i], 'black', linewidth=linewidth)

plt.axis('equal')
plt.axis('off')

plt.savefig('big_hilbert.svg' , format='svg', dpi=2400)

#%%

num_dims = 2
num_bits = 5
locs_3 = generate_locs(num_dims, num_bits)

offset = 32

fig = plt.figure(1)

linewidth = 0.5

x_offset = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])*32
y_offset = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])*32

for i in range(len(x_offset)):
    plt.plot(locs_3[:, 0] + x_offset[i], locs_3[:, 1] + y_offset[i], 'black', linewidth=linewidth)

plt.axis('equal')
plt.axis('off')

plt.savefig('small_hilbert.svg' , format='svg', dpi=2400)

#%%

num_dims = 2
num_bits = 5
locs_3 = generate_locs(num_dims, num_bits)

offset = 32

fig = plt.figure(1)

linewidth = 0.5

x_offset = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])*32
y_offset = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])*32

for i in range(len(x_offset)):
    plt.plot(locs_3[:, 0] + x_offset[i], locs_3[:, 1] + y_offset[i], 'black', linewidth=linewidth)

plt.axis('equal')
plt.axis('off')

plt.savefig('smallest_hilbert.svg' , format='svg', dpi=2400)

#%%

num_dims = 2
num_bits = 5
locs_3 = generate_locs(num_dims, num_bits)

offset = 32

fig = plt.figure(1)

linewidth = 0.5

x_offset = np.array([0, 0, 0, 0, 1, 1, 1, 1])*32
y_offset = np.array([0, 1, 0, 1, 0, 1, 0, 1])*32

for i in range(len(x_offset)):
    plt.plot(locs_3[:, 0] + x_offset[i], locs_3[:, 1] + y_offset[i], 'black', linewidth=linewidth)

plt.axis('equal')
plt.axis('off')

plt.savefig('smallestest_hilbert.svg' , format='svg', dpi=2400)