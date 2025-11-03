import numpy as np
import sys
import torch
import matplotlib.pyplot as plt

def xu2u_ufile(xu,z,length_of_z,u):
    xu2u = []
    for i in range(int(len(xu) / length_of_z) - 1):
        xu2u.append((xu[3 * i] + xu[3 * (i + 1)]) / 2)
    interpolated_z = np.zeros(len(z) - length_of_z)
    interpolated_u = np.zeros(len(u) - length_of_z)
    for j in range(len(z) - length_of_z):
        interpolated_z[j] = (z[j] + z[j + length_of_z]) / 2
        interpolated_u[j] = (u[j] + u[j + length_of_z]) / 2
    interpolated_x = np.zeros(len(xu2u) * length_of_z)
    for k in range(len(xu2u)):
        for l in range(length_of_z):
            interpolated_x[(k * length_of_z) + l] = xu2u[k]
    ### error message print-out
    if len(interpolated_x) != len(interpolated_z):
        print('Something is wrong! Dimension not matched')
    return interpolated_x,interpolated_z,interpolated_u


def yw2w_wfile(length_of_x,length_of_z,x,yw,w):
    interpolated_x = np.zeros(length_of_x*length_of_z)
    interpolated_z = []
    interpolated_w = []
    temp_x = []
    for i in range(length_of_x):
        temp_x.append(x[(length_of_z+1)*i])
        for j in range(length_of_z):
            interpolated_z.append((yw[(length_of_z+1)*i+j] + yw[(length_of_z+1)*i+j+1]) / 2)
            interpolated_w.append((w[(length_of_z+1)*i+j] + w[(length_of_z+1)*i+j+1]) / 2)
    for k in range(len(temp_x)):
        for l in range(length_of_z):
            interpolated_x[(k * length_of_z) + l] = temp_x[k]
    ### error message print-out
    if len(interpolated_x) != len(interpolated_z):
        print('Something is wrong! Dimension not matched')
    return interpolated_x,interpolated_z,interpolated_w

def reshape_hafile(ha_array):
    new_ha_array = np.zeros(512*128)
    for i in range(len(ha_array)):
        new_ha_array[i:512*128:128] = ha_array[i]
    return new_ha_array

# def normalize2rgb(array):
#     normalized_array = (255 * (array - np.min(array)) / np.ptp(array)).astype(int)
#     return normalized_array
def normalize2rgb(array):
    if np.ptp(array) == 0:
        return np.zeros_like(array, dtype=int)  # All values are the same; return 0
    normalized_array = (255 * (array - np.min(array)) / np.ptp(array)).astype(int)
    return normalized_array


def generate_rgb(array1,array2,array3,length_of_z,length_of_x):
    array1_grid = np.reshape(array1, (length_of_z, length_of_x), order='F')
    array1_grid_flip_v = np.flip(array1_grid, 0)  # vertical flip
    array2_grid = np.reshape(array2, (length_of_z, length_of_x), order='F')
    array2_grid_flip_v = np.flip(array2_grid, 0)  # vertical flip
    array3_grid = np.reshape(array3, (length_of_z, length_of_x), order='F')
    array3_grid_flip_v = np.flip(array3_grid, 0)  # vertical flip
    # Stack the three color channels together to form a single RGB image.
    rgb_image = np.stack((array1_grid_flip_v, array2_grid_flip_v, array3_grid_flip_v), axis=2)
    return rgb_image

def read_files(N, model):
    # Read f file
    i = '/f' + N
    ffile_name = model + i
    ffile = np.loadtxt(ffile_name)
    # Read w file
    j = '/w' + N
    wfile_name = model + j
    wfile = np.loadtxt(wfile_name)
    # Read ha file
    k = '/ha' + N
    hafile_name = model + k
    hafile = np.loadtxt(hafile_name)
    l = '/t' + N
    tfile_name = model + l
    tfile = np.loadtxt(tfile_name)

    temp = ffile[:, 2]
    alpha = ffile[:, 3]
    mu = ffile[:,5]
    x = wfile[:, 0]
    yw = wfile[:, 1]
    w = wfile[:, 2]
    ha_temp = hafile[:, 2]
    tau_tot = tfile[:,2]

    return temp, ha_temp, alpha, x, yw, w, mu, tau_tot


def process_files(temp_num, model1, model_name, length_of_x, length_of_z, offset=0):
    # Convert temp_num to string with leading zeros if necessary
    input_temp_num = 'temp' + str(temp_num)
    output_temp_num = f'temp{temp_num + offset:04d}'
    N1 = f"{temp_num:03d}"

    # Read the necessary files
    temp, ha_temp, alpha, x, yw, w, mu, tau = read_files(N1, model1)

    # Perform interpolation and reshaping
    interpolated_x, interpolated_z, interpolated_w = yw2w_wfile(length_of_x, length_of_z, x, yw, w)
    new_ha_temp = reshape_hafile(ha_temp)

    # Normalize and generate RGB channels
    rgb_temp = normalize2rgb(temp - new_ha_temp)
    rgb_alpha = normalize2rgb(alpha)
    rgb_mu = normalize2rgb(mu)
    rgb_w = normalize2rgb(interpolated_w)
    rgb_tau = normalize2rgb(tau)

    # Create RGB image
    # Order: Blue, Green, Red
    a = generate_rgb(rgb_temp, rgb_tau, rgb_w, length_of_z, length_of_x)

    # Print status message
    print(f'Temp-ha #{output_temp_num} is generated!')

    # Save the image and RGB tensor
    plt.figure(figsize=(8, 2))
    plt.axis('off')

    # Save the image as .png
    filename = f'/rubin/s1/hxc5400/data/{model_name}/rgb_ha_{output_temp_num}.png'
    plt.imshow(a)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    # Save the RGB torch tensor
    t = torch.from_numpy(a)
    torch.save(t, f'/rubin/s1/hxc5400/data/{model_name}/rgb_ha_{output_temp_num}.pt')

    plt.close()  # Close the plot to prevent memory leaks


# Set your parameters
model_name = 'dam1_diss_1.4_new_icmobR_crust0975_vis100'
model1 = f'/rubin/s1/scratch/hxc5400/model_output/{model_name}'
model2 = f'/rubin/s1/scratch/hxc5400/model_output/{model_name}_2'
model3 = f'/rubin/s1/scratch/hxc5400/model_output/{model_name}_3'
model4 = f'/rubin/s1/scratch/hxc5400/model_output/{model_name}_4'
model5 = f'/rubin/s1/scratch/hxc5400/model_output/{model_name}_5'
m1 = 121
m2 = 85
m3 = 281
m4 = 61
m5 = 153
length_of_x = 512
length_of_z = 128

# Loop over different temp_num values
for temp_num in range(m1):
    process_files(temp_num, model1, model_name, length_of_x, length_of_z)
for temp_num in range(m2):
    process_files(temp_num, model2, model_name, length_of_x, length_of_z, offset=m1)
for temp_num in range(m3):
    process_files(temp_num, model3, model_name, length_of_x, length_of_z, offset=m1+m2)
for temp_num in range(m4):
    process_files(temp_num, model4, model_name, length_of_x, length_of_z, offset=m1+m2+m3)
for temp_num in range(m5):  # Adjust the range as needed
    process_files(temp_num, model5, model_name, length_of_x, length_of_z, offset=m1+m2+m3+m4)
