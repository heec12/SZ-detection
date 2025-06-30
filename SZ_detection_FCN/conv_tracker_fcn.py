import numpy as np
import sys
import torch
import torch.optim as optim
from conv_tracker_fcn_func import *
from conv_tracker_trck_func import *

# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# Initialize model, criterion, optimizer, and scheduler
def initialize_model():
    model = FCN().to(device)
    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    return model, criterion, optimizer, scheduler

# Humongous empty arrays
ct_rows, ct_columns = 2000, 1000
conv_tracker_index = np.zeros((ct_rows, ct_columns))

# Track continental location
# def initialize_cont_index(m1):
#     return np.zeros((m1 + 1, 3))

# Main tracking function
def track_subduction(model_name, total_timestep):
    model_path = f'/rubin/s1/hxc5400/data/{model_name}/'
    file_path = f'/rubin/s1/scratch/hxc5400/model_output/{model_name}'
    conv_tracker_index = np.zeros((ct_rows, ct_columns))

    # Define average_u inside this function
    average_u = 0.00774687768014917 * 512 / 4  # Define within function scope

    # Initial setup
    top_row_boolean, left_sz0, right_sz0, sz0 = apply_fcn(str("{:0>4d}".format(0)), model, model_path, device)
    # left_cont0, right_cont0, _ = find_cont_loc(file_path, str("{:0>3d}".format(0)))
    print("It started")
    print(left_sz0, right_sz0, sz0)

    for i in range(len(sz0)):
        conv_tracker_index[i, 0] = 1  # Record time
        conv_tracker_index[i, 1] = sz0[i]  # Assign values of sz0 to the second column
        conv_tracker_index[i, 2] = -1 # Keep active array active
        # print("I'm here", conv_tracker_index[i,:])

    # Track continent location for the first time step
    # cont_index[0, 0] = 1
    # cont_index[0, 1] = left_cont0
    # cont_index[0, 2] = right_cont0
    # print(cont_index)

    for i in range(1, total_timestep):
        N = str("{:0>3d}".format(i))
        print(f"\nProcessing timestep {i} - Image: {N}")

        top_row_boolean1, left_sz1, right_sz1, sz1 = apply_fcn(N, model, model_path, device)
        # left_cont1, right_cont1, _ = find_cont_loc(file_path, N)
        # print("Left SZ:", left_sz1, "Right SZ:", right_sz1, "Subduction Zones:", sz1)

        # cont_index[i, 0] = i + 1
        # cont_index[i, 1] = left_cont1
        # cont_index[i, 2] = right_cont1

        # Subduction zone tracking logic
        handle_subduction_zones(i, sz0, sz1, conv_tracker_index, average_u)

        # Deactivate the unupdated array
        deactivate_active_array(conv_tracker_index)

        # Update the previous timestep info
        top_row_boolean0, left_sz0, right_sz0, sz0 = top_row_boolean1, left_sz1, right_sz1, sz1

    return conv_tracker_index#, cont_index

def deactivate_active_array(conv_tracker_index):
    for row in range(conv_tracker_index.shape[0]):
        for col in range(conv_tracker_index.shape[1]):
            if conv_tracker_index[row, col] == -1:
                conv_tracker_index[row, col] = 0 # Stop tracking

    for row in range(conv_tracker_index.shape[0]):
        for col in range(conv_tracker_index.shape[1]):
            if conv_tracker_index[row, col] == -10:
                conv_tracker_index[row, col] = -1 # Keep tracking

# Helper function to handle subduction zones
def handle_subduction_zones(i, sz0, sz1, conv_tracker_index, average_u):
    if len(sz0) == len(sz1):
        # print("Same number")
        update_tracker_same_sz(i, sz1, conv_tracker_index, average_u)
    elif len(sz0) > len(sz1):
        update_tracker_decreased_sz(i, sz1, conv_tracker_index, average_u)
    else:
        update_tracker_increased_sz(i, sz0, sz1, conv_tracker_index, average_u)  # Ensure proper argument order


def update_tracker_same_sz(i, sz1, conv_tracker_index, average_u):
    """Update conv_tracker_index for the case when the number of SZ remains the same."""
    temp_active_matrix = find_active_array(conv_tracker_index)
    for j in range(len(sz1)):
        # Find closest subduction zone and attach the new SZ to the active array
        index_closest, leftover_element = find_closest(average_u,temp_active_matrix[2, :], sz1[j])

        if index_closest == -99:  # Same number of SZ but new SZ is required
            new_row = find_empty_array(ct_rows, conv_tracker_index)
            conv_tracker_index[new_row, 0] = i + 1
            conv_tracker_index[new_row, 1] = leftover_element
            conv_tracker_index[new_row, 2] = -10  # Keep active array active
        else:
            # Update the SZ in the active array and keep it active
            # print("I'm here and active")
            row, col = int(temp_active_matrix[0, index_closest]), int(temp_active_matrix[1, index_closest])
            conv_tracker_index[row, col] = sz1[j]
            conv_tracker_index[row, col + 1] = -10  # Mark the end to keep active
    # print(conv_tracker_index[0, :])

def update_tracker_decreased_sz(i, sz1, conv_tracker_index, average_u):
    """Update conv_tracker_index for the case when the number of SZ decrease."""
    temp_active_matrix = find_active_array(conv_tracker_index)
    for j in range(len(sz1)):
        # Find closest & attach the new SZ to the active array
        index_cloest, leftover_element = find_closest(average_u,temp_active_matrix[2, :], sz1[j])
        if index_cloest == -99:  # The closest SZ in the previous timestep is too far; new SZ is required
            new_row = find_empty_array(ct_rows, conv_tracker_index)
            conv_tracker_index[new_row, 0] = i + 1
            conv_tracker_index[new_row, 1] = leftover_element
            conv_tracker_index[new_row, 2] = -10  # keep active array active
        else:
            conv_tracker_index[int(temp_active_matrix[0,index_cloest]),int(temp_active_matrix[1,index_cloest])]=sz1[j]
            # Keep active array active by changing the last node to -10
            conv_tracker_index[int(temp_active_matrix[0,index_cloest]),int(temp_active_matrix[1,index_cloest])+1]=-10


def update_tracker_increased_sz(i, sz0, sz1, conv_tracker_index, average_u):
    """Update conv_tracker_index for the case when the number of SZ increase."""
    num_new_conv = len(sz1) - len(sz0)
    temp_active_matrix = np.concatenate(
        (find_active_array(conv_tracker_index),
         new_active_array(ct_rows, conv_tracker_index, num_new_conv)),
        axis=1)
    #leftover_element = 0
    for j in range(len(sz1)):
        # Find closest & attach the new SZ to the active array
        index_cloest, leftover_element = find_closest(average_u,temp_active_matrix[2, :], sz1[j])
        if index_cloest == -99:  # same num. of SZ; but new SZ is required
            new_row = find_empty_array(ct_rows, conv_tracker_index)
            conv_tracker_index[new_row, 0] = i + 1
            conv_tracker_index[new_row, 1] = leftover_element
            conv_tracker_index[new_row, 2] = -10  # keep active array active
        else:
            conv_tracker_index[int(temp_active_matrix[0, index_cloest]), int(temp_active_matrix[1, index_cloest])] = sz1[j]
            # Keep active array active by changing the last node to -10
            conv_tracker_index[int(temp_active_matrix[0, index_cloest]), int(temp_active_matrix[1, index_cloest]) + 1] = -10

# Running Part
model, criterion, optimizer, scheduler = initialize_model()
m1 = 764
m2 = 528
# m3 = 999
# m4 = 529
# m5 = 153
mnumber = m1+m2
# cont_index = initialize_cont_index(m1=mnumber)  # (m1=1192)
modelname = 'dam1_diss_1.3_new_icmobR_crust090_vis100'

# Load the trained model instead of training
model_path = "trained_model_fcn_final.pt"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # Set to evaluation mode

# Proceed to find subduction zones
# conv_tracker_index1, cont_index1 = track_subduction(f'{modelname}', mnumber, cont_index=cont_index)
conv_tracker_index1 = track_subduction(f'{modelname}', mnumber)
# To find continent locations
cont_index1 = np.zeros((mnumber + 1, 4))
# average_u = 0.0181868278962208 * 512/4
for i in range(m1):
    N = str("{:0>3d}".format(i))
    file_path = f'/rubin/s1/scratch/hxc5400/model_output/{modelname}'
    left_cont1, right_cont1, rtime1 = find_cont_loc(file_path, N)
    cont_index1[i, 0] = int(i + 1)
    cont_index1[i, 1] = left_cont1
    cont_index1[i, 2] = right_cont1
    cont_index1[i, 3] = rtime1
for i in range(m2):
    N = str("{:0>3d}".format(i))
    file_path = f'/rubin/s1/scratch/hxc5400/model_output/{modelname}_2'
    left_cont2, right_cont2, rtime2 = find_cont_loc(file_path, N)
    cont_index1[m1+i, 0] = int(m1+ i + 1)
    cont_index1[m1+i, 1] = left_cont2
    cont_index1[m1+i, 2] = right_cont2
    cont_index1[m1+i, 3] = rtime1+rtime2
# for i in range(m3):
#     N = str("{:0>3d}".format(i))
#     file_path = f'/rubin/s1/scratch/hxc5400/model_output/{modelname}_3'
#     left_cont3, right_cont3, rtime3 = find_cont_loc(file_path, N)
#     cont_index1[m1+m2+i, 0] = m1+m2 + i + 1
#     cont_index1[m1+m2+i, 1] = left_cont3
#     cont_index1[m1+m2+i, 2] = right_cont3
#     cont_index1[m1+m2+i, 3] = rtime1 + rtime2 +rtime3
# for i in range(m4):
#     N = str("{:0>3d}".format(i))
#     file_path = f'/rubin/s1/scratch/hxc5400/model_output/{modelname}_4'
#     left_cont4, right_cont4, rtime4 = find_cont_loc(file_path, N)
#     cont_index1[m1+m2+m3+i, 0] = m1+m2+m3 + i + 1
#     cont_index1[m1+m2+m3+i, 1] = left_cont4
#     cont_index1[m1+m2+m3+i, 2] = right_cont4
#     cont_index1[m1+m2+m3+i, 3] = rtime1 + rtime2 +rtime3 + rtime4
# for i in range(m5):
#     N = str("{:0>3d}".format(i))
#     file_path = f'/rubin/s1/scratch/hxc5400/model_output/{modelname}_5'
#     left_cont5, right_cont5, rtime5 = find_cont_loc(file_path, N)
#     cont_index1[m1+m2+m3+m4+i, 0] = m1+m2+m3+m4 + i + 1
#     cont_index1[m1+m2+m3+m4+i, 1] = left_cont5
#     cont_index1[m1+m2+m3+m4+i, 2] = right_cont5
#     cont_index1[m1+m2+m3+m4+i, 3] = rtime1 + rtime2 +rtime3 + rtime4 + rtime5

# Save multiple matrices
np.savez(f'sz_tracker_output_{modelname}.npz', conv_tracker_index=conv_tracker_index1, cont_tracker_index=cont_index1)
