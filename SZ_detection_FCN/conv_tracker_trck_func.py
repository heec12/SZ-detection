import numpy as np

def find_empty_array(ct_rows, conv_tracker_index):
    first_empty = ct_rows + 1
    for i in range(2, ct_rows):
        if conv_tracker_index[i, 0] == 0:
            # print(f"Checking row {i} for emptiness:")
            # print(conv_tracker_index[i, :])  # Print the whole row to check
            first_empty = i
            break
    return first_empty

def find_active_array(conv_tracker_index):
    temp_active_row = []
    temp_active_column = []
    temp_active_index = []
    for i in range(conv_tracker_index.shape[0]):
        for j in range(conv_tracker_index.shape[1]):
            if conv_tracker_index[i,j] == -1:# look for -1
                temp_active_row.append(i)
                temp_active_column.append(j)
                temp_active_index.append(conv_tracker_index[i,j-1])
    temp_active_matrix = np.array([temp_active_row, temp_active_column, temp_active_index])
    return temp_active_matrix

def new_active_array(ct_rows, conv_tracker_index, num_new_conv):
    temp_new_active_row = []
    temp_new_active_column = []
    temp_new_active_index = []

    for i in range(num_new_conv):
        new_row = find_empty_array(ct_rows, conv_tracker_index)  # Ensure it's an empty row
        temp_new_active_row.append(new_row)
        temp_new_active_column.append(0)
        temp_new_active_index.append(-2)  # Mark as active

    temp_new_active_matrix = np.array([temp_new_active_row, temp_new_active_column, temp_new_active_index])
    return temp_new_active_matrix

def find_closest(average_u,arr,element):
    node_difference = 20 + average_u
    leftover_element = -100
    #leftover_element_mag = -100
    index_cloest= []
    diff_array = np.absolute(arr-element)
    for i in range(len(diff_array)):
        if np.min(diff_array) <= node_difference: # if the difference is less than node diff criteria
            index_cloest.append(diff_array.argmin())
            #print('Im here; the diff is less than 10')
        else:
            diff_array2 = np.absolute(arr-(element+512))
            if np.min(diff_array2) <= node_difference: # in case of SZs migrated over the edge
                index_cloest.append(diff_array2.argmin())
                #print("It's working; the diff is less than 10", index_cloest)
            else:                        # if the node diff is still larger
                index_cloest.append(-99) # so it can't be recognized by the program
                leftover_element = element
                #print('IThe diff is larger than 10')
    return index_cloest[0], leftover_element

def find_cont_loc(model, N):
    i = '/w' + N
    j = '/ha' + N
    # Read u, w, ha files
    wfile = model + i
    hafile = model + j
    # Create empty arrays for later purpose
    x_coord = []
    y_coord = []
    chemical = []
    rtime = []

    # From w files, read chemical info
    with open(wfile) as fp:
        line = fp.readline()
        while line:
            sline = line.split()
            x = float(sline[0])  # Convert to float
            x_coord.append(x)
            y = float(sline[1])  # Convert to float
            y_coord.append(y)
            chem = sline[3]
            chemical.append(chem)
            line = fp.readline()

    # From ha files, read a real time of each output
    with open(hafile) as fp:
        line = fp.readline()
        while line:
            sline = line.split()
            time = sline[0]
            line = fp.readline()
    rtime = time

    # Convert character arrays to float arrays for calculation #
    chemical = np.array(chemical).astype(float)
    rtime = np.array(rtime).astype(float)

    temp_margin_loc = []
    temp_margin_loc_index = []
    cont_margin_loc = []
    cont_margin_loc_index = []
    temp_margin_loc_left = []
    temp_margin_loc_right = []
    # print(chemical)
    # Find y-coordinate of where continental margins are
    for i in range(len(chemical)):
        if 0.0 < chemical[i] < 1.0:
            temp_margin_loc.append(x_coord[i])
            temp_margin_loc_index.append(i)
    # print(temp_margin_loc)
    # print(temp_margin_loc_index)
    if len(temp_margin_loc) < 2:
        print("something is wrong; margins are not correctly detected")
    elif len(temp_margin_loc) >= 2:
        cont_margin_loc.append(min(temp_margin_loc))
        cont_margin_loc_index.append(int(min(temp_margin_loc) * 512 / 4))
        cont_margin_loc.append(max(temp_margin_loc))
        cont_margin_loc_index.append(int(max(temp_margin_loc) * 512 / 4))
    # Sort left(0-1) and right(1-0) boundary
    if cont_margin_loc_index[1] - cont_margin_loc_index[0] > 192:
        left_cont = cont_margin_loc_index[1]
        right_cont = cont_margin_loc_index[0]
    else:
        left_cont = cont_margin_loc_index[0]
        right_cont = cont_margin_loc_index[1]
    # print(left_cont,right_cont)
    # Re-find the continental margin if it's splited
    if left_cont == 511 and right_cont == 0:
        # print("left is 511 and right is 0")
        for i in range(len(temp_margin_loc)):
            if temp_margin_loc[i] <= 2.0:
                temp_margin_loc_left.append(temp_margin_loc[i])
            else:
                temp_margin_loc_right.append(temp_margin_loc[i])
        # print('Left = ',temp_margin_loc_left,'Right = ',temp_margin_loc_right)
        # print('Max of left = ',max(temp_margin_loc_index_left))
        # print('Min of right = ',min(temp_margin_loc_index_right))
        left_cont = int(min(temp_margin_loc_right) * 512 / 4)
        right_cont = int(max(temp_margin_loc_left) * 512 / 4)

    return left_cont, right_cont, rtime
