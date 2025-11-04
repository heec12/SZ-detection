import numpy as np
import matplotlib.pyplot as plt

# Python code for generating histograms and scatter plots
# Load SZ and cont tracker matrices
modelname = 'dam1_diss_1.2_new_icmobR_crust095_vis100'
fcn_output_file_path = f"../example/new_sz_tracker_output_{modelname}.npz"
dudx_output_file_path = f"../example/dudx_output_arrays_{modelname}.npz"

# Extract the arrays
fcn_loaded_data = np.load(fcn_output_file_path)

fcn_longevity = fcn_loaded_data['counts_before_zero']
fcn_init_tau = fcn_loaded_data['minimum_distance']
fcn_init_tau = np.where(fcn_init_tau > 256, 512 - fcn_init_tau, fcn_init_tau)
fcn_count_new_cropped_arr = fcn_loaded_data['count_new_cropped_arr']

dudx_loaded_data = np.load(dudx_output_file_path)

dudx_longevity = dudx_loaded_data['new_cropped_arr']
dudx_init_tau = dudx_loaded_data['init_taus']
dudx_init_tau = np.where(dudx_init_tau > 256, 512 - dudx_init_tau, dudx_init_tau)
dudx_count_new_cropped_arr = dudx_loaded_data['count_new_cropped_arr']


# Convert timestep to non-dimensional time
dt = 1e-4
fcn_time = np.array(fcn_longevity) * dt
dudx_time = np.array(dudx_longevity) * dt

# Compute common bin edges based on both datasets
bin_edges = np.histogram_bin_edges(
    np.concatenate([fcn_time, dudx_time]), bins=20)

# Plot histograms of longevity (in non-dimensional time)
plt.figure(figsize=(8, 6))
plt.tight_layout()
plt.hist(fcn_time, bins=bin_edges, alpha=0.5, label="FCN longevity", color='#005AB5')
plt.hist(dudx_time, bins=bin_edges, alpha=0.5, label="du/dx longevity", color='#DC3220')

# Horizontal lines (if these represent counts at specific longevity bins, scale accordingly)
plt.hlines(y=fcn_count_new_cropped_arr, xmin=bin_edges[0], xmax=bin_edges[1],
           color='#005AB5', linewidth=2)
plt.hlines(y=dudx_count_new_cropped_arr, xmin=bin_edges[0], xmax=bin_edges[1],
           color='#DC3220', linewidth=2)
# Labels, legend, and axis formatting
# plt.xlabel("Longevity [non-dimensional time]", fontsize=24)
# plt.ylabel("Count", fontsize=24)
plt.xlim(0, 300 * dt)  # adjust to same scale as before, but in non-dim time
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=36)
# plt.legend(fontsize=20)
plt.savefig(f'ltime_comb_longevity_histogram_{modelname}.pdf', bbox_inches='tight', pad_inches=0.02)
