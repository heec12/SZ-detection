import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Python code for generating box plots to compare
# Load SZ and cont tracker matrices
Q1 = '1.2'
Q2 = '1.3'
Q3 = '1.4'
dcont = '0975'
fcn_output_file_path1 = f"/rubin/s1/hxc5400/ml_connect_conv_py_code/output_arrays_dam1_diss_{Q1}_new_icmobR_crust{dcont}_vis100.npz"
fcn_output_file_path2 = f"/rubin/s1/hxc5400/ml_connect_conv_py_code/output_arrays_dam1_diss_{Q2}_new_icmobR_crust{dcont}_vis100.npz"
fcn_output_file_path3 = f"/rubin/s1/hxc5400/ml_connect_conv_py_code/output_arrays_dam1_diss_{Q3}_new_icmobR_crust{dcont}_vis100.npz"

# Extract the arrays
fcn_loaded_data1 = np.load(fcn_output_file_path1)

fcn_longevity1 = fcn_loaded_data1['counts_before_zero']
fcn_init_tau1 = fcn_loaded_data1['minimum_distance']
fcn_init_tau1 = np.where(fcn_init_tau1 > 256, 512 - fcn_init_tau1, fcn_init_tau1)

fcn_loaded_data2 = np.load(fcn_output_file_path2)

fcn_longevity2 = fcn_loaded_data2['counts_before_zero']
fcn_init_tau2 = fcn_loaded_data2['minimum_distance']
fcn_init_tau2 = np.where(fcn_init_tau2 > 256, 512 - fcn_init_tau2, fcn_init_tau2)

fcn_loaded_data3 = np.load(fcn_output_file_path3)

fcn_longevity3 = fcn_loaded_data3['counts_before_zero']
fcn_init_tau3 = fcn_loaded_data3['minimum_distance']
fcn_init_tau3 = np.where(fcn_init_tau3 > 256, 512 - fcn_init_tau3, fcn_init_tau3)


# Split data into categories
fcn_longevity_tau_leq_101 = fcn_longevity1[fcn_init_tau1 <= 10]
fcn_longevity_tau_gt_101 = fcn_longevity1[fcn_init_tau1 > 10]

fcn_longevity_tau_leq_102 = fcn_longevity2[fcn_init_tau2 <= 10]
fcn_longevity_tau_gt_102 = fcn_longevity2[fcn_init_tau2 > 10]

fcn_longevity_tau_leq_103 = fcn_longevity3[fcn_init_tau3 <= 10]
fcn_longevity_tau_gt_103 = fcn_longevity3[fcn_init_tau3 > 10]


# Prepare data for boxplot
data_near = [
    fcn_longevity_tau_leq_101,
    fcn_longevity_tau_leq_102,
    fcn_longevity_tau_leq_103
]
data_far = [
    fcn_longevity_tau_gt_101,
    fcn_longevity_tau_gt_102,
    fcn_longevity_tau_gt_103
]

data_near_time = [arr * 1e-4 for arr in data_near]
data_far_time  = [arr * 1e-4 for arr in data_far]

labels = ["10","15","20"]

# Create the box plot
plt.figure(figsize=(8, 6))
sns.boxplot(data=data_near_time, showfliers=False, palette=["#3976B7", "#3976B7", "#3976B7"], width=0.3)
# Customize the plot
plt.xticks(ticks=range(3), labels=labels, fontsize=24)
plt.yticks(fontsize=24)
eps = 1e-9
plt.ylim(0.0, 0.026)
plt.yticks(np.arange(0.0, 0.026 + eps, 0.005))
plt.savefig(f'new_figures5/new_near_boxplot_rmoutlier_dam1_crust{dcont}.pdf', bbox_inches='tight', pad_inches=0.02)

plt.figure(figsize=(8, 6))
sns.boxplot(data=data_near_time, palette=["#3976B7", "#3976B7", "#3976B7"], width=0.3)
# Customize the plot
plt.xticks(ticks=range(3), labels=labels, fontsize=24)
plt.yticks(fontsize=24)
plt.savefig(f'new_figures5/new_near_boxplot_dam1_crust{dcont}.pdf')

plt.figure(figsize=(8, 6))
sns.boxplot(data=data_far_time, showfliers=False, palette=["#C5D9EF","#C5D9EF","#C5D9EF"], width=0.3)
# Customize the plot
plt.xticks(ticks=range(3), labels=labels, fontsize=24)
plt.yticks(fontsize=24)
eps = 1e-9
plt.ylim(0.0, 0.015)
plt.yticks(np.arange(0.0, 0.015 + eps, 0.005))
plt.savefig(f'new_figures5/new_far_boxplot_rmoutlier_dam1_crust{dcont}.pdf', bbox_inches='tight', pad_inches=0.02)

plt.figure(figsize=(8, 6))
sns.boxplot(data=data_far_time, palette=["#C5D9EF","#C5D9EF","#C5D9EF"], width=0.3)
# Customize the plot
plt.xticks(ticks=range(3), labels=labels, fontsize=24)
plt.yticks(fontsize=24)
plt.savefig(f'new_figures5/new_far_boxplot_dam1_crust{dcont}.pdf')
