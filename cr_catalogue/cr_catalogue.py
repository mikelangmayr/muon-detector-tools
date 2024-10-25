import glob
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.signal import find_peaks


def cr_catalogue(data_dir=""):
    # Get image list
    cr_list = sorted(glob.glob(f'{data_dir}/img*_cr.cat'))
    if not cr_list:
        print("No cr files found.")
        return

    all_rms_b = []

    # Loop over images
    for cat_file in cr_list:
        # Read CR.cat file with the updated format
        cat_data = np.loadtxt(cat_file, usecols=(0, 1, 2, 3, 4, 5), comments='#')

        sids, xs, ys, rms_a, rms_b, thetas = cat_data.T

        # Add cr_file values to list
        all_rms_b.extend(rms_b)


    # print(f'All yrms list: {all_yrms}')
    counts, bin_edges = np.histogram(all_rms_b, bins=100, range=(0,3), density=True)

    peaks, _ = find_peaks(counts, prominence=0.001)

    # Get the main spike (the one with the highest count)
    # if len(peaks) > 0:
    #     main_spike = peaks[np.argmax(counts[peaks])]
    #
    #     # Finding the beginning of the main spike (left side)
    #     spike_start = main_spike
    #     while spike_start > 0 and counts[spike_start - 1] <= counts[spike_start]:
    #         spike_start -= 1
    #
    #     # Finding the end of the main spike (right side)
    #     spike_end = main_spike
    #     while spike_end < len(counts) - 1 and counts[spike_end + 1] <= counts[spike_end]:
    #         spike_end += 1
    #
    #     # Get the bin edges corresponding to the start and end of the main spike
    #     spike_start_edge = bin_edges[spike_start]
    #     spike_end_edge = bin_edges[spike_end + 1]
    #
    #     print(f"The main spike starts at: {spike_start_edge}")
    #     print(f"The main spike ends at: {spike_end_edge}")
    #
    #     # Plotting the histogram with main spike highlighted
    #     plt.hist(all_rms_b, bins=100, range=(0, 3), edgecolor='black', density=True, color='blue')
    #
    #     # Highlighting the main spike range in red
    #     plt.axvspan(spike_start_edge, spike_end_edge, color='red', alpha=0.5)
    #
    #     # Adding labels and title
    #     plt.xlabel('Yrms (px)')
    #     plt.ylabel('N (norm)')
    #     plt.title('Histogram with Main Spike Range Highlighted')
    #
    #     plt.show()
    # else:
    #     print("No spikes found.")

    # Calculate z-scores for the bin counts
    z_scores = zscore(counts)

    # Identify bins that are considered spikes (e.g., z-score > 2)
    spike_bins = np.where(z_scores > 2)[0]

    if len(spike_bins) > 0:
        # Assuming the main spike is the one with the highest count in the detected spike bins
        main_spike = spike_bins[np.argmax(counts[spike_bins])]

        # Finding the beginning of the main spike (left side)
        # spike_start = main_spike
        # while spike_start > 0 and counts[spike_start - 1] <= counts[spike_start]:
        #     spike_start -= 1

        # Finding the end of the main spike (right side)
        # spike_end = main_spike
        # while spike_end < len(counts) - 1 and counts[spike_end + 1] <= counts[spike_end]:
        #     spike_end += 1

        # Get the bin edges corresponding to the start and end of the main spike
        spike_start_edge = float(bin_edges[spike_bins[0]])
        spike_end_edge = float(bin_edges[spike_bins[len(spike_bins) -1 ] + 1])

        print(f"The main spike starts at: {spike_start_edge}")
        print(f"The main spike ends at: {spike_end_edge}")

        # Plotting the histogram with main spike highlighted
        plt.hist(all_rms_b, bins=100, range=(0, 3), edgecolor='black', density=True, color='blue')

        # Highlighting the main spike range in red
        plt.axvspan(spike_start_edge, spike_end_edge, color='red', alpha=0.5)

        # Adding labels and title
        plt.xlabel('Yrms (px)')
        plt.ylabel('N (norm)')
        plt.title('Histogram with Main Spike Range Highlighted (Z-Score)')

        plt.show()

        return [spike_start_edge, spike_end_edge]
    else:
        print("No spikes found.")



    # # # Plotting the histogram with spikes highlighted
    # plt.hist(all_rms_b, bins=100, range=(0, 3), edgecolor='black', density=True, color='blue')
    #
    # for bin_index in spike_bins:
    #     plt.bar(bin_edges[bin_index], counts[bin_index], width=bin_edges[1] - bin_edges[0], color='red', edgecolor='black')
    #
    # print(f'spike bins: {spike_bins}')
    #
    # # Adding labels and title
    # plt.xlabel('Rmsb (px)')
    # plt.ylabel('N (norm)')
    # plt.title('Histogram of RMSb Values from entire CR')
    #
    # # Show the plot
    # plt.show()
