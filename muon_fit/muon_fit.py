import glob
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from muon_fit.muon_sample import muon_sample


def muon_fit(data_dir="", mw0=0.62, mw1=0.73, gplot=None, hplot=None, ps=None, crplot=None, verbose=None):
    # Initial parameters
    bin_size = 0.025

    # Get image list
    imlist = sorted(glob.glob(f'{data_dir}/img*_00.fits'))
    if not imlist:
        print("No image files found.")
        return

    allbig = np.empty(0, dtype=np.float32)
    allsmall = np.empty(0, dtype=np.float32)
    p = 0

    # Loop over images
    for imf in imlist:
        print(f"Processing: {imf}")
        # Read image and segmentation files
        im = fits.getdata(imf)
        seg = fits.getdata(f"{imf.split('.')[0]}_seg.fits")

        # Read the corresponding CR.cat file with the updated format
        cat_file = f"{imf.split('.')[0]}_cr.cat"
        cat_data = np.loadtxt(cat_file, usecols=(0, 1, 2, 3, 4, 5), comments='#')

        sids, xs, ys, rms_a, rms_b, thetas = cat_data.T
        thetas = np.deg2rad(thetas)  # Convert angles from degrees to radians

        nseg = len(sids)
        big = np.full(nseg, -1.0, dtype=np.float32)
        small = np.full(nseg, -1.0, dtype=np.float32)

        for j in range(nseg):
            sno = sids[j]
            costh = np.cos(thetas[j])

            t = np.where(seg == sno)[1]
            if len(t) > 0:
                ind = np.array(np.unravel_index(t, seg.shape))

                # Get the range of x-points
                xs_range = [np.min(ind[1]), np.max(ind[1])]
                nx = xs_range[1] - xs_range[0] + 1

                if mw0 <= rms_b[j] <= mw1 and costh >= 0.7 and rms_a[j] > 9.0:
                    if verbose:
                        print(f"Segment {sno} passed")

                    sg = np.zeros(nx, dtype=np.float32)
                    for i in range(nx):
                        # Call muon_sample to fit CR
                        sig = muon_sample(im, seg, sno, xs_range[0] + i, gplot=gplot)
                        sg[i] = sig

                    # Ensure we have enough valid points to fit the data
                    valid = (sg > 0)
                    print(f'{np.sum(valid)} valid points for sno {sno}')
                    if np.sum(valid) >= 9:  # Minimum valid points to perform a fit
                        xx = np.arange(xs_range[0], xs_range[1] + 1)
                        sgth = sg * costh

                        # Attempt to fit a line to the valid points
                        coef = np.polyfit(xx[valid], sgth[valid], 1)

                        # Evaluate the polynomial fit
                        yfit = np.polyval(coef, xx)

                        # Calculate residuals and apply a weight mask
                        diff = sgth - yfit
                        weight_mask = np.abs(diff) < 2  # Example weight filter

                        # Ensure there are valid weights
                        if np.any(weight_mask):
                            big[j] = np.max(yfit[weight_mask])
                            small[j] = np.min(yfit[weight_mask])

                            if big[j] > 0 and small[j] > 0:
                                allbig = np.append(allbig, big[j])
                                allsmall = np.append(allsmall, small[j])
                                p += 1

                            # Optional crplot
                            if crplot:
                                plot_muon(xx, sg, yfit, sno)
                        else:
                            print(f"No valid weights for fitting in segment {sno}")
                    else:
                        print(f"Not enough valid points for fitting in segment {sno}")
            else:
                print(f"Could not find segment {sno}")

        big = big[big > 0]
        small = small[small > 0]
        print(f'{imf}: N CRs = {len(big)}')

        # Optional histogram plotting
        if hplot:
            plt.hist(big, bins=np.arange(0, 1.5, bin_size), alpha=0.7, label='Big')
            plt.hist(small, bins=np.arange(0, 1.5, bin_size), alpha=0.7, label='Small', linestyle='--')
            plt.title(imf)
            plt.legend()
            plt.show()


    allbig = allbig[allbig > 0]
    allsmall = allsmall[allsmall > 0]

    # Final histogram over all files
    print(f"allbig and allsmall lengths: {len(allbig)} vs {len(allsmall)}")
    diflen = np.sqrt(allbig ** 2 - allsmall ** 2) * 15.0
    diflen2 = np.sqrt(allbig ** 2 - (1/12)) * 15.0 # (1/12^(1/2))**2 is 1/12

    print(f"Processed {len(imlist)} images, {p} muon CRs")

    # Create the histograms
    bin_width = 0.25
    bins = np.arange(0, 50 + bin_width, bin_width)

    # First histogram (σMAX - σMIN)
    hist1, bin_edges1 = np.histogram(diflen[diflen <= 50], bins=bins)

    # Second histogram (σMAX - 1/12^(1/2))
    hist2, bin_edges2 = np.histogram(diflen2[diflen2 <= 50], bins=bins)

    # Plotting the histograms
    plt.figure(figsize=(10, 6))

    # Plot first histogram
    plt.hist(bin_edges1[:-1], bins, histtype='step', weights=hist1, label='σMAX - σMIN', color='black', linestyle='-')

    # Plot second histogram
    plt.hist(bin_edges2[:-1], bins, histtype='step', weights=hist2, label='σMAX - 1/12^(1/2)', color='black', linestyle='--')

    # Adding labels and title
    plt.title('Diffusion Lengths')
    plt.xlabel('Diffusion Length (µm)')
    plt.ylabel('N')
    plt.legend(loc='upper right')
    plt.xlim(2,30)

    # Display the plot
    plt.show()

# Cr_plot
def plot_muon(xx, sg, yfit, sno, psfile=None):
    # Set up plot appearance
    fig, ax = plt.subplots()
    ax.set_facecolor('white')
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')

    # Font size and thickness
    th = 5
    si = 1.75
    ax.set_title(f'CR#: {sno}', fontsize=si * 10)
    ax.set_xlabel('X (px)', fontsize=si * 10, weight='bold')
    ax.set_ylabel('Signal (px)', fontsize=si * 10, weight='bold')
    ax.tick_params(axis='both', which='major', labelsize=si * 10, width=th)
    ax.set_xlim([xx.min() - 1, xx.max() + 1])
    ax.set_ylim([sg.min() - 1, sg.max() + 1])

    print(f"xx: {xx}, sg: {sg}, yfit: {yfit}")

    # Plot the signal points
    ax.plot(xx, sg, 'o', label='Signal', color='blue', markersize=5)

    # Plot the linear fit
    ax.plot(xx, yfit, '--', label='Fit', linewidth=3, color='red')

    # Plot vertical lines marking the big and small values
    ax.axvline(x=xx.max(), linestyle=':', color='black', linewidth=2, label='Big Value')
    ax.axvline(x=xx.min(), linestyle=':', color='black', linewidth=2, label='Small Value')

    # Horizontal reference line at 1/sqrt(12)
    ax.axhline(y=1 / np.sqrt(12), linestyle='-.', color='black', linewidth=3)

    # # Add labels for big/small and 1/12^0.5
    # ax.text(1305, 0.3, '1/12$^{1/2}$', fontsize=si * 10)
    # ax.text(1284, 0.2, 'MAX', fontsize=si * 10)
    # ax.text(1325, 0.2, 'MIN', fontsize=si * 10)

    # Add legend
    ax.legend()

    # Save the figure if a filename is provided
    if psfile:
        plt.savefig(psfile)

    # Show the plot
    plt.show()
