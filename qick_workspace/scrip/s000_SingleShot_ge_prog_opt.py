# ===================================================================
# 1. Standard & Third-Party Scientific Libraries
# ===================================================================
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from tqdm.auto import tqdm


# ===================================================================
# 2. QICK Libraries
# ===================================================================
from qick import *
from qick.pyro import make_proxy
from qick.asm_v2 import AveragerProgramV2, QickSpan, QickSweep1D

# ===================================================================
# 3. User/Local Libraries
# ===================================================================
from ..tools.system_cfg import *
from ..tools.system_cfg import DATA_PATH
from ..tools.system_tool import get_next_filename_labber, hdf5_generator
from ..tools.fitting import fit_doublegauss, double_gaussian


##################
# plot hist
##################


# Use np.hist and plt.plot to accomplish plt.hist with less memory usage
default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
linestyle_cycle = ["solid", "dashed", "dotted", "dashdot"]
marker_cycle = ["o", "*", "s", "^"]


def plot_hist(
    data,
    bins,
    ax=None,
    xlims=None,
    color=None,
    linestyle=None,
    label=None,
    alpha=None,
    normalize=True,
):
    if color is None:
        color_cycle = cycle(default_colors)
        color = next(color_cycle)
    hist_data, bin_edges = np.histogram(data, bins=bins, range=xlims)
    if normalize:
        hist_data = hist_data / hist_data.sum()
    for i in range(len(hist_data)):
        if i > 0:
            label = None
        ax.plot(
            [bin_edges[i], bin_edges[i + 1]],
            [hist_data[i], hist_data[i]],
            color=color,
            linestyle=linestyle,
            label=label,
            alpha=alpha,
            linewidth=0.9,
        )
        if i < len(hist_data) - 1:
            ax.plot(
                [bin_edges[i + 1], bin_edges[i + 1]],
                [hist_data[i], hist_data[i + 1]],
                color=color,
                linestyle=linestyle,
                alpha=alpha,
                linewidth=0.9,
            )
    ax.relim()
    ax.set_ylim((0, None))
    return hist_data, bin_edges


# ===================================================================== #
def general_hist(
    iqshots,
    state_labels,
    g_states,
    e_states,
    e_label="e",
    check_qubit_label=None,
    numbins=200,
    amplitude_mode=False,
    ps_threshold=None,
    theta=None,
    plot=True,
    verbose=True,
    fid_avg=False,
    fit=False,
    gauss_overlap=False,
    fitparams=None,
    normalize=True,
    title=None,
    export=False,
    check_qnd=False,
):
    """
    span: histogram limit is the mean +/- span
    theta given and returned in deg
    assume iqshots = [(idata, qdata)]*len(check_states), idata=[... *num_shots]*num_qubits_sample
    g_states are indices to the check_states to categorize as "g" (the rest are "e")
    e_label: label to put on the cumulative counts for the "e" state, i.e. the state relative to which the angle/fidelity is calculated
    check_qubit_label: label to indicate which qubit is being measured
    fid_avg: determines the method of calculating the fidelity (whether to average the mis-categorized e/g or count the total number of miscategorized over total counts)
    normalize: normalizes counts by total counts
    """
    if numbins is None:
        numbins = 200

    # total histograms for shots listed as g or e
    Ig_tot = []
    Qg_tot = []
    Ie_tot = []
    Qe_tot = []

    # the actual total histograms of everything
    Ig_tot_tot = []
    Qg_tot_tot = []
    Ie_tot_tot = []
    Qe_tot_tot = []
    for check_i, data_check in enumerate(iqshots):
        I, Q = data_check
        Ig_tot_tot = np.concatenate((Ig_tot_tot, I))
        Qg_tot_tot = np.concatenate((Qg_tot_tot, Q))
        Ie_tot_tot = np.concatenate((Ig_tot_tot, I))
        Qe_tot_tot = np.concatenate((Qg_tot_tot, Q))
        if check_i in g_states:
            Ig_tot = np.concatenate((Ig_tot, I))
            Qg_tot = np.concatenate((Qg_tot, Q))
        elif check_i in e_states:
            Ie_tot = np.concatenate((Ig_tot, I))
            Qe_tot = np.concatenate((Qg_tot, Q))

    if not amplitude_mode:
        """Compute the rotation angle"""
        if theta is None:
            xg, yg = np.average(Ig_tot), np.average(Qg_tot)
            xe, ye = np.average(Ie_tot), np.average(Qe_tot)
            theta = -np.arctan2((ye - yg), (xe - xg))
        else:
            theta *= np.pi / 180
        Ig_tot_tot_new = Ig_tot_tot * np.cos(theta) - Qg_tot_tot * np.sin(theta)
        Qg_tot_tot_new = Ig_tot_tot * np.sin(theta) + Qg_tot_tot * np.cos(theta)
        Ie_tot_tot_new = Ie_tot_tot * np.cos(theta) - Qe_tot_tot * np.sin(theta)
        Qe_tot_tot_new = Ie_tot_tot * np.sin(theta) + Qe_tot_tot * np.cos(theta)
        I_tot_tot_new = np.concatenate((Ie_tot_tot_new, Ig_tot_tot_new))
        span = (np.max(I_tot_tot_new) - np.min(I_tot_tot_new)) / 2
        midpoint = (np.max(I_tot_tot_new) + np.min(I_tot_tot_new)) / 2
    else:
        theta = 0
        amp_g_tot_tot = np.abs(Ig_tot_tot + 1j * Qg_tot_tot)
        amp_e_tot_tot = np.abs(Ie_tot_tot + 1j * Qe_tot_tot)
        amp_tot_tot = np.concatenate((amp_g_tot_tot, amp_e_tot_tot))
        span = (np.max(amp_tot_tot) - np.min(amp_tot_tot)) / 2
        midpoint = (np.max(amp_tot_tot) + np.min(amp_tot_tot)) / 2
    xlims = [midpoint - span, midpoint + span]

    if plot:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(9, 6))
        if title is None:
            title = f"Readout Fidelity" + (
                f" on Q{check_qubit_label}" if check_qubit_label is not None else ""
            )
        fig.suptitle(title)
        fig.tight_layout()
        axs[0, 0].set_ylabel("Q [ADC levels]", fontsize=11)
        axs[0, 0].set_title("Unrotated", fontsize=13)
        axs[0, 0].axis("equal")
        axs[0, 0].tick_params(axis="both", which="major", labelsize=10)
        axs[0, 0].set_xlabel("I [ADC levels]", fontsize=11)

        axs[0, 1].axis("equal")
        axs[0, 1].tick_params(axis="both", which="major", labelsize=10)
        axs[0, 1].set_xlabel("I [ADC levels]", fontsize=11)

        threshold_axis = "I" if not amplitude_mode else "Amplitude"
        axs[1, 0].set_ylabel("Counts", fontsize=12)
        axs[1, 0].set_xlabel(f"{threshold_axis} [ADC levels]", fontsize=11)
        axs[1, 0].tick_params(axis="both", which="major", labelsize=10)

        axs[1, 1].set_title("Cumulative Counts", fontsize=13)
        axs[1, 1].set_xlabel(f"{threshold_axis} [ADC levels]", fontsize=11)
        axs[1, 1].tick_params(axis="both", which="major", labelsize=10)
        plt.subplots_adjust(hspace=0.35, wspace=0.15)

    y_max = 0
    n_tot_g = [0] * numbins
    n_tot_e = [0] * numbins

    # FIX: Create a flag to ensure fitting runs if either fit=True OR overlap=True
    do_fit = fit or gauss_overlap

    if do_fit:
        popts = [None] * len(state_labels)
        pcovs = [None] * len(state_labels)

    """
    Loop over check states
    """
    y_max = 0
    for check_i, data_check in enumerate(iqshots):
        state_label = state_labels[check_i]

        I, Q = data_check
        amp = np.abs(I + 1j * Q)

        xavg, yavg, amp_avg = np.average(I), np.average(Q), np.average(amp)

        """Rotate the IQ data"""
        I_new = I * np.cos(theta) - Q * np.sin(theta)
        Q_new = I * np.sin(theta) + Q * np.cos(theta)

        """New means of each blob"""
        xavg_new, yavg_new = np.average(I_new), np.average(Q_new)

        if verbose:
            print(state_label, "unrotated averages:")
            if not amplitude_mode:
                print(
                    f"I {xavg:.3f} +/- {np.std(I):.3f} \t Q {yavg:.3f} +/- {np.std(Q):.3f} \t Amp {amp_avg:.3f} +/- {np.std(amp):.3f}"
                )
                print(
                    f"Rotated (theta={theta:.3f}):"
                )  # Assuming theta also needs limiting
                print(
                    f"I {xavg_new:.3f} +/- {np.std(I_new):.3f} \t Q {yavg_new:.3f} +/- {np.std(Q_new):.3f} \t Amp {np.abs(xavg_new + 1j * yavg_new):.3f} +/- {np.std(amp):.3f}"
                )
            else:
                print(f"Amps {amp_avg:.3f} +/- {np.std(amp):.3f}")

        if plot:
            axs[0, 0].scatter(
                I,
                Q,
                label=state_label,
                color=default_colors[check_i % len(default_colors)],
                marker=".",
                edgecolor="None",
                alpha=0.3,
            )
            axs[0, 0].plot(
                [xavg],
                [yavg],
                color="k",
                linestyle=":",
                marker="o",
                markerfacecolor=default_colors[check_i % len(default_colors)],
                markersize=5,
            )

            axs[0, 1].scatter(
                I_new,
                Q_new,
                label=state_label,
                color=default_colors[check_i % len(default_colors)],
                marker=".",
                edgecolor="None",
                alpha=0.3,
            )
            axs[0, 1].plot(
                [xavg_new],
                [yavg_new],
                color="k",
                linestyle=":",
                marker="o",
                markerfacecolor=default_colors[check_i % len(default_colors)],
                markersize=5,
            )

            if check_i in g_states or check_i in e_states:
                linestyle = linestyle_cycle[0]
            else:
                linestyle = linestyle_cycle[1]

            # n, bins, p = axs[1,0].hist(I_new, bins=numbins, range=xlims, color=default_colors[check_i % len(default_colors)], label=label, histtype='step', linestyle=linestyle)
            n, bins = plot_hist(
                I_new if not amplitude_mode else amp,
                bins=numbins,
                ax=axs[1, 0],
                xlims=xlims,
                color=default_colors[check_i % len(default_colors)],
                label=state_label,
                linestyle=linestyle,
                normalize=normalize,
            )
            y_max = max(y_max, max(n))
            axs[1, 0].set_ylim((0, y_max * 1.1))
            # n, bins = np.histogram(I_new, bins=numbins, range=xlims)
            # axs[1,0].plot(bins[:-1], n/n.sum(), color=default_colors[check_i % len(default_colors)], linestyle=linestyle)

            axs[1, 1].plot(
                bins[:-1],
                np.cumsum(n) / n.sum(),
                color=default_colors[check_i % len(default_colors)],
                linestyle=linestyle,
            )

        else:  # just getting the n, bins for data processing
            n, bins = np.histogram(
                I_new if not amplitude_mode else amp, bins=numbins, range=xlims
            )

        if check_i in g_states:
            n_tot_g += n
            bins_g = bins
        elif check_i in e_states:
            n_tot_e += n
            bins_e = bins

        if check_qnd:
            if state_label == "g_0":
                n_g_0 = n
            if state_label == "g_1":
                n_g_1 = n

    if check_qnd:
        n_diff = np.abs((n_g_0 - n_g_1))
        n_diff_qnd = np.sum(n_diff) / 2 / np.sum(n_g_0)

    # FIX: Use do_fit flag here
    if do_fit:
            # 1. Determine Global Anchors
            # Use the total accumulated histograms (n_tot_g, n_tot_e) to find the
            # robust locations of the Ground (g) and Excited (e) states.
            # This prevents the fitter from getting lost on individual shots with low populations.
            xmax_g = bins_g[np.argmax(n_tot_g)]
            xmax_e = bins_e[np.argmax(n_tot_e)]
            
            # 2. Estimate Sigma Dynamically
            # Calculate the distance between G and E. A good rule of thumb is that 
            # the peaks are separated by roughly 4 to 6 sigmas.
            dist_ge = abs(xmax_e - xmax_g)
            if dist_ge > 0:
                sigma_guess = dist_ge / 6.0
            else:
                # Fallback: use a fraction of the total span
                sigma_guess = (bins[-1] - bins[0]) / 20.0

            # Safety check to prevent division by zero
            if sigma_guess < 1e-3: sigma_guess = 1.0

            for check_i, data_check in enumerate(iqshots):
                state_label = state_labels[check_i]
                I, Q = data_check
                
                # --- Data Rotation & Preparation ---
                I_new = I * np.cos(theta) - Q * np.sin(theta)
                Q_new = I * np.sin(theta) + Q * np.cos(theta)
                amp = np.abs(I_new + 1j * Q_new)

                target_data = amp if amplitude_mode else I_new
                n, bins = np.histogram(target_data, bins=numbins, range=xlims)
                
                # Use Bin Centers (more accurate than edges for curve fitting)
                bin_centers = (bins[:-1] + bins[1:]) / 2

                # --- Smart Initial Guesses ---
                # Instead of searching for peaks in this specific trace (which might lack one state),
                # we look up the amplitude at the *known* global G and E positions.
                
                # Find indices in current bins closest to global xmax_g/e
                idx_g = np.argmin(np.abs(bin_centers - xmax_g))
                idx_e = np.argmin(np.abs(bin_centers - xmax_e))
                
                # Read height at those positions
                ymax_g_guess = n[idx_g]
                ymax_e_guess = n[idx_e]

                # Ensure guess is non-zero for log-likelihood stability
                if ymax_g_guess < 1: ymax_g_guess = 1
                if ymax_e_guess < 1: ymax_e_guess = 1

                # Construct parameters: [Amp_g, Mean_g, Sigma_g, Amp_e, Mean_e, Sigma_e]
                # Note: We use the GLOBAL xmax_g/e as the mean guess.
                fitparams = [
                    ymax_g_guess, xmax_g, sigma_guess, 
                    ymax_e_guess, xmax_e, sigma_guess
                ]

                # Perform the Fit
                popt, pcov = fit_doublegauss(xdata=bin_centers, ydata=n, fitparams=fitparams)

                if plot:
                    # Generate the fitted curve
                    y_fit = double_gaussian(bin_centers, *popt)
                    
                    # Normalize the curve for plotting if necessary
                    # (Adjust this based on whether your 'plot_hist' function normalizes the data)
                    if normalize:
                        y_fit_norm = y_fit / y_fit.sum()
                    else:
                        y_fit_norm = y_fit

                    # Plot the fitted curve
                    axs[1, 0].plot(
                        bin_centers,
                        y_fit_norm,
                        "-",
                        linewidth=2.0,
                        color=default_colors[check_i % len(default_colors)],
                        alpha=0.8,
                        label=f"{state_label} fit"
                    )

                    # --- Calculate Gaussian Overlap Fidelity ---
                    _, b1, c1, _, b2, c2 = popt
                    
                    # Helper to calc PDF
                    def gaussian_pdf(x, mean, sigma):
                        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

                    # Helper to find intersection
                    def overlap_integrand(x):
                        return np.minimum(gaussian_pdf(x, b1, c1), gaussian_pdf(x, b2, c2))
                    
                    # Integrate over a relevant range (5 sigmas)
                    int_min = min(b1 - 5*c1, b2 - 5*c2)
                    int_max = max(b1 + 5*c1, b2 + 5*c2)
                    
                    area, _ = quad(overlap_integrand, int_min, int_max)
                    gauss_fit_fidelity = 1 - area

                popts[check_i] = popt
                pcovs[check_i] = pcov

        if plot:
            y = double_gaussian(bins[:-1], *popt)
            y_norm = y / y.sum()

            axs[1, 0].plot(
                bins[:-1],
                y_norm,
                "-",
                color=default_colors[check_i % len(default_colors)],
            )

            def gaussian_norm(x, b, c):
                a = 1 / (np.sqrt(2 * np.pi) * c)
                return a * np.exp(-((x - b) ** 2) / (2 * c**2))

            def overlap_area_norm(b1, c1, b2, c2):
                def min_func(x):
                    return np.minimum(
                        gaussian_norm(x, b1, c1), gaussian_norm(x, b2, c2)
                    )

                x_min = min(b1 - 5 * c1, b2 - 5 * c2)
                x_max = max(b1 + 5 * c1, b2 + 5 * c2)
                area, _ = quad(min_func, x_min, x_max)
                return area

            def readout_fidelity_norm(b1, c1, b2, c2):
                overlap = overlap_area_norm(b1, c1, b2, c2)
                return 1 - overlap

            _, b1, c1, _, b2, c2 = popt
            gauss_fit_fidelity = readout_fidelity_norm(b1, c1, b2, c2)

        popts[check_i] = popt
        pcovs[check_i] = pcov

    """Compute the fidelity using overlap of the histograms"""
    fids = []
    thresholds = []
    # this method calculates fidelity as 1-2(Neg + Nge)/N
    contrast = np.abs(
        (
            (np.cumsum(n_tot_g) - np.cumsum(n_tot_e))
            / (0.5 * n_tot_g.sum() + 0.5 * n_tot_e.sum())
        )
    )
    tind = contrast.argmax()
    thresholds.append(bins[tind])
    # thresholds.append(np.average([bins_e[idx_e], bins_g[idx_g]]))
    if not fid_avg:
        fids.append(contrast[tind])
    else:
        # this method calculates fidelity as
        # (Ngg+Nee)/N = Ngg/N + Nee/N=(0.5N-Nge)/N + (0.5N-Neg)/N = 1-(Nge+Neg)/N
        fids.append(
            0.5
            * (
                1
                - n_tot_g[tind:].sum() / n_tot_g.sum()
                + 1
                - n_tot_e[:tind].sum() / n_tot_e.sum()
            )
        )

    if plot:
        axs[0, 1].set_title(
            f"Rotated ($\\theta={theta * 180 / np.pi:.5}^\\circ$)", fontsize=13
        )

        axs[1, 0].axvline(thresholds[0], color="0.2", linestyle="--")
        title = (
            "$\overline{F}_{g" + e_label + "}$" if fid_avg else "$F_{g" + e_label + "}$"
        )
        if gauss_overlap:
            # gauss_fit_fidelity is now guaranteed to be defined because do_fit ensured the block ran
            axs[1, 0].set_title(f"{title}: {100 * gauss_fit_fidelity:.3}%", fontsize=13)
        else:
            axs[1, 0].set_title(f"{title}: {100 * fids[0]:.3}%", fontsize=13)
        if ps_threshold is not None:
            axs[1, 0].axvline(ps_threshold, color="0.2", linestyle="-.")

        axs[1, 1].plot(bins[:-1], np.cumsum(n_tot_g) / n_tot_g.sum(), "b", label="g")
        axs[1, 1].plot(
            bins[:-1], np.cumsum(n_tot_e) / n_tot_e.sum(), "r", label=e_label
        )
        axs[1, 1].axvline(thresholds[0], color="0.2", linestyle="--")

        prop = {"size": 8}
        axs[0, 0].legend(prop=prop)
        axs[0, 1].legend(prop=prop)
        axs[1, 0].legend(prop=prop)
        axs[1, 1].legend(prop=prop)

        if export:
            plt.savefig("multihist.jpg", dpi=1000)
            print("exported multihist.jpg")
            plt.close()
        else:
            plt.show()

    # fids: ge, gf, ef
    if gauss_overlap:
        return_data = [gauss_fit_fidelity, thresholds, theta * 180 / np.pi]
    else:
        return_data = [fids, thresholds, theta * 180 / np.pi]
    if fit:
        return_data += [popts, pcovs]
    if check_qnd:
        return_data += [n_diff_qnd]
    if verbose:
        print(
            f"fidelity:{fids} \nthressholds:{thresholds} \ntheta:{theta * 180 / np.pi}"
        )
        gg = 100 * (1 - n_tot_g[tind:].sum() / n_tot_g.sum())
        ge = 100 * (n_tot_g[tind:].sum() / n_tot_g.sum())
        eg = 100 * (1 - n_tot_e[tind:].sum() / n_tot_e.sum())
        ee = 100 * (n_tot_e[tind:].sum() / n_tot_e.sum())
        print(f"""
            Fidelity Matrix:
            -----------------
            | {gg:.3f}% | {ge:.3f}% |
            ----------------
            | {eg:.3f}% | {ee:.3f}% |
            -----------------
            IQ plane rotated by: {180 / np.pi * theta:.1f}{chr(176)}
            Threshold: {thresholds[0]:.3e}
            Fidelity: {100 * fids[0]:.3f}%
            """)
    return return_data


# ===================================================================== #


def hist(
    data,
    amplitude_mode=False,
    ps_threshold=None,
    theta=None,
    plot=True,
    verbose=True,
    fid_avg=False,
    fit=False,
    gauss_overlap=False,
    fitparams=None,
    normalize=True,
    title=None,
    export=False,
):
    Ig = data["Ig"]
    Qg = data["Qg"]
    Ie = data["Ie"]
    Qe = data["Qe"]
    iqshots = [(Ig, Qg), (Ie, Qe)]
    state_labels = ["g", "e"]
    g_states = [0]
    e_states = [1]

    if "If" in data.keys():
        If = data["If"]
        Qf = data["Qf"]
        iqshots.append((If, Qf))
        state_labels.append("f")
        e_states = [2]

    return general_hist(
        iqshots=iqshots,
        state_labels=state_labels,
        g_states=g_states,
        e_states=e_states,
        amplitude_mode=amplitude_mode,
        ps_threshold=ps_threshold,
        theta=theta,
        plot=plot,
        verbose=verbose,
        fid_avg=fid_avg,
        fit=fit,
        gauss_overlap=gauss_overlap,
        fitparams=fitparams,
        normalize=normalize,
        title=title,
        export=export,
    )


# ===================================================================== #


def multihist(
    data,
    check_qubit,
    check_states,
    play_pulses_list,
    g_states,
    e_states,
    numbins=200,
    amplitude_mode=False,
    ps_threshold=None,
    theta=None,
    plot=True,
    verbose=True,
    fid_avg=False,
    fit=False,
    fitparams=None,
    normalize=True,
    title=None,
    export=False,
    check_qnd=False,
):
    """
    Assumes data is passed in via data["iqshots"] = [(idata, qdata)]*len(check_states), idata=[... *num_shots]*num_qubits_sample

    These are mostly for labeling purposes:
    check_states: an array of strs of the init_state specifying each configuration to plot a histogram for
    play_pulses_list: list of play_pulses corresponding to check_states, see code for play_pulses
    """
    state_labels = []
    assert len(play_pulses_list) == len(check_states)
    for i in range(len(check_states)):
        check_state = check_states[i]
        play_pulses = play_pulses_list[i]
        label = f"{check_state}"
        # print('check state', check_state)
        if len(play_pulses) > 1 or play_pulses[0] != 0:
            label += f" play {play_pulses}"
        state_labels.append(label)
    all_q_iqshots = data["iqshots"]
    iqshots = []
    for check_i, data_check in enumerate(all_q_iqshots):
        I, Q = data_check
        I = I[check_qubit]
        Q = Q[check_qubit]
        iqshots.append((I, Q))
    check_qubit_label = check_qubit
    return_data = general_hist(
        iqshots=iqshots,
        check_qubit_label=check_qubit_label,
        state_labels=state_labels,
        g_states=g_states,
        e_states=e_states,
        numbins=numbins,
        amplitude_mode=amplitude_mode,
        ps_threshold=ps_threshold,
        theta=theta,
        plot=plot,
        verbose=verbose,
        fid_avg=fid_avg,
        fit=fit,
        fitparams=fitparams,
        normalize=normalize,
        title=title,
        export=export,
        check_qnd=check_qnd,
    )
    if check_qnd:
        data["n_diff_qnd"] = return_data[-1]
    return return_data



##################
# Define Program #
##################

# Separate g and e per each experiment defined.


class SingleShotProgram_g(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]
        qubit_ch = cfg["qubit_ch"]

        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])
        if self.soccfg["gens"][qubit_ch]["type"] == "axis_sg_int4_v2":
            self.declare_gen(
                ch=qubit_ch, nqz=cfg["nqz_qubit"], mixer_freq=cfg["qmixer_freq"]
            )
        else:
            self.declare_gen(ch=qubit_ch, nqz=cfg["nqz_qubit"])

        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])
        self.add_readoutconfig(
            ch=ro_ch, name="myro", freq=cfg["res_freq_ge"], gen_ch=res_ch
        )

        self.add_loop("shotloop", cfg["shots"])

        self.add_gauss(
            ch=res_ch,
            name="readout",
            sigma=cfg["res_sigma"],
            length=5 * cfg["res_sigma"],
            even_length=True,
        )
        self.add_pulse(
            ch=res_ch,
            name="res_pulse",
            ro_ch=ro_ch,
            style="flat_top",
            envelope="readout",
            length=cfg["res_length"],
            freq=cfg["res_freq_ge"],
            phase=cfg["res_phase"],
            gain=cfg["res_gain_ge"],
        )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        self.delay_auto(0.01, tag="wait")
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class SingleShotProgram_e(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]
        qubit_ch = cfg["qubit_ch"]

        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])
        if self.soccfg["gens"][qubit_ch]["type"] == "axis_sg_int4_v2":
            self.declare_gen(
                ch=qubit_ch, nqz=cfg["nqz_qubit"], mixer_freq=cfg["qmixer_freq"]
            )
        else:
            self.declare_gen(ch=qubit_ch, nqz=cfg["nqz_qubit"])

        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])
        self.add_readoutconfig(
            ch=ro_ch, name="myro", freq=cfg["res_freq_ge"], gen_ch=res_ch
        )

        self.add_loop("shotloop", cfg["shots"])

        self.add_gauss(
            ch=res_ch,
            name="readout",
            sigma=cfg["res_sigma"],
            length=5 * cfg["res_sigma"],
            even_length=True,
        )
        self.add_pulse(
            ch=res_ch,
            name="res_pulse",
            ro_ch=ro_ch,
            style="flat_top",
            envelope="readout",
            length=cfg["res_length"],
            freq=cfg["res_freq_ge"],
            phase=cfg["res_phase"],
            gain=cfg["res_gain_ge"],
        )

        self.add_gauss(
            ch=qubit_ch,
            name="ramp",
            sigma=cfg["sigma"],
            length=cfg["sigma"] * 5,
            even_length=True,
        )
        if cfg["qubit_ge_pulse_style"] == "arb":
            self.add_pulse(
                ch=qubit_ch,
                name="qubit_pulse",
                ro_ch=ro_ch,
                style="arb",
                envelope="ramp",
                freq=cfg["qubit_freq_ge"],
                phase=cfg["qubit_phase"],
                gain=cfg["qubit_pi_gain_ge"],
            )
        elif cfg["qubit_ge_pulse_style"] == "flat_top":
            self.add_pulse(
                ch=qubit_ch,
                name="qubit_pulse",
                ro_ch=ro_ch,
                style="flat_top",
                envelope="ramp",
                freq=cfg["qubit_freq_ge"],
                phase=cfg["qubit_phase"],
                gain=cfg["qubit_pi_gain_ge"],
                length=cfg["qubit_flat_top_length_ge"],
            )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=0)
        self.delay_auto(0.01, tag="wait")
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class SingleShot_ge_opt:
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(self, SHOTS, sweep_para: dict):
        self.cfg["shots"] = SHOTS

        raw_length = sweep_para.get("length")
        self.length_sweep = (
            raw_length
            if isinstance(raw_length, (list, tuple, np.ndarray))
            else [raw_length]
        )

        raw_gain = sweep_para.get("gain")
        self.gain_sweep = (
            raw_gain if isinstance(raw_gain, (list, tuple, np.ndarray)) else [raw_gain]
        )

        raw_freq = sweep_para.get("freq")
        self.freq_sweep = (
            raw_freq if isinstance(raw_freq, (list, tuple, np.ndarray)) else [raw_freq]
        )

        final_shape = (
            len(self.length_sweep),
            len(self.gain_sweep),
            len(self.freq_sweep),
            SHOTS,
        )

        self.I_g_array = np.full(final_shape, np.nan)
        self.Q_g_array = np.full(final_shape, np.nan)
        self.I_e_array = np.full(final_shape, np.nan)
        self.Q_e_array = np.full(final_shape, np.nan)

        # --- TQDM 動態設定 ---
        is_l_sweep = len(self.length_sweep) > 1
        is_g_sweep = len(self.gain_sweep) > 1
        is_f_sweep = len(self.freq_sweep) > 1

        outermost_real_sweep = None
        if is_l_sweep:
            outermost_real_sweep = "l"
        elif is_g_sweep:
            outermost_real_sweep = "g"
        elif is_f_sweep:
            outermost_real_sweep = "f"

        # 1. 建立 L (最外層) 迭代器
        l_iter = self.length_sweep
        if "l" == outermost_real_sweep:
            l_iter = tqdm(self.length_sweep, desc="Length loop")

        for l_idx, l_val in enumerate(l_iter):
            # --- 修正點：G 迭代器在 L 迴圈內建立 ---
            g_iter = self.gain_sweep
            if "g" == outermost_real_sweep:
                g_iter = tqdm(self.gain_sweep, desc="Gain loop")
            elif is_g_sweep:
                g_iter = tqdm(self.gain_sweep, desc="Gain loop", leave=False)

            for g_idx, g_val in enumerate(g_iter):
                # --- 修正點：F 迭代器在 G 迴圈內建立 ---
                f_iter = self.freq_sweep
                if "f" == outermost_real_sweep:
                    f_iter = tqdm(self.freq_sweep, desc="Freq loop")
                elif is_f_sweep:
                    f_iter = tqdm(self.freq_sweep, desc="Freq loop", leave=False)

                for f_idx, f_val in enumerate(f_iter):
                    cfg_update = {"steps": SHOTS}
                    if l_val is not None:
                        cfg_update["ro_length"] = l_val
                    if g_val is not None:
                        cfg_update["res_gain_ge"] = g_val
                    if f_val is not None:
                        cfg_update["res_freq_ge"] = f_val

                    self.cfg.update(cfg_update)

                    ssp_g = SingleShotProgram_g(
                        self.soccfg,
                        reps=1,
                        final_delay=self.cfg["relax_delay"],
                        cfg=self.cfg,
                    )
                    iq_list_g = ssp_g.acquire(self.soc, rounds=1, progress=False)

                    ssp_e = SingleShotProgram_e(
                        self.soccfg,
                        reps=1,
                        final_delay=self.cfg["relax_delay"],
                        cfg=self.cfg,
                    )
                    iq_list_e = ssp_e.acquire(self.soc, rounds=1, progress=False)

                    I_g = iq_list_g[0][0].T[0]
                    Q_g = iq_list_g[0][0].T[1]
                    I_e = iq_list_e[0][0].T[0]
                    Q_e = iq_list_e[0][0].T[1]

                    self.I_g_array[l_idx, g_idx, f_idx, :] = I_g
                    self.Q_g_array[l_idx, g_idx, f_idx, :] = Q_g
                    self.I_e_array[l_idx, g_idx, f_idx, :] = I_e
                    self.Q_e_array[l_idx, g_idx, f_idx, :] = Q_e

        self.data = {
            "Ig": self.I_g_array,
            "Qg": self.Q_g_array,
            "Ie": self.I_e_array,
            "Qe": self.Q_e_array,
        }

    def analyze(self):
        try:
            from scipy.interpolate import RegularGridInterpolator
            from scipy.optimize import minimize

            SCIPY_AVAILABLE = True
        except ImportError:
            SCIPY_AVAILABLE = False

        try:
            len_L = len(self.length_sweep)
            len_G = len(self.gain_sweep)
            len_F = len(self.freq_sweep)
        except AttributeError:
            print("Error: 'run' method must be called first to define sweep axes.")
            return

        fid_Array = np.zeros((len_L, len_G, len_F))

        I_g_data = self.data["Ig"]
        Q_g_data = self.data["Qg"]
        I_e_data = self.data["Ie"]
        Q_e_data = self.data["Qe"]

        for l_idx in tqdm(range(len_L), desc="Analyze max fidelity"):
            for g_idx in range(len_G):
                for f_idx in range(len_F):
                    I_g = I_g_data[l_idx, g_idx, f_idx]
                    Q_g = Q_g_data[l_idx, g_idx, f_idx]
                    I_e = I_e_data[l_idx, g_idx, f_idx]
                    Q_e = Q_e_data[l_idx, g_idx, f_idx]

                    vec_I = np.mean(I_e) - np.mean(I_g)
                    vec_Q = np.mean(Q_e) - np.mean(Q_g)
                    denom = np.abs(vec_I + 1j * vec_Q) ** 2

                    if denom == 0:
                        fid_Array[l_idx, g_idx, f_idx] = 0.0
                        continue

                    gstate = (
                        (I_g - np.mean(I_g)) * vec_I + (Q_g - np.mean(Q_g)) * vec_Q
                    ) / denom
                    estate = (
                        (I_e - np.mean(I_g)) * vec_I + (Q_e - np.mean(Q_g)) * vec_Q
                    ) / denom

                    if gstate.size == 0 or estate.size == 0:
                        fid_Array[l_idx, g_idx, f_idx] = 0.0
                        continue

                    min_g = np.min(gstate)
                    max_e = np.max(estate)
                    if min_g >= max_e:
                        min_g, max_e = min(min_g, max_e), max(min_g, max_e)
                        if min_g == max_e:
                            max_e += 1e-9

                    th_list = np.linspace(min_g, max_e, 1000).reshape(-1, 1)
                    g_proj = np.array(gstate).reshape(1, -1)
                    e_proj = np.array(estate).reshape(1, -1)

                    g_cdf = np.sum(g_proj < th_list, axis=1) / g_proj.shape[1]
                    e_cdf = np.sum(e_proj > th_list, axis=1) / e_proj.shape[1]
                    fidelity = (g_cdf + e_cdf) / 2

                    fid_Array[l_idx, g_idx, f_idx] = np.max(fidelity)

        max_idx = np.unravel_index(np.argmax(fid_Array), fid_Array.shape)
        max_l_idx, max_g_idx, max_f_idx = max_idx

        max_fid_grid = fid_Array[max_idx]
        max_length_grid = self.length_sweep[max_l_idx]
        max_gain_grid = self.gain_sweep[max_g_idx]
        max_freq_grid = self.freq_sweep[max_f_idx]

        print(f"\n--- Grid Search Result ---")
        print(f"Max fidelity (on grid): {max_fid_grid:.4f}")

        length_str_grid = (
            f"{max_length_grid:.3f} us" if max_length_grid is not None else "default"
        )
        gain_str_grid = (
            f"{max_gain_grid:.5f} DAC" if max_gain_grid is not None else "default"
        )
        freq_str_grid = (
            f"{max_freq_grid:.5f} MHz" if max_freq_grid is not None else "default"
        )

        print(
            f"At length={length_str_grid}, gain={gain_str_grid}, freq={freq_str_grid}"
        )

        max_length, max_gain, max_freq = max_length_grid, max_gain_grid, max_freq_grid

        if not SCIPY_AVAILABLE:
            print("\nWarning: `scipy` not found. Skipping interpolation.")
            print("To enable, run: pip install scipy")
        else:
            try:
                all_axes_data = [
                    (self.length_sweep, len_L, max_length_grid),
                    (self.gain_sweep, len_G, max_gain_grid),
                    (self.freq_sweep, len_F, max_freq_grid),
                ]

                opt_param_indices = []
                opt_axes = []
                opt_initial_guess = []
                opt_bounds = []
                fixed_params = {}

                for i, (axis, length, initial_val) in enumerate(all_axes_data):
                    if length > 1 and axis[0] is not None:
                        opt_param_indices.append(i)
                        opt_axes.append(np.array(axis))
                        opt_initial_guess.append(initial_val)
                        opt_bounds.append((np.min(axis), np.max(axis)))
                    else:
                        fixed_params[i] = initial_val

                if not opt_param_indices:
                    print(
                        "\nNo parameters to optimize (all sweeps have length 1 or are None)."
                    )
                    print("Returning grid search result.")
                else:
                    squeezed_fid_Array = np.squeeze(fid_Array)

                    if squeezed_fid_Array.ndim == 0:
                        squeezed_fid_Array = squeezed_fid_Array.reshape(
                            (1,) * len(opt_param_indices)
                        )

                    try:
                        interpolator = RegularGridInterpolator(
                            tuple(opt_axes), squeezed_fid_Array, method="cubic"
                        )
                    except ValueError:
                        print(
                            "Warning: Not enough data for cubic interpolation. Falling back to linear."
                        )
                        interpolator = RegularGridInterpolator(
                            tuple(opt_axes), squeezed_fid_Array, method="linear"
                        )

                    def objective_func(opt_params):
                        return -interpolator(opt_params)[0]

                    result = minimize(
                        objective_func,
                        opt_initial_guess,
                        method="L-BFGS-B",
                        bounds=opt_bounds,
                    )

                    if not result.success:
                        print(
                            f"\nWarning: Interpolation optimization failed. {result.message}"
                        )
                        print("Returning grid search result.")
                    else:
                        max_fid_interp = -result.fun

                        final_params = [None, None, None]

                        for i, param_val in enumerate(result.x):
                            final_params[opt_param_indices[i]] = param_val

                        for i, param_val in fixed_params.items():
                            final_params[i] = param_val

                        max_length, max_gain, max_freq = final_params

                        print(f"\n--- Interpolated Result ---")
                        print(f"Max fidelity (interpolated): {max_fid_interp:.4f}")

                        length_str = (
                            f"{max_length:.3f} us"
                            if max_length is not None
                            else "default"
                        )
                        gain_str = (
                            f"{max_gain:.5f} DAC" if max_gain is not None else "default"
                        )
                        freq_str = (
                            f"{max_freq:.5f} MHz" if max_freq is not None else "default"
                        )

                        print(
                            f"At length={length_str}, gain={gain_str}, freq={freq_str}"
                        )

            except Exception as e:
                print(f"\nAn error occurred during interpolation: {e}")
                print("Returning grid search result.")

        self.fid_Array = fid_Array  # 將 fid_Array 存儲在 self 中 供 plot 函數使用

        return_L = round(max_length, 3) if max_length is not None else None
        return_G = round(max_gain, 6) if max_gain is not None else None
        return_F = round(max_freq, 6) if max_freq is not None else None

        return return_L, return_G, return_F

    def plot_top_fidelity_histograms(self, top_n=9):
        if not hasattr(self, "fid_Array"):
            print("Running analyze() to generate fidelity data...")
            self.analyze()
            if not hasattr(self, "fid_Array"):
                print(
                    "Error: Fidelity data not available after running analyze(). Cannot plot."
                )
                return

        fid_Array = self.fid_Array

        if fid_Array.ndim != 3:
            print("Error: fid_Array must be a 3D array (length, gain, freq).")
            return

        flat_fid_Array = fid_Array.flatten()

        top_n_flat_indices = np.argsort(flat_fid_Array)[-top_n:][::-1]

        top_n_indices = np.unravel_index(top_n_flat_indices, fid_Array.shape)

        top_n_I_g = [self.data["Ig"][idx] for idx in zip(*top_n_indices)]
        top_n_Q_g = [self.data["Qg"][idx] for idx in zip(*top_n_indices)]
        top_n_I_e = [self.data["Ie"][idx] for idx in zip(*top_n_indices)]
        top_n_Q_e = [self.data["Qe"][idx] for idx in zip(*top_n_indices)]

        all_I_data = np.concatenate(top_n_I_g + top_n_I_e)
        all_Q_data = np.concatenate(top_n_Q_g + top_n_Q_e)

        overall_min = min(np.min(all_I_data), np.min(all_Q_data))
        overall_max = max(np.max(all_I_data), np.max(all_Q_data))
        range_span = overall_max - overall_min
        padding_val = range_span * 0.05

        plot_min = overall_min - padding_val
        plot_max = overall_max + padding_val

        plot_extent = [plot_min, plot_max, plot_min, plot_max]

        hexbin_gridsize = 50

        grid_size = int(np.ceil(np.sqrt(top_n)))
        fig, axes = plt.subplots(
            grid_size, grid_size, figsize=(5 * grid_size, 5 * grid_size)
        )

        axes = axes.flatten()

        print(f"\nPlotting top {top_n} fidelity points using hexbin...")
        for i in range(min(top_n, len(axes))):
            l_idx, g_idx, f_idx = (
                top_n_indices[0][i],
                top_n_indices[1][i],
                top_n_indices[2][i],
            )

            I_g = self.data["Ig"][l_idx, g_idx, f_idx]
            Q_g = self.data["Qg"][l_idx, g_idx, f_idx]
            I_e = self.data["Ie"][l_idx, g_idx, f_idx]
            Q_e = self.data["Qe"][l_idx, g_idx, f_idx]

            current_fid = fid_Array[l_idx, g_idx, f_idx]

            length = self.length_sweep[l_idx]
            gain = self.gain_sweep[g_idx]
            freq = self.freq_sweep[f_idx]

            ax = axes[i]

            ax.hexbin(
                I_e,
                Q_e,
                gridsize=hexbin_gridsize,
                cmap="Reds",
                alpha=0.6,
                extent=plot_extent,
                mincnt=1,
            )

            ax.hexbin(
                I_g,
                Q_g,
                gridsize=hexbin_gridsize,
                cmap="Blues",
                alpha=0.6,
                extent=plot_extent,
                mincnt=1,
            )

            ax.set_xlim(plot_min, plot_max)
            ax.set_ylim(plot_min, plot_max)

            ax.set_aspect("equal", adjustable="box")

            title_str = (
                f"Fidelity: {current_fid:.4f}\n"
                f"L={length:.3f} us, G={gain:.5f} DAC, F={freq:.5f} MHz"
            )
            ax.set_title(title_str, fontsize=10)
            ax.set_xlabel("I")
            ax.set_ylabel("Q")

        for j in range(top_n, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
