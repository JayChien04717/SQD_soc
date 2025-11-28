# ----- Qick package ----- #
from tabnanny import verbose
from qick import *
from qick.pyro import make_proxy
from qick.asm_v2 import AveragerProgramV2
from qick.asm_v2 import QickSpan, QickSweep1D

# ----- Library ----- #
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

# ----- User Library ----- #
from ..tools.system_cfg import *
from ..tools.system_cfg import DATA_PATH
from ..tools.system_tool import get_next_filename_labber, hdf5_generator
from ..tools.yamltool import yml_comment

# from .singleshotplot import hist
from ..tools.fitting import fit_doublegauss, double_gaussian, fit_gauss, gaussian
from scipy.integrate import quad

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
        # Avoid division by zero error
        hist_sum = hist_data.sum()
        if hist_sum > 0:
            hist_data = hist_data / hist_sum

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
    plotoverlap=False,  # 新增選項
    fitparams=None,
    normalize=True,
    title=None,
    export=False,
    check_qnd=False,
):
    if numbins is None:
        numbins = 200

    # Detect states
    has_f_state = len(iqshots) > 2

    # --- 1. Data Aggregation ---
    data_map = {"g": np.array([]), "e": np.array([]), "f": np.array([])}
    I_tot_all = np.array([])
    Q_tot_all = np.array([])

    for check_i, data_check in enumerate(iqshots):
        I, Q = data_check
        I_tot_all = np.concatenate((I_tot_all, I))
        Q_tot_all = np.concatenate((Q_tot_all, Q))

        if check_i in g_states:
            cat = "g"
        elif check_i in e_states:
            cat = "e"
        else:
            cat = "f"

        if data_map[cat].size == 0:
            data_map[cat] = I + 1j * Q
        else:
            data_map[cat] = np.concatenate((data_map[cat], I + 1j * Q))

    # --- 2. Rotation Calculation ---
    if not amplitude_mode:
        if theta is None:
            xg = np.mean(np.real(data_map["g"])) if data_map["g"].size > 0 else 0
            yg = np.mean(np.imag(data_map["g"])) if data_map["g"].size > 0 else 0
            xe = np.mean(np.real(data_map["e"])) if data_map["e"].size > 0 else 1
            ye = np.mean(np.imag(data_map["e"])) if data_map["e"].size > 0 else 1
            theta = -np.arctan2((ye - yg), (xe - xg))
        else:
            theta *= np.pi / 180

        def rotate_iq(c_data, ang):
            i_new = np.real(c_data) * np.cos(ang) - np.imag(c_data) * np.sin(ang)
            q_new = np.real(c_data) * np.sin(ang) + np.imag(c_data) * np.cos(ang)
            return i_new, q_new

        I_all_new, _ = rotate_iq(I_tot_all + 1j * Q_tot_all, theta)
        span = (np.max(I_all_new) - np.min(I_all_new)) / 2
        midpoint = (np.max(I_all_new) + np.min(I_all_new)) / 2
    else:
        theta = 0
        amp_all = np.abs(I_tot_all + 1j * Q_tot_all)
        span = (np.max(amp_all) - np.min(amp_all)) / 2
        midpoint = (np.max(amp_all) + np.min(amp_all)) / 2

    xlims = [midpoint - span, midpoint + span]

    # --- 3. Plot Setup ---
    if plot:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(9, 7))
        if title is None:
            title = f"Readout Fidelity" + (
                f" on Q{check_qubit_label}" if check_qubit_label is not None else ""
            )
        fig.suptitle(title)
        fig.tight_layout()

        # Row 0: IQ Scatter
        axs[0, 0].set_ylabel("Q [ADC levels]", fontsize=11)
        axs[0, 0].set_title("Unrotated", fontsize=13)
        axs[0, 0].set_xlabel("I [ADC levels]", fontsize=11)
        axs[0, 0].axis("equal")

        axs[0, 1].set_title(
            f"Rotated ($\\theta={theta * 180 / np.pi:.1f}^\\circ$)", fontsize=13
        )
        axs[0, 1].set_xlabel("I [ADC levels]", fontsize=11)
        axs[0, 1].axis("equal")

        # Row 1: Histogram & Confusion Matrix
        threshold_axis = "I" if not amplitude_mode else "Amplitude"
        axs[1, 0].set_ylabel("Counts", fontsize=12)
        axs[1, 0].set_xlabel(f"{threshold_axis} [ADC levels]", fontsize=11)

        plt.subplots_adjust(hspace=0.35, wspace=0.15)

    # Variables
    n_dist = {"g": None, "e": None, "f": None}
    bins_dist = None
    gauss_fit_fidelity = 0
    popts = []
    pcovs = []

    # Store extracted params for overlap plotting
    b_g_plot, c_g_plot = None, None
    b_e_plot, c_e_plot = None, None

    # --- 4. Process Each Input State ---
    for check_i, data_check in enumerate(iqshots):
        state_label = state_labels[check_i]
        I, Q = data_check
        complex_data = I + 1j * Q

        # Apply styles based on cycle
        this_color = default_colors[check_i % len(default_colors)]
        this_marker = marker_cycle[check_i % len(marker_cycle)]
        this_linestyle = linestyle_cycle[0]

        # Rotation
        if not amplitude_mode:
            I_new, Q_new = rotate_iq(complex_data, theta)
            data_to_hist = I_new
        else:
            I_new, Q_new = I, Q
            data_to_hist = np.abs(complex_data)

        # Scatter Plot
        if plot:
            axs[0, 0].scatter(
                I,
                Q,
                label=state_label,
                color=this_color,
                marker=".",
                edgecolor="None",
                alpha=0.1,
            )
            axs[0, 0].plot(
                [np.mean(I)],
                [np.mean(Q)],
                color="k",
                marker=this_marker,
                markerfacecolor=this_color,
                markersize=6,
            )
            axs[0, 1].scatter(
                I_new,
                Q_new,
                label=state_label,
                color=this_color,
                marker=".",
                edgecolor="None",
                alpha=0.1,
            )
            axs[0, 1].plot(
                [np.mean(I_new)],
                [np.mean(Q_new)],
                color="k",
                marker=this_marker,
                markerfacecolor=this_color,
                markersize=6,
            )

        # Histogram Accumulation
        if plot:
            n, bins = plot_hist(
                data_to_hist,
                bins=numbins,
                ax=axs[1, 0],
                xlims=xlims,
                color=this_color,
                linestyle=this_linestyle,
                label=state_label,
                alpha=0.6,
                normalize=False,
            )
        else:
            n, bins = np.histogram(data_to_hist, bins=numbins, range=xlims)

        bins_dist = bins

        # Accumulate
        if check_i in g_states:
            cat = "g"
        elif check_i in e_states:
            cat = "e"
        else:
            cat = "f"

        if n_dist[cat] is None:
            n_dist[cat] = n
        else:
            n_dist[cat] += n

    # --- 5. Fitting (Modified) ---
    def gaussian_norm(x, b, c):
        a = 1 / (np.sqrt(2 * np.pi) * c)
        return a * np.exp(-((x - b) ** 2) / (2 * c**2))

    def overlap_area_norm(b1, c1, b2, c2):
        def min_func(x):
            return np.minimum(gaussian_norm(x, b1, c1), gaussian_norm(x, b2, c2))

        x_min = min(b1 - 5 * c1, b2 - 5 * c2)
        x_max = max(b1 + 5 * c1, b2 + 5 * c2)
        area, _ = quad(min_func, x_min, x_max)
        return area

    def readout_fidelity_norm(b1, c1, b2, c2):
        return 1 - overlap_area_norm(b1, c1, b2, c2)

    do_fit = fit or gauss_overlap or plotoverlap

    if do_fit and n_dist["g"] is not None and n_dist["e"] is not None:
        bin_centers = (bins_dist[:-1] + bins_dist[1:]) / 2
        n_g = n_dist["g"]
        n_e = n_dist["e"]

        xmax_g_idx = np.argmax(n_g)
        xmax_e_idx = np.argmax(n_e)
        xmax_g_val = bin_centers[xmax_g_idx]
        xmax_e_val = bin_centers[xmax_e_idx]

        sigma_guess = abs(xmax_e_val - xmax_g_val) / 5.0
        if sigma_guess < 1e-3:
            sigma_guess = (bins_dist[-1] - bins_dist[0]) / 20.0

        if gauss_overlap:
            # --- DOUBLE GAUSSIAN FIT ---
            fit_func = double_gaussian
            guess_g = [
                np.max(n_g),
                xmax_g_val,
                sigma_guess,
                np.max(n_g) * 0.1,
                xmax_e_val,
                sigma_guess,
            ]
            guess_e = [
                np.max(n_e),
                xmax_e_val,
                sigma_guess,
                np.max(n_e) * 0.2,
                xmax_g_val,
                sigma_guess,
            ]

            try:
                popt_g, pcov_g = fit_doublegauss(bin_centers, n_g, guess_g)
                popt_e, pcov_e = fit_doublegauss(bin_centers, n_e, guess_e)

                # Extract main peak
                if popt_g[0] > popt_g[3]:
                    b_g, c_g = popt_g[1], abs(popt_g[2])
                else:
                    b_g, c_g = popt_g[4], abs(popt_g[5])

                if popt_e[0] > popt_e[3]:
                    b_e, c_e = popt_e[1], abs(popt_e[2])
                else:
                    b_e, c_e = popt_e[4], abs(popt_e[5])

                gauss_fit_fidelity = readout_fidelity_norm(b_g, c_g, b_e, c_e)
                popts = [popt_g, popt_e]
                pcovs = [pcov_g, pcov_e]

                # Save for plotting
                b_g_plot, c_g_plot = b_g, c_g
                b_e_plot, c_e_plot = b_e, c_e

            except Exception as e:
                print(f"Double Gaussian fit failed: {e}")
                gauss_fit_fidelity = 0
                popt_g, popt_e = None, None

        else:
            # --- SINGLE GAUSSIAN FIT ---
            fit_func = gaussian
            guess_g = [np.max(n_g), xmax_g_val, sigma_guess, 0]
            guess_e = [np.max(n_e), xmax_e_val, sigma_guess, 0]

            try:
                popt_g, pcov_g = fit_gauss(bin_centers, n_g, guess_g)
                popt_e, pcov_e = fit_gauss(bin_centers, n_e, guess_e)
                popts = [popt_g, popt_e]
                pcovs = [pcov_g, pcov_e]

                # Extract simple gaussian params for overlap plot if requested
                b_g_plot, c_g_plot = popt_g[1], abs(popt_g[2])
                b_e_plot, c_e_plot = popt_e[1], abs(popt_e[2])

            except Exception as e:
                print(f"Gaussian fit failed: {e}")
                popt_g, popt_e = None, None

        if plot and popt_g is not None and popt_e is not None:
            x_dense = np.linspace(bins_dist[0], bins_dist[-1], 500)
            y_fit_g = fit_func(x_dense, *popt_g)
            y_fit_e = fit_func(x_dense, *popt_e)

            axs[1, 0].plot(
                x_dense,
                y_fit_g,
                color=default_colors[0],
                linestyle="-",
                linewidth=2,
                label="Fit G",
            )
            axs[1, 0].plot(
                x_dense,
                y_fit_e,
                color=default_colors[1],
                linestyle="-",
                linewidth=2,
                label="Fit E",
            )

            # --- Plot Overlap Area ---
            if plotoverlap and b_g_plot is not None and b_e_plot is not None:
                # Reconstruct the single gaussians used for fidelity calculation
                # Scale them to counts: N_counts = PDF * Total_Counts * Bin_Width
                bin_width = bins_dist[1] - bins_dist[0]

                # Scale G
                norm_g = gaussian_norm(x_dense, b_g_plot, c_g_plot)
                scaled_g = norm_g * np.sum(n_g) * bin_width

                # Scale E
                norm_e = gaussian_norm(x_dense, b_e_plot, c_e_plot)
                scaled_e = norm_e * np.sum(n_e) * bin_width

                # Calculate overlap of these scaled distributions
                y_overlap = np.minimum(scaled_g, scaled_e)

                axs[1, 0].fill_between(
                    x_dense, 0, y_overlap, color="purple", alpha=0.3, label="Overlap"
                )

    # --- 6. Thresholds & Confusion Matrix ---
    fids = []
    thresholds = []

    contrast_ge = np.abs(
        (np.cumsum(n_dist["g"]) - np.cumsum(n_dist["e"]))
        / (np.sum(n_dist["g"]) + np.sum(n_dist["e"]))
    )
    tind_ge = contrast_ge.argmax()
    threshold_ge = bins_dist[tind_ge]
    thresholds.append(threshold_ge)

    if not fid_avg:
        fids.append(contrast_ge[tind_ge])
    else:
        fids.append(
            0.5
            * (
                1
                - n_dist["g"][tind_ge:].sum() / n_dist["g"].sum()
                + 1
                - n_dist["e"][:tind_ge].sum() / n_dist["e"].sum()
            )
        )

    # --- Matrix Calculation ---
    if not has_f_state:
        matrix_size = 2
        labels = ["|g>", f"|{e_label}>"]
        n00 = n_dist["g"][:tind_ge].sum()
        n01 = n_dist["g"][tind_ge:].sum()
        n10 = n_dist["e"][:tind_ge].sum()
        n11 = n_dist["e"][tind_ge:].sum()
        raw_matrix = np.array([[n00, n01], [n10, n11]])
    else:
        matrix_size = 3
        labels = ["|g>", f"|{e_label}>", "|f>"]
        if n_dist["f"] is not None:
            contrast_ef = np.abs(
                (np.cumsum(n_dist["e"]) - np.cumsum(n_dist["f"]))
                / (np.sum(n_dist["e"]) + np.sum(n_dist["f"]))
            )
            tind_ef = contrast_ef.argmax()
            threshold_ef = bins_dist[tind_ef]
            thresholds.append(threshold_ef)
            sorted_t_indices = sorted([tind_ge, tind_ef])
            t1, t2 = sorted_t_indices[0], sorted_t_indices[1]

            def classify_counts(n_arr):
                c0 = n_arr[:t1].sum()
                c1 = n_arr[t1:t2].sum()
                c2 = n_arr[t2:].sum()
                return [c0, c1, c2]

            row_g = classify_counts(n_dist["g"])
            row_e = classify_counts(n_dist["e"])
            row_f = classify_counts(n_dist["f"])
            raw_matrix = np.array([row_g, row_e, row_f])
        else:
            raw_matrix = np.zeros((3, 3))

    row_sums = raw_matrix.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1
    conf_matrix = 100 * raw_matrix / row_sums

    # --- 7. Finalize Plots ---
    if plot:
        for th in thresholds:
            axs[1, 0].axvline(th, color="k", linestyle="--", label="Threshold")

        fid_title = "$\overline{F}_{ge}$" if fid_avg else "$F_{ge}$"
        if gauss_overlap:
            axs[1, 0].set_title(
                f"{fid_title} (Gauss): {100 * gauss_fit_fidelity:.2f}%", fontsize=13
            )
        else:
            axs[1, 0].set_title(
                f"{fid_title} (Thresh): {100 * fids[0]:.2f}%", fontsize=13
            )

        if ps_threshold is not None:
            axs[1, 0].axvline(ps_threshold, color="gray", linestyle="-.")

        axs[1, 0].legend(fontsize=8, loc="upper right")
        axs[0, 0].legend(fontsize=8)
        axs[0, 1].legend(fontsize=8)

        # Confusion Matrix
        ax_cm = axs[1, 1]
        ax_cm.clear()
        im = ax_cm.imshow(conf_matrix, cmap="Reds", vmin=0, vmax=100)
        ax_cm.set_xticks(np.arange(matrix_size))
        ax_cm.set_yticks(np.arange(matrix_size))
        ax_cm.set_xticklabels(labels)
        ax_cm.set_yticklabels(labels)
        ax_cm.set_xlabel("Declared output", fontsize=11)
        ax_cm.set_ylabel("Input state", fontsize=11)
        ax_cm.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

        for i in range(matrix_size):
            for j in range(matrix_size):
                val = conf_matrix[i, j]
                text_color = "white" if val > 50 else "black"
                ax_cm.text(
                    j,
                    i,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=12,
                )

        if title is not None:
            ax_cm.set_title("Readout Fidelity Matrix (%)")

        if export:
            plt.savefig("multihist.jpg", dpi=1000)
            print("exported multihist.jpg")
            plt.close()
        else:
            plt.show()

    # --- 8. Returns ---
    if gauss_overlap:
        return_data = [gauss_fit_fidelity, thresholds, theta * 180 / np.pi]
    else:
        return_data = [fids, thresholds, theta * 180 / np.pi]

    if fit or gauss_overlap or plotoverlap:
        return_data += [popts, pcovs]

    if check_qnd:
        n_diff = np.abs((n_g_0 - n_g_1)) if "n_g_0" in locals() else 0
        n_diff_qnd = (
            np.sum(n_diff) / 2 / np.sum(n_dist["g"]) if np.sum(n_dist["g"]) > 0 else 0
        )
        return_data += [n_diff_qnd]

    if verbose:
        print(f"Theta: {theta * 180 / np.pi:.2f} deg")
        print(f"Threshold Fidelity: {100 * fids[0]:.3f}%")
        print("Fidelity Matrix (%):\n", conf_matrix)
        if gauss_overlap:
            print(f"Gaussian Fit Fidelity: {100 * gauss_fit_fidelity:.3f}%")

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
    plotoverlap=False,
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
        plotoverlap=plotoverlap,
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
        qb_ch = cfg["qb_ch"]

        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])
        if self.soccfg["gens"][qb_ch]["type"] == "axis_sg_int4_v2":
            self.declare_gen(ch=qb_ch, nqz=cfg["nqz_qb"], mixer_freq=cfg["qb_mixer"])
        else:
            self.declare_gen(ch=qb_ch, nqz=cfg["nqz_qb"])

        # pynq configured
        # self.declare_readout(ch=ro_ch, length=cfg['ro_len'], freq=cfg['f_res'], gen_ch=res_ch)

        # tproc configured
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
        qb_ch = cfg["qb_ch"]

        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])

        if self.soccfg["gens"][qb_ch]["type"] == "axis_sg_int4_v2":
            self.declare_gen(ch=qb_ch, nqz=cfg["nqz_qb"], mixer_freq=cfg["qb_mixer"])
        else:
            self.declare_gen(ch=qb_ch, nqz=cfg["nqz_qb"])
        # pynq configured
        # self.declare_readout(ch=ro_ch, length=cfg['ro_len'], freq=cfg['f_res'], gen_ch=res_ch)

        # tproc configured
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
            ch=qb_ch,
            name="ramp",
            sigma=cfg["sigma"],
            length=cfg["sigma"] * 5,
            even_length=True,
        )
        if cfg["pulse_type"] == "arb":
            self.add_pulse(
                ch=qb_ch,
                name="qb_pulse",
                ro_ch=ro_ch,
                style="arb",
                envelope="ramp",
                freq=cfg["qb_freq_ge"],
                phase=cfg["qb_phase"],
                gain=cfg["pi_gain_ge"],
            )
        elif cfg["pulse_type"] == "flat_top":
            self.add_pulse(
                ch=qb_ch,
                name="qb_pulse",
                ro_ch=ro_ch,
                style="flat_top",
                envelope="ramp",
                freq=cfg["qb_freq_ge"],
                phase=cfg["qb_phase"],
                gain=cfg["pi_gain_ge"],
                length=cfg["qb_flat_top_length_ge"],
            )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        self.pulse(ch=self.cfg["qb_ch"], name="qb_pulse", t=0)
        self.delay_auto(0.01, tag="wait")
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class SingleShotProgram_f(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]
        qb_ch = cfg["qb_ch"]
        qb_eh_ef = cfg["qb_eh_ef"]
        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])
        if self.soccfg["gens"][qb_ch]["type"] == "axis_sg_int4_v2":
            self.declare_gen(ch=qb_ch, nqz=cfg["nqz_qb"], mixer_freq=cfg["qb_mixer"])
        else:
            self.declare_gen(ch=qb_ch, nqz=cfg["nqz_qb"])

        if self.soccfg["gens"][qb_eh_ef]["type"] == "axis_sg_int4_v2":
            self.declare_gen(
                ch=qb_eh_ef, nqz=cfg["nqz_qb"], mixer_freq=cfg["qb_mixer_ef"]
            )
        else:
            self.declare_gen(ch=qb_eh_ef, nqz=cfg["nqz_qb"])
        # pynq configured
        # self.declare_readout(ch=ro_ch, length=cfg['ro_len'], freq=cfg['f_res'], gen_ch=res_ch)

        # tproc configured
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
            ch=qb_ch,
            name="ramp_ge",
            sigma=cfg["sigma"],
            length=cfg["sigma"] * 5,
            even_length=True,
        )
        self.add_pulse(
            ch=qb_ch,
            name="qb_ge_pulse",
            style="arb",
            envelope="ramp_ge",
            freq=cfg["qb_freq_ge"],
            phase=cfg["qb_phase"],
            gain=cfg["pi_gain_ge"],
        )

        self.add_gauss(
            ch=qb_ch,
            name="ramp_ef",
            sigma=cfg["sigma"],
            length=cfg["sigma_ef"] * 5,
            even_length=True,
        )
        self.add_pulse(
            ch=qb_ch,
            name="qb_ef_pulse",
            style="arb",
            envelope="ramp_ef",
            freq=cfg["qb_freq_ef"],
            phase=cfg["qb_phase"],
            gain=cfg["pi_gain_ef"],
        )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        self.pulse(ch=self.cfg["qb_ch"], name="qb_ge_pulse", t=0)
        self.delay_auto(0.01, tag="wait1")
        self.pulse(ch=self.cfg["qb_ch"], name="qb_ef_pulse", t=0)
        self.delay_auto(0.01)
        self.pulse(ch=self.cfg["qb_ch"], name="qb_ge_pulse", t=0)
        self.delay_auto(0.01)
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class SingleShot_gef:
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(self, SHOTS, shot_f=False):
        self.cfg["shots"] = SHOTS
        shot_g = SingleShotProgram_g(
            self.soccfg, reps=1, final_delay=self.cfg["relax_delay"], cfg=self.cfg
        )
        shot_e = SingleShotProgram_e(
            self.soccfg, reps=1, final_delay=self.cfg["relax_delay"], cfg=self.cfg
        )

        iq_list_g = shot_g.acquire(self.soc, rounds=1, progress=True)
        iq_list_e = shot_e.acquire(self.soc, rounds=1, progress=True)

        I_g = iq_list_g[0][0].T[0]
        Q_g = iq_list_g[0][0].T[1]
        I_e = iq_list_e[0][0].T[0]
        Q_e = iq_list_e[0][0].T[1]
        if shot_f:
            shot_f = SingleShotProgram_f(
                self.soccfg,
                reps=1,
                final_delay=self.cfg["relax_delay"],
                cfg=self.cfg,
            )
            iq_list_f = shot_f.acquire(self.soc, rounds=1, progress=True)
            I_f = iq_list_f[0][0].T[0]
            Q_f = iq_list_f[0][0].T[1]

        if shot_f:
            self.data = {
                "Ig": I_g,
                "Qg": Q_g,
                "Ie": I_e,
                "Qe": Q_e,
                "If": I_f,
                "Qf": Q_f,
            }
        else:
            self.data = {"Ig": I_g, "Qg": Q_g, "Ie": I_e, "Qe": Q_e}

    def plot(
        self, fid_avg=False, fit=False, normalize=False, verbose=True, overlap=False
    ):
        return hist(
            self.data,
            amplitude_mode=False,
            ps_threshold=None,
            theta=None,
            plot=True,
            verbose=verbose,
            gauss_overlap=overlap,
            fid_avg=fid_avg,
            fit=fit,
            fitparams=[None, None, 5, None, None, 5],
            normalize=normalize,
            title=None,
            export=False,
        )

    def autorun(self, SHOTS, shot_f=False):
        self.cfg["shots"] = SHOTS
        shot_g = SingleShotProgram_g(
            self.soccfg, reps=1, final_delay=self.cfg["relax_delay"], cfg=self.cfg
        )
        shot_e = SingleShotProgram_e(
            self.soccfg, reps=1, final_delay=self.cfg["relax_delay"], cfg=self.cfg
        )

        iq_list_g = shot_g.acquire(self.soc, rounds=1, progress=False)
        iq_list_e = shot_e.acquire(self.soc, rounds=1, progress=False)

        I_g = iq_list_g[0][0].T[0]
        Q_g = iq_list_g[0][0].T[1]
        I_e = iq_list_e[0][0].T[0]
        Q_e = iq_list_e[0][0].T[1]
        if shot_f:
            shot_f = SingleShotProgram_f(
                self.soccfg,
                reps=1,
                final_delay=self.cfg["relax_delay"],
                cfg=self.cfg,
            )
            iq_list_f = shot_f.acquire(self.soc, rounds=1, progress=True)
            I_f = iq_list_f[0][0].T[0]
            Q_f = iq_list_f[0][0].T[1]

        if shot_f:
            self.data = {
                "Ig": I_g,
                "Qg": Q_g,
                "Ie": I_e,
                "Qe": Q_e,
                "If": I_f,
                "Qf": Q_f,
            }
        else:
            self.data = {"Ig": I_g, "Qg": Q_g, "Ie": I_e, "Qe": Q_e}

        data = hist(
            self.data,
            amplitude_mode=False,
            ps_threshold=None,
            theta=None,
            plot=False,
            verbose=False,
            fid_avg=False,
            fit=False,
            fitparams=[None, None, 20, None, None, 20],
            normalize=False,
            title=None,
            export=False,
        )
        return data

    def saveLabber(self, qb_idx, yoko_value=None):
        expt_name = "s000_singleshot" + f"_{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name, yoko_value)

        print("Current data file: " + file_path)

        shotdata = np.array(
            [
                self.data["Ig"] + 1j * self.data["Qg"],
                self.data["Ie"] + 1j * self.data["Qe"],
            ]
        )
        dict_val = yml_comment(self.cfg)
        hdf5_generator(
            filepath=file_path,
            x_info={
                "name": "# shot",
                "unit": "#",
                "values": np.arange(self.cfg["shots"]),
            },
            y_info={"name": "State", "unit": "", "values": [0, 1]},
            z_info={"name": "Signal", "unit": "ADC unit", "values": shotdata},
            comment=(f"{dict_val}"),
            tag="OneTone",
        )
