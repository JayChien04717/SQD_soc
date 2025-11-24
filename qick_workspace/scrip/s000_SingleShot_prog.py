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
    if fit:
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
                    f"I {xavg} +/- {np.std(I)} \t Q {yavg} +/- {np.std(Q)} \t Amp {amp_avg} +/- {np.std(amp)}"
                )
                print(f"Rotated (theta={theta}):")
                print(
                    f"I {xavg_new} +/- {np.std(I_new)} \t Q {yavg_new} +/- {np.std(Q_new)} \t Amp {np.abs(xavg_new + 1j * yavg_new)} +/- {np.std(amp)}"
                )
            else:
                print(f"Amps {amp_avg} +/- {np.std(amp)}")

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

    if fit:
        xmax_g = bins_g[np.argmax(n_tot_g)]
        xmax_e = bins_e[np.argmax(n_tot_e)]

        # a bit stupid but we need to know what the g and e states are to fit the gaussians, and
        # that requires having already looped through all the states once
        for check_i, data_check in enumerate(iqshots):
            state_label = state_labels[check_i]

            I, Q = data_check

            xavg, yavg = np.average(I), np.average(Q)

            """Rotate the IQ data"""
            I_new = I * np.cos(theta) - Q * np.sin(theta)
            Q_new = I * np.sin(theta) + Q * np.cos(theta)
            amp = np.abs(I_new + 1j * Q_new)

            n, bins = np.histogram(
                I_new if not amplitude_mode else amp, bins=numbins, range=xlims
            )

            idx_g = np.argmin(np.abs(bins[:-1] - xmax_g))
            idx_e = np.argmin(np.abs(bins[:-1] - xmax_e))
            ymax_g = n[idx_g]
            ymax_e = n[idx_e]
            fitparams = [ymax_g, xmax_g, 5, ymax_e, xmax_e, 5]

            popt, pcov = fit_doublegauss(xdata=bins[:-1], ydata=n, fitparams=fitparams)

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

    print(f"fidelity:{fids} \nthressholds:{thresholds} \ntheta:{theta * 180 / np.pi}")
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

        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])
        if self.soccfg["gens"][qb_ch]["type"] == "axis_sg_int4_v2":
            self.declare_gen(ch=qb_ch, nqz=cfg["nqz_qb"], mixer_freq=cfg["qmixer_freq"])
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
            name="ramp_ge",
            sigma=cfg["sigma"],
            length=cfg["sigma"] * 5,
            even_length=True,
        )
        self.add_pulse(
            ch=qb_ch,
            name="qb_ge_pulse",
            ro_ch=ro_ch,
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
            ro_ch=ro_ch,
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

    def plot(self, fid_avg=False, fit=False, normalize=False):
        hist(
            self.data,
            amplitude_mode=False,
            ps_threshold=None,
            theta=None,
            plot=True,
            verbose=True,
            fid_avg=fid_avg,
            fit=fit,
            fitparams=[None, None, 20, None, None, 20],
            normalize=normalize,
            title=None,
            export=False,
        )

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
