import threading
import queue
import numpy as np
import matplotlib.pyplot as plt
import pyvisa
from IPython.display import display, clear_output, update_display
from tqdm.auto import tqdm

from ..system_tool.YOKOGS200 import YOKOGS200
from ..system_tool.system_tool import auto_unit


def liveplotfun(
    prog=None,  # 適用於: sw_avg, yoko
    soc=None,  # 適用於: sw_avg, yoko
    py_avg=1,  # 適用於: ALL
    x_axis_vals=None,  # 適用於: sw_avg (作為X軸), yoko (作為內層迴圈X軸)
    y_axis_vals=None,  # 適用於: sw_avg (作為2D圖的Y軸), yoko (作為外層迴圈Yoko值)
    x_label="X Axis",
    y_label="Y Axis",
    title_prefix="Experiment",
    # --- Yoko-specific ---
    yoko_inst_addr=None,
    yoko_mode="current",
    # --- 1D Scan specific ---
    scan_x_axis=None,  # 如果提供此參數，則啟用 1D 參數掃描模式
    get_prog_callback=None,  # 用於 1D 掃描時動態產生 program 的回呼函式
    # --- General ---
    show_final_plot=True,
):
    """
    General-purpose live plotter (Facade pattern).
    Dispatches to specialized internal functions based on provided arguments.
    """

    # Mode 1: Yoko Parameter Sweep
    if yoko_inst_addr is not None:
        if y_axis_vals is None:
            raise ValueError("y_axis_vals must be provided for a Yoko sweep.")
        return _liveplot_sweep_yoko(
            prog=prog,
            soc=soc,
            py_avg=py_avg,
            x_axis_vals=x_axis_vals,
            y_axis_vals_yoko=y_axis_vals,
            yoko_inst_addr=yoko_inst_addr,
            yoko_mode=yoko_mode,
            x_label=x_label,
            y_label=y_label,
            title_prefix=title_prefix,
            show_final_plot=show_final_plot,
        )

    # Mode 2: 1D Parameter Scan (e.g., Length Rabi, T1)
    elif scan_x_axis is not None:
        if get_prog_callback is None:
            raise ValueError(
                "get_prog_callback must be provided for 1D parameter scan."
            )
        return _liveplot_1d_scan(
            soc=soc,
            py_avg=py_avg,
            scan_x_axis=scan_x_axis,
            get_prog_callback=get_prog_callback,
            x_label=x_label,
            title_prefix=title_prefix,
            show_final_plot=show_final_plot,
        )

    # Mode 3: Software Averaging (Default 1D or 2D repeat)
    else:
        return _liveplot_sw_avg(
            prog=prog,
            soc=soc,
            py_avg=py_avg,
            x_axis_vals=x_axis_vals,
            y_axis_vals=y_axis_vals,
            x_label=x_label,
            y_label=y_label,
            title_prefix=title_prefix,
            show_final_plot=show_final_plot,
        )


# ===================================================================
# 2. Internal Function: Software Averaging
# ===================================================================
def _liveplot_sw_avg(
    prog,
    soc,
    py_avg,
    x_axis_vals,
    y_axis_vals=None,
    x_label="X Axis",
    y_label="Y Axis",
    title_prefix="Experiment",
    show_final_plot=False,
):
    """
    [Internal function] Executes a software-averaged live plot (1D or 2D) using a separate, persistent thread.
    """
    data_queue = queue.LifoQueue(maxsize=1)
    stop_event = threading.Event()

    iq = 0
    iqdata = None
    interrupted = False
    last_i = 0

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_display_id = f"live-plot-optimized-{np.random.randint(1e9)}"
    display(fig, display_id=plot_display_id)

    is_2d = y_axis_vals is not None
    plot_artist = None

    if is_2d:
        plot_artist = ax.pcolormesh(
            x_axis_vals,
            y_axis_vals,
            np.zeros((len(y_axis_vals), len(x_axis_vals))),
            cmap="viridis",
        )
        fig.colorbar(plot_artist, ax=ax, label="Normalized Amplitude")
        ax.set_ylabel(y_label)
    else:
        (plot_artist,) = ax.plot(
            x_axis_vals, np.zeros_like(x_axis_vals), "o-", markersize=5, alpha=0.7
        )
        ax.set_ylabel("ADC Units (Abs)")

    ax.set_xlabel(x_label)
    ax.set_title(f"{title_prefix} (Initializing...)")

    def plotter_thread_func():
        while not stop_event.is_set():
            try:
                current_i, data = data_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if is_2d:
                plot_artist.set_array(data.ravel())
                plot_artist.set_clim(vmin=np.min(data), vmax=np.max(data))
            else:
                plot_artist.set_ydata(data)
                ax.set_ylim(np.min(data) * 0.98, np.max(data) * 1.02)

            ax.set_title(f"{title_prefix} | Average: {current_i + 1} / {py_avg}")
            display(fig, display_id=plot_display_id, update=True)
            data_queue.task_done()

    plot_thread = threading.Thread(target=plotter_thread_func, daemon=True)
    plot_thread.start()

    try:
        for i in tqdm(range(py_avg), desc="Software Average Count", mininterval=0.1):
            last_i = i
            iq_list = prog.acquire(soc, rounds=1, progress=False)
            iq_data = iq_list[0][0].dot([1, 1j])
            iq = iq_data if i == 0 else iq + iq_data
            iqdata = iq / (i + 1)
            plot_data_abs = np.abs(iqdata)

            if is_2d:
                row_mins = plot_data_abs.min(axis=1, keepdims=True)
                row_maxs = plot_data_abs.max(axis=1, keepdims=True)
                ranges = row_maxs - row_mins
                ranges[ranges == 0] = 1
                data_to_push = (plot_data_abs - row_mins) / ranges
            else:
                data_to_push = plot_data_abs

            try:
                data_queue.put_nowait((i, data_to_push))
            except queue.Full:
                try:
                    data_queue.get_nowait()
                    data_queue.put_nowait((i, data_to_push))
                except:
                    pass

    except KeyboardInterrupt:
        interrupted = True
    finally:
        stop_event.set()
        if plot_thread.is_alive():
            plot_thread.join(timeout=1.0)

    clear_output(wait=True)
    plt.close(fig)

    if show_final_plot:
        final_fig, final_ax = plt.subplots(figsize=(6, 4))
        title_status = "Interrupted" if interrupted else "Completed"
        final_ax.set_title(f"{title_prefix} ({title_status} at avg {last_i + 1})")
        final_ax.set_xlabel(x_label)

        if iqdata is not None:
            plot_data_abs = np.abs(iqdata)
            if is_2d:
                row_mins = plot_data_abs.min(axis=1, keepdims=True)
                row_maxs = plot_data_abs.max(axis=1, keepdims=True)
                ranges = row_maxs - row_mins
                ranges[ranges == 0] = 1
                final_data = (plot_data_abs - row_mins) / ranges
                im = final_ax.pcolormesh(
                    x_axis_vals, y_axis_vals, final_data, cmap="viridis"
                )
                final_fig.colorbar(im, ax=final_ax, label="Normalized Amplitude")
                final_ax.set_ylabel(y_label)
            else:
                final_ax.plot(x_axis_vals, plot_data_abs, "o-", markersize=5, alpha=0.7)
                final_ax.set_ylabel("ADC Units (Abs)")
        else:
            final_ax.text(
                0.5,
                0.5,
                "No data acquired",
                ha="center",
                va="center",
                transform=final_ax.transAxes,
            )
        display(final_fig)

    return iqdata, interrupted, last_i + 1


# ===================================================================
# 3. Internal Function: Yoko Parameter Sweep
# ===================================================================
def _liveplot_sweep_yoko(
    prog,
    soc,
    py_avg,
    x_axis_vals,
    y_axis_vals_yoko,
    yoko_inst_addr,
    yoko_mode="current",
    x_label="X Axis",
    y_label="Y Axis",
    title_prefix="Experiment",
    show_final_plot=True,
):
    """
    [Internal function] Executes a Yoko parameter sweep live plot.
    """
    rm = None
    yoko = None
    iqdata_full = np.zeros((len(y_axis_vals_yoko), len(x_axis_vals)), dtype=complex)
    data_to_plot = np.zeros_like(iqdata_full, dtype=float)
    interrupted = False
    last_idx = 0

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_display_id = f"live-plot-yoko-{np.random.randint(1e9)}"
    display(fig, display_id=plot_display_id)

    try:
        rm = pyvisa.ResourceManager()
        yoko = YOKOGS200(yoko_inst_addr, rm)

        if yoko_mode == "current":
            yoko.SetMode("current")
            yoko_unit = "A"
        else:
            yoko.SetMode("voltage")
            yoko_unit = "V"
        yoko.SetOutput(1)

        try:
            value_info = auto_unit(y_axis_vals_yoko, yoko_unit)
            y_plot_vals = value_info["value"]
            dynamic_y_label = f"{y_label} ({value_info['unit']})"
        except Exception:
            y_plot_vals = y_axis_vals_yoko
            dynamic_y_label = y_label

        mesh = ax.pcolormesh(
            y_plot_vals, x_axis_vals, data_to_plot.T, shading="nearest"
        )
        ax.set_xlabel(dynamic_y_label)
        ax.set_ylabel(x_label)

        for idx, val in enumerate(tqdm(y_axis_vals_yoko, desc=f"Sweeping {yoko_mode}")):
            last_idx = idx
            if yoko_mode == "current":
                yoko.SetCurrent(val)
            else:
                yoko.SetVoltage(val)

            iq_list = prog.acquire(soc, rounds=py_avg, progress=False)
            iq_data_row = iq_list[0][0].dot([1, 1j])
            iqdata_full[idx, :] = iq_data_row
            data_to_plot[idx, :] = np.abs(iq_data_row)

            mesh.set_array(data_to_plot.T.ravel())
            current_max = np.max(data_to_plot)
            if current_max > 0:
                mesh.set_clim(vmin=0, vmax=current_max)

            ax.set_title(f"Step: {idx + 1} / {len(y_axis_vals_yoko)}")
            update_display(fig, display_id=plot_display_id)

    except KeyboardInterrupt:
        interrupted = True
    except Exception as e:
        print(f"An error occurred: {e}")
        interrupted = True
    finally:
        if yoko:
            try:
                yoko.close()
            except:
                pass
        if rm:
            try:
                rm.close()
            except:
                pass

    clear_output(wait=True)
    if interrupted:
        print(f"Scan interrupted at step: {last_idx + 1}")

    if show_final_plot:
        fig_final, ax_final = plt.subplots(figsize=(6, 4))
        title_status = "Interrupted" if interrupted else "Completed"
        ax_final.set_title(f"{title_prefix} ({title_status})")
        ax_final.set_xlabel(dynamic_y_label)
        ax_final.set_ylabel(x_label)
        im = ax_final.pcolormesh(
            y_plot_vals, x_axis_vals, data_to_plot.T, shading="nearest"
        )
        fig_final.colorbar(im, ax=ax_final, label="Amplitude (Abs)")
        display(fig_final)
        plt.close(fig_final)

    plt.close(fig)
    return iqdata_full, interrupted, last_idx + 1


# ===================================================================
# 4. Internal Function: 1D Parameter Scan (NEW)
# ===================================================================
def _liveplot_1d_scan(
    soc,
    py_avg,
    scan_x_axis,
    get_prog_callback,
    x_label="Scan Parameter",
    title_prefix="1D Scan",
    show_final_plot=True,
):
    """
    [Internal function] Executes a 1D parameter scan (e.g., Length Rabi) with live plotting.
    'get_prog_callback' is a function that takes a single value from 'scan_x_axis' and returns a ready-to-run program.
    """
    iq_sum = 0
    iqdata = None
    interrupted = False
    last_avg = 0

    fig, ax = plt.subplots(figsize=(6, 4))
    (line,) = ax.plot(
        scan_x_axis, np.zeros_like(scan_x_axis), "o-", markersize=5, alpha=0.7
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("ADC unit (Abs)")
    title = ax.set_title(f"{title_prefix} (Initializing...)")

    plot_display_id = f"live-plot-1d-{np.random.randint(1e9)}"
    display(fig, display_id=plot_display_id)

    try:
        for avg in tqdm(range(py_avg), desc="Average Count"):
            last_avg = avg
            iqlst = []
            # Scan through each point in the X axis
            for val in scan_x_axis:
                # Use the callback to get a new program for this specific scan value
                prog = get_prog_callback(val)
                iq_list = prog.acquire(soc, rounds=1, progress=False)
                iqlst.append(iq_list[0][0].dot([1, 1j]))

            current_iq_data = np.array(iqlst)
            iq_sum = current_iq_data if avg == 0 else iq_sum + current_iq_data
            iqdata = iq_sum / (avg + 1)

            # Update plot
            plot_data = np.abs(iqdata)
            line.set_ydata(plot_data)
            current_min, current_max = np.min(plot_data), np.max(plot_data)
            range_span = current_max - current_min
            if range_span == 0:
                range_span = 1  # Avoid zero range
            ax.set_ylim(
                current_min - 0.05 * range_span, current_max + 0.05 * range_span
            )

            title.set_text(f"{title_prefix} | Average: {avg + 1} / {py_avg}")
            update_display(fig, display_id=plot_display_id)

    except KeyboardInterrupt:
        interrupted = True
    except Exception as e:
        print(f"An error occurred during 1D scan: {e}")
        interrupted = True

    clear_output(wait=True)
    if interrupted:
        print(f"Scan interrupted at average: {last_avg + 1}")

    if show_final_plot:
        fig_final, ax_final = plt.subplots(figsize=(6, 4))
        title_status = "Interrupted" if interrupted else "Completed"
        ax_final.set_title(f"{title_prefix} ({title_status} at avg {last_avg + 1})")
        ax_final.set_xlabel(x_label)
        ax_final.set_ylabel("ADC unit (Abs)")

        if iqdata is not None:
            ax_final.plot(scan_x_axis, np.abs(iqdata), "o-", markersize=5, alpha=0.7)
        else:
            ax_final.text(
                0.5,
                0.5,
                "No data acquired",
                ha="center",
                va="center",
                transform=ax_final.transAxes,
            )

        display(fig_final)
        plt.close(fig_final)

    plt.close(fig)
    return iqdata, interrupted, last_avg + 1
