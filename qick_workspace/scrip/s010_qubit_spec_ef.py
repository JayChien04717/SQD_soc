# ----- Qick package ----- #
from qick import *
from qick.pyro import make_proxy
from qick.asm_v2 import AveragerProgramV2
from qick.asm_v2 import QickSpan, QickSweep1D

# ----- Library ----- #
import matplotlib.pyplot as plt
import numpy as np

# ----- User Library ----- #
from ..tools.system_cfg import *
from ..tools.system_cfg import DATA_PATH
from ..tools.system_tool import get_next_filename_labber, hdf5_generator
from tqdm.auto import tqdm
from ..tools.module_fitzcu import spectrum_analyze, post_rotate
from ..tools.fitting import *
from ..system_tool.yamltool import yml_comment
from IPython.display import display, clear_output

##################
# Define Program #
##################


class PulseProbeSpectroscopyProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]
        qubit_ch = cfg["qubit_ch"]
        qubit_ch_ef = cfg["qubit_ch_ef"]

        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])
        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])

        if self.soccfg["gens"][qubit_ch]["type"] == "axis_sg_int4_v2":
            self.declare_gen(
                ch=qubit_ch, nqz=cfg["nqz_qubit"], mixer_freq=cfg["qmixer_freq"]
            )
        else:
            self.declare_gen(ch=qubit_ch, nqz=cfg["nqz_qubit"])

        if self.soccfg["gens"][qubit_ch_ef]["type"] == "axis_sg_int4_v2":
            self.declare_gen(
                ch=qubit_ch_ef,
                nqz=cfg["nqz_qubit_ef"],
                mixer_freq=cfg["qmixer_freq_ef"],
            )
        else:
            self.declare_gen(ch=qubit_ch_ef, nqz=cfg["nqz_qubit_ef"])

        self.add_loop("freqloop", cfg["steps"])
        self.add_readoutconfig(
            ch=ro_ch, name="myro", freq=cfg["res_freq_ef"], gen_ch=res_ch
        )

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
        self.add_pulse(
            ch=qubit_ch,
            name="qubit_pi_pulse",
            style="arb",
            envelope="ramp",
            freq=cfg["qubit_freq_ge"],
            phase=cfg["qubit_phase"],
            gain=cfg["qubit_pi_gain_ge"],
        )

        self.add_pulse(
            ch=qubit_ch_ef,
            name="qubit_pulse_ef",
            style="const",
            length=cfg["qubit_length_ef"],
            freq=cfg["qubit_freq_ef"],
            phase=0,
            gain=cfg["qubit_gain_ef"],
        )

    def apply_cool(self, cfg):
        cool_ch1 = cfg["cool_ch1"]
        cool_ch2 = cfg["cool_ch2"]
        if self.soccfg["gens"][cool_ch1]["type"] == "axis_sg_int4_v2":
            self.declare_gen(
                ch=cool_ch1, nqz=cfg["nqz_cool_ch1"], mixer_freq=cfg["cool_mixer1"]
            )
        else:
            self.declare_gen(ch=cool_ch1, nqz=cfg["nqz_cool_ch1"])

        if self.soccfg["gens"][cool_ch2]["type"] == "axis_sg_int4_v2":
            self.declare_gen(
                ch=cool_ch2, nqz=cfg["nqz_cool_ch2"], mixer_freq=cfg["cool_mixer2"]
            )
        else:
            self.declare_gen(ch=cool_ch2, nqz=cfg["nqz_cool_ch2"])

        self.add_pulse(
            ch=cool_ch1,
            name="cool_pulse1",
            style="const",
            length=cfg["cool_length"],
            freq=cfg["cool_freq_1"],
            phase=0,
            gain=cfg["cool_gain_1"],
        )
        self.add_pulse(
            ch=cool_ch2,
            name="cool_pulse2",
            style="const",
            length=cfg["cool_length"],
            freq=cfg["cool_freq_2"],
            phase=0,
            gain=cfg["cool_gain_2"],
        )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg["cooling"] is True:
            self.apply_cool(cfg)
            self.pulse(ch=self.cfg["cool_ch1"], name="cool_pulse1", t=0)
            self.pulse(ch=self.cfg["cool_ch2"], name="cool_pulse2", t=0)
            self.delay_auto(0.5, tag="Ring down")

        self.pulse(ch=cfg["qubit_ch"], name="qubit_pi_pulse", t=0)
        self.delay_auto(0.01)
        self.pulse(
            ch=self.cfg["qubit_ch_ef"], name="qubit_pulse_ef", t=0
        )  # play probe pulse
        self.delay_auto(0.01)

        if cfg["ge_ref"] is True:
            self.pulse(ch=cfg["qubit_ch"], name="qubit_pi_pulse", t=0)
            self.delay_auto(0.01)
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class Qubit_Twotone_ef:
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(self, py_avg, liveplot=False):
        if liveplot:
            return self.liveplot(py_avg)
        else:
            prog = PulseProbeSpectroscopyProgram(
                self.soccfg,
                reps=self.cfg["reps"],
                final_delay=self.cfg["relax_delay"],
                cfg=self.cfg,
            )
            iq_list = prog.acquire(self.soc, soft_avgs=py_avg, progress=True)
            self.iqdata = iq_list[0][0].dot([1, 1j])
            self.freqs = prog.get_pulse_param("qubit_pulse_ef", "freq", as_array=True)

    def plot(self):
        f_q = spectrum_analyze(self.freqs, self.iqdata)
        return f_q

    def liveplot(self, py_avg):
        iq = 0
        prog = PulseProbeSpectroscopyProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )
        self.freqs = prog.get_pulse_param("qubit_pulse_ef", "freq", as_array=True)

        marker_style = {
            "marker": "o",
            "markersize": 5,
            "alpha": 0.7,
            "linestyle": "-",
        }
        fig, ax = plt.subplots(figsize=(6, 4))

        for i in tqdm(range(py_avg), desc="average count"):
            self.iq_list = prog.acquire(self.soc, soft_avgs=1, progress=False)
            iq_data = self.iq_list[0][0].dot([1, 1j])
            iq = iq_data if i == 0 else iq + iq_data
            self.iqdata = iq / (i + 1)

            ax.cla()
            ax.plot(self.freqs, np.abs(post_rotate(self.iqdata)), **marker_style)
            ax.set_title(f"average: {i + 1} / {py_avg}")
            ax.set_xlabel("Frequency (MHz)")
            ax.set_ylabel("ADC unist")

            clear_output(wait=True)
            display(fig)
        clear_output(wait=True)
        ax.set_title(f"Qubit ef Spectrum")
        ax.plot(self.freqs, np.abs(post_rotate(self.iqdata)), **marker_style)
        pOpt, _ = fitlor(self.freqs, np.abs(post_rotate(self.iqdata)))
        res = pOpt[2]  # Extract resonance frequency

        ax.plot(self.freqs, lorfunc(self.freqs, *pOpt), label="Fit")
        ax.axvline(res, color="r", ls="--", label=f"$f_{{res}}$ = {res:.2f} MHz")
        ax.legend()
        return round(res, 4)

    def saveLabber(self, qb_idx, yoko_value=None):
        expt_name = "s010_qubit_spec_ef" + f"_Q{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name, yoko_value)
        try:
            self.cfg.pop("qubit_freq_ef")
        except:
            pass

        dict_val = yml_comment(self.cfg)

        hdf5_generator(
            filepath=file_path,
            x_info={"name": "Frequency", "unit": "Hz", "values": self.freqs * 1e6},
            z_info={"name": "Signal", "unit": "ADC unit", "values": self.iqdata},
            comment=(f"{dict_val}"),
            tag="TwoTone",
        )
        print(f"Data save to {file_path}")


if __name__ == "__main__":
    ###################
    # Run the Program
    ###################

    START_FREQ = 4000  # [MHz]
    STOP_FREQ = 6000  # [MHz]
    STEPS = 101
    config.update(
        [
            ("steps", STEPS),
            ("qubit_freq_ge", QickSweep1D("freqloop", START_FREQ, STOP_FREQ)),
        ]
    )

    ###################
    # Run the Program
    ###################

    qspec = PulseProbeSpectroscopyProgram(
        soccfg, reps=1000, final_delay=0.5, cfg=config
    )
    py_avg = config["py_avg"]
    iq_list = qspec.acquire(soc, soft_avgs=py_avg, progress=True)
    freqs = qspec.get_pulse_param("qubit_pulse_ef", "freq", as_array=True)
    amps = np.abs(iq_list[0][0].dot([1, 1j]))

    ###################
    # Plot
    ###################

    Plot = True

    if Plot:
        # plt.plot(freqs,  iq_list[0][0].T[0])
        # plt.plot(freqs,  iq_list[0][0].T[1])
        plt.plot(freqs, amps)
        plt.show()

    #####################################
    # ----- Saves data to a file ----- #
    #####################################

    Save = True
    if Save:
        data_path = "./data"
        labber_data = "./data/Labber"
        exp_name = expt_name + "_Q" + str(QubitIndex)
        print("Experiment name: " + exp_name)
        file_path = get_next_filename(data_path, exp_name, suffix=".h5")
        print("Current data file: " + file_path)

        data_dict = {
            "x_name": "x_axis",
            "x_value": freqs,
            "z_name": "iq_list",
            "z_value": iq_list[0][0].dot([1, 1j]),
        }

        result = {"T1": "350us", "T2": "130us"}

        saveh5(file_path, data_dict, result)
