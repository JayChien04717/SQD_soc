# ----- Qick package ----- #
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
from ..tools.module_fitzcu import amprabi_analyze, post_rotate, pipulse_analyze
from ..tools.fitting import decaysin, fitdecaysin
from ..tools.yamltool import yml_comment
from IPython.display import display, clear_output

##################
# Define Program #
##################


class AmplitudeRabiProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]
        qubit_ch = cfg["qubit_ch"]
        qubit_ch_ef = cfg["qubit_ch_ef"]

        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])

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

        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])
        self.add_readoutconfig(
            ch=ro_ch, name="myro", freq=cfg["res_freq_ef"], gen_ch=res_ch
        )

        self.add_loop("gainloop", cfg["steps"])

        self.add_pulse(
            ch=res_ch,
            name="res_pulse",
            ro_ch=ro_ch,
            style="const",
            length=cfg["res_length"],
            freq=cfg["res_freq_ef"],
            phase=cfg["res_phase"],
            gain=cfg["res_gain_ef"],
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

        self.add_gauss(
            ch=qubit_ch_ef,
            name="ramp2",
            sigma=cfg["sigma_ef"],
            length=cfg["sigma_ef"] * 5,
            even_length=True,
        )
        self.add_pulse(
            ch=qubit_ch_ef,
            name="qubit_pulse_ef",
            style="arb",
            envelope="ramp2",
            freq=cfg["qubit_freq_ef"],
            phase=cfg["qubit_phase"],
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

        self.pulse(ch=self.cfg["qubit_ch_ef"], name="qubit_pulse_ef", t=0)
        self.delay_auto(0.01)
        if cfg["ge_ref"] is True:
            self.pulse(ch=cfg["qubit_ch"], name="qubit_pi_pulse", t=0)
            self.delay_auto(0.01)
        self.delay_auto(0.01)

        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class Amp_Rabi_ef:
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(self, py_avg, liveplot=False):
        if liveplot:
            return self.liveplot(py_avg)
        else:
            prog = AmplitudeRabiProgram(
                self.soccfg,
                reps=self.cfg["reps"],
                final_delay=self.cfg["relax_delay"],
                cfg=self.cfg,
            )
            iq_list = prog.acquire(self.soc, soft_avgs=py_avg, progress=True)
            self.iqdata = iq_list[0][0].dot([1, 1j])
            self.gains = prog.get_pulse_param("qubit_pulse_ef", "gain", as_array=True)

    def plot(self):
        pi_gain, pi2_gain = amprabi_analyze(self.gains, self.iqdata)
        return pi_gain, pi2_gain

    def liveplot(self, py_avg):
        iq = 0

        marker_style = {
            "marker": "o",
            "markersize": 5,
            "alpha": 0.7,
            "linestyle": "-",
        }
        fig, ax = plt.subplots(figsize=(6, 4))
        prog = AmplitudeRabiProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )
        self.gains = prog.get_pulse_param("qubit_pulse_ef", "gain", as_array=True)

        for avg in tqdm(range(py_avg), desc="average count"):
            self.iq_list = prog.acquire(self.soc, soft_avgs=1, progress=False)

            iq_data = self.iq_list[0][0].dot([1, 1j])
            iq = iq_data if avg == 0 else iq + iq_data
            self.iqdata = iq / (avg + 1)

            ax.cla()
            ax.plot(self.gains, np.abs(post_rotate(self.iqdata)), **marker_style)
            ax.set_title(f"average: {avg + 1} / {py_avg}")
            ax.set_xlabel("Gain (Dac unit)")
            ax.set_ylabel("Signal (ADC unit)")

            clear_output(wait=True)
            display(fig)

        clear_output(wait=True)
        ax.set_title("Power Rabi ef")
        ax.plot(self.gains, np.abs(post_rotate(self.iqdata)), **marker_style)
        pOpt, _ = fitdecaysin(self.gains, np.abs(post_rotate(self.iqdata)))
        pi_gain, pi2_gain = pipulse_analyze(pOpt)
        ax.plot(self.gains, decaysin(self.gains, *pOpt), label="Fit")
        ax.axvline(pi_gain, color="r", ls="--", label=f"pi gain = {pi_gain:.6f}")
        ax.axvline(pi2_gain, color="r", ls="--", label=f"pi2 gain = {pi2_gain:.6f}")
        ax.legend()
        self.sim = decaysin(self.gains, *pOpt)
        return round(pi_gain, 6), round(pi2_gain, 6)

    def saveLabber(self, qb_idx, yoko_current=None, save_sim=False):
        expt_name = "s011_power_rabi_ef" + f"_Q{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name, yoko_current)
        try:
            self.cfg.pop("qubit_gain_ge")
        except:
            pass

        dict_val = yml_comment(self.cfg)

        if save_sim:
            hdf5_generator(
                filepath=file_path,
                x_info={"name": "Gain", "unit": "DAC unit", "values": self.gains},
                y_info={"name": "simulate", "unit": "None", "values": np.array([0, 1])},
                z_info={
                    "name": "Signal",
                    "unit": "ADC unit",
                    "values": np.array([self.iqdata, self.sim]),
                },
                comment=(f"{dict_val}"),
                tag="Rabi",
            )
        else:
            hdf5_generator(
                filepath=file_path,
                x_info={"name": "Gain", "unit": "DAC unit", "values": self.gains},
                z_info={"name": "Signal", "unit": "ADC unit", "values": self.iqdata},
                comment=(f"{dict_val}"),
                tag="Rabi",
            )

        print(f"Data save to {file_path}")


if __name__ == "__main__":
    ###################
    # Experiment sweep parameter
    ###################

    START_GAIN = 0.0  # [DAC units]
    STOP_GAIN = 0.5  # [DAC units]
    STEPS = 200
    config.update(
        [
            ("steps", STEPS),
            ("qubit_gain_ef", QickSweep1D("gainloop", START_GAIN, STOP_GAIN)),
        ]
    )

    ###################
    # Run the Program
    ###################

    amp_rabi = AmplitudeRabiProgram(
        soccfg, reps=10, final_delay=config["relax_delay"], cfg=config
    )
    py_avg = 10
    iq_list = np.array(amp_rabi.acquire(soc, soft_avgs=py_avg, progress=True))
    gains = amp_rabi.get_pulse_param("qubit_pulse_ef", "gain", as_array=True)
    amps = np.abs(iq_list[0][0].dot([1, 1j]))

    ###################
    # Plot
    ###################

    Plot = True

    if Plot:
        # plt.plot(freqs,  iq_list[0][0].T[0])
        # plt.plot(freqs,  iq_list[0][0].T[1])
        plt.plot(gains, iq_list[0][0].dot([1, 1j]))
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
            "x_name": "Gain(a.u)",
            "x_value": gains,
            "z_name": "iq_list",
            "z_value": iq_list[0][0].dot([1, 1j]),
        }

        result = {"T1": "350us", "T2": "130us"}

        saveh5(file_path, data_dict, result)
