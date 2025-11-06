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
from ..tools.module_fitzcu import T1_analyze, post_rotate
from ..tools.fitting import expfunc, fitexp
from ..system_tool.yamltool import yml_comment
from IPython.display import display, clear_output
##################
# Define Program #
##################


class T1Program(AveragerProgramV2):
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
        # pynq configured
        # self.declare_readout(ch=ro_ch, length=cfg['ro_len'], freq=cfg['f_res'], gen_ch=res_ch)

        # tproc configured
        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])
        self.add_readoutconfig(
            ch=ro_ch, name="myro", freq=cfg["res_freq_ge"], gen_ch=res_ch
        )

        self.add_loop("waitloop", cfg["steps"])

        self.add_pulse(
            ch=res_ch,
            name="res_pulse",
            ro_ch=ro_ch,
            style="const",
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
            ro_ch=ro_ch,
            style="arb",
            envelope="ramp",
            freq=cfg["qubit_freq_ge"],
            phase=cfg["qubit_phase"],
            gain=cfg["qubit_pi_gain_ge"],
        )

        self.add_gauss(
            ch=qubit_ch_ef,
            name="ramp1",
            sigma=cfg["sigma_ef"],
            length=cfg["sigma_ef"] * 5,
            even_length=True,
        )
        self.add_pulse(
            ch=qubit_ch_ef,
            name="qubit_pulse",
            ro_ch=ro_ch,
            style="arb",
            envelope="ramp1",
            freq=cfg["qubit_freq_ef"],
            phase=cfg["qubit_phase"],
            gain=cfg["qubit_pi_gain_ef"],
        )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        self.pulse(ch=cfg["qubit_ch"], name="qubit_pi_pulse", t=0)
        self.delay_auto(0.01)
        self.pulse(ch=self.cfg["qubit_ch_ef"], name="qubit_pulse", t=0)
        self.delay_auto(cfg["wait_time"] + 0.05, tag="wait")
        if cfg["ge_ref"] is True:
            self.pulse(ch=cfg["qubit_ch"], name="qubit_pi_pulse", t=0)
            self.delay_auto(0.01)
        self.delay_auto(0.01)
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class T1_ef:
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(self, py_avg, liveplot=False):
        if liveplot:
            self.liveplot(py_avg)

        else:
            prog = T1Program(
                self.soccfg,
                reps=self.cfg["reps"],
                final_delay=self.cfg["relax_delay"],
                cfg=self.cfg,
            )
            self.iq_list = prog.acquire(self.soc, soft_avgs=py_avg, progress=True)
            self.iq_list[0][0].dot([1, 1j])
            self.delay_times = prog.get_time_param("wait", "t", as_array=True)

    def plot(self):
        T1_analyze(self.delay_times, self.iq_list[0][0].dot([1, 1j]))

    def liveplot(self, py_avg):
        iq = 0

        marker_style = {
            "marker": "o",
            "markersize": 5,
            "alpha": 0.7,
            "linestyle": "-",
        }
        fig, ax = plt.subplots(figsize=(6, 4))
        prog = T1Program(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )
        self.delay_times = prog.get_time_param("wait", "t", as_array=True)

        for avg in tqdm(range(py_avg), desc="average count"):
            self.iq_list = prog.acquire(self.soc, soft_avgs=1, progress=False)

            iq_data = self.iq_list[0][0].dot([1, 1j])
            iq = iq_data if avg == 0 else iq + iq_data
            self.iqdata = iq / (avg + 1)

            ax.cla()
            ax.plot(self.delay_times, np.abs(post_rotate(self.iqdata)), **marker_style)
            ax.set_title(f"average: {avg + 1} / {py_avg}")
            ax.set_xlabel("Times (us)")
            ax.set_ylabel("Signal (ADC unit)")

            clear_output(wait=True)
            display(fig)

        clear_output(wait=True)

        pOpt, pCov = fitexp(self.delay_times, np.abs(post_rotate(self.iqdata)))
        p_sigma = np.sqrt(np.diag(pCov))
        ax.plot(self.delay_times, expfunc(self.delay_times, *pOpt), label="Fit")
        ax.set_title(f"T1 ef = {pOpt[3]:.2f} +-{p_sigma[3]:.2f}$\mu s$", fontsize=15)
        ax.legend()

    def saveLabber(self, qb_idx, yoko_value=None):
        expt_name = "s013_T1_ef" + f"_Q{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name, yoko_value)

        try:
            self.cfg.pop("wait_time")
        except:
            pass

        dict_val = yml_comment(self.cfg)

        hdf5_generator(
            filepath=file_path,
            x_info={"name": "Times", "unit": "us", "values": self.delay_times},
            z_info={"name": "Signal", "unit": "ADC unit", "values": self.iqdata},
            comment=(f"{dict_val}"),
            tag="T1",
        )

        print(f"Data save to {file_path}")
