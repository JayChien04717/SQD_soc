# ----- Qick package ----- #
from qick import *
from qick.pyro import make_proxy
from qick.asm_v2 import AveragerProgramV2
from qick.asm_v2 import QickSpan, QickSweep1D

# ----- Library ----- #
import matplotlib.pyplot as plt
import numpy as np

# ----- User Library ----- #
from ..system_cfg import *
from ..system_cfg import DATA_PATH
from ..system_tool import get_next_filename_labber, hdf5_generator
from tqdm.auto import tqdm
from ..fitting import *
from ..yamltool import yml_comment
from IPython.display import display, clear_output

##################
# Define Program #
##################


class SingleToneSpectroscopyCoolingProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]
        cool_ch1 = cfg["cool_ch1"]
        cool_ch2 = cfg["cool_ch2"]

        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])
        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])

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

        self.add_loop("freqloop2", cfg["f_steps1"])
        self.add_loop("freqloop1", cfg["f_steps2"])
        self.add_readoutconfig(
            ch=ro_ch, name="myro", freq=cfg["res_freq_ge"], gen_ch=res_ch
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
            ch=cool_ch1,
            name="cool1",
            sigma=cfg["res_sigma"],
            length=5 * cfg["res_sigma"],
            even_length=True,
        )
        self.add_pulse(
            ch=cool_ch1,
            name="cool_pulse1",
            style="flat_top",
            envelope="cool1",
            length=cfg["cool_length"],
            freq=cfg["cool_freq_1"],
            phase=0,
            gain=cfg["cool_gain_1"],
        )
        self.add_gauss(
            ch=cool_ch2,
            name="cool2",
            sigma=cfg["res_sigma"],
            length=5 * cfg["res_sigma"],
            even_length=True,
        )
        self.add_pulse(
            ch=cool_ch2,
            name="cool_pulse2",
            style="flat_top",
            envelope="cool2",
            length=cfg["cool_length"],
            freq=cfg["cool_freq_2"],
            phase=0,
            gain=cfg["cool_gain_2"],
        )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        self.pulse(ch=cfg["cool_ch1"], name="cool_pulse1", t=0)
        self.pulse(ch=cfg["cool_ch2"], name="cool_pulse2", t=0)
        self.delay_auto(0.05, tag="Ring down")
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class SingleToneSpectroscopyCooling:
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(self, py_avg, liveplot=False):
        if liveplot:
            max_point = self.liveplot(py_avg)
            return max_point
        else:
            prog = SingleToneSpectroscopyCoolingProgram(
                self.soccfg,
                reps=self.cfg["reps"],
                final_delay=self.cfg["relax_delay"],
                cfg=self.cfg,
            )

            self.iq_list = prog.acquire(self.soc, soft_avgs=py_avg, progress=True)
            self.iqdata = self.iq_list[0][0].dot([1, 1j])
            self.freqs1 = prog.get_pulse_param("cool_pulse1", "freq", as_array=True)
            self.freqs2 = prog.get_pulse_param("cool_pulse2", "freq", as_array=True)

    def plot(self):
        data = np.abs(post_rotate(self.iqdata))  # shape: (n_gain, n_freq)
        pcm = plt.pcolormesh(self.freqs1, self.freqs2, data)
        plt.title("Resonator Cooling")
        plt.ylabel(r"$|f,0\rangle - |g,1\rangle$")
        plt.xlabel(r"$|f,0\rangle - |e,0\rangle$")
        plt.colorbar(pcm)

    def liveplot(self, py_avg):
        iq = 0
        prog = SingleToneSpectroscopyCoolingProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )
        self.freqs1 = prog.get_pulse_param("cool_pulse1", "freq", as_array=True)
        self.freqs2 = prog.get_pulse_param("cool_pulse2", "freq", as_array=True)

        fig, ax = plt.subplots(figsize=(6, 4))

        for i in tqdm(range(py_avg), desc="average count"):
            self.iq_list = prog.acquire(self.soc, soft_avgs=1, progress=False)
            iq_data = self.iq_list[0][0].dot([1, 1j])
            iq = iq_data if i == 0 else iq + iq_data
            self.iqdata = iq / (i + 1)

            data = np.abs(self.iqdata)  # shape: (n_gain, n_freq)

            ax.cla()
            im = ax.pcolorfast(self.freqs1, self.freqs2, data)
            # ax.pcolorfast(self.freqs1, self.freqs2, data)
            ax.set_title(f"average: {i + 1} / {py_avg}")
            ax.set_ylabel(r"$|f,0\rangle - |g,1\rangle$")
            ax.set_xlabel(r"$|f,0\rangle - |e,0\rangle$")

            clear_output(wait=True)
            display(fig)

        clear_output(wait=True)

        ax.set_title("Resonator ge cooling")
        ax.pcolorfast(self.freqs1, self.freqs2, data)
        idx1, idx2 = np.unravel_index(
            np.argmax((data - np.mean(data)) ** 2), data.shape
        )
        max1 = self.freqs1[idx2]
        max2 = self.freqs2[idx1]
        ax.plot(
            max1,
            max2,
            "*",
            color="red",
            markersize=10,
            label=f"freq={max1:.1f}, {max2:.1f} MHz",
        )
        ax.legend()
        fig.colorbar(im, ax=ax, label="Normalized Amplitude")
        return dict(f12=max1, f0g1=max2)

    def saveLabber(self, qb_idx, yoko_current=None):
        expt_name = "002b_res_ge_punchout" + f"_Q{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name, yoko_current)
        try:
            self.cfg.pop("cool_freq_1")
            self.cfg.pop("cool_freq_2")

        except:
            pass

        dict_val = yml_comment(self.cfg)

        hdf5_generator(
            filepath=file_path,
            x_info={
                "name": r"$|2,0\rangle - |0,1\rangle$",
                "unit": "Hz",
                "values": self.freqs1 * 1e6,
            },
            y_info={
                "name": r"$|2,0\rangle - |1,0\rangle$",
                "unit": "Hz",
                "values": self.freqs2 * 1e6,
            },
            z_info={"name": "Signal", "unit": "ADC unit", "values": self.iqdata},
            comment=(f"{dict_val}"),
            tag="OneTone",
        )
        print(f"Data save to {file_path}")
