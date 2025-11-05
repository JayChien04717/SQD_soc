# ----- Qick package ----- #
from qick import *
from qick.pyro import make_proxy

# for now, all the tProc v2 classes need to be individually imported (can't use qick.*)
# the main program class
from qick.asm_v2 import AveragerProgramV2

# for defining sweeps
from qick.asm_v2 import QickSpan, QickSweep1D

# ----- Library ----- #
import matplotlib.pyplot as plt
import numpy as np
import datetime
from system_cfg import *
from system_tool import select_config_idx, saveh5, get_next_filename
from pprint import pprint
from tqdm import tqdm

# ----- Experiment configurations ----- #
expt_name = "002a_res_spec_ge_mux"
config = {**hw_cfg, **readout_cfg, **qubit_cfg, **expt_cfg}


##################
# Define Program #
##################


class SingleToneSpectroscopyProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_chs = cfg["mux_ro_chs"]
        gen_ch = cfg["mux_ch"]

        self.declare_gen(
            ch=gen_ch,
            nqz=cfg["nqz_res"],
            ro_ch=ro_chs[0],
            mux_freqs=cfg["res_freq_ge"],
            mux_gains=cfg["res_gain_ge"],
            mux_phases=cfg["res_phase"],
            mixer_freq=cfg["mixer_freq"],
        )
        for ch, f, ph in zip(ro_chs, cfg["res_freq_ge"], cfg["res_phase"]):
            self.declare_readout(
                ch=ch, length=cfg["res_length"], freq=f, phase=ph, gen_ch=gen_ch
            )

        self.add_pulse(
            ch=gen_ch,
            name="res_pulse",
            style="const",
            length=cfg["res_length"],
            mask=[0, 1, 2, 3, 4, 5],
        )

    def _body(self, cfg):
        self.pulse(ch=cfg["mux_ch"], name="res_pulse", t=0)
        self.trigger(ros=cfg["mux_ro_chs"], pins=[0], t=cfg["trig_time"])


###################
# Run the Program
###################

expt = {"start": -10, "step": 10, "expts": 51}
fpts = [expt["start"] + ii * expt["step"] for ii in range(expt["expts"])]
fcenter = np.array(config["res_freq_ge"])

avgi = np.zeros((len(fcenter), len(fpts)))
avgq = np.zeros((len(fcenter), len(fpts)))
amps = np.zeros((len(fcenter), len(fpts)))
for index, f in enumerate(tqdm(fpts)):
    config["res_freq_ge"] = fcenter + f
    prog = SingleToneSpectroscopyProgram(soccfg, reps=10, final_delay=0.5, cfg=config)
    iq_list = prog.acquire(soc, soft_avgs=config["py_avg"], progress=False)
    for i in range(len(fcenter)):
        avgi[i][index] = iq_list[i][:, 0]
        avgq[i][index] = iq_list[i][:, 1]
        amps[i][index] = np.abs(iq_list[i][:, 0] + 1j * iq_list[i][:, 1])
amps = np.array(amps)
avgi = np.array(avgi)
avgq = np.array(avgq)

###################
# Plot
###################
Plot = True

if Plot:
    res_freqs = []
    plt.figure(figsize=(18, 9))
    plt.rcParams.update({"font.size": 12})  # Set base font size

    plt.subplot(231, xlabel="Freq (MHz)", ylabel="Amp. (adc level)")
    plt.plot(fpts + fcenter[0], amps[0], "o-", label="amp")
    freq_r1 = fpts[np.argmin(amps[0])] + fcenter[0]
    res_freqs.append(freq_r1)
    plt.axvline(freq_r1, linestyle="--", color="red")
    plt.title("Resonator 0: " + str(freq_r1) + "MHz")

    plt.subplot(232, xlabel="Freq (MHz)", ylabel="Amp. (adc level)")
    plt.plot(fpts + fcenter[1], amps[1], "o-", label="amp")
    freq_r2 = fpts[np.argmin(amps[1])] + fcenter[1]
    res_freqs.append(freq_r2)
    plt.axvline(freq_r2, linestyle="--", color="red")
    plt.title("Resonator 1: " + str(freq_r2) + "MHz")

    plt.subplot(233, xlabel="Freq (MHz)", ylabel="Amp. (adc level)")
    plt.plot(fpts + fcenter[2], amps[2], "o-", label="amp")
    freq_r3 = fpts[np.argmin(amps[2])] + fcenter[2]
    res_freqs.append(freq_r3)
    plt.axvline(freq_r3, linestyle="--", color="red")
    plt.title("Resonator 2: " + str(freq_r3) + "MHz")

    plt.subplot(234, xlabel="Freq (MHz)", ylabel="Amp. (adc level)")
    plt.plot(fpts + fcenter[3], amps[3], "o-", label="amp")
    freq_r4 = fpts[np.argmin(amps[3])] + fcenter[3]
    res_freqs.append(freq_r4)
    plt.axvline(freq_r4, linestyle="--", color="red")
    plt.title("Resonator 3: " + str(freq_r4) + "MHz")

    plt.subplot(235, xlabel="Freq (MHz)", ylabel="Amp. (adc level)")
    plt.plot(fpts + fcenter[4], amps[4], "o-", label="amp")
    freq_r5 = fpts[np.argmin(amps[4])] + fcenter[4]
    res_freqs.append(freq_r5)
    plt.axvline(freq_r5, linestyle="--", color="red")
    plt.title("Resonator 4: " + str(freq_r5) + "MHz")

    plt.subplot(236, xlabel="Freq (MHz)", ylabel="Amp. (adc level)")
    plt.plot(fpts + fcenter[5], amps[5], "o-", label="amp")
    freq_r6 = fpts[np.argmin(amps[5])] + fcenter[5]
    res_freqs.append(freq_r6)
    plt.axvline(freq_r6, linestyle="--", color="red")
    plt.title("Resonator 5: " + str(freq_r6) + "MHz")

    plt.tight_layout()
    plt.show()
Save = True
if Save:
    data_path = "./data"
    labber_data = "./data/Labber"
    print("Experiment name: " + expt_name)
    file_path = get_next_filename(data_path, expt_name, suffix=".h5")
    print("Current data file: " + file_path)

    data_dict = {
        "x_name": "x_axis",
        "x_value": fpts,
        "z_name": "iq_list",
        "z_value": iq_list[i][:, 0] + 1j * iq_list[i][:, 1],
    }

    result = {"T1": "350us", "T2": "130us"}

    saveh5(file_path, data_dict, result)

# %%
