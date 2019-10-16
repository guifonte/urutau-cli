import time
import os
import collections
import argparse
import textwrap

import numpy as np
from librosa import output
import matplotlib.pyplot as plt

import solver
import body
import strings


def run(phikyz_path, fmk_path, string_name, tf, xp,
        method='cfc', pol=2, damp='paiva', fs=44100,
        phikyzp_path=None, nb=-1, xsi=0.1,
        string_len=0.65, string_fmax=10000, string_f0=-1,
        pluck_ti=0.001, pluck_dp=0.008, pluck_F0=10, pluck_gamma=1.57079632679,
        verbose=False, graphic=False, fftwindow=-1, pluckingpoint=False, displ=False, vel=False, acc=True,
        wav=True, mp3=False):

    # ------------------------------------------------------------------------------------------------------------------
    # Overwriteable values
    # You can uncomment this values if you feel it's easier to test than with the command line
    # Just set dummy values to the required tags --phikyz --fmk --string --tf --xp so the program is allowed to run

    # Simulation Parameters
    # method =      # 'cfc' or 'fft'
    # pol =         # 1 or 2
    # damp =        # 'paiva' or 'woodhouse'
    # fs =
    # tf =

    # Body Parameters
    # nb =
    # phikyz_path =
    # fmk_path =
    # xsi =
    # phikyzp_path =

    # String
    # string_name =
    # string_len =
    # string_fmax =
    # string_f0 =

    # Pluck Parameters
    # xp =
    # pluck_ti =
    # pluck_dp =
    # pluck_F0 =
    # pluck_gamma =

    # Output Parameters
    # verbose =
    # graphic =
    # fftwindow =
    # pluckingpoint =
    # displ =
    # vel =
    # acc =
    # wav =
    # mp3 =

    # ------------------------------------------------------------------------------------------------------------------
    # Now starts the program (Load inputs for the solver)
    print("START METHOD RUNNER")
    print("LOAD INPUTS")
    output_path = 'data/outputs/'

    # Paiva's Damping model parameters
    neta = 1.8 * 10 ** (-5)     # air dynamic viscosity - (Valette)
    rho = 1.2                   # air density - (Valette)
    Qd = 5500                   # damping due to the dislocation phenomenom(Adjusted in Pathe) -
    # Depends strongly on the history of the material 7000 - 80000 for brass strings(Cuesta)!
    dVETE = 1 * 10 ** (-3)      # thermo and visco - elastic effects constant(Valette) -
    # It is a problem, high uncertainty in determination!

    # Pluck
    Pluck = collections.namedtuple('Pluck', ['xp', 'Ti', 'dp', 'F0', 'gamma'])
    pluck = Pluck(xp, pluck_ti, pluck_dp, pluck_F0, pluck_gamma)

    # Strings
    stringdb_path = 'data/inputs/stringdb.json'
    string = strings.get_string_params(string_name, string_len, string_f0, stringdb_path)
    if string is None:
        return
    Ns = round(string_fmax / string.f0)  # number of string modes

    # Calculate string losses
    if damp == 'paiva':
        Qt = strings.damping_paiva(Ns, string, neta, rho, dVETE, Qd)
    else:
        Qt = strings.damping_woodhouse(Ns, string)

    # Load body parameters
    if phikyzp_path is not None:
        body_params = body.load_body_2cp(phikyz_path, phikyzp_path, fmk_path, nb, xsi)
    else:
        body_params = body.load_body(phikyz_path, fmk_path, nb, xsi)
    if body_params is None:
        return
    # ------------------------------------------------------------------------------------------------------------------
    # Start Solver
    print("START SOLVER")
    timestamp = str(time.time()).split('.')[0]

    start_time = time.time()
    # Start solvers
    if method == 'cfc':
        if pol == 2:
            if phikyzp_path is not None:
                res = solver.cfc_solve_2p_2cp(body_params, Qt, Ns, string, fs, pluck, tf, verbose)
            else:
                res = solver.cfc_solve_2p(body_params, Qt, Ns, string, fs, pluck, tf, verbose)
        else:  # pol == 1
            if phikyzp_path is not None:
                res = solver.cfc_solve_1p_2cp(body_params, Qt, Ns, string, fs, pluck, tf, verbose)
            else:
                res = solver.cfc_solve_1p(body_params, Qt, Ns, string, fs, pluck, tf, verbose)
    else:  # method = 'fft'
        if pol == 2:
            res = solver.fft_solve_2p(body_params, tf, fs, Qt, Ns, string, xp, pluck_gamma)
        else:  # pol == 1
            res = solver.fft_solve_1p(body_params, tf, fs, Qt, Ns, string, xp, pluck_gamma)

    end_time = time.time()
    print("Elapsed time = {:.5f}s".format(end_time - start_time))

    # ------------------------------------------------------------------------------------------------------------------
    # Outputs
    output_path = output_path+timestamp+'_'+method+'_'+damp+'_'+string_name+'_'+str(string.f0).replace('.', '-')+'Hz_'\
                  +str(pol)+'p_fs'+str(fs)+'_tf'+str(int(1000*tf))
    if phikyzp_path is not None:
        output_path = output_path + '_peg'
    os.makedirs(output_path)
    output_path = output_path + '/'
    print("GENERATE OUTPUT FILES AT: " + output_path)

    # wav
    if wav:
        if method == 'fft':
            if acc:
                output.write_wav(output_path + timestamp + '_fft_z_acc.wav', res.z_c, sr=fs, norm=True)
                if pol == 2:
                    output.write_wav(output_path + timestamp + '_fft_y_acc.wav', res.y_c, sr=fs, norm=True)
        else:
            if displ:
                output.write_wav(output_path + timestamp + '_cfc_z_displ.wav', res.z_b1, sr=fs, norm=True)
                if pol == 2:
                    output.write_wav(output_path + timestamp + '_cfc_y_displ.wav', res.y_b1, sr=fs, norm=True)
                if pluckingpoint:
                    output.write_wav(output_path + timestamp + '_cfc_z_displ_pluck.wav', res.z_p, sr=fs, norm=True)
                    if pol == 2:
                        output.write_wav(output_path + timestamp + '_cfc_y_displ_pluck.wav', res.y_p, sr=fs, norm=True)
            if vel:
                res_z_vel = np.concatenate(([0], derivative(res.z_b1, 1 / fs)))
                output.write_wav(output_path + timestamp + '_cfc_z_vel.wav', res_z_vel, sr=fs, norm=True)
                if pol == 2:
                    res_y_vel = np.concatenate(([0], derivative(res.y_b1, 1 / fs)))
                    output.write_wav(output_path + timestamp + '_cfc_y_vel.wav', res_y_vel, sr=fs, norm=True)
                if pluckingpoint:
                    res_z_p_vel = np.concatenate(([0], derivative(res.z_p, 1 / fs)))
                    output.write_wav(output_path + timestamp + '_cfc_z_vel_pluck.wav', res_z_p_vel, sr=fs, norm=True)
                    if pol == 2:
                        res_y_p_vel = np.concatenate(([0], derivative(res.y_p, 1 / fs)))
                        output.write_wav(output_path + timestamp + '_cfc_y_vel_pluck.wav', res_y_p_vel, sr=fs, norm=True)
            if acc:
                if vel:
                    res_z_acc = np.concatenate(([0], derivative(res_z_vel, 1 / fs)))
                    if pol == 2:
                        res_y_acc = np.concatenate(([0], derivative(res_y_vel, 1 / fs)))
                    if pluckingpoint:
                        res_z_p_acc = np.concatenate(([0], derivative(res_z_p_vel, 1 / fs)))
                        if pol == 2:
                            res_y_p_acc = np.concatenate(([0], derivative(res_y_p_vel, 1 / fs)))
                else:
                    res_z_acc = np.concatenate(([0], [0], derivative(derivative(res.z_b1, 1 / fs), 1 / fs)))
                    if pol == 2:
                        res_y_acc = np.concatenate(([0], [0], derivative(derivative(res.y_b1, 1 / fs), 1 / fs)))
                    if pluckingpoint:
                        res_z_p_acc = np.concatenate(([0], [0], derivative(derivative(res.z_p, 1 / fs), 1 / fs)))
                        if pol == 2:
                            res_y_p_acc = np.concatenate(([0], [0], derivative(derivative(res.y_p, 1 / fs), 1 / fs)))
                output.write_wav(output_path + timestamp + '_cfc_z_acc.wav', res_z_acc, sr=fs, norm=True)
                if pol == 2:
                    output.write_wav(output_path + timestamp + '_cfc_y_acc.wav', res_y_acc, sr=fs, norm=True)
                if pluckingpoint:
                    output.write_wav(output_path + timestamp + '_cfc_z_acc_pluck.wav', res_z_p_acc, sr=fs, norm=True)
                    if pol == 2:
                        output.write_wav(output_path + timestamp + '_cfc_y_acc_pluck.wav', res_y_p_acc, sr=fs, norm=True)

    # mp3
    # AudioSegment.from_wav(output_path + timestamp + '_fft_z.wav').export(output_path + timestamp + '_fft_z.mp3',format="mp3")
    if mp3:
        if method == 'fft':
            if acc:
                print("")
                if pol == 2:
                    print("")
        else:
            if displ:
                print("")
                if pol == 2:
                    print("")
                if pluckingpoint:
                    print("")
                    if pol == 2:
                        print("")
            if vel:
                print("")
                if pol == 2:
                    print("")
                if pluckingpoint:
                    print("")
                    if pol == 2:
                        print("")
            if acc:
                print("")
                if pol == 2:
                    print("")
                if pluckingpoint:
                    print("")
                    if pol == 2:
                        print("")

    # graphic
    if graphic:
        if fftwindow == -1:
            df = 1 / tf
            winlen_obs = ''
            winlen_path = ''
        else:
            df = 1 / (fftwindow*1e-3)
            winlen_obs = ' - ' + str(int(fftwindow)) + ' ms'
            winlen_path = '_' + str(int(fftwindow)) + 'ms'
        f = np.linspace(-fs / 2, fs / 2 - df, np.floor(fs / df))
        winlen = len(f)
        a0 = 10e-6
        v0 = 10e-9
        s0 = 10e-12

        if method == 'fft':
            if acc:
                if pol != 2:
                    plot_1p(output_path + timestamp + '_fft_acc' + winlen_path, res.z_c, f, winlen, a0,
                            'FFT - Acceleration' + winlen_obs, 'acceleration (dB re ' + r'$10\mu m/s^2$' + ')')
                else:
                    plot_2p(output_path + timestamp + '_fft_2p_acc' + winlen_path, res.z_c, res.y_c, f, winlen, a0,
                            'FFT - Acceleration - 2 polarizations' + winlen_obs,
                            'acceleration (dB re ' + r'$10\mu m/s^2$' + ')')
        else:
            if displ:
                if pol != 2:
                    plot_1p(output_path + timestamp + '_cfc_displ' + winlen_path, res.z_b1, f, winlen, s0,
                            'CFC - Displacement' + winlen_obs, 'displacement (dB re ' + r'$10p m$' + ')')
                else:
                    plot_2p(output_path + timestamp + '_cfc_2p_displ' + winlen_path, res.z_b1, res.y_b1, f, winlen, s0,
                            'CFC - Displacement - 2 polarizations' + winlen_obs,
                            'displacement (dB re ' + r'$10p m$' + ')')
                if pluckingpoint:
                    if pol != 2:
                        plot_1p(output_path + timestamp + '_cfc_displ_pluck' + winlen_path, res.z_p, f, winlen, s0,
                                'CFC - Displacement - Plucking Point' + winlen_obs,
                                'displacement (dB re ' + r'$10p m$' + ')')
                    else:
                        plot_2p(output_path + timestamp + '_cfc_2p_displ_pluck' + winlen_path, res.z_p, res.y_p,
                                f, winlen, s0, 'CFC - Displacement - Plucking Point - 2 polarizations' + winlen_obs,
                                'displacement (dB re ' + r'$10p m$' + ')')
            if vel:
                if pol != 2:
                    plot_1p(output_path + timestamp + '_cfc_vel' + winlen_path, res_z_vel, f, winlen, v0,
                            'CFC - Velocity' + winlen_obs, 'velocity (dB re ' + r'$10n m/s$' + ')')
                else:
                    plot_2p(output_path + timestamp + '_cfc_2p_vel' + winlen_path, res_z_vel, res_y_vel, f, winlen, v0,
                            'CFC - Velocity - 2 polarizations' + winlen_obs,
                            'velocity (dB re ' + r'$10n m/s$' + ')')
                if pluckingpoint:
                    if pol != 2:
                        plot_1p(output_path + timestamp + '_cfc_vel_pluck' + winlen_path, res_z_p_vel, f, winlen, v0,
                                'CFC - Velocity - Plucking Point' + winlen_obs,
                                'velocity (dB re ' + r'$10n m/s$' + ')')
                    else:
                        plot_2p(output_path + timestamp + '_cfc_2p_vel_pluck' + winlen_path, res_z_p_vel, res_y_p_vel,
                                f, winlen, v0, 'CFC - Velocity - Plucking Point - 2 polarizations' + winlen_obs,
                                'velocity (dB re ' + r'$10n m/s$' + ')')
            if acc:
                if pol != 2:
                    plot_1p(output_path + timestamp + '_cfc_acc' + winlen_path, res_z_acc, f, winlen, a0,
                            'CFC - Acceleration' + winlen_obs, 'acceleration (dB re ' + r'$10\mu m/s^2$' + ')')
                else:
                    plot_2p(output_path + timestamp + '_cfc_2p_acc' + winlen_path, res_z_acc, res_y_acc, f, winlen, a0,
                            'CFC - Acceleration - 2 polarizations' + winlen_obs,
                            'acceleration (dB re ' + r'$10\mu m/s^2$' + ')')
                if pluckingpoint:
                    if pol != 2:
                        plot_1p(output_path + timestamp + '_cfc_acc_pluck' + winlen_path, res_z_p_acc, f, winlen, a0,
                                'CFC - Acceleration - Plucking Point' + winlen_obs,
                                'acceleration (dB re ' + r'$10\mu m/s^2$' + ')')
                    else:
                        plot_2p(output_path + timestamp + '_cfc_2p_acc_pluck' + winlen_path, res_z_p_acc, res_y_p_acc,
                                f, winlen, a0, 'CFC - Acceleration - Plucking Point - 2 polarizations' + winlen_obs,
                                'acceleration (dB re ' + r'$10\mu m/s^2$' + ')')

    print("METHOD RUNNER FINISHED")
    # ------------------------------------------------------------------------------------------------------------------


def plot_1p(path, z, f, winlen, ref, title, y_label):
    plt.ioff()
    plt.figure()
    z_fft = 20*np.abs(np.log10(np.fft.fftshift(np.fft.fft(z[0:winlen] / np.max(z[0:winlen])))/ref))
    plt.semilogx(f, z_fft, lw=1, label='z')
    plt.xlabel('frequency (Hz)', fontsize=10)
    plt.ylabel(y_label, fontsize=10)
    plt.legend(fontsize=10)
    plt.title(title, fontsize=14)
    plt.xlim(10 ** 2, 10 ** 4)
    plt.ylim(20, 180)
    plt.tight_layout()
    plt.savefig(path + '.pdf')

    return


def plot_2p(path, z, y, f, winlen, ref, title, y_label):
    plt.ioff()
    plt.figure()

    z_fft = 20 * np.abs(np.log10(np.fft.fftshift(np.fft.fft(z[0:winlen] / np.max(z[0:winlen]))) / ref))
    y_fft = 20 * np.abs(np.log10(np.fft.fftshift(np.fft.fft(y[0:winlen] / np.max(y[0:winlen]))) / ref))

    plt.subplot(2, 1, 1)
    plt.title(title, fontsize=14)
    plt.xlim(10 ** 2, 10 ** 4)
    #plt.ylim(20, 180)
    plt.semilogx(f, z_fft, lw=1, label='z')
    plt.ylabel(y_label, fontsize=10)
    plt.legend(fontsize=10)

    plt.subplot(2, 1, 2)
    plt.xlim(10 ** 2, 10 ** 4)
    #plt.ylim(20, 180)
    plt.semilogx(f, y_fft, lw=1, label='y')
    plt.xlabel('frequency (Hz)', fontsize=10)
    plt.ylabel(y_label, fontsize=10)
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(path + '.pdf')

    return


def derivative(a, dt):
    b = np.zeros(len(a) - 1)

    for i in range(1, len(a) - 1):
        b[i] = (a[i] - a[i-1])/dt

    return b


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''\
                                     Hybrid Modal Synthesis Method Runner
                                     ------------------------------------
                                        Required arguments:
                                        --phikyz <path>
                                        --fmk <path>
                                        --string <name>
                                        --tf <dur>
                                        --xp <pos>
                                     ------------------------------------
                                     '''))

    body_group = parser.add_argument_group('body')
    body_group.add_argument("--phikyz", help="Path of the PHIKYZ file at the bridge (REQUIRED)",
                            required=True, metavar="<path>")
    body_group.add_argument("--phikyzp", help="Path of the PHIKYZ file at the peg",
                            metavar="<path>", default=None)
    body_group.add_argument("--fmk", help="Path of the FMK file (REQUIRED)", required=True, metavar="<path>")
    body_group.add_argument("--nb", help="body modes (default: max)", type=int, metavar="<num>", default=-1)
    xsi_group = body_group.add_mutually_exclusive_group()
    xsi_group.add_argument("--xsi-fixed", help="fixed xsi for all modes (default: %(default)s)",
                           type=float, metavar="<val>", default=0.1)
    xsi_group.add_argument("--xsi-path", help="Path of the XSI file", metavar="<path>", default=None)

    string_group = parser.add_argument_group('string')
    string_group.add_argument("--string", help="string name (REQUIRED)", required=True, metavar="<name>")
    string_group.add_argument("--string-len", help="string length (meters) (default: %(default)s)", type=float,
                              default=0.65, metavar="<len>")
    string_group.add_argument("--string-fmax", help="max frequency of string modes (Hz) (default: %(default)s)",
                              type=float, default=10000, metavar="<len>")
    string_group.add_argument("--string-f0", help="fundamental frequency (Hz) (default: string default f0)",
                              type=float, default=-1, metavar="<freq>")

    pluck_group = parser.add_argument_group('pluck', description="ramp function")
    pluck_group.add_argument("--xp", help="pluck position (meters)(0 ref: peg)(REQUIRED)",
                             required=True, metavar="<pos>", type=float)
    pluck_group.add_argument("--pluck-ti", help="starting time of the ramp (seconds)(default: %(default)s)",
                             default=0.001, metavar="<time>", type=float)
    pluck_group.add_argument("--pluck-dp", help="ramp length (seconds)(default: %(default)s)",
                             default=0.008, metavar="<len>", type=float)
    pluck_group.add_argument("--pluck-F0", help="height of the ramp (N)(default: %(default)s)",
                             default=10, metavar="<val>", type=float)
    pluck_group.add_argument("--pluck-gamma", help="pluck angle (radians)(default: pi/2)",
                             default=1.57079632679, metavar="<rad>", type=float)
    simulation_group = parser.add_argument_group('simulation')
    simulation_group.add_argument("--tf",
                                  help="Duration of the simulation (seconds) (REQUIRED)",
                                  type=float, required=True, metavar="<dur>")
    simulation_group.add_argument("--method",
                                  help="simulation method: 'cfc' or 'fft' (default: %(default)s)",
                                  choices=['cfc', 'fft'], default="cfc", metavar="<method>")
    simulation_group.add_argument("--pol", help="the number of polarizations: 1 or 2 (default: %(default)s)",
                                  choices=[1, 2], type=int, default=2, metavar="<num>")
    simulation_group.add_argument("--damp",
                                  help="damping method: 'woodhouse' or 'paiva' (default: %(default)s)",
                                  choices=['paiva', 'woodhouse'], default="paiva", metavar="<method>")
    simulation_group.add_argument("--fs",
                                  help="sample frequency (default: %(default)s)",
                                  default=44100, type=int, metavar="<freq>")

    output_group = parser.add_argument_group('output')
    output_group.add_argument("-v", "--verbose", action="store_true", help="print progress percentage for cfc")
    output_group.add_argument("-g", "--graphic", action="store_true", help="generate pdf with graphics")
    output_group.add_argument("--fftwindow", type=float, help="size of the fft window (ms)(default: full signal)",
                              default=-1, metavar="<dur>")
    output_group.add_argument("--pluckingpoint", action="store_true", help="generate files for plucking point when cfc")
    output_group.add_argument("--displ", action="store_true", help="generate files for displacement when cfc")
    output_group.add_argument("--vel", action="store_true", help="generate files for velocity when cfc")
    output_group.add_argument("--acc-no", action="store_true", help="does not generate files for acceleration")
    mp3_group = output_group.add_mutually_exclusive_group()
    mp3_group.add_argument("--mp3", action="store_true", help="generate mp3 file")
    mp3_group.add_argument("--mp3-only", action="store_true", help="generate only mp3 file (no wav)")
    args = parser.parse_args()

    if args.mp3_only:
        mp3 = True
        wav = False
    elif args.mp3:
        mp3 = True
        wav = True
    else:
        mp3 = False
        wav = True

    if args.acc_no:
        acc = False
    else:
        acc = True

    if args.xsi_path:
        xsi = args.xsi_path
    else:
        xsi = args.xsi_fixed

    run(args.phikyz, args.fmk, args.string, args.tf, args.xp,
        args.method, args.pol, args.damp, args.fs,
        args.phikyzp, args.nb, xsi,
        args.string_len, args.string_fmax, args.string_f0,
        args.pluck_ti, args.pluck_dp, args.pluck_F0, args.pluck_gamma,
        args.verbose, args.graphic, args.fftwindow, args.pluckingpoint, args.displ, args.vel, acc, wav, mp3)

