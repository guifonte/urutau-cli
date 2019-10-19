import time
import os
import collections

import numpy as np
from librosa import output

import plot
import solver
import body
import strings


def run(phikyz_path, fmk_path, string_name, tf, xp,
        method='cfc', pol=2, damp='paiva', fs=44100,
        phikyzp_path=None, nb=-1, xsi=0.1,
        string_len=0.65, string_fmax=10000, string_f0=-1,
        pluck_ti=0.001, pluck_dp=0.008, pluck_F0=10, pluck_gamma=1.57079632679,
        verbose=False, fft=False, fft_window=-1, pluckingpoint=False, displ=False, vel=False, acc=True,
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
    sel_string = strings.get_string_params(string_name, string_len, string_f0, stringdb_path, string_fmax)
    if sel_string is None:
        return

    #Ns = int(string_fmax / sel_string.f0)  # number of string modes

    # Calculate string losses
    if damp == 'paiva':
        Qt = strings.damping_paiva(sel_string, neta, rho, dVETE, Qd)
    else:
        Qt = strings.damping_woodhouse(sel_string)

    Qt = Qt/2

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
                res = solver.cfc_solve_2p_2cp(body_params, Qt, sel_string, fs, pluck, tf, verbose)
            else:
                res = solver.cfc_solve_2p(body_params, Qt, sel_string, fs, pluck, tf, verbose)
        else:  # pol == 1
            if phikyzp_path is not None:
                res = solver.cfc_solve_1p_2cp(body_params, Qt, sel_string, fs, pluck, tf, verbose)
            else:
                res = solver.cfc_solve_1p(body_params, Qt, sel_string, fs, pluck, tf, verbose)
    else:  # method = 'fft'
        if pol == 2:
            res = solver.fft_solve_2p(body_params, tf, fs, Qt, sel_string, xp, pluck_gamma)
        else:  # pol == 1
            res = solver.fft_solve_1p(body_params, tf, fs, Qt, sel_string, xp, pluck_gamma)

    end_time = time.time()
    print("Elapsed time = {:.5f}s".format(end_time - start_time))

    # ------------------------------------------------------------------------------------------------------------------
    # Outputs
    output_path = output_path+timestamp+'_'+method+'_'+damp+'_'+string_name+'_'\
                  +str(sel_string.f0).replace('.', '-')+'Hz_'+str(xp).replace('.', '-')+'xp_'\
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
    if fft:
        if fft_window == -1:
            df = 1 / tf
            winlen_obs = ''
            winlen_path = ''
        else:
            df = 1 / (fft_window*1e-3)
            winlen_obs = ' - ' + str(int(fft_window)) + ' ms'
            winlen_path = '_' + str(int(fft_window)) + 'ms'
        f = np.linspace(-fs / 2, fs / 2 - df, np.floor(fs / df))
        winlen = len(f)
        a0 = 10e-6
        v0 = 10e-9
        s0 = 10e-12

        if method == 'fft':
            if acc:
                if pol != 2:
                    plot.plot_1p(output_path + timestamp + '_fft_acc' + winlen_path, res.z_c, f, winlen, a0,
                            'FFT - Acceleration' + winlen_obs, 'acceleration (dB re ' + r'$10\mu m/s^2$' + ')')
                else:
                    plot.plot_2p(output_path + timestamp + '_fft_2p_acc' + winlen_path, res.z_c, res.y_c, f, winlen, a0,
                            'FFT - Acceleration - 2 polarizations' + winlen_obs,
                            'acceleration (dB re ' + r'$10\mu m/s^2$' + ')')
        else:
            if displ:
                if pol != 2:
                    plot.plot_1p(output_path + timestamp + '_cfc_displ' + winlen_path, res.z_b1, f, winlen, s0,
                            'CFC - Displacement' + winlen_obs, 'displacement (dB re ' + r'$10p m$' + ')')
                else:
                    plot.plot_2p(output_path + timestamp + '_cfc_2p_displ' + winlen_path, res.z_b1, res.y_b1, f, winlen, s0,
                            'CFC - Displacement - 2 polarizations' + winlen_obs,
                            'displacement (dB re ' + r'$10p m$' + ')')
                if pluckingpoint:
                    if pol != 2:
                        plot.plot_1p(output_path + timestamp + '_cfc_displ_pluck' + winlen_path, res.z_p, f, winlen, s0,
                                'CFC - Displacement - Plucking Point' + winlen_obs,
                                'displacement (dB re ' + r'$10p m$' + ')')
                    else:
                        plot.plot_2p(output_path + timestamp + '_cfc_2p_displ_pluck' + winlen_path, res.z_p, res.y_p,
                                f, winlen, s0, 'CFC - Displacement - Plucking Point - 2 polarizations' + winlen_obs,
                                'displacement (dB re ' + r'$10p m$' + ')')
            if vel:
                if pol != 2:
                    plot.plot_1p(output_path + timestamp + '_cfc_vel' + winlen_path, res_z_vel, f, winlen, v0,
                            'CFC - Velocity' + winlen_obs, 'velocity (dB re ' + r'$10n m/s$' + ')')
                else:
                    plot.plot_2p(output_path + timestamp + '_cfc_2p_vel' + winlen_path, res_z_vel, res_y_vel, f, winlen, v0,
                            'CFC - Velocity - 2 polarizations' + winlen_obs,
                            'velocity (dB re ' + r'$10n m/s$' + ')')
                if pluckingpoint:
                    if pol != 2:
                        plot.plot_1p(output_path + timestamp + '_cfc_vel_pluck' + winlen_path, res_z_p_vel, f, winlen, v0,
                                'CFC - Velocity - Plucking Point' + winlen_obs,
                                'velocity (dB re ' + r'$10n m/s$' + ')')
                    else:
                        plot.plot_2p(output_path + timestamp + '_cfc_2p_vel_pluck' + winlen_path, res_z_p_vel, res_y_p_vel,
                                f, winlen, v0, 'CFC - Velocity - Plucking Point - 2 polarizations' + winlen_obs,
                                'velocity (dB re ' + r'$10n m/s$' + ')')
            if acc:
                if pol != 2:
                    plot.plot_1p(output_path + timestamp + '_cfc_acc' + winlen_path, res_z_acc, f, winlen, a0,
                            'CFC - Acceleration' + winlen_obs, 'acceleration (dB re ' + r'$10\mu m/s^2$' + ')')
                else:
                    plot.plot_2p(output_path + timestamp + '_cfc_2p_acc' + winlen_path, res_z_acc, res_y_acc, f, winlen, a0,
                            'CFC - Acceleration - 2 polarizations' + winlen_obs,
                            'acceleration (dB re ' + r'$10\mu m/s^2$' + ')')
                if pluckingpoint:
                    if pol != 2:
                        plot.plot_1p(output_path + timestamp + '_cfc_acc_pluck' + winlen_path, res_z_p_acc, f, winlen, a0,
                                'CFC - Acceleration - Plucking Point' + winlen_obs,
                                'acceleration (dB re ' + r'$10\mu m/s^2$' + ')')
                    else:
                        plot.plot_2p(output_path + timestamp + '_cfc_2p_acc_pluck' + winlen_path, res_z_p_acc, res_y_p_acc,
                                f, winlen, a0, 'CFC - Acceleration - Plucking Point - 2 polarizations' + winlen_obs,
                                'acceleration (dB re ' + r'$10\mu m/s^2$' + ')')

    print("METHOD RUNNER FINISHED")
    # ------------------------------------------------------------------------------------------------------------------


def derivative(a, dt):
    b = np.zeros(len(a) - 1)

    for i in range(1, len(a) - 1):
        b[i] = (a[i] - a[i-1])/dt

    return b