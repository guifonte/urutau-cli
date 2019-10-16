import numpy as np
import collections


def fft_solve_2p(body, Tf, fs, Qt, Ns, string, xp, gamma):
    # Frequency domain vector[Hz]
    df = 1 / Tf
    f = np.linspace(-fs / 2, fs / 2 - df, np.floor(fs / df))

    # Calculate body admitance matrix
    Yb = calc_Y_FRF_2p(body, Tf, fs)

    Zs, H = Ys_and_H_gen(Qt, 2*np.pi*f, Ns, string, xp)
    Zs = Zs
    H = H
    G = np.zeros(Yb.shape, dtype=np.complex)
    Y = np.zeros(Yb.shape, dtype=np.complex)

    for j in range(0, len(f)):
        if 1/np.linalg.cond(Yb[:, :, j]) < 2 * np.spacing(1):
            Y[:, :, j] = Yb[:, :, j]
        else:
            Yb_inv = np.linalg.inv(Yb[:, :, j])
            Y[:, :, j] = np.linalg.inv(Zs[:, :, j] + Yb_inv)
        #if 1 / np.linalg.cond(Zs[:, :, j]) < 2 * np.spacing(1):
        #    Y[:, :, j] = Zs[:, :, j]
        #else:
        #    Y[:, :, j] = np.linalg.inv(Zs[:, :, j])
        G[:, :, j] = Y[:, :, j]@H[:, :, j]

    G[:, 0, :] = G[:, 0, :] * np.cos(gamma)
    G[:, 1, :] = G[:, 1, :] * np.sin(gamma)
    G_temp = np.fft.ifftshift(G, 2)
    y = np.fft.ifft(G_temp, len(G_temp[1, 1, :]), 2)
    # Tf = 1/(f[1]-f[2])
    h = np.real(np.sum(y, axis=1))

    Accelerations = collections.namedtuple('Accelerations', ['z_c', 'y_c'])
    return Accelerations(h[1, :], h[0, :])


def fft_solve_1p(body, Tf, fs, Qt, Ns, string, xp, ratio):
    # Frequency domain vector[Hz]
    df = 1 / Tf
    f = np.linspace(-fs / 2, fs / 2 - df, np.floor(fs / df))

    # Calculate body admitance matrix
    Yb = calc_Y_FRF_1p(body, Tf, fs)

    Zs, H = Ys_and_H_gen_1p(Qt, 2*np.pi*f, Ns, string, xp)
    Zs = Zs*ratio
    H = H*ratio
    G = np.zeros(Yb.shape, dtype=np.complex)
    Y = np.zeros(Yb.shape, dtype=np.complex)

    for j in range(0, len(f)):
        if Yb[j] == 0:
            Y[j] = Yb[j]
        else:
            Y[j] = 1/(Zs[j] + 1/Yb[j])
        #if Zs[j] == 0:
        #    Y[j] = Zs[j]
        #else:
        #    Y[j] = 1/Zs[j]
        G[j] = Y[j]*H[j]

    G_temp = np.fft.ifftshift(G)
    y = np.fft.ifft(G_temp)
    # Tf = 1/(f[1]-f[2])
    h = np.real(y)
    Accelerations = collections.namedtuple('Accelerations', ['z_c'])
    return Accelerations(h)


def calc_Y_FRF_2p(body, Tf, fs):
    mk = body.mk
    xsi = body.xsi
    phiky = body.phiky
    phikz = body.phikz
    wk = body.wk
    n_modes = body.Nb

    w = np.linspace((-fs*np.pi), (fs*np.pi-2*np.pi/Tf), np.floor(2*fs*np.pi / (2*np.pi/Tf)))
    Y = np.zeros((2, 2, len(w)), dtype=np.complex)

    etak = 2*xsi

    for mn in range(0, n_modes):
        aux = (mk[mn] * (wk[mn]**2 + 1j * w * wk[mn] * etak[mn] - w**2))
        Y[0, 0, :] = Y[0, 0, :] + 1j * w * phiky[mn] * phiky[mn] / aux
        Y[1, 0, :] = Y[1, 0, :] + 1j * w * phiky[mn] * phikz[mn] / aux
        Y[1, 1, :] = Y[1, 1, :] + 1j * w * phikz[mn] * phikz[mn] / aux

    Y[0, 1, :] = Y[1, 0, :]

    return Y


def calc_Y_FRF_1p(body, Tf, fs):
    mk = body.mk
    xsi = body.xsi
    phikz = body.phikz
    wk = body.wk
    n_modes = body.Nb

    w = np.linspace((-fs*np.pi), (fs*np.pi-2*np.pi/Tf), np.floor(2*fs*np.pi / (2*np.pi/Tf)))
    Y = np.zeros(len(w), dtype=np.complex)

    etak = 2*xsi

    for mn in range(0, n_modes):
        aux = (mk[mn] * (wk[mn]**2 + 1j * w * wk[mn] * etak[mn] - w**2))
        Y = Y + 1j * w * phikz[mn] * phikz[mn] / aux

    return Y


def Ys_and_H_gen(Qt, w, Ns, string, xp):
    c = string.c
    L = string.L
    T = string.T

    Zs = 1/(w+np.spacing(1))
    Hi = 1/(w+np.spacing(1))

    for mn in range(1, Ns+1):
        wj = mn*np.pi*c/L
        aux = 1j*1/Qt[mn-1]/2
        aux2 = 1/(w-wj*(1+aux)) + 1/(w + wj*(1-aux))
        Zs = Zs + aux2
        Hi = Hi + (-1)**mn*aux2
    Zs = Zs*(-1j*T/L)
    Hi = Hi*(c/L*np.sin(w*xp/c))
    H = np.zeros((2, 2, len(w)), dtype=np.complex)
    Z = np.zeros((2, 2, len(w)), dtype=np.complex)

    Z[0, 0, :] = Zs
    Z[0, 1, :] = Zs
    Z[1, 0, :] = Zs
    Z[1, 1, :] = Zs

    H[0, 0, :] = Hi
    H[1, 1, :] = Hi

    return Z, H


def Ys_and_H_gen_1p(Qt, w, Ns, string, xp):
    c = string.c
    L = string.L
    T = string.T

    Zs = 1/(w+np.spacing(1))
    Hi = 1/(w+np.spacing(1))

    for mn in range(1, Ns+1):
        wj = mn*np.pi*c/L
        aux = 1j*1/Qt[mn-1]/2
        aux2 = 1/(w-wj*(1+aux)) + 1/(w + wj*(1-aux))
        Zs = Zs + aux2
        Hi = Hi + (-1)**mn*aux2
    Zs = Zs*(-1j*T/L)
    Hi = Hi*(c/L*np.sin(w*xp/c))

    return Zs, Hi
