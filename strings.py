import json
from collections import namedtuple
import numpy as np


def get_string_params(name, length, fund_freq, stringdb_path):
    data = json.load(open(stringdb_path), object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))

    found_string = None
    for string in data:
        if string.name == name:
            found_string = string
            break

    if found_string is None:
        print("ERROR: String " + name + " was not found!")
        return None

    if fund_freq == -1:
        f0 = found_string.f0_std
    else:
        f0 = fund_freq

    if found_string.L_std != length or found_string.f0_std != f0:
        L = length
        c = L * 2 * f0
        mu = found_string.mu_std*L/found_string.L_std
        T = mu * c ** 2
    else:
        L = found_string.L_std
        T = found_string.T_std
        c = found_string.c_std
        mu = found_string.mu_std

    String = namedtuple('String', ['f0', 'L', 'c', 'T', 'mu', 'D', 'B', 'etaf', 'etab', 'etaa'])
    return String(f0, L, c, T, mu, found_string.D, found_string.B, found_string.etaf, found_string.etab, found_string.etaa)


def damping_paiva(Ns, string, neta, rho, dVETE, Qd):

    Qt = np.ones(Ns)

    T = string.T
    L = string.L
    B = string.B
    c = string.c  # transverse wave velocity
    mu = string.mu
    D = string.D

    # c = np.sqrt(T/mu)

    for j in range(1, Ns+1):

        omegaj = (j * np.pi / L) * c * np.sqrt(1 + (j ** 2) * ((B * np.pi ** 2) / (T * L ** 2)))
        # modal angular frequencies(considering inharmonic effects)
        fj = omegaj / (2 * np.pi)

        # DAMPING PARAMETERS
        R = 2 * np.pi * neta + 2 * np.pi * D * np.sqrt(np.pi * neta * rho * fj)  # air friction coefficient
        Qf = (2 * np.pi * mu * fj) / R  # damping due to the air friction

        Qvt = ((T**2) * c)/((4 * np.pi ** 2) * B * dVETE * fj ** 2)  # damping due to the thermo and visco - elasticity

        Qt[j-1] = (Qf * Qvt * Qd) / (Qvt * Qd + Qf * Qd + Qvt * Qf)

    return Qt


def damping_woodhouse(Ns, string):

    T = string.T
    L = string.L
    B = string.B
    c = string.c
    etaf = string.etaf
    etab = string.etab
    etaa = string.etaa

    Qt = np.ones(Ns)

    for j in range(1, Ns+1):
        wj = j * np.pi * c / L * np.sqrt(1 + (j ** 2) * ((B * np.pi ** 2) / (T * L ** 2)))
        # facteurs de pertes J.Woodhouse AAA2004, eq(8)
        etaJ = (T * (etaf + etaa/wj) + B * etab * (j * np.pi / L)**2) / (T + B * (j * np.pi / L)**2)

        Qt[j-1] = 1/etaJ

    return Qt
