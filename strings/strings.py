import json
from collections import namedtuple
import numpy as np


def get_string_params(name, length, fund_freq, stringdb_path, fhmax):
    data = json.load(open(stringdb_path), object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))

    found_string = None
    for sel_string in data:
        if sel_string.name == name:
            found_string = sel_string
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

    Ns = get_string_ns(L, c, T, found_string.B, fhmax)

    SelString = namedtuple('SelString', ['f0', 'L', 'c', 'T', 'mu', 'D', 'B', 'etaf', 'etab', 'etaa', 'Ns'])
    return SelString(f0, L, c, T, mu, found_string.D, found_string.B, found_string.etaf, found_string.etab, found_string.etaa, Ns)


def damping_paiva(sel_string, neta, rho, dVETE, Qd):

    Ns = sel_string.Ns
    T = sel_string.T
    L = sel_string.L
    B = sel_string.B
    c = sel_string.c  # transverse wave velocity
    mu = sel_string.mu
    D = sel_string.D

    Qt = np.ones(Ns)

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


def damping_woodhouse(sel_string):

    Ns = sel_string.Ns
    T = sel_string.T
    L = sel_string.L
    B = sel_string.B
    c = sel_string.c
    etaf = sel_string.etaf
    etab = sel_string.etab
    etaa = sel_string.etaa

    Qt = np.ones(Ns)

    for j in range(1, Ns+1):
        wj = j * np.pi * c / L * np.sqrt(1 + (j ** 2) * ((B * np.pi ** 2) / (T * L ** 2)))
        # facteurs de pertes J.Woodhouse AAA2004, eq(8)
        etaJ = (T * (etaf + etaa/wj) + B * etab * (j * np.pi / L)**2) / (T + B * (j * np.pi / L)**2)

        Qt[j-1] = 1/etaJ

    return Qt


def get_string_ns(L, c, T, B, fhmax):

    j = 1
    fj = 0
    while fhmax > fj:
        wj = j * np.pi * c / L * np.sqrt(1 + (j ** 2) * ((B * np.pi ** 2) / (T * L ** 2)))
        fj = wj/2/np.pi
        j = j + 1

    return j - 1
