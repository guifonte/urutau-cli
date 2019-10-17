import collections
import numpy as np


def load_body(phik_file, fmk_file, Nb, xsi_values):

    phikyz = np.load(phik_file)
    fmk = np.load(fmk_file)

    Nb_max = len(phikyz[:, 0])

    if Nb == -1:
        Nb = Nb_max
    elif Nb > Nb_max:
        Nb = Nb_max
        print("ERROR - Choosen Nb is bigger than the quantity of available modes in the phikyz file!")
        print("ERROR - Using Nb = " + str(Nb_max))

    phiky = phikyz[0:Nb, 0]
    phikz = phikyz[0:Nb, 1]

    wk = 2*np.pi*fmk[0:Nb, 0]
    mk = fmk[0:Nb, 1]

    try:
        xsi_all = np.load(xsi_values)
        xsi = xsi_all[0:Nb]
    except:
        xsi = np.ones(Nb) * xsi_values

    Body = collections.namedtuple('Body', ['Nb', 'phiky', 'phikz', 'wk', 'mk', 'xsi'])
    return Body(Nb, phiky, phikz, wk, mk, xsi)


def load_body_2cp(phikb_file, phikp_file, fmk_file, Nb, xsi_values):

    phikyzb = np.load(phikb_file)
    phikyzp = np.load(phikp_file)
    fmk = np.load(fmk_file)

    Nb_max = len(phikyzb[:, 0])

    if Nb == -1:
        Nb = Nb_max
    elif Nb > Nb_max:
        Nb = Nb_max
        print("ERROR - Choosen Nb is bigger than the quantity of available modes in the phikyz file!")
        print("ERROR - Using Nb = " + str(Nb_max))

    phikyb = phikyzb[0:Nb, 0]
    phikzb = phikyzb[0:Nb, 1]
    phikyp = phikyzp[0:Nb, 0]
    phikzp = phikyzp[0:Nb, 1]

    wk = 2*np.pi*fmk[0:Nb, 0]
    mk = fmk[0:Nb, 1]

    if len(xsi_values) == 1:
        xsi = np.ones(Nb) * xsi_values
    else:
        xsi = xsi_values[0:Nb]

    Body = collections.namedtuple('Body', ['phikyb', 'phikzb', 'phikyp', 'phikzp', 'wk', 'mk', 'xsi'])
    return Body(phikyb, phikzb, phikyp, phikzp, wk, mk, xsi)
