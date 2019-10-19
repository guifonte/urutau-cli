import numpy as np
import collections


# Coupled Forces Computation
def cfc_solve_2p_2cp(body, Qt, string, Fs, pluck, Tf, verbose):
    # Simulation parameters
    dt = 1 / (Fs*5)
    t = np.linspace(0, Tf, np.floor(Tf / dt)) * dt

    # Pluck parameters
    Ti = pluck.Ti
    dp = pluck.dp
    F0 = pluck.F0
    xp = pluck.xp
    gamma = pluck.gamma
    Tr = Ti + dp

    strings = cfc_strings_2cp(Qt, string, xp, dt)
    body_matrix = cfc_body_2p_2cp(body, dt)
    # Input parameters

    # String
    A1 = strings.A1
    A2 = strings.A2
    GSe = strings.GSe
    GSb = strings.GSb
    GSp = strings.GSp
    PhiSe = strings.PhiSe
    PhiSb = strings.PhiSb
    PhiSp = strings.PhiSp
    Ns = string.Ns

    j = np.linspace(1, Ns, Ns)

    # Body

    B1 = body_matrix.B1
    B2 = body_matrix.B2
    GBzb = body_matrix.GBzb
    PhiBzb = body_matrix.PhiBzb
    GByb = body_matrix.GByb
    PhiByb = body_matrix.PhiByb
    GBzp = body_matrix.GBzp
    PhiBzp = body_matrix.PhiBzp
    GByp = body_matrix.GByp
    PhiByp = body_matrix.PhiByp
    Nb = body.Nb

    # Initialisation

    # Modal quantities

    #az = np.ones((Ns + 1, int(len(t) / 5)), dtype=np.float64)
    #ay = np.ones((Ns + 1, int(len(t) / 5)), dtype=np.float64)
    #b = np.ones((Nb, int(len(t) / 5)), dtype=np.float64)

    #az[:, 0] = 0
    #ay[:, 0] = 0
    #b[:, 0] = 0

    z_e = np.zeros(int(len(t) / 5), dtype=np.float64)
    z_b1_b = np.zeros(int(len(t) / 5), dtype=np.float64)
    y_e = np.zeros(int(len(t) / 5), dtype=np.float64)
    y_b1_b = np.zeros(int(len(t) / 5), dtype=np.float64)

    az_1 = np.zeros(Ns + 2, dtype=np.float64)
    az_2 = np.zeros(Ns + 2, dtype=np.float64)
    ay_1 = np.zeros(Ns + 2, dtype=np.float64)
    ay_2 = np.zeros(Ns + 2, dtype=np.float64)
    b_1 = np.zeros(Nb, dtype=np.float64)
    b_2 = np.zeros(Nb, dtype=np.float64)

    # Physical quantity
    #Fcz = np.zeros(int(len(t) / 5))
    #Fcy = np.zeros(int(len(t) / 5))

    # ITERATIVE COMPUTATION OF SOLUTION

    B1_diag = np.diagonal(B1)
    B2_diag = np.diagonal(B2)

    PhiSbGSb = PhiSb @ GSb
    PhiBb = np.vstack((PhiBzb, PhiByb))
    GBb = np.vstack((GBzb, GByb))
    PhiBbGBb = PhiBb @ GBb.T
    PhiBbGBb[0, 0] = PhiBbGBb[0, 0] + PhiSbGSb
    PhiBbGBb[1, 1] = PhiBbGBb[1, 1] + PhiSbGSb
    Fb1 = np.linalg.inv(PhiBbGBb)

    PhiSpGSp = PhiSp @ GSp
    PhiBp = np.vstack((PhiBzp, PhiByp))
    GBp = np.vstack((GBzp, GByp))
    PhiBpGBp = PhiBp @ GBp.T
    PhiBpGBp[0, 0] = PhiBpGBp[0, 0] + PhiSpGSp
    PhiBpGBp[1, 1] = PhiBpGBp[1, 1] + PhiSpGSp
    Fp1 = np.linalg.inv(PhiBpGBp)

    F0_dp = F0 / dp

    percentage = 0
    total_len = len(t)
    for i in range(1, total_len):

        Fe = F0_dp * (i * dt - Ti) * (np.heaviside((i * dt - Ti), 0.5) - np.heaviside((i * dt - Tr), 0.5))
        Fez = Fe * np.sin(gamma)
        Fey = Fe * np.cos(gamma)

        if i % 2 == 0:
            Aalphaz = A1 @ az_1 + A2 @ az_2 + GSe * Fez
            Aalphay = A1 @ ay_1 + A2 @ ay_2 + GSe * Fey
            Bbeta = B1_diag * b_1 + B2_diag * b_2

            Fb2z = PhiSb @ Aalphaz - PhiBzb @ Bbeta
            Fb2y = PhiSb @ Aalphay - PhiByb @ Bbeta
            Fbz_temp = Fb1[0, 0] * Fb2z + Fb1[0, 1] * Fb2y
            Fby_temp = Fb1[1, 0] * Fb2z + Fb1[1, 1] * Fb2y

            Fp2z = PhiSp @ Aalphaz - PhiBzp @ Bbeta
            Fp2y = PhiSp @ Aalphay - PhiByp @ Bbeta
            Fpz_temp = Fp1[0, 0] * Fp2z + Fp1[0, 1] * Fp2y
            Fpy_temp = Fp1[1, 0] * Fp2z + Fp1[1, 1] * Fp2y

            az_2 = Aalphaz - GSb * Fbz_temp - GSp * Fpz_temp
            ay_2 = Aalphay - GSb * Fby_temp - GSp * Fpy_temp
            b_2 = Bbeta + GBzb * Fbz_temp + GByb * Fby_temp + GBzp * Fpz_temp + GByp * Fpy_temp

        else:
            Aalphaz = A1 @ az_2 + A2 @ az_1 + GSe * Fez
            Aalphay = A1 @ ay_2 + A2 @ ay_1 + GSe * Fey
            Bbeta = B1_diag * b_2 + B2_diag * b_1

            Fb2z = PhiSb @ Aalphaz - PhiBzb @ Bbeta
            Fb2y = PhiSb @ Aalphay - PhiByb @ Bbeta
            Fbz_temp = Fb1[0, 0] * Fb2z + Fb1[0, 1] * Fb2y
            Fby_temp = Fb1[1, 0] * Fb2z + Fb1[1, 1] * Fb2y

            Fp2z = PhiSp @ Aalphaz - PhiBzp @ Bbeta
            Fp2y = PhiSp @ Aalphay - PhiByp @ Bbeta
            Fpz_temp = Fp1[0, 0] * Fp2z + Fp1[0, 1] * Fp2y
            Fpy_temp = Fp1[1, 0] * Fp2z + Fp1[1, 1] * Fp2y

            az_1 = Aalphaz - GSb * Fbz_temp - GSp * Fpz_temp
            ay_1 = Aalphay - GSb * Fby_temp - GSp * Fpy_temp
            b_1 = Bbeta + GBzb * Fbz_temp + GByb * Fby_temp + GBzp * Fpz_temp + GByp * Fpy_temp

        if i % 5 == 0:
            temp_i = [int(i / 5)]
            if i % 2 == 0:
                #az[:, int(i / 5)] = az_2
                #ay[:, int(i / 5)] = ay_2
                #b[:, int(i / 5)] = b_2
                z_e[temp_i] = np.real(PhiSe @ az_2)
                y_e[temp_i] = np.real(PhiSe @ ay_2)
                z_b1_b[temp_i] = np.real(PhiSb @ az_2)
                y_b1_b[temp_i] = np.real(PhiSb @ ay_2)
            else:
                #az[:, int(i / 5)] = az_1
                #ay[:, int(i / 5)] = ay_1
                #b[:, int(i / 5)] = b_1
                z_e[temp_i] = np.real(PhiSe @ az_1)
                y_e[temp_i] = np.real(PhiSe @ ay_1)
                z_b1_b[temp_i] = np.real(PhiSb @ az_1)
                y_b1_b[temp_i] = np.real(PhiSb @ ay_1)
                if verbose:
                    temp_percentage = int(i/total_len*100)
                    if temp_percentage > percentage:
                        print("Progress: " + str(temp_percentage) + "%")
                        percentage = temp_percentage

            #Fcz[int(i / 5)] = Fcz_temp
            #Fcy[int(i / 5)] = Fcy_temp

    # String displacement

    # String displacement at the plucking point
    #z_p = np.real(PhiSe @ az) * 1e3  # in mm
    #y_p = np.real(PhiSe @ ay) * 1e3  # in mm

    # String displacement at the coupling point
    #z_b1 = np.real(PhiSc @ az) * 1e3  # in mm
    #y_b1 = np.real(PhiSc @ ay) * 1e3  # in mm

    # Body displacement at the coupling point
    #z_b2 = np.real(PhiBz @ b) * 1e3  # in mm
    #y_b2 = np.real(PhiBy @ b) * 1e3  # in mm

    #Displacements = collections.namedtuple('Displacements', ['z_p', 'y_p', 'z_b1', 'y_b1', 'z_b2', 'y_b2'])
    #return Displacements(z_p, y_p, z_b1, y_b1, z_b2, y_b2)

    Displacements = collections.namedtuple('Displacements', ['z_p', 'y_p', 'z_b1', 'y_b1'])
    return Displacements(z_e, y_e, z_b1_b, y_b1_b)


def cfc_solve_1p_2cp(body, Qt, sel_string, Fs, pluck, Tf, verbose):
    # Simulation parameters
    dt = 1 / (Fs*5)
    t = np.linspace(0, Tf, np.floor(Tf / dt)) * dt

    # Pluck parameters

    Ti = pluck.Ti
    dp = pluck.dp
    F0 = pluck.F0
    xp = pluck.xp
    gamma = pluck.gamma
    Tr = Ti + dp

    strings = cfc_strings_2cp(Qt, sel_string, xp, dt)
    body_matrix = cfc_body_1p_2cp(body, dt)

    # String
    A1 = strings.A1
    A2 = strings.A2
    GSe = strings.GSe
    GSb = strings.GSb
    GSp = strings.GSp
    PhiSe = strings.PhiSe
    PhiSb = strings.PhiSb
    PhiSp = strings.PhiSp
    Ns = strings.Ns

    j = np.linspace(1, Ns, Ns)

    # Body

    B1 = body_matrix.B1
    B2 = body_matrix.B2
    GBzb = body_matrix.GBzb
    GBzp = body_matrix.GBzp
    PhiBzb = body_matrix.PhiBzb
    PhiBzp = body_matrix.PhiBzp
    Nb = body.Nb

    # Initialisation

    # Modal quantities

    z_e = np.zeros(int(len(t) / 5), dtype=np.float64)
    z_b1_b = np.zeros(int(len(t) / 5), dtype=np.float64)

    az_1 = np.zeros(Ns + 2, dtype=np.float64)
    az_2 = np.zeros(Ns + 2, dtype=np.float64)
    b_1 = np.zeros(Nb, dtype=np.float64)
    b_2 = np.zeros(Nb, dtype=np.float64)

    # Physical quantity
    # Fcz = np.zeros(int(len(t) / 5))

    # ITERATIVE COMPUTATION OF SOLUTION

    B1_diag = np.diagonal(B1)
    B2_diag = np.diagonal(B2)
    PhiSbGSb = PhiSb @ GSb
    PhiSpGSp = PhiSp @ GSp
    PhiBbGBb = PhiBzb @ GBzb.T + PhiSbGSb
    PhiBpGBp = PhiBzp @ GBzp.T + PhiSpGSp
    Fb1 = 1 / PhiBbGBb
    Fp1 = 1 / PhiBpGBp

    F0_dp = F0 / dp

    percentage = 0
    total_len = len(t)
    for i in range(1, total_len):

        Fe = F0_dp * (i * dt - Ti) * (np.heaviside((i * dt - Ti), 0.5) - np.heaviside((i * dt - Tr), 0.5))
        Fez = Fe * np.sin(gamma)

        if i % 2 == 0:
            Aalphaz = A1 @ az_1 + A2 @ az_2 + GSe * Fez
            Bbeta = B1_diag * b_1 + B2_diag * b_2
            Fb2z = PhiSb @ Aalphaz - PhiBzb @ Bbeta
            Fp2z = PhiSp @ Aalphaz - PhiBzp @ Bbeta

            Fbz_temp = Fb1 * Fb2z
            Fpz_temp = Fp1 * Fp2z

            az_2 = Aalphaz - GSb * Fbz_temp - GSp * Fpz_temp
            b_2 = Bbeta + GBzb * Fbz_temp + GBzp * Fpz_temp

        else:
            Aalphaz = A1 @ az_2 + A2 @ az_1 + GSe * Fez
            Bbeta = B1_diag * b_2 + B2_diag * b_1
            Fb2z = PhiSb @ Aalphaz - PhiBzb @ Bbeta
            Fp2z = PhiSp @ Aalphaz - PhiBzp @ Bbeta

            Fbz_temp = Fb1 * Fb2z
            Fpz_temp = Fp1 * Fp2z

            az_1 = Aalphaz - GSb * Fbz_temp - GSp * Fpz_temp
            b_1 = Bbeta + GBzb * Fbz_temp + GBzp * Fpz_temp

        if i % 5 == 0:
            temp_i = [int(i / 5)]
            if i % 2 == 0:
                # az[:, int(i / 5)] = az_2
                # b[:, int(i / 5)] = b_2
                z_e[temp_i] = np.real(PhiSe @ az_2)
                z_b1_b[temp_i] = np.real(PhiSb @ az_2)
                # z_b2[temp_i] = np.real(PhiBz @ b_2)
            else:
                # az[:, int(i / 5)] = az_1
                # b[:, int(i / 5)] = b_1
                z_e[temp_i] = np.real(PhiSe @ az_1)
                z_b1_b[temp_i] = np.real(PhiSb @ az_1)
                # z_b2[int(i / 5)] = np.real(PhiBz @ b_1)
                if verbose:
                    temp_percentage = int(i/total_len*100)
                    if temp_percentage > percentage:
                        print("Progress: " + str(temp_percentage) + "%")
                        percentage = temp_percentage

            # Fcz[int(i / 5)] = Fcz_temp

    # String displacement at the plucking point
    # z_p = np.real(PhiSe @ az) * 1e3  # in mm

    # String displacement at the coupling point
    # z_b1 = np.real(PhiSc @ az) * 1e3  # in mm

    # Body displacement at the coupling point
    # z_b2 = np.real(PhiBz @ b) * 1e3  # in mm

    # Displacements = collections.namedtuple('Displacements', ['z_p', 'z_b1', 'z_b2'])
    # return Displacements(z_p, z_b1, z_b2)
    Displacements = collections.namedtuple('Displacements', ['z_p', 'z_b1'])
    return Displacements(z_e, z_b1_b)


# Coupled Forces Computation
def cfc_solve_2p(body, Qt, sel_string, Fs, pluck, Tf, verbose):
    # Simulation parameters
    dt = 1 / (Fs*5)
    t = np.linspace(0, Tf, np.floor(Tf / dt)) * dt

    # Pluck parameters
    Ti = pluck.Ti
    dp = pluck.dp
    F0 = pluck.F0
    xp = pluck.xp
    gamma = pluck.gamma
    Tr = Ti + dp

    strings = cfc_strings(Qt, sel_string, xp, dt)
    body_matrix = cfc_body(body, dt)
    # Input parameters

    # String
    A1 = strings.A1
    A2 = strings.A2
    GSe = strings.GSe
    GSc = strings.GSc
    PhiSe = strings.PhiSe
    PhiSc = strings.PhiSc
    Ns = strings.Ns

    j = np.linspace(1, Ns, Ns)

    # Body

    B1 = body_matrix.B1
    B2 = body_matrix.B2
    GBz = body_matrix.GBz
    PhiBz = body_matrix.PhiBz
    GBy = body_matrix.GBy
    PhiBy = body_matrix.PhiBy
    Nb = body.Nb

    # Initialisation

    # Modal quantities

    #az = np.ones((Ns + 1, int(len(t) / 5)), dtype=np.float64)
    #ay = np.ones((Ns + 1, int(len(t) / 5)), dtype=np.float64)
    #b = np.ones((Nb, int(len(t) / 5)), dtype=np.float64)

    #az[:, 0] = 0
    #ay[:, 0] = 0
    #b[:, 0] = 0

    z_p = np.zeros(int(len(t) / 5), dtype=np.float64)
    z_b1 = np.zeros(int(len(t) / 5), dtype=np.float64)
    y_p = np.zeros(int(len(t) / 5), dtype=np.float64)
    y_b1 = np.zeros(int(len(t) / 5), dtype=np.float64)

    az_1 = np.zeros(Ns + 1, dtype=np.float64)
    az_2 = np.zeros(Ns + 1, dtype=np.float64)
    ay_1 = np.zeros(Ns + 1, dtype=np.float64)
    ay_2 = np.zeros(Ns + 1, dtype=np.float64)
    b_1 = np.zeros(Nb, dtype=np.float64)
    b_2 = np.zeros(Nb, dtype=np.float64)

    # Physical quantity
    #Fcz = np.zeros(int(len(t) / 5))
    #Fcy = np.zeros(int(len(t) / 5))

    # ITERATIVE COMPUTATION OF SOLUTION

    B1_diag = np.diagonal(B1)
    B2_diag = np.diagonal(B2)
    PhiScGSc = PhiSc @ GSc
    PhiB = np.vstack((PhiBz, PhiBy))
    GB = np.vstack((GBz, GBy))
    PhiBcGBc = PhiB @ GB.T
    PhiBcGBc[0, 0] = PhiBcGBc[0, 0] + PhiScGSc
    PhiBcGBc[1, 1] = PhiBcGBc[1, 1] + PhiScGSc
    Fc1 = np.linalg.inv(PhiBcGBc)

    F0_dp = F0 / dp

    percentage = 0
    total_len = len(t)
    for i in range(1, total_len):

        Fe = F0_dp * (i * dt - Ti) * (np.heaviside((i * dt - Ti), 0.5) - np.heaviside((i * dt - Tr), 0.5))
        Fez = Fe * np.sin(gamma)
        Fey = Fe * np.cos(gamma)

        if i % 2 == 0:
            Aalphaz = A1 @ az_1 + A2 @ az_2 + GSe * Fez
            Aalphay = A1 @ ay_1 + A2 @ ay_2 + GSe * Fey
            Bbeta = B1_diag * b_1 + B2_diag * b_2
            Fc2z = PhiSc @ Aalphaz - PhiBz @ Bbeta
            Fc2y = PhiSc @ Aalphay - PhiBy @ Bbeta
            Fcz_temp = Fc1[0, 0] * Fc2z + Fc1[0, 1] * Fc2y
            Fcy_temp = Fc1[1, 0] * Fc2z + Fc1[1, 1] * Fc2y

            az_2 = Aalphaz - GSc * Fcz_temp
            ay_2 = Aalphay - GSc * Fcy_temp
            b_2 = Bbeta + GBz * Fcz_temp + GBy * Fcy_temp

        else:
            Aalphaz = A1 @ az_2 + A2 @ az_1 + GSe * Fez
            Aalphay = A1 @ ay_2 + A2 @ ay_1 + GSe * Fey
            Bbeta = B1_diag * b_2 + B2_diag * b_1
            Fc2z = PhiSc @ Aalphaz - PhiBz @ Bbeta
            Fc2y = PhiSc @ Aalphay - PhiBy @ Bbeta
            Fcz_temp = Fc1[0, 0] * Fc2z + Fc1[0, 1] * Fc2y
            Fcy_temp = Fc1[1, 0] * Fc2z + Fc1[1, 1] * Fc2y

            az_1 = Aalphaz - GSc * Fcz_temp
            ay_1 = Aalphay - GSc * Fcy_temp
            b_1 = Bbeta + GBz * Fcz_temp + GBy * Fcy_temp

        if i % 5 == 0:
            temp_i = [int(i / 5)]
            if i % 2 == 0:
                #az[:, int(i / 5)] = az_2
                #ay[:, int(i / 5)] = ay_2
                #b[:, int(i / 5)] = b_2
                z_p[temp_i] = np.real(PhiSe @ az_2)
                y_p[temp_i] = np.real(PhiSe @ ay_2)
                z_b1[temp_i] = np.real(PhiSc @ az_2)
                y_b1[temp_i] = np.real(PhiSc @ ay_2)
            else:
                #az[:, int(i / 5)] = az_1
                #ay[:, int(i / 5)] = ay_1
                #b[:, int(i / 5)] = b_1
                z_p[temp_i] = np.real(PhiSe @ az_1)
                y_p[temp_i] = np.real(PhiSe @ ay_1)
                z_b1[temp_i] = np.real(PhiSc @ az_1)
                y_b1[temp_i] = np.real(PhiSc @ ay_1)
                if verbose:
                    temp_percentage = int(i/total_len*100)
                    if temp_percentage > percentage:
                        print("Progress: " + str(temp_percentage) + "%")
                        percentage = temp_percentage

            #Fcz[int(i / 5)] = Fcz_temp
            #Fcy[int(i / 5)] = Fcy_temp

    # String displacement

    # String displacement at the plucking point
    #z_p = np.real(PhiSe @ az) * 1e3  # in mm
    #y_p = np.real(PhiSe @ ay) * 1e3  # in mm

    # String displacement at the coupling point
    #z_b1 = np.real(PhiSc @ az) * 1e3  # in mm
    #y_b1 = np.real(PhiSc @ ay) * 1e3  # in mm

    # Body displacement at the coupling point
    #z_b2 = np.real(PhiBz @ b) * 1e3  # in mm
    #y_b2 = np.real(PhiBy @ b) * 1e3  # in mm

    #Displacements = collections.namedtuple('Displacements', ['z_p', 'y_p', 'z_b1', 'y_b1', 'z_b2', 'y_b2'])
    #return Displacements(z_p, y_p, z_b1, y_b1, z_b2, y_b2)

    Displacements = collections.namedtuple('Displacements', ['z_p', 'y_p', 'z_b1', 'y_b1'])
    return Displacements(z_p, y_p, z_b1, y_b1)


def cfc_solve_1p(body, Qt, sel_string, Fs, pluck, Tf, verbose):
    # Simulation parameters
    dt = 1 / (Fs*5)
    t = np.linspace(0, Tf, np.floor(Tf / dt)) * dt

    # Pluck parameters
    Ti = pluck.Ti
    dp = pluck.dp
    F0 = pluck.F0
    xp = pluck.xp
    gamma = pluck.gamma
    Tr = Ti + dp

    strings = cfc_strings(Qt, sel_string, xp, dt)
    body_matrix = cfc_body(body, dt)
    # Input parameters

    # String
    A1 = strings.A1
    A2 = strings.A2
    GSe = strings.GSe
    GSc = strings.GSc
    PhiSe = strings.PhiSe
    PhiSc = strings.PhiSc
    Ns = strings.Ns

    j = np.linspace(1, Ns, Ns)

    # Body

    B1 = body_matrix.B1
    B2 = body_matrix.B2
    GBz = body_matrix.GBz
    PhiBz = body_matrix.PhiBz
    Nb = body.Nb

    # Initialisation

    # Modal quantities

    #az = np.ones((Ns + 1, int(len(t) / 5)), dtype=np.float64)
    #b = np.ones((Nb, int(len(t) / 5)), dtype=np.float64)
    z_p = np.zeros(int(len(t) / 5), dtype=np.float64)
    z_b1 = np.zeros(int(len(t) / 5), dtype=np.float64)
    #z_b2 = np.zeros(int(len(t) / 5), dtype=np.float64)

    #az[:, 0] = 0
    #b[:, 0] = 0

    az_1 = np.zeros(Ns + 1, dtype=np.float64)
    az_2 = np.zeros(Ns + 1, dtype=np.float64)
    b_1 = np.zeros(Nb, dtype=np.float64)
    b_2 = np.zeros(Nb, dtype=np.float64)

    # Physical quantity
    #Fcz = np.zeros(int(len(t) / 5))

    # ITERATIVE COMPUTATION OF SOLUTION

    B1_diag = np.diagonal(B1)
    B2_diag = np.diagonal(B2)
    PhiScGSc = PhiSc @ GSc
    PhiBcGBc = PhiBz @ GBz.T + PhiScGSc
    Fc1 = 1/PhiBcGBc

    F0_dp = F0 / dp

    percentage = 0
    total_len = len(t)
    for i in range(1, total_len):

        Fe = F0_dp * (i * dt - Ti) * (np.heaviside((i * dt - Ti), 0.5) - np.heaviside((i * dt - Tr), 0.5))
        Fez = Fe * np.sin(gamma)

        if i % 2 == 0:
            Aalphaz = A1 @ az_1 + A2 @ az_2 + GSe * Fez
            Bbeta = B1_diag * b_1 + B2_diag * b_2
            Fc2z = PhiSc @ Aalphaz - PhiBz @ Bbeta
            Fcz_temp = Fc1 * Fc2z

            az_2 = Aalphaz - GSc * Fcz_temp
            b_2 = Bbeta + GBz * Fcz_temp

        else:
            Aalphaz = A1 @ az_2 + A2 @ az_1 + GSe * Fez
            Bbeta = B1_diag * b_2 + B2_diag * b_1
            Fc2z = PhiSc @ Aalphaz - PhiBz @ Bbeta
            Fcz_temp = Fc1 * Fc2z

            az_1 = Aalphaz - GSc * Fcz_temp
            b_1 = Bbeta + GBz * Fcz_temp

        if i % 5 == 0:
            temp_i = [int(i / 5)]
            if i % 2 == 0:
                #az[:, int(i / 5)] = az_2
                #b[:, int(i / 5)] = b_2
                z_p[temp_i] = np.real(PhiSe @ az_2)
                z_b1[temp_i] = np.real(PhiSc @ az_2)
                #z_b2[temp_i] = np.real(PhiBz @ b_2)
            else:
                #az[:, int(i / 5)] = az_1
                #b[:, int(i / 5)] = b_1
                z_p[temp_i] = np.real(PhiSe @ az_1)
                z_b1[temp_i] = np.real(PhiSc @ az_1)
                #z_b2[int(i / 5)] = np.real(PhiBz @ b_1)
                if verbose:
                    temp_percentage = int(i/total_len*100)
                    if temp_percentage > percentage:
                        print("Progress: " + str(temp_percentage) + "%")
                        percentage = temp_percentage

            #Fcz[int(i / 5)] = Fcz_temp

    # String displacement at the plucking point
    #z_p = np.real(PhiSe @ az) * 1e3  # in mm

    # String displacement at the coupling point
    #z_b1 = np.real(PhiSc @ az) * 1e3  # in mm

    # Body displacement at the coupling point
    #z_b2 = np.real(PhiBz @ b) * 1e3  # in mm

    #Displacements = collections.namedtuple('Displacements', ['z_p', 'z_b1', 'z_b2'])
    #return Displacements(z_p, z_b1, z_b2)
    Displacements = collections.namedtuple('Displacements', ['z_p', 'z_b1'])
    return Displacements(z_p, z_b1)


def cfc_body(body, dt):
    mk = body.mk
    wdk = body.wk
    xsi = body.xsi
    wnk = wdk / np.sqrt(1 - xsi ** 2)
    ck = 2 * mk * xsi * wnk
    kk = mk * wnk ** 2

    MK = np.diagflat(mk)
    KK = np.diagflat(kk)
    CK = np.diagflat(ck)
    PhiBz = body.phikz
    PhiBy = body.phiky

    BI = np.linalg.inv(MK / dt ** 2 + CK / (2 * dt))
    B1 = BI @ (2 * MK / dt ** 2 - KK)
    B2 = BI @ (CK / (2 * dt) - MK / dt ** 2)
    B3 = BI

    GBz = B3 @ PhiBz
    GBy = B3 @ PhiBy

    #B1 = np.ones(B1.shape)
    #B2 = np.ones(B2.shape)
    #GBz = np.ones(GBz.shape)
    #GBy = np.ones(GBy.shape)
    #PhiBy = np.ones(PhiBy.shape)
    #PhiBz = np.ones(PhiBz.shape)

    #B1 = B1*0
    #B2 = B2*0
    #GBz = GBz*0
    #GBy = GBy*0
    #PhiBy = PhiBy*0
    #PhiBz = PhiBz*0

    Body_matrix = collections.namedtuple('Body_matrix', ['B1', 'B2', 'PhiBz', 'PhiBy', 'GBz', 'GBy'])
    return Body_matrix(B1, B2, PhiBz, PhiBy, GBz, GBy)


def cfc_body_2p_2cp(body, dt):
    mk = body.mk
    wdk = body.wk
    xsi = body.xsi
    wnk = wdk / np.sqrt(1 - xsi ** 2)
    ck = 2 * mk * xsi * wnk
    kk = mk * wnk ** 2

    MK = np.diagflat(mk)
    KK = np.diagflat(kk)
    CK = np.diagflat(ck)
    PhiBzb = body.phikzb
    PhiByb = body.phikyb
    PhiBzp = body.phikzp
    PhiByp = body.phikyp

    BI = np.linalg.inv(MK / dt ** 2 + CK / (2 * dt))
    B1 = BI @ (2 * MK / dt ** 2 - KK)
    B2 = BI @ (CK / (2 * dt) - MK / dt ** 2)
    B3 = BI

    GBzb = B3 @ PhiBzb
    GByb = B3 @ PhiByb
    GBzp = B3 @ PhiBzp
    GByp = B3 @ PhiByp

    Body_matrix = collections.namedtuple('Body_matrix', ['B1', 'B2', 'PhiBzb', 'PhiByb', 'GBzb', 'GByb', 'PhiBzp', 'PhiByp', 'GBzp', 'GByp'])
    return Body_matrix(B1, B2, PhiBzb, PhiByb, GBzb, GByb, PhiBzp, PhiByp, GBzp, GByp)


def cfc_body_1p_2cp(body, dt):
    mk = body.mk
    wdk = body.wk
    xsi = body.xsi
    wnk = wdk / np.sqrt(1 - xsi ** 2)
    ck = 2 * mk * xsi * wnk
    kk = mk * wnk ** 2

    MK = np.diagflat(mk)
    KK = np.diagflat(kk)
    CK = np.diagflat(ck)
    PhiBzb = body.phikzb
    PhiBzp = body.phikzp

    BI = np.linalg.inv(MK / dt ** 2 + CK / (2 * dt))
    B1 = BI @ (2 * MK / dt ** 2 - KK)
    B2 = BI @ (CK / (2 * dt) - MK / dt ** 2)
    B3 = BI

    GBzb = B3 @ PhiBzb
    GBzp = B3 @ PhiBzp

    Body_matrix = collections.namedtuple('Body_matrix', ['B1', 'B2', 'PhiBzb', 'GBzb', 'PhiBzp', 'GBzp'])
    return Body_matrix(B1, B2, PhiBzb, GBzb, PhiBzp, GBzp)


def cfc_strings_2cp(Qt, sel_string, xe, dt):
    L = sel_string.L
    mu = sel_string.mu
    B = sel_string.B
    T = sel_string.T
    Ns = sel_string.Ns

    xb = L
    xp = 0

    mj = np.ones(Ns) * L * mu / 2  # modal masses
    kj = np.ones(Ns)  # modal stiffness
    cj = np.ones(Ns)  # modal damping

    for j in range(1, Ns + 1):
        kj[j - 1] = (j ** 2) * np.pi ** 2 * T / (2 * L) + (j ** 4) * B * np.pi ** 4 / (2 * L ** 3)
        cj[j - 1] = (j * np.pi * np.sqrt(T * mu)) / (2 * Qt[j-1])

    kj = np.insert(kj, 0, T / L)
    kj = np.insert(kj, 1, T / L)
    KJ = np.diagflat(kj)
    KJ[0, 1] = -T / L
    KJ[1, 0] = -T / L

    cj = np.insert(cj, 0, 0)
    cj = np.insert(cj, 1, 0)
    CJ = np.diagflat(cj)

    MJ = np.ones((Ns+2, Ns+2))
    MJ[0, 0] = L * mu / 3
    MJ[1, 1] = MJ[0, 0]
    MJ[0, 1] = L * mu / 6
    MJ[1, 0] = MJ[0, 1]

    for j in range(2, Ns+2):
        MJ[0, j] = L * mu / (np.pi * (j-1))
        MJ[j, 0] = MJ[0, j]
        MJ[1, j] = MJ[0, j]
        MJ[j, 1] = MJ[0, j]

    mjdiag = np.diagflat(mj)

    MJ[2:, 2:] = mjdiag

    # Mode Shapes
    PhiSb = np.ones(Ns + 2)
    PhiSe = np.ones(Ns + 2)
    PhiSp = np.ones(Ns + 2)

    PhiSb[0] = xb/L
    PhiSe[0] = xe/L
    PhiSp[0] = xp/L
    PhiSb[1] = 1-xb/L
    PhiSe[1] = 1-xe/L
    PhiSp[1] = 1-xp/L

    for j in range(2, Ns + 2):
        PhiSb[j] = 0 #np.sin((j-1) * np.pi * xb / L)
        PhiSe[j] = np.sin((j-1) * np.pi * xe / L)
        PhiSp[j] = 0 #np.sin((j-1) * np.pi * xp / L)

    # Resolution Matrices
    AI = np.linalg.inv(MJ/dt**2 + CJ/(2*dt))
    A1 = AI @ (2*MJ/dt**2 - KJ)
    A2 = AI @ (CJ/(2*dt) - MJ/dt**2)
    A3 = AI

    GSe = (A3 @ PhiSe).T
    GSb = (A3 @ PhiSb).T
    GSp = (A3 @ PhiSp).T

    String = collections.namedtuple('String', ['A1', 'A2', 'GSe', 'GSb', 'GSp', 'PhiSe', 'PhiSb', 'PhiSp', 'Ns'])
    return String(A1, A2, GSe, GSb, GSp, PhiSe, PhiSb, PhiSp, Ns)


def cfc_strings(Qt, sel_string, xp, dt):
    L = sel_string.L
    mu = sel_string.mu
    B = sel_string.B
    T = sel_string.T
    Ns = sel_string.Ns

    xb = L

    mj = np.ones(Ns) * L * mu / 2  # modal masses
    kj = np.ones(Ns)  # modal stiffness
    cj = np.ones(Ns)  # modal damping

    for j in range(1, Ns + 1):
        kj[j - 1] = (j ** 2) * np.pi ** 2 * T / (2 * L) + (j ** 4) * B * np.pi ** 4 / (2 * L ** 3)
        cj[j - 1] = (j * np.pi * np.sqrt(T * mu)) / (2 * Qt[j-1])

    kj = np.insert(kj, 0, T / L)
    KJ = np.diagflat(kj)

    cj = np.insert(cj, 0, 0)
    CJ = np.diagflat(cj)

    MJ = np.ones((Ns+1, Ns+1))
    MJ[0, 0] = L * mu / 3

    for j in range(1, Ns+1):
        MJ[0, j] = L * mu / (np.pi * j)
        MJ[j, 0] = MJ[0, j]

    mjdiag = np.diagflat(mj)
    MJ[1:, 1:] = mjdiag

    # Mode Shapes
    PhiSc = np.ones(Ns + 1)
    PhiSe = np.ones(Ns + 1)
    PhiSc[0] = xb/L
    PhiSe[0] = xp/L
    for j in range(1, Ns + 1):
        PhiSc[j] = np.sin(j * np.pi * xb / L)
        PhiSe[j] = np.sin(j * np.pi * xp / L)

    # Resolution Matrices
    AI = np.linalg.inv(MJ/dt**2 + CJ/(2*dt))
    A1 = AI @ (2*MJ/dt**2 - KJ)
    A2 = AI @ (CJ/(2*dt) - MJ/dt**2)
    A3 = AI

    GSe = (A3 @ PhiSe).T
    GSc = (A3 @ PhiSc).T

    String = collections.namedtuple('String', ['A1', 'A2', 'GSe', 'GSc', 'PhiSe', 'PhiSc', 'Ns'])
    return String(A1, A2, GSe, GSc, PhiSe, PhiSc, Ns)

