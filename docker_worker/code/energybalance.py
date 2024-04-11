import numpy as np
from scipy import integrate

from base import DotDict

from functools import lru_cache, partial
from dataclasses import dataclass
import dataclasses as dc
import math

np.set_printoptions(linewidth=300)



def photosynthesis(
    Iphoton,
    cca,
    tlk,
    ra,
    rb_co2,
    P_kPa,
    eair_Pa,
    vcopt,
    kball,
    bprime,
    SM,
    rho_mol,
    jmax_coeff,
    #switch_sm,
    ID_TOY,
):
    #  Baldocchi, D.D. 1994. An analytical solution for coupled leaf photosynthesis
    #  and stomatal conductance models. Tree Physiology 14: 1069-1079.
    #  After Farquhar, von Caemmerer and Berry (1980) Planta.
    #  149: 78-90.
    #
    #  rd25 - Dark respiration at 25 degrees C (umol m-2 s-1)
    #  tlk - leaf temperature, Kelvin
    #  jmax - optimal rate of electron transport
    #  vcopt - maximum rate of RuBP Carboxylase/oxygenase
    #  iphoton - incident photosynthetically active photon flux (mmols m-2 s-1)
    #  gs - stomatal conductance (mol m-2 s-1), typically 0.01-0.20
    #  pstat-station pressure, bars
    #  aphoto - net photosynthesis  (umol m-2 s-1)
    #  ps - gross photosynthesis (umol m-2 s-1)
    #  aps - net photosynthesis (mg m-2 s-1)

    np.seterr(invalid="ignore") # results outside the domain are set to nan

    Iphoton = Iphoton * 4.17  # convert to umol m-2 s-1

    vcopt = vcopt * SM

    jmax_coeff = np.array([jmax_coeff])
    app_jmax_coeff = np.repeat(jmax_coeff, vcopt.shape[1])

    jmopt = np.multiply(app_jmax_coeff, vcopt)  # % jmax_coeff from param inp

    # Universal gas constant
    # d = {"rugc": [], "hkin": [], "skin": []}

    rugc = 8.314  # % J mole-1 K-1
    rgc1000 = 8314  # % gas constant times 1000.

    # %          Consts for Photosynthesis model and kinetic equations.
    # %          for Vcmax and Jmax.  Taken from Harley and Baldocchi (1995, PCE)
    hkin = 200000.0  # % enthalpy term, J mol-1
    skin = 710.0  # % entropy term, J K-1 mol-1
    ejm = 45000.0  # % activation energy for electron transport, J mol-1
    evc = 58520.0  # % activation energy for carboxylation, J mol-1

    # %         Enzyme constants & partial pressure of O2 and CO2
    # %         Michaelis-Menten K values. From survey of literature.
    kc25 = 460  # % kinetic coef for CO2 at 25 C, microbars
    ko25 = 330  # % kinetic coef for O2 at 25C,  millibars
    o2 = 210.0  # % oxygen concentration  mmol mol-1

    # %         tau is computed on the basis of the Specificity factor (102.33)
    # %         times Kco2/Kh2o (28.38) to convert for value of C in solution
    # %         to that based in air/
    # %
    # %         New value for Quercus robor from Balaguer et al. 1996
    # %         Similar number from Dreyer et al. 2001, Tree Physiol, tau= 2710
    tau25 = 2904.12  # %  tau coefficient

    # %          Arrhenius constants
    # %          Eact for Michaelis-Menten const. for KC, KO and dark respiration
    # %          These values are from Harley
    ekc = 59356.0  # % Activation energy for K of CO2; J mol-1
    erd = 38000.0  # % activation energy for dark respiration, eg Q10  = 2
    ektau = -29000.0  # % J mol-1 (Jordan and Ogren, 1984)
    tk_25 = 298.16  # % absolute temperature at 25 C
    toptvc = 311.0  # % optimum temperature for maximum carboxylation
    toptjm = 311.0  # % optimum temperature for maximum electron transport
    rsm = 145.0  # % Minimum stomatal resistance, s m-1.
    brs = 60.0  # % curvature coeffient for light response
    qalpha = 0.22  # % leaf quantum yield, electrons
    qalpha2 = 0.0484  # % qalpha squared, qalpha2                            = pow(qalpha, 2.0);

    rt = rugc * tlk
    tprime25 = tlk - tk_25  # temperature difference

    ttemp = np.exp(((tlk * skin) - hkin) / rt) + 1

    kct = get_arhennius_temperature(kc25, ekc, tprime25, tk_25, tlk)
    tau = get_arhennius_temperature(tau25, ektau, tprime25, tk_25, tlk)

    # fix the Ko and O2 values with same units
    ko25_Pa = ko25 * 100  # % Pa
    o2_Pa = o2 * 101.3  # % Pa
    k25 = 1.0 + o2_Pa / ko25_Pa
    bc = kct * k25

    # gammac is the CO2 compensation point due to photorespiration, umol mol-1
    # Recalculate gammac with the new temperature dependent KO and KC
    # coefficients..C at Vc = 0.5 Vo
    # gammac = O2/(2 tau)
    # O2 has units of kPa so multiplying 0.5 by 1000 gives units of Pa

    gammac = (500.0 * o2) / tau

    index_rdz = Iphoton > 10
    rdz = vcopt * 0.004657
    rdz = np.ones(index_rdz.shape) * rdz

    rdz[index_rdz] = rdz[index_rdz] * 0.4

    rd = get_arhennius_temperature(rdz, erd, tprime25, tk_25, tlk)
    jmax = get_boltzmann(jmopt, ejm, toptjm, tlk)
    vcmax = get_boltzmann(vcopt, evc, toptvc, tlk)
    rb_mole = (ra + rb_co2) * tlk * 101.3 * 0.022624 / (273.15 * P_kPa)
    gb_mole = np.divide(np.ones((rb_mole.shape[0], rb_mole.shape[1])), rb_mole)

    dd = gammac

    b8_dd = 8 * dd

    # APHOTO = PG - rd, net photosynthesis is the difference
    # between gross photosynthesis and dark respiration. Note
    # photorespiration is already factored into PG.
    #
    # coefficients for Ball-Berry stomatal conductance model
    # Gs = k A rh/cs + b'
    # rh is relative humidity, which comes from a coupled
    # leaf energy balance model

    # ES_ret = ES(tlk)

    rh_leaf = eair_Pa / get_saturation_vapor_pressure(tlk)
    k_rh = np.multiply(rh_leaf, kball)
    k_rh = np.divide(k_rh, 1.6)

    gb_k_rh = np.multiply(gb_mole, k_rh)

    ci_guess = cca * 0.7  # % // initial guess of internal CO2 to estimate Wc and Wj

    # cubic coefficients that are only dependent on CO2 levels
    # app_bprime = np.repeat(bprime[0],gb_mole.size)
    # app_bprime = app_bprime.reshape((gb_mole.shape[0],gb_mole.shape[1]))

    app_bprime = np.divide(bprime, gb_mole)

    alpha_ps = np.add(1.0, app_bprime)
    alpha_ps = np.subtract(alpha_ps, k_rh)

    bbeta_app_gb = np.subtract(gb_k_rh, gb_mole)
    bbeta_app_2 = np.subtract(bbeta_app_gb, (bprime * 2))

    bbeta = np.multiply(cca, bbeta_app_2)

    gamma = cca * cca
    gamma = np.multiply(gamma, gb_mole)
    gamma = np.multiply(gamma, bprime)

    theta_ps = np.subtract(gb_k_rh, bprime)

    #
    # theta_ps = gb_k_rh - bprime

    # Test for the minimum of Wc and Wj.  Both have the form:
    # W = (a ci - ad)/(e ci + b)
    # after the minimum is chosen set a, b, e and d for the cubic solution.
    # estimate of J according to Farquhar and von Cammerer (1981)
    # J photon from Harley

    j_photon = np.repeat(np.NaN, gb_mole.size)
    j_photon = j_photon.reshape((gb_mole.shape[0], gb_mole.shape[1]))

    # j_photon                  = nan(size(jmax))

    index_j_photon = jmax > 0

    j_photon[index_j_photon] = (
        qalpha
        * Iphoton[index_j_photon]
        / np.sqrt(
            1.0
            + (
                qalpha2
                * Iphoton[index_j_photon]
                * Iphoton[index_j_photon]
                / (jmax[index_j_photon] * jmax[index_j_photon])
            )
        )
    )
    j_photon[np.isnan(j_photon)] = 0

    wj = j_photon * (ci_guess - dd) / ((4 * ci_guess) + b8_dd)
    wc = vcmax * (ci_guess - dd) / (ci_guess + bc)

    psguess = np.repeat(np.NaN, wc.size)
    psguess = psguess.reshape((wc.shape[0], wc.shape[1]))


    E_ps = np.array(psguess)

    B_ps = np.array(psguess)
    a_ps = np.array(psguess)
    index_psguess = wj < wc

    # for Harley and Farquhar type model for Wj

    psguess[index_psguess] = wj[index_psguess]

    B_ps[index_psguess] = b8_dd[index_psguess]

    a_ps[index_psguess] = j_photon[index_psguess]

    E_ps[index_psguess] = 4.0

    j_sucrose = vcmax / 2  # %C <l1646> j_sucrose = vcmax / 2. - rd;

    psguess[~index_psguess] = wc[~index_psguess]
    B_ps[~index_psguess] = bc[~index_psguess]
    a_ps[~index_psguess] = vcmax[~index_psguess]
    E_ps[~index_psguess] = 1.0


    # PREALLOCATION

    gs_leaf_mole = np.repeat(np.NaN, Iphoton.size)
    gs_leaf_mole = gs_leaf_mole.reshape((Iphoton.shape[0], Iphoton.shape[1]))

    gs_co2 = np.array(gs_leaf_mole)

    ps_1 = np.array(gs_leaf_mole)
    delta_1 = np.array(gs_leaf_mole)
    Aquad1 = np.array(gs_leaf_mole)
    Bquad1 = np.array(gs_leaf_mole)
    Cquad1 = np.array(gs_leaf_mole)
    product = np.array(gs_leaf_mole)
    sqrprod = np.array(gs_leaf_mole)
    ci = np.array(gs_leaf_mole)
    cs = np.array(gs_leaf_mole)
    denom = np.array(gs_leaf_mole)
    aphoto = np.array(gs_leaf_mole)
    gs_m_s = np.array(gs_leaf_mole)

    subindex = np.zeros((Iphoton.shape[0], Iphoton.shape[1]))

    # INDEX FOR QUADRATIC AND CUBIC SOLUTION

    index1 = np.logical_or(wj <= rd, wc <= rd)

    index2 = ~index1  # cubic solution

    # PHOTOSYNTHESIS: CUBIC SOLUTION (INDEX2)
    if np.any(np.any(index2)):  # %! any(any...) is required for matrices
        denom = E_ps * alpha_ps

        Pcube = (
            (E_ps * bbeta)
            + (B_ps * theta_ps)
            - (a_ps * alpha_ps)
            + (E_ps * rd * alpha_ps)
        )
        Pcube = Pcube / denom
        Qcube = (
            E_ps * gamma
            + (B_ps * gamma / cca)
            - a_ps * bbeta
            + a_ps * dd * theta_ps
            + E_ps * rd * bbeta
            + rd * B_ps * theta_ps
        )
        Qcube = Qcube / denom
        Rcube = (
            -a_ps * gamma
            + a_ps * dd * (gamma / cca)
            + E_ps * rd * gamma
            + rd * B_ps * gamma / cca
        )
        Rcube = Rcube / denom

        # Use solution from Numerical Recipes from Press
        P2 = Pcube**2
        P3 = P2 * Pcube
        Q = (P2 - (3.0 * Qcube)) / 9.0
        R = (2.0 * P3 - 9.0 * Pcube * Qcube + 27.0 * Rcube) / 54.0

        rr = R * R
        qqq = Q * Q * Q

        # real roots
        arg_U = R / np.sqrt(qqq)
        ang_L = np.arccos(arg_U)

        root1 = -2.0 * np.sqrt(Q) * np.cos(ang_L / 3.0) - Pcube / 3.0
        root2 = -2.0 * np.sqrt(Q) * np.cos((ang_L + np.pi * 2) / 3.0) - Pcube / 3.0
        root3 = -2.0 * np.sqrt(Q) * np.cos((ang_L - np.pi * 2) / 3.0) - Pcube / 3.0

        # rank roots #1,#2 and #3 according to the minimum, intermediate and maximum
        # value
        roots = np.dstack([root1, root2, root3])
        maxroot = roots.max(axis=2)
        minroot = roots.min(axis=2)
        midroot = np.median(roots, axis=2)

        indexAphoto1 = (minroot > 0) & (midroot > 0) & (maxroot > 0)
        indexAphoto2 = (minroot < 0) & (midroot < 0) & (maxroot > 0)
        indexAphoto3 = (minroot < 0) & (midroot > 0) & (maxroot > 0)
        index_debug = (indexAphoto1 + indexAphoto2 + indexAphoto3) == 0
        aphoto = np.zeros(indexAphoto1.shape)
        aphoto[indexAphoto1] = minroot[indexAphoto1]
        aphoto[indexAphoto2] = maxroot[indexAphoto2]
        aphoto[indexAphoto3] = midroot[indexAphoto3]
        aphoto[index_debug] = midroot[index_debug]

        # new solution to imaginary numbers in aphoto (nan in Python) from: Press, William H., et al. Numerical recipes 3rd edition: The art of scientific computing. Cambridge university press, 2007.
        # happening probably when Q is negative
        imaginary_index = np.isnan(aphoto)
        if np.any(imaginary_index):
            # get sign. eq. 5.6.13  (not used: Q and R are both real)
            sign = (R >= 0).astype(np.float32)
            sign[sign == 0] = -1
            a_im = -sign * (np.abs(R) + (R**2 - Q**3) ** 0.5) ** (1 / 3)
            b_im = np.zeros(a_im.shape)
            mask = a_im != 0
            b_im[mask] = (Q / a_im)[mask]
            b_im[~mask] = 0
            root_real = (a_im + b_im) - Pcube / 3  # only one real solution  (a = Pcube)
            root_real[root_real < 0] = 0
            aphoto[imaginary_index] = root_real[imaginary_index]

        cs[index2] = cca - aphoto[index2] / gb_mole[index2]

        gs_leaf_mole[index2] = (
            (float(kball) * rh_leaf[index2] * aphoto[index2] / cs[index2])
            + float(bprime)
        ) * 1.6 # stomatal conductance is mol(h20) m-2 s-1

        # convert Gs from vapor to CO2 diffusion coefficient
        gs_co2[index2] = (
            gs_leaf_mole[index2] / 1.6
        )  # stomatal conductance is mol(CO2) m-2 s-1
        gs_m_s[index2] = (
            gs_leaf_mole[index2] / rho_mol
        )
        # to compute ci, Gs must be in terms for CO2 transfer
        ci[index2] = cs[index2] - aphoto[index2] / gs_co2[index2]



    # the quadratic solution is needed if Aphoto < 0
    # PHOTOSYNTHESIS: QUADRATIC SOLUTION (INDEX1)
    if np.any(np.any(index1)):
        gs_co2[index1] = bprime
        gs_leaf_mole[index1] = bprime * 1.6
        # stomatal conductance is mol m-2 s-1
        # convert back to resistance (s/m) for energy balance routine
        gs_m_s[index1] = np.divide(gs_leaf_mole[index1], rho_mol)

        # a quadratic solution of A is derived if gs=ax, but a cubic form occurs
        # if gs =ax + b.  Use quadratic case when A is less than zero because gs will be
        # negative, which is nonsense
        ps_1[index1] = cca * gb_mole[index1] * gs_co2[index1]
        delta_1[index1] = gs_co2[index1] + gb_mole[index1]
        denom[index1] = gb_mole[index1] * gs_co2[index1]

        Aquad1[index1] = delta_1[index1] * E_ps[index1]
        Bquad1[index1] = (
            -ps_1[index1] * E_ps[index1]
            - a_ps[index1] * delta_1[index1]
            + E_ps[index1] * rd[index1] * delta_1[index1]
            - B_ps[index1] * denom[index1]
        )
        Cquad1[index1] = (
            a_ps[index1] * ps_1[index1]
            - a_ps[index1] * dd[index1] * denom[index1]
            - E_ps[index1] * rd[index1] * ps_1[index1]
            - rd[index1] * B_ps[index1] * denom[index1]
        )
        product[index1] = (
            Bquad1[index1] * Bquad1[index1] - 4.0 * Aquad1[index1] * Cquad1[index1]
        )

        subindex = product >= 0
        sqrprod[subindex] = np.sqrt(product[subindex])

        aphoto[index1] = (-Bquad1[index1] - sqrprod[index1]) / (2.0 * Aquad1[index1])
        cs[index1] = cca - aphoto[index1] / gb_mole[index1]
        ci[index1] = cs[index1] - aphoto[index1] / gs_co2[index1]

        wj[index1] = (
            j_photon[index1]
            * (ci[index1] - dd[index1])
            / (4 * ci_guess + b8_dd[index1])
        )
        wc[index1] = vcmax[index1] * (ci[index1] - dd[index1]) / (ci_guess + bc[index1])

    # for the transition between wj and wc, when there is colimitation.  this
    # is important because if one looks at the light response curves of the
    # current code one see jumps in A at certain Par values
    # theta wp^2 - wp (wj + wc) + wj wc = 0
    # a x^2 + b x + c = 0
    # x = [-b +/- sqrt(b^2 - 4 a c)]/2a

    a = 0.98
    b = -(wj + wc)
    c = wj * wc

    wp1 = (-b + np.sqrt((b * b) - (4 * a * c))) / (2 * a)
    wp2 = (-b - np.sqrt((b * b) - (4 * a * c))) / (2 * a)

    wp = np.minimum(wp1, wp2)

    aa = 0.95
    bb = -(wp + j_sucrose)
    cc = wp * j_sucrose

    Aps1 = (-bb + np.sqrt(bb * bb - 4 * aa * cc)) / (2 * aa)
    Aps2 = (-bb - np.sqrt(bb * bb - 4 * aa * cc)) / (2.0 * aa)

    Aps = np.minimum(Aps1, Aps2)

    index_aps = np.logical_and(Aps < aphoto, Aps > 0)
    aphoto[index_aps] = Aps[index_aps] - rd[index_aps]

    t_a = bprime * 1.6

    #if switch_sm == 2:
    # gs_leaf_mole = gs_leaf_mole * SM  # keenan et al 2010

    gs_co2 = gs_leaf_mole / 1.6
    gs_m_s = gs_leaf_mole / rho_mol

    # TODO: Palozzi?
    # if PALOZZI["O3CORR"][0]==1 :  #% @@1
    #    [aphoto_O3,gs_leaf_mole_O3,idx_O3,PALOZZI]=lombardozzi(aphoto, gs_leaf_mole,PALOZZI,ID_TOY)
    #    gs_co2_O3= gs_leaf_mole_O3 /1.57           #%! stomal conductance to CO2
    #    gs_ms_O3 = gs_leaf_mole_O3 / rho_mol
    #    aphoto[idx_O3]=aphoto_O3[idx_O3]
    #    gs_m_s[idx_O3]=gs_ms_O3[idx_O3]
    #    gs_co2[idx_O3]=gs_co2_O3[idx_O3]
    #    gs_leaf_mole[idx_O3]=gs_leaf_mole_O3[idx_O3]

    return (
        aphoto,
        ci,
        gs_co2,
        gs_leaf_mole,
        gs_m_s,
        wj,
        wc,
        wp,
        j_sucrose,
        vcmax,
        jmax,
        rd,
        index2,
    )



def get_rbh(t_air, windspeed, leaf_length, leaf_width, P, rho_kg):
    """
    Calculate the boundary layer resistance to heat.

    Parameters:
    t_air (float): Air temperature in Kelvin.
    windspeed (float): Wind speed in m/s.
    leaf_length (float): Length of the leaf in meters.
    leaf_width (float): Width of the leaf in meters.
    P (float): Air pressure in Pascal.
    rho_kg (float): Density of air in kg/m^3.

    Returns:
    rbh (float): Boundary layer resistance to heat in s/m.
    """
    leaf_dim = 0.6 * leaf_length + 0.4 * leaf_width  # % meters
    cp = 1004.2  # %specific heat of air at constant pressure, J kg-1 K-1
    dynamic_viscosity = (1.458e-6 * (t_air ** (3.0 / 2))) / (
        t_air + 110.4
    )  # % dynamic viscosity ((kg m-1 s-1 K-1./2) .* K.^3./2) ./ K
    kinematic_viscosity = dynamic_viscosity / rho_kg  #  % m2 s-1

    reynolds_num = windspeed * leaf_dim / kinematic_viscosity

    air_thermal_conductivity_coeff = (
        (
            -0.9474
            + 11.961 * (t_air / 100)
            - 2.3632 * (t_air / 100) ** 2
            + 0.8406 * (t_air / 100) ** 3
            - 0.1747 * (t_air / 100) ** 4
            + 1.904e-2 * (t_air / 100) ** 5
            - 1.035e-3 * (t_air / 100) ** 6
            + 2.228e-5 * (t_air / 100) ** 7
        )
        / 1000
    )  # % Handbook of Thermal Conductivity of Liquids and Gases (Natan B. Vargaftik)
    thermal_diffusivity = air_thermal_conductivity_coeff / (rho_kg * cp)  # % m2 s-1
    prandtl_num = kinematic_viscosity / thermal_diffusivity
    # % Schymanski, S. J., Or, D., and Zwieniecki, M., 2013. Stomatal Control and Leaf Thermal and Hydraulic Capacitances under Rapid Environmental Fluctuations. PLoS ONE, 8 (1), e54231.
    reynolds_critical = 10**5  # % ranges from 10^5 to 3*10^6
    C1 = (
        0.037
        * (
            reynolds_num
            + reynolds_critical
            - np.absolute(reynolds_critical - reynolds_num)
        )
        ** (4 / 5)
        - 0.664
        * (
            reynolds_num
            + reynolds_critical
            - np.absolute(reynolds_critical - reynolds_num)
        )
        ** 0.5
    )

    nusselt_num = (0.037 * reynolds_num ** (4 / 5) - C1) * prandtl_num
    # % boundary layer resistance % s m-1

    if nusselt_num == 0:
        nusselt_num = 1e-12

    app_thrmal_nuss = thermal_diffusivity * nusselt_num

    rbh = leaf_dim / app_thrmal_nuss  # % 0.5 *
    return rbh


# function to define the sensible heat of tree leaves
# tleaf: leaf temperature (K)
# tair: air temperature (K)
# P: air pressure (Pa)
# RH: relative humidity (%)
# res_sto: stomatal resistance (s m-1)
# lai_layer: leaf area index (m2 m-2)
# rbh: boundary layer resistance (s m-1)
# ra: aerodynamic resistance (s m-1)
# net_radiation: net radiation (W m-2)
# aereodynamic_resistance: aerodynamic resistance (s m-1)
# rho_kg: air density (kg m-3)
# lai_coeff: leaf area index coefficient (m2 m-2)
def get_heat_fluxes(
    tleaf,
    tair,
    P,
    RH,
    res_sto,
    lai_layer,
    rbh,
    ra,
    net_radiation,
    aereodynamic_resistance,
    rho_kg,
    lai_coeff,
):
    # Lhomme, Jean Paul. “Extension of Penman’s Formulae to Multi-Layer Models.” Boundary-Layer Meteorology 42, no. 4 (1988): 281–91.
    gb = 1 / rbh  # % boundary layer conductance of leaves (m s-1)
    tair_C = tair - 273.16
    tleaf_C = tleaf - 273.16
    cp = 1004.2  # specific heat of air at constant pressure, J kg-1 K-1

    Vap2 = 3149000 - 2370 * tleaf
    gamma = (cp * P) / (Vap2 * 0.622)  # adiabatic psychrometric coefficient

    # Vapor pressure computation
    # from Katul
    # tetens formula (Buck, 1981)
    a = 0.611  # kPa
    b = 17.502  # unitless
    c = 240.97  # C
    e_sat_leaf = a * np.exp((b * tleaf_C) / (tleaf_C + c)) * 1000
    e_sat_air = a * np.exp((b * tair_C) / (tair_C + c)) * 1000
    e_air = (RH / 100) * e_sat_air  # Pa

    vpd_air = e_sat_air - e_air
    delta = (e_sat_leaf - e_sat_air) / (tleaf - tair)
    if np.isnan(delta):
        delta = 0

    ge_sensible = lai_layer * 2 * gb
    sensible_heat = rho_kg * cp * ge_sensible * (tleaf_C - tair_C)

    resistance_lh_leaf = (rbh + res_sto + aereodynamic_resistance)
    conductance_lh_leaf = 1 / resistance_lh_leaf

    resistance_lh_evap = (rbh + aereodynamic_resistance)
    conductance_lh_evap = 1 / resistance_lh_evap

    # TODO: check if this is correct
    # reference?
    #latent_heat = (
    #    lai_coeff
    #    * lai_layer
    #    * (delta * net_radiation + rho_kg * cp * vpd_air / aereodynamic_resistance)
    #    / (delta + gamma * (1 + res_sto / aereodynamic_resistance))
    #)
    latent_heat = (
        lai_coeff
        * lai_layer
        * (delta * net_radiation + rho_kg * cp * vpd_air * conductance_lh_leaf)
        / (delta + gamma)
    )

    # EXP stomatal conductance = inf (Lhomme 1988)
    # boundary layer restistance = rbh + aereodynamic_resistance
    latent_heat_evaporation = (
        lai_coeff
        * lai_layer
        * (delta * net_radiation + rho_kg * cp * vpd_air * conductance_lh_evap)
        / (delta + gamma)
    )
    return latent_heat, sensible_heat, delta, gamma, e_sat_leaf, e_sat_air, latent_heat_evaporation




# def lombardozzi(aphoto, gs_leaf_mole, PALOZZI, ID_TOY):
#     # % aphoto = photosynthesis (umol CO2 m-2 s-1) --> otput from 'photosynthesis' function (BALL BERRY)
#     # % gs_leaf_mole =  stomatal conductance to water(mol H2O m-2 s-1) --> otput from 'photosynthesis' function (BALL BERRY)
#     # % species = (100 for CPZ ilex parametrization; otherwise put 1,2,3,4,5 for broadleaf sensitive or nonsentivie plant, see the comments in the code)
#     # % CO2Flux = NEE, as measured from eddy covariance data( umol m-2 s-1)
#     # % O3= O3 concentrations (ppb)
#     # % PAR = or PPFD (umol photon m-2 s-1)
#     # % thr_O3 = thr_O3eshold ozone flux for plant detoxyfication (nmol m-2 s-1) -> species-specific it is the Y of the PODY. ilex is 1 nmol m-2 s-1
#     # % VPD = Vapour pressure deficit (kPa)
#     # % Press = air pressure (Kpa)
#     o3vulnerability = PALOZZI["o3vulnerability"]
#     if o3vulnerability == 100:  # %ilex Adriano
#         ac = -0.0009  # %linear regression intercept between Treatment/Control gsto and CUO - calculated from Alonso et al
#         bc = 0.86  # %linear regression slope between Treatment/Control gsto and CUO - calculated
#         ap = -0.00027  # %linear regression intercept between Treatment/Control photosinthesys and CUO - calculated from Alonso et al
#         bp = 0.79  # %linear regression intercept between Treatment/Control photosinthesys and CUO - alculated from Alonso et al
#     elif o3vulnerability == 0:  # %all plant types
#         ap = -0.00098
#         bp = 0.8434
#         ac = 0.0
#         bc = 0.8444
#     elif o3vulnerability == 1:  # % High vulnerability Broadleaf
#         ap = 0
#         bp = 0.8502
#         ac = 0
#         bc = 0.89
#     elif o3vulnerability == 2:  # % High vulnerability Needleaf
#         ap = -0.038
#         bp = 1.083
#         ac = -0.0144
#         bc = 0.8874
#     elif o3vulnerability == 3:  # % Low vulnerability Broadleaf
#         ap = 0
#         bp = 0.9798
#         ac = -0
#         bc = 0.9425
#     elif o3vulnerability == 4:  # % Low vulnerability Needleaf
#         ap = 0
#         bp = 0.8595
#         ac = 0.0067
#         bc = 0.7574
#     elif o3vulnerability == 5:  # % Low vulnerability Needleaf
#         ap = 0
#         bp = 0.8595
#         ac = 0.0067
#     elif o3vulnerability == 101:  # %pinus ponderosa
#         ac = -0.000737  # %linear regression intercept between Treatment/Control gsto and CUO - calculated from Alonso et al
#         bc = 1.06708  # %linear regression slope between Treatment/Control gsto and CUO - calculated
#         ap = -0.00046  # %linear regression intercept between Treatment/Control photosinthesys and CUO - calculated from Alonso et al
#         bp = 0.91517  # %linear regression intercept between Treatment/Control photosinthesys and CUO - alculated from Alonso et
#         bc = 0.7574
#     else:
#         raise NotImplementedError
#
#     # %% -----------%
#
#     CEO3 = PALOZZI["CEO3"]
#
#     CUO = (
#         CEO3 * gs_leaf_mole * 1.67 * 3600 * (10**-6)
#     )  # %Lomardozzi methods need stomatal conductance measured after the treatment I think that usining gsto simulated without ozone effect would be better
#     FO3p = CUO * ap + bp
#     FO3c = CUO * ac + bc
#
#     # %Now let's apply the thr_O3eshold value
#     O3Conc = PALOZZI["O3CONC"]
#     Ozone_Flux = 0.6 * gs_leaf_mole * O3Conc
#     indx_O3 = Ozone_Flux > PALOZZI["thr_O3"]
#
#     if np.sum(indx_O3) > 0:
#         PALOZZI["CEO3"] = CEO3 + O3Conc / 2
#
#     PALOZZI["CUO"] = CUO
#     gs_leaf_mole_O3 = gs_leaf_mole * FO3c
#     aphoto_O3 = aphoto * FO3p
#
#     return aphoto_O3, gs_leaf_mole_O3, indx_O3, PALOZZI

@dataclass
class InputPhoto:
    '''input for photosynthesis'''
    ppfd_sun               : float
    ppfd_shade             : float
    CO2conc                : float
    atmospheric_resistance : float
    bound_layer_res_co2    : float
    P_kPa                  : float
    eair_Pa                : float
    vcopt                  : float
    m                      : float
    bprime                 : float
    SM                     : float
    rho_mol_tmp            : float
    jmax_coeff             : float
    # switch_sm              : int
    ID_TOY                 : int


@dataclass
class InputHeat:
    tairK                      : float
    P_Pa                       : float
    RH                         : float
    lai_layer_sun              : float
    lai_layer_shade            : float
    rbh                        : float
    boundary_layer_resistance  : float
    rad_tot_sun                : float
    rad_tot_shade              : float
    aerodynamic_resistance     : float
    rho_kg                     : float
    lai_coeff                  : float


@dataclass
class InputThermal:
    layers                : int
    t_air_k               : float
    leafremissivity       : float
    atmosphere_emissivity : float
    backscattering_veg    : float
    thermal_atm           : float
    boltzman              : float
    penetration_veg       : float


@dataclass
class OutPhoto:
    A                    : np.typing.ArrayLike
    Ci                   : np.typing.ArrayLike
    Gs_co2               : np.typing.ArrayLike
    Gs_leaf_mole         : np.typing.ArrayLike
    Gs_m_s               : np.typing.ArrayLike
    wj                   : np.typing.ArrayLike
    wc                   : np.typing.ArrayLike
    wp                   : np.typing.ArrayLike
    j_sucrose            : np.typing.ArrayLike
    vcmax                : np.typing.ArrayLike
    jmax                 : np.typing.ArrayLike
    rd                   : np.typing.ArrayLike
    index2               : np.typing.ArrayLike

    def as_dict(self):
        return dc.asdict(self)

    def set_temperature_array(self, temperature_array):
        self.temperature_array = temperature_array

    def gs_from_temperature(self, temperature):
        arr = self.temperature_array - temperature
        index = np.argmin(np.abs(arr))
        return self.Gs_m_s[index, :]

    def dict_from_temperature(self, temperature, layer, prefix=""):
        arr = self.temperature_array - temperature
        index = np.argmin(np.abs(arr))

        out1 = self.as_dict()
        out = {}
        for key in out1:
            val =  out1[key][index, layer]
            k1 = prefix + key
            out[k1] = val
        return out


        arr = self.temperature_array - temperature
        index = np.argmin(np.abs(arr))
        return self.Gs_m_s[index, :]


@dataclass
class OutHeat:
    LH        : float
    SH        : float
    delta     : float
    gamma     : float
    esat_leaf : float
    esat_air  : float
    lh_evap   : float

    def to_dict(self, prefix=""):
        out1 = dc.asdict(self)
        out = {}
        for key in out1:
            val =  out1[key]
            k1 = prefix + key
            out[k1] = val
        return out


@dataclass
class OutThermal:
    net     : np.typing.ArrayLike
    net0    : np.typing.ArrayLike
    lwd0    : np.typing.ArrayLike
    lwu0    : np.typing.ArrayLike

    def to_dict(self, prefix=""):
        out1 = dc.asdict(self)
        out = {}
        for key in out1:
            val =  out1[key]
            k1 = prefix + key
            out[k1] = val
        return out

class Thermal:
    def __init__(self, in_thermal: InputThermal):
        self.cache : dict[float, OutThermal] = {}
        self.inp = in_thermal

    def get_thermal(
        self,
        Tleaf,
        array_mode = False
    ):
        if not array_mode:
            leaf_t = float(Tleaf[0])
            if leaf_t in self.cache:
                return self.cache[leaf_t]

        out_thermal = OutThermal(
            *self._get_thermal(
                self.inp.layers,
                Tleaf,
                self.inp.t_air_k,
                self.inp.leafremissivity,
                self.inp.atmosphere_emissivity,
                self.inp.backscattering_veg,
                self.inp.thermal_atm,
                self.inp.boltzman,
                self.inp.penetration_veg,
            )
        )
        if not array_mode:
            self.cache[leaf_t] = out_thermal
        return out_thermal

    def _get_thermal(
        self,
        canopy_layers,
        Tleaf,
        tairK,
        canopy_emissivity,
        atmosphere_emissivity,
        backscattering_veg,
        thermal_atm,
        boltzman,
        penetration_veg,
    ):
        # % GET THERMAL CANOPY
        # Zhao, W. and Qualls, R. J., 2006. Modeling of long-wave and net radiation energy distribution within a homogeneous plant canopy via multiple scattering processes: LONG-WAVE AND NET RADIATION DISTRIBUTION MODEL. Water Resources Research, 42 (8), n/a–n/a.

        soil_emissivity = 0.96
        Tsoil = tairK   # % TODO model soil temperature
        penetration_soil = 0
        penetration_air = 1
        # % for spherical distribution (see Zhao..)
        backscattering_soil = 1
        backscattering_air = 0

        # % lw from soil surface boundary
        # % fine-grain soils; Axelsson, S. R., 1984. Thermal-IR emissivity of soils and its dependence on porosity, surface roughness and soil-moisture. 5th Int. Congr. on Photogramm. and Remote Sensing, Brazil.
        # % buildin vectors of tau, backscattering and emissivity
        layers = canopy_layers + 2

        tau = np.zeros(layers)
        backscattering = np.zeros(layers)
        emissivity = np.zeros(layers)

        tau[0] = penetration_soil
        tau[-1] = penetration_air
        backscattering[0] = backscattering_soil
        backscattering[-1] = backscattering_air
        emissivity[0] = soil_emissivity
        emissivity[-1] = atmosphere_emissivity

        # reverse layer order convention
        penetration_veg = np.flip(penetration_veg)
        for i in range(1, layers - 1):
            tau[i] = penetration_veg[i - 1]
            backscattering[i] = backscattering_veg
            emissivity[i] = canopy_emissivity

        # buinding vectors of C and A; Eq 25, 26; Zhao, W. and Qualls, R. J., 2006. Modeling of long-wave and net radiation energy distribution within a homogeneous plant canopy via multiple scattering processes: LONG-WAVE AND NET RADIATION DISTRIBUTION MODEL. Water Resources Research, 42 (8), n/a–n/a.
        matrix_length = 2 * canopy_layers + 2
        vect_c = np.zeros((matrix_length, 1))
        mat_a = np.zeros((matrix_length, matrix_length))

        vect_c[0] = (soil_emissivity) * boltzman * Tsoil**4
        vect_c[-1] = thermal_atm

        mat_a[0, 0] = 1
        mat_a[-1, -1] = 1

        i = 1  # layers index; start from canopy
        m = 1  # matrix index; start from canopy
        ix_tleaf = 0

        # --------------------------------------
        # TODO: CHECK lai correction
        # apply lai correction, missing on reference paper
        while m != matrix_length - 1:
            ix_tleaf -= 1
            tl_1 = Tleaf[ix_tleaf]
            vect_c[m] = (
                (
                    1
                    - backscattering[i - 1]
                    * backscattering[i]
                    * (1 - emissivity[i - 1])
                    * (1 - tau[i - 1])
                    * (1 - emissivity[i])
                    * (1 - tau[i])
                )
                * (1 - tau[i])
                * emissivity[i]
                * boltzman
                * tl_1**4
            )

            vect_c[m + 1] = (
                (
                    1
                    - backscattering[i]
                    * backscattering[i + 1]
                    * (1 - emissivity[i])
                    * (1 - tau[i])
                    * (1 - emissivity[i + 1])
                    * (1 - tau[i + 1])
                )
                * (1 - tau[i])
                * emissivity[i]
                * boltzman
                * tl_1**4
            )
            m = m + 2
            i = i + 1
        # --------------------------------------
        for i in range(1, canopy_layers + 1):
            d = (i * 2) - 1
            mat_a[d, d - 1] = -(
                tau[i] + (1 - backscattering[i])
                * (1 - emissivity[i]) * (1 - tau[i])
            )
            mat_a[d, d] = (
                -backscattering[i - 1]
                * (tau[i] + (1 - backscattering[i])
                   * (1 - emissivity[i]) * (1 - tau[i]))
                * (1 - emissivity[i - 1])
                * (1 - tau[i - 1])
            )
            mat_a[d, d + 1] = 1 - backscattering[i - 1] * backscattering[i] * (
                1 - emissivity[i - 1]
            ) * (1 - tau[i - 1]) * (1 - emissivity[i]) * (1 - tau[i])
            mat_a[d + 1, d] = 1 - backscattering[i] * backscattering[i - 1] * (
                1 - emissivity[i]
            ) * (1 - tau[i]) * (1 - emissivity[i - 1]) * (1 - tau[i - 1])
            mat_a[d + 1, d + 1] = (
                -backscattering[i - 1]
                * (tau[i] + (1 - backscattering[i])
                   * (1 - emissivity[i]) * (1 - tau[i]))
                * (1 - emissivity[i - 1])
                * (1 - tau[i + 1])
            )
            mat_a[d + 1, d + 2] = -(
                tau[i] + (1 - backscattering[i])
                * (1 - emissivity[i]) * (1 - tau[i])
            )

        mat_a[np.isnan(mat_a)] = 0
        # https://research.wmz.ninja/articles/2018/10/notes-on-migrating-doa-tools-from-matlab-to-python.html   Risolvere sistemi di equazioni lineari
        # Risolvere sistemi di equazioni lineari
        # longwave0 = np.linalg.lstsq(mat_a, vect_c, rcond=None)[0]
        vect_c = vect_c.squeeze()
        longwave0 = np.linalg.solve(mat_a, vect_c).squeeze()

        # ----------------------------------------------------------------------------
        # % divide upward and downward radiation
        lwu0 = np.zeros(layers)
        lwd0 = np.zeros(layers)

        lwu0[0] = longwave0[0]
        lwd0[-1] = longwave0[-1]

        i = -1
        while i != canopy_layers:
            i += 1
            # y += 1
            lwd0[i + 1] = longwave0[(i * 2) + 1]
            lwu0[i] = longwave0[(i * 2) ]

        # ----------------------------------------------------------------------------
        # apply multiple scattering using eq8 and eq9
        lwd = np.zeros(layers)
        lwu = np.zeros(layers)

        for i in range(canopy_layers + 1):
            lwd[i + 1] = (
                lwd0[i + 1]
                + backscattering[i + 1]
                * (1 - emissivity[i + 1])
                * (1 - tau[i + 1])
                * lwu0[i]
            ) / (
                1
                - backscattering[i]
                * backscattering[i + 1]
                * (1 - emissivity[i])
                * (1 - tau[i])
                * (1 - emissivity[i + 1])
                * (1 - tau[i + 1])
            )

        for i in range(canopy_layers + 1):
            lwu[i] = (
                lwu0[i]
                + backscattering[i] * (
                    1 - emissivity[i]
                ) * (1 - tau[i]) * lwd0[i + 1]
            ) / (
                1
                - backscattering[i]
                * backscattering[i + 1]
                * (1 - emissivity[i])
                * (1 - tau[i])
                * (1 - emissivity[i + 1])
                * (1 - tau[i + 1])
            )

        # ----------------------------------------------------------------------------
        # calculate net radiation
        net_tmp = np.zeros(len(lwu))
        net0_tmp = np.zeros(len(lwu))

        # +sign convention for net radiation absorbed by layer
        for i in range(1, len(lwd) -1):
            net_tmp[i] = lwd[i] + lwu[i] - lwu[i - 1] - lwd[i + 1]
            net0_tmp[i] = lwd0[i] + lwu0[i] - lwu0[i - 1] - lwd0[i + 1]

        # reverse matrix and eliminate soil and sky layers
        net = net_tmp[1:-1]
        net = net[::-1]
        net0 = net0_tmp[1:-1]
        net0 = net0[::-1]
        return net, net0, lwd0, lwu0



class EnergyBalance:
    def __init__(self, PROJECT, CANOPY, ATM, rad, res, ID_TOY):
        self.PROJECT = PROJECT
        self.CANOPY = CANOPY
        self.ATM = ATM
        self.rad = rad
        self.res = res
        self.ID_TOY = ID_TOY
        super().__init__()


    def add_totals(self, CANOPY, rad, ceb, ID_TOY):
        '''compute:
            - sums of sun and shade values
            - total values
            '''
        sunFrac = rad["fraction_sun_leaves"]
        ceb.all_LH = ceb.hsn_LH + ceb.hsd_LH
        ceb.all_SH = ceb.hsn_SH + ceb.hsd_SH

        leafTransp = ceb.all_LH / 40.65  # CONTROLLARE CEB
        lai_layer = CANOPY.lai_layer

        ceb.all_A = (
            (ceb.psn_A * sunFrac + ceb.psh_A * (1 - sunFrac))
            * lai_layer
            * CANOPY["lai_coeff"]
        )

        ceb.all_Gs_leaf_mole = (
            (
                ceb.psn_Gs_leaf_mole * sunFrac
                + ceb.psh_Gs_leaf_mole * (1 - sunFrac)
            )
            * lai_layer
            * CANOPY["lai_coeff"]
        )

        ceb.all_Gs_leaf_m_s = (
            (ceb.psn_Gs_m_s * sunFrac + ceb.psn_Gs_m_s * (1 - sunFrac))
            * lai_layer
            * CANOPY["lai_coeff"]
        )

        ceb.tot_A = np.nansum(ceb.all_A)
        ceb.tot_Gs_leaf_mole = np.nansum(ceb.all_Gs_leaf_mole)
        ceb.tot_Gs_leaf_m_s = np.nansum(ceb.all_Gs_leaf_m_s)
        ceb.tot_LH = np.nansum(ceb.all_LH)
        ceb.tot_SH = np.nansum(ceb.all_SH)
        ceb.tot_LeafTransp = np.nansum(leafTransp)
        return ceb

    def get_energy_fluxes(self, PROJECT, CANOPY, ATM, rad, res, ID_TOY):
        # Code developed by Alex Guenther, based on Goudrian and Laar (1994), Leuning (1997)
        # Initial code 8-99, modified 7-2000 and 12-2001
        # Note: iy denotes an array containing a vertical profile through the canopy with 0 (above canopy conditions) plus 1 to number of canopy layers
        #
        # INPUT
        # --tree data--
        # CANOPY.layers : canopy layers
        # CANOPY : data structure from canopy.csv file
        # CANOPY.shape : LAI fraction per layer
        # CANOPY.dist : distance from top per canopy layer
        # --atmospheric data--
        # ATM.Tairk0 :  air temperature at canopy top (K)
        # ATM.eair_Pa : atmospheric vapor pressure (Pa)
        # ATM.P_kPa : air pressure (kPa)
        # ATM.CO2conc : Carbone dioxide (ppm)
        # --radiation data--
        # rad.ShadePPFD : incoming (NOT absorbed) PPFD on a shade leaf (Umol m-2 s-1)
        # rad.trate :  output from stability function
        # rad.SunPPFD : incoming (NOT absorbed) PPFD on a sun leaf (Umol m-2 s-1)
        # rad.SunQv : sum of visible radiation (in and out) fluxes on sun leaves
        # rad.ShadeQv : sum of absorbed visible radiation (in and out) fluxes on shade leaves
        # rad.SunQn : sum of absorbed near IR radiation (in and out) fluxes on sun leaves
        # rad.ShadeQn : sum of absorbed near IR radiation (in and out) fluxes on shade leaves
        # --resistances data--
        # res.HumidairPa : data from resistances class
        # res.WS :  data from resistances class
        # res.boundary_layer_res_CO2 : Boundary layer resistance

        # OUTPUT
        # ------------------------------------------------------
        # sunleafTK(iy) is an array of leaf temperature (K) for sun leaves
        # sunleafSH(iy) is an array of sensible heat flux (W m-2) for sun leaves
        # sunleafLH(iy) is an array of latent heat flux (W m-2) for sun leaves
        # sunleafIR(iy) is an array of Infrared flux (W m-2) for sun leaves
        # shadeleafTK(iy) is an array of leaf temperature (K) for shade leaves
        # shadeleafSH(iy) is an array of sensible heat flux (W m-2) for shade leaves
        # shadeleafLH(iy) is an array of latent heat flux (W m-2) for shade leaves
        # shadeleafIR(iy) is an array of Infrared flux (W m-2) for shade leaves

        tairK = ATM["Tairk0"][ID_TOY]  # % TODO: model air temperature in the canopy
        theta_avg = ATM["theta_avg"][ID_TOY]

        sunFrac = rad["fraction_sun_leaves"]
        ppfd_sun = rad["par_sun"] * 4.17
        ppfd_shade = rad["par_shaded"] * 4.17
        radtot = rad["rad_tot"]
        rad_tot_sun = rad["rad_tot_sun"]
        rad_tot_shade = rad["rad_tot_shaded"]

        RH = res["rh"]
        WS = res[
            "ws"
        ]  # %%An equation that describes the wind speed (uZ) within the top layer of most plant canopies (i.e., for the uppermost 36%), as well as the top and middle layer of plant stands with uniform canopies, is given by Campbell (1977)? and Niklas (1992)
        Ra = res["atmospheric_resistance"]
        RbCO2 = res["boundary_layer_res_CO2"]
        RbH2O = res["boundary_layer_res_H2O"]
        aereodynamic_conductance = res["aereodynamic_conductance"]

        # %Here i Calculate soil moisture reducing factor
        # % Keenan, T., Sabate, S., and Gracia, C., 2010. Soil water stress and coupled photosynthesis–conductance models: Bridging the gap between conflicting reports on the relative roles of stomatal, mesophyll conductance and biochemical limitations to photosynthesis. Agricultural and Forest Meteorology, 150 (3), 443–453.
        # % \cite{Keenan2010Soilwaterstress}

        #     %Keenan, T., García, R., Friend, A. D., Zaehle, S., Gracia, C., & Sabate, S. (2009).
        # % Improved understanding of drought controls on seasonal variation in Mediterranean forest canopy CO 2 and water...
        # %     fluxes through combined in situ measurements and ecosystem modelling. Biogeosciences, 6(8), 1423-1444.

        # photozero = 0.7  # % CSWC/FC and minimum SWC for photosynthesis/WP  (same ratio observed in Conte et al. 2021 )
        # breakpoint()
        # photozero = 0.3  # Testing

        # tetaw = CANOPY["WP"] * photozero
        # tetac = CANOPY["FC"] * photozero
        tetaw = CANOPY["WP"]
        tetac = CANOPY["FC"]
        #qteta = 0.48  # %average value for all pft
        #qteta = 0.48
        qteta = 0.85  # roccarespampani site in Keenan et al 2009

        # breakpoint()
        # if theta_avg < 0.1:
        #     breakpoint()

        SM = 1
        if theta_avg >= tetac or PROJECT["irrigation"]:
            SM = 1
        elif theta_avg < tetac:  # % && theta_avg>CANOPY.tetaw
            SM = ((theta_avg - tetaw) / (tetac - tetaw)) ** qteta
            if np.isreal(SM) == 0:
                SM = 0

        # TODO: find a better solution with workgroup
        SM = np.maximum(SM, 0.1)

        obj = self.balance_energy_fluxes(
            ppfd_sun,
            ppfd_shade,
            radtot,
            tairK,
            WS,
            RH,
            Ra,
            RbCO2,
            sunFrac,
            ATM["P_kPa"][ID_TOY],
            ATM["P_Pa"][ID_TOY],
            ATM["eair_Pa"][ID_TOY],
            ATM["CO2conc"][ID_TOY],
            RbH2O,
            SM,
            rad["kb"],
            CANOPY["lai"],
            rad_tot_sun,
            rad_tot_shade,
            aereodynamic_conductance,
            CANOPY,
            ATM,
            theta_avg,
            ID_TOY,
        )
        obj.SM = SM
        return obj

    def balance_energy_fluxes(
        self,
        ppfd_sun,
        ppfd_shade,
        radtot,
        tairK,
        ws,
        RH,
        atmospheric_resistance,
        bound_layer_res_co2,
        sunFrac,
        P_kPa,
        P_Pa,
        eair_Pa,
        CO2conc,
        bound_layer_res_h2o,
        SM,
        kb,
        LAI,
        rad_tot_sun,
        rad_tot_shade,
        aereodynamic_conductance,
        CANOPY,
        ATM,
        theta_avg,
        ID_TOY
    ):
        # INPUT
        # ppfd : incoming (NOT absorbed) PPFD on a leaf (Umol m-2 s-1)
        # radtot: absorbed shortwave radiation (W m-2)
        # CANOPY.leafIRemissivity : leafIRemissivity
        # CANOPY.leafLength : leaf lentgh
        # tairK : air temperature at canopy top (K)
        # humidairPa : data from resistances class
        # ws: data from resistances class
        # bound_layer_res_co2: data from resistances class
        # ATM.P_kPa :   air pressure (kPa)
        # ATM.eair_Pa :  atmospheric vapor pressure (Pa)
        # CANOPY.vcopt : rubisco carboxylation velocity
        # m : Ball-Berry coefficient
        # CANOPY.bprime : Ball-Berry intercept
        # ATM.CO2conc : Carbone dioxide (ppm)
        #
        # OUTPUT
        # ------------------------------------------------------------
        # aphoto : net photosynthesis  (umol m-2 s-1)
        # wj : RuBP limited rate of carboxylation
        # wc : minimum of RuBP saturated rate of carboxylation
        # wp : dummy variable for quadratic model of Collatz
        # jmax : optimal rate of electron transport

        layers = CANOPY["layers"]
        canopy_x = CANOPY["canopy_x"]

        aereodynamic_resistance = 1 / aereodynamic_conductance
        boltzman = 5.67051e-8
        rho_kg = (P_Pa / 100) / 1013 * 28.95 / 0.0821 / tairK  # % air density  [kg/m3]
        rho_mol = P_Pa / (8.314 * (tairK))

        # Crawford, T. M. and Duchon, C. E., 1999. An improved parameterization for estimating effective atmospheric emissivity for use in calculating daytime downwelling longwave radiation. Journal of Applied Meteorology, 38 (4), 474–480.
        # altered to use the extraterrestrial radiation
        # Meyer, W. S., Smith, D. J., and Shell, G., 1999. Estimating reference evaporation and crop evapotranspiration from weather data and crop coefficients. CSIRO Land and Water Technical Report, 34, 98.
        # cloudiness_factor = 1.1 * clearness_index +  0.18; for
        cloudiness_factor = 1
        laiStep = CANOPY.lai_layer

        step = 0.5

        # Brutsaert, W., 1975. On a derivable formula for long‐wave radiation from clear skies. Water Resources Research, 11 (5), 742–744.
        atmosphere_emissivity = 1.24 * (eair_Pa / (100 * tairK)) ** (1 / 7)
        # !! include cloudiness Song et al 2009
        thermal_atm = atmosphere_emissivity * cloudiness_factor * boltzman * tairK**4

        # Zhao, W. and Qualls, R. J., 2006. Modeling of long-wave and net radiation energy distribution within a homogeneous plant canopy via multiple scattering processes: LONG-WAVE AND NET RADIATION DISTRIBUTION MODEL. Water Resources Research, 42 (8), n/a–n/a.
        penetration_veg = []
        for i in range(0, layers):
            lai = laiStep[i]
            pen1 = get_penetration_veg(canopy_x, lai)
            penetration_veg.append(pen1[0])

        t_min1 = CANOPY["t_leaf_delta_min"] + tairK  # value from parameter file
        diff1 = (tairK - t_min1) % step
        t_min = t_min1 - diff1

        t_max = CANOPY["t_leaf_delta_max"] + tairK
        t_1 = np.arange(start=t_min, stop=t_max, step=step)
        shp_1 = (CANOPY["layers"], 1)

        Tleaf_mat_tmp = np.flip(np.transpose(np.tile(t_1, shp_1)), axis=0)

        # adapt vector to matrices
        c1 = Tleaf_mat_tmp.shape[0]
        ppfd_sun_tmp = np.tile(ppfd_sun, (c1, 1))
        ppfd_shade_tmp = np.tile(ppfd_shade, (c1, 1))
        bound_layer_res_co2_tmp = np.tile(bound_layer_res_co2, (c1, 1))
        atmospheric_resistance_tmp = atmospheric_resistance
        rho_mol_tmp = rho_mol

        # % -- scaling vcopt ---------------------------------------------------------
        # Bonan, G. B., Lawrence, P. J., Oleson, K. W., Levis, S., Jung, M., Reichstein, M., ... & Swenson, S. C. (2011). Improving canopy processes in the Community Land Model version 4 (CLM4) using global flux fields empirically inferred from FLUXNET data. Journal of Geophysical Research: Biogeosciences, 116(G2).
        lai_cumulative = np.multiply(CANOPY["layer_volume_ratio"], LAI)
        for i in range(1, layers):
            app_lai_cumul = lai_cumulative[i - 1] + lai_cumulative[i]
            lai_cumulative[i] = app_lai_cumul

        toy = ATM[ID_TOY]["TOY"][0]
        vcopt = CANOPY["vcopt"] * 1.16  # %per portarlo al vcmax osservato
        vcopt = np.ones(ppfd_sun_tmp.shape) * vcopt

        # if not CANOPY["evergreen"]:
        #     if toy < CANOPY["leafing"] or toy > CANOPY["defoliation"]:
        #         vcopt[:] = 0

        vcopt = vcopt * np.exp(-CANOPY["kn"] * lai_cumulative)

        # TODO
        #switch_sm_string = CANOPY["switch_model"]  # % see ModelData module
        #switch_sm = int(switch_sm_string[0])
        #switch_stomata = int(switch_sm_string[1])  # % see ModelData module

        out_dict = {}
        in_photo = InputPhoto(
                # phosynthesys input
                ppfd_sun               =         ppfd_sun_tmp,
                ppfd_shade             =         ppfd_shade_tmp,
                CO2conc                =         CO2conc,
                atmospheric_resistance =         atmospheric_resistance_tmp,
                bound_layer_res_co2    =         bound_layer_res_co2_tmp,
                P_kPa                  =         P_kPa,
                eair_Pa                =         eair_Pa,
                vcopt                  =         vcopt,
                m                      =         CANOPY.m,
                bprime                 =         CANOPY.bprime,
                SM                     =         SM,
                rho_mol_tmp            =         rho_mol_tmp,
                jmax_coeff             =         CANOPY.jmax_coeff,
                #switch_sm              =         switch_sm,
                ID_TOY                 =         ID_TOY,
        )
        out_photo_sun = OutPhoto(
            *photosynthesis(
                in_photo.ppfd_sun,
                in_photo.CO2conc,
                Tleaf_mat_tmp,
                in_photo.atmospheric_resistance,
                in_photo.bound_layer_res_co2,
                in_photo.P_kPa,
                in_photo.eair_Pa,
                in_photo.vcopt,
                in_photo.m,
                in_photo.bprime,
                in_photo.SM,
                in_photo.rho_mol_tmp,
                in_photo.jmax_coeff,
                #in_photo.switch_sm,
                in_photo.ID_TOY
            )
        )
        out_photo_shade = OutPhoto(
            *photosynthesis(
                in_photo.ppfd_shade,
                in_photo.CO2conc,
                Tleaf_mat_tmp,
                in_photo.atmospheric_resistance,
                in_photo.bound_layer_res_co2,
                in_photo.P_kPa,
                in_photo.eair_Pa,
                in_photo.vcopt,
                in_photo.m,
                in_photo.bprime,
                in_photo.SM,
                in_photo.rho_mol_tmp,
                in_photo.jmax_coeff,
                #in_photo.switch_sm,
                in_photo.ID_TOY
            )
        )

        in_thermal = InputThermal(
            layers                   =         int(CANOPY["layers"]),
            t_air_k                  =         tairK,
            leafremissivity          =         CANOPY["leafremissivity"],
            atmosphere_emissivity    =         atmosphere_emissivity,
            backscattering_veg       =         CANOPY["backscattering_veg"],
            thermal_atm              =         thermal_atm,
            boltzman                 =         boltzman,
            penetration_veg          =         tuple(penetration_veg),
        )
        thermal = Thermal(in_thermal)

        out_photo_sun.set_temperature_array(Tleaf_mat_tmp[:, 0])
        out_photo_shade.set_temperature_array(Tleaf_mat_tmp[:, 0])

        boundary_layer_resistance = (
            np.array(atmospheric_resistance + bound_layer_res_h2o)
        )

        for j in range(CANOPY.layers):
            l_ppfd_sun = ppfd_sun[0]
            l_ppfd_shade = ppfd_shade[0]

            l_ppfd_sun = np.array([[l_ppfd_sun]])
            l_ppfd_shade = np.array([[l_ppfd_shade]])

            l_vcopt = CANOPY["vcopt"] * 1.16
            l_vcopt = l_vcopt * np.exp(-CANOPY["kn"] * lai_cumulative)
            l_vcopt = np.array([[l_vcopt[j]]])

            l_rbh = get_rbh(
                tairK,
                ws[j],
                CANOPY["leaflength"],
                CANOPY["leafwidth"],
                P_Pa,
                rho_kg,
            )

            l_boundary_layer_resistance = np.array([boundary_layer_resistance[j]])

            in_heat = InputHeat(
                tairK                     = tairK,
                P_Pa                      = P_Pa,
                RH                        = RH[j],
                lai_layer_sun             = (LAI * CANOPY["layer_volume_ratio"] * sunFrac)[j],
                lai_layer_shade           = (LAI * CANOPY["layer_volume_ratio"] * (1 - sunFrac))[j],
                rbh                       = l_rbh,
                boundary_layer_resistance = l_boundary_layer_resistance,
                rad_tot_sun               = rad_tot_sun[j],
                rad_tot_shade             = rad_tot_shade[j],
                aerodynamic_resistance    = aereodynamic_resistance[j],
                rho_kg                    = rho_kg,
                lai_coeff = CANOPY["lai_coeff"]
            )

            # energy balance
            kwargs = {
                'photo_sun'   : out_photo_sun,
                'photo_shade' : out_photo_shade,
                'in_heat'     : in_heat,
                'thermal'     : thermal,
                'rad_tot'     : radtot[j],
                'layer'       : j,
                'mode'        : 0
                    }
            func_energy = partial(balance_leaf_temperature, **kwargs)
            t_leaf_max = tairK + CANOPY["t_leaf_delta_max"]
            t_leaf_min = tairK + CANOPY["t_leaf_delta_min"]
            leaf_temperature_ridders = ridders_arr(
                func_energy,
                t_leaf_min,
                t_leaf_max,
                Tleaf_mat_tmp[:, 0]
            )

            # get layer values for photosynthesis and heat fluxes
            photo_sun3 = out_photo_sun.dict_from_temperature(leaf_temperature_ridders, j, "psn_")
            photo_shade3 = out_photo_shade.dict_from_temperature(leaf_temperature_ridders, j, "psh_")

            _sun_Gs_m_s = photo_sun3['psn_Gs_m_s']
            _shade_Gs_m_s = photo_shade3['psh_Gs_m_s']

            res_sto_sunleaf = 1 / _sun_Gs_m_s
            res_sto_shadeleaf = 1 / _shade_Gs_m_s

            heat_sun = OutHeat(
                *get_heat_fluxes(
                    leaf_temperature_ridders,
                    in_heat.tairK,
                    in_heat.P_Pa,
                    in_heat.RH,
                    res_sto_sunleaf,
                    in_heat.lai_layer_sun,
                    in_heat.rbh,
                    in_heat.boundary_layer_resistance,
                    in_heat.rad_tot_sun,
                    in_heat.aerodynamic_resistance,
                    in_heat.rho_kg,
                    in_heat.lai_coeff
                )
            ).to_dict('hsn_')
            heat_shade = OutHeat(
                *get_heat_fluxes(
                    leaf_temperature_ridders,
                    in_heat.tairK,
                    in_heat.P_Pa,
                    in_heat.RH,
                    res_sto_shadeleaf,
                    in_heat.lai_layer_shade,
                    in_heat.rbh,
                    in_heat.boundary_layer_resistance,
                    in_heat.rad_tot_shade,
                    in_heat.aerodynamic_resistance,
                    in_heat.rho_kg,
                    in_heat.lai_coeff
                )
            ).to_dict('hsd_')
            for dt3 in [photo_sun3, photo_shade3, heat_sun, heat_shade]:
                for key, value in dt3.items():
                    if key not in out_dict:
                        out_dict[key] = []
                    out_dict[key].append(value)
            # add leaf temperature [K]
            if 'leaf_temperature' not in out_dict:
                out_dict['leaf_temperature'] = []
                out_dict['leaf_temperature_delta_air'] = []
                out_dict['energy_balance'] = []

            out_dict['leaf_temperature'].append(leaf_temperature_ridders)
            out_dict['leaf_temperature_delta_air'].append(tairK - leaf_temperature_ridders)
        out_dict['rho_mol'] = rho_mol
        for key, value in out_dict.items():
            out_dict[key] = np.array(value)
        # add thermal data to output
        out_dict.update(
                thermal.get_thermal(
                    out_dict['leaf_temperature'],
                    array_mode=True
                ).to_dict('trm_')
                )

        out_dict['energy_balance'] = radtot - out_dict['trm_net'] - out_dict['hsn_SH'] - out_dict['hsn_LH'] - out_dict['hsd_SH'] - out_dict['hsd_LH']
        return DotDict(out_dict)




    def energybalance(self):
        CANOPY = self.CANOPY
        ATM = self.ATM
        rad = self.rad
        res = self.res
        ID_TOY = self.ID_TOY

        # compute the energy balance
        ceb = self.get_energy_fluxes(self.PROJECT, CANOPY, ATM, rad, res, ID_TOY)
        obj_ceb = self.add_totals(CANOPY, rad, ceb, ID_TOY)
        return obj_ceb


def balance_leaf_temperature(
    leaf_temperature: float,
    photo_sun: OutPhoto,
    photo_shade: OutPhoto,
    in_heat: InputHeat,
    thermal: Thermal,
    rad_tot: float,
    layer: int,
    mode: bool = False,
) -> float:
    sun_Gs_m_s = photo_sun.gs_from_temperature(leaf_temperature)
    shade_Gs_m_s = photo_shade.gs_from_temperature(leaf_temperature)

    res_sto_sunleaf = 1 / sun_Gs_m_s[layer]
    res_sto_shadeleaf = 1 / shade_Gs_m_s[layer]

    #leaf_temperature = np.array([[leaf_temperature]])

    heat_sun = OutHeat(
        *get_heat_fluxes(
            leaf_temperature,
            in_heat.tairK,
            in_heat.P_Pa,
            in_heat.RH,
            res_sto_sunleaf,
            in_heat.lai_layer_sun,
            in_heat.rbh,
            in_heat.boundary_layer_resistance,
            in_heat.rad_tot_sun,
            in_heat.aerodynamic_resistance,
            in_heat.rho_kg,
            in_heat.lai_coeff
        )
    )
    heat_shade = OutHeat(
        *get_heat_fluxes(
            leaf_temperature,
            in_heat.tairK,
            in_heat.P_Pa,
            in_heat.RH,
            res_sto_shadeleaf,
            in_heat.lai_layer_shade,
            in_heat.rbh,
            in_heat.boundary_layer_resistance,
            in_heat.rad_tot_shade,
            in_heat.aerodynamic_resistance,
            in_heat.rho_kg,
            in_heat.lai_coeff
        )
    )

    #leaf_temperature_arr = np.ones(5) * leaf_temperature[0]
    if not mode:
        leaf_temperature_arr = np.ones(5) * leaf_temperature
    else:
        leaf_temperature_arr = leaf_temperature
    out_thermal = thermal.get_thermal(leaf_temperature_arr, array_mode=mode)

    if mode == 0:
        trm = float(out_thermal.net[layer])
        sh = float(heat_sun.SH + heat_shade.SH)
        lh = float(heat_sun.LH + heat_shade.LH)
        balance = float(
            rad_tot - trm - sh - lh
        )
        return balance

    return OutEnergy(photo_sun, photo_shade, heat_sun, heat_shade, out_thermal)


@lru_cache(maxsize=3000)
def get_penetration_veg(canopy_x, lai):
    fun = lambda y: np.exp(
        -(lai * (canopy_x**2 - 1 + 1 / y) ** 0.5)
        / (canopy_x + 1.774 * (canopy_x + 1.182) ** -0.733)
    )
    return integrate.quad(fun, 0, 1)


def to_same_sign(x, y):
    ''' return value with the same magnitude as x but with the sign of y'''
    sign_y = y > 0
    sign_x = x > 0
    if sign_y == sign_x:
        return x
    else:
        return -x


def find_closest(arr, val):
    return arr[np.abs(arr - val).argmin()]


def ridders_arr(func, xl, xh, arr, xacc=50):
    """
    Compute the root of a given function within a specified interval, using Ridders' formula.
    the formual is modified to return a value contained in the input array (arr)

    Parameters:
    - func (function): The function for which the root needs to be found.
    - xl (float): The lower bound of the interval.
    - xh (float): The upper bound of the interval.
    - arr (list): The array of values to search for the closest value in.
    - xacc (float): The desired accuracy of the root (default value is 50).

    Returns:
    - ans (float): The improved root x.

    Raises:
    - ValueError: If the root is not bracketed within the interval.
    - ValueError: If the maximum number of iterations is exceeded.
    """




    # Compute the improved root x from Ridders' formula (pp. 477)
    fl = func(xl)
    fh = func(xh)

    ans = -9.99e99
    if fl * fh < 0:
        for _ in range(30):
            xm = 0.5 * (xl + xh)
            xm_arr = find_closest(arr, xm)   # point to value in array
            fm = func(xm_arr)
            s = math.sqrt(fm**2 - fl * fh)
            sign = 1
            if fl < fh:
                sign = -1
            xnew=xm + (xm - xl) * (sign * fm / s)
            if (abs(xnew-ans) <= xacc):
                return ans
            ans = xnew
            ans_arr = find_closest(arr, ans)  # point to value in array
            fnew = func(ans_arr)
            #if fnew == 0.0:
            if abs(fnew) < xacc:
                return ans
            if to_same_sign(fm, fnew) != fm:
                xl = xm
                fl = fm
                xh = ans
                fh = fnew
            elif to_same_sign(fl, fnew) != fl:
                xh = ans
                fh = fnew
            elif to_same_sign(fh, fnew) != fh:
                xl = ans
                fl = fnew
            else:
                raise ValueError("never get here")
            if abs(xh - xl) <= xacc:
                return ans
        raise ValueError("Maximum number of iterations exceeded in rtflsp")
    else:
        if fl == 0.0:
            return xl
        if fh == 0.0:
            return xh
        # raise ValueError("root must be bracketed in rtflsp")
        # get best solution
        if abs(fl) < abs(fh):
            return xl
        return xh

    return None


def get_saturation_vapor_pressure(temperature):
    "return Saturation vapor pressure,  Pa"

    nn1 = np.log(temperature) * 5.02808
    nnn = 54.8781919 - (6790.4985 / temperature) - nn1
    fT = 100 * np.exp(nnn)
    return fT


def get_boltzmann(rate, eakin, topt, tl):
    "return Boltzmann temperature distribution for photosynthesis"

    Rstar = 8.3144
    hkin = 200000.0  # % enthalpy term, J mol-1

    dtlopt = tl - topt
    prodt = Rstar * topt * tl

    numm = rate * hkin * np.exp(eakin * dtlopt / prodt)
    denom = hkin - eakin * (1.0 - np.exp(hkin * (dtlopt) / (prodt)))

    fT = np.divide(numm, denom)

    return fT


def get_arhennius_temperature(rate, eact, tprime, tref, t_lk):
    "Arhennius temperature function"

    num = tprime * eact
    den = tref * 8.3144 * t_lk
    ft = rate * np.exp(num / den)
    return ft



# def ridder(func, xl, xh, xacc=50):
#     # Compute the improved root x from Ridders' formula (pp. 477)
#     fl = func(xl)
#     fh = func(xh)
#
#     ans = -9.99e99
#     if fl * fh < 0:
#         for _ in range(30):
#             xm = 0.5 * (xl + xh)
#             fm = func(xm)
#             s = math.sqrt(fm**2 - fl * fh)
#             sign = 1
#             if fl < fh:
#                 sign = -1
#             xnew=xm + (xm - xl) * (sign * fm / s)
#             if (abs(xnew-ans) <= xacc):
#                 return ans
#             ans = xnew
#             fnew = func(ans)
#             #if fnew == 0.0:
#             if abs(fnew) < xacc:
#                 return ans
#             if to_same_sign(fm, fnew) != fm:
#                 xl = xm
#                 fl = fm
#                 xh = ans
#                 fh = fnew
#             elif to_same_sign(fl, fnew) != fl:
#                 xh = ans
#                 fh = fnew
#             elif to_same_sign(fh, fnew) != fh:
#                 xl = ans
#                 fl = fnew
#             else:
#                 raise ValueError("never get here")
#             if abs(xh - xl) <= xacc:
#                 return ans
#         raise ValueError("Maximum number of iterations exceeded in rtflsp")
#     else:
#         if fl == 0.0:
#             return xl
#         if fh == 0.0:
#             return xh
#         # raise ValueError("root must be bracketed in rtflsp")
#         # get best solution
#         if abs(fl) < abs(fh):
#             return xl
#         return xh
#
#     return None
