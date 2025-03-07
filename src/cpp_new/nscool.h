/*
  -------------------------------------------------------------------

  Copyright (C) 2015-2020, Andrew W. Steiner and Satyajit Roy

  This file is part of nscool.

  nscool is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.

  nscool is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with O2scl. If not, see <http://www.gnu.org/licenses/>.

  -------------------------------------------------------------------
*/
#ifndef NSCOOL_H
#define NSCOOL_H

#include <o2scl/interp.h>
#include <vector>

#include <boost/numeric/ublas/vector.hpp>

#include <o2scl/eos_had_base.h>
#include <o2scl/fermion_deriv_rel.h>
#include <o2scl/hdf_file.h>
#include <o2scl/hdf_io.h>
#include <o2scl/interp.h>
#include <o2scl/interp2_direct.h>
#include <o2scl/lib_settings.h>
#include <o2scl/nstar_cold.h>
#include <o2scl/permutation.h>
#include <o2scl/table.h>

#include "tc.h"

// Ublas vector typedef
typedef boost::numeric::ublas::vector<double> ubvector;

/** \brief The HHJ parameterized EOS
 */
class eos_had_hhj : public o2scl::eos_had_eden_base {

public:
  double s;
  double gamma;
  double eps0;
  double S0;
  double n0;

  /** \brief The coefficient of the pressure modification
   */
  double delta_P_K;

  /** \brief The exponent of the pressure modification
   */
  double delta_P_Gamma;

  eos_had_hhj() {
    s = 0.1;
    gamma = 0.7;
    eps0 = 15.8 / o2scl_const::hc_mev_fm;
    S0 = 32.0 / o2scl_const::hc_mev_fm;
    n0 = 0.16;
    delta_P_K = 0.0;
    delta_P_Gamma = 0.0;
  }

  virtual int calc_temp_f_gen(double nB, double nQ, double nS, double T,
                              o2scl::thermo &th) {
    std::cerr << "Unimplemented." << std::endl;
    exit(-1);
    return 0;
  }
  /** \brief Equation of state as a function of density
   */
  virtual int calc_e(o2scl::fermion &ln, o2scl::fermion &lp,
                     o2scl::thermo &lth) {

    double barn = ln.n + lp.n;

    double xp;
    if (barn <= 0.0) {
      xp = 0.0;
      lth.ed = 0.0;
      ln.mu = 0.0;
      lp.mu = 0.0;
      lth.pr = 0.0;
      return 0;
    } else {
      xp = lp.n / barn;
    }

    double u = barn / n0;
    double sym = S0 * pow(u, gamma);
    double symp = S0 * gamma * pow(u, gamma - 1.0);

    lth.ed = ln.m * ln.n + lp.m * lp.n +
             barn * (eps0 * u * (u - 2.0 - s) / (1.0 + s * u) +
                     sym * pow((1.0 - 2.0 * xp), 2.0));
    ln.mu =
        ln.m + (lth.ed - ln.m * ln.n - lp.m * lp.n) / barn +
        u * (eps0 * (u - 2.0 - s) / (1.0 + s * u) + eps0 * u / (1.0 + s * u) -
             eps0 * u * (u - 2.0 - s) * s / pow(1.0 + s * u, 2.0) +
             pow(1 - 2.0 * xp, 2.0) * symp) +
        xp * 4.0 * (1 - 2.0 * xp) * sym;
    lp.mu = lp.m + ln.mu - ln.m - 4.0 * (1.0 - 2.0 * xp) * sym;
    lth.pr = -lth.ed + ln.mu * ln.n + lp.mu * lp.n;

    ln.kf = cbrt(3.0 * o2scl_const::pi2 * ln.n);
    lp.kf = cbrt(3.0 * o2scl_const::pi2 * lp.n);

    if (delta_P_Gamma > 0.0 && lth.ed > 1.5) {
      lth.pr += (delta_P_K * pow(lth.ed, delta_P_Gamma) -
                 delta_P_K * pow(1.5, delta_P_Gamma));
    }

    return 0;
  }
};

/** \brief Base wrapper for neutron star cooling

    <b>SVN benchmark commits:
    2160, 2552, 4036

    <b>General notes</b>

    The purpose for this (somewhat obtuse) wrapper of Dany's Fortran
    NSCool code is to allow the use of O2scl EOS routines and to
    implement some MPI calls while still allowing C++ extensions of
    the original functionality with inheritance and virtual functions.

    Basic usage is to instantiate the class and set the global
    pointer list <tt>nscool_ptrs</tt>, change the input EOS
    or structure tables, and then call \ref run() .

    <hr>
    <b>Grid definition</b>

    The temperature is defined at odd indices and the luminosities at
    even indices. The values \c rhocore, \c rhodrip, and \c rhomax are
    the densities at the indexes defined by \c icore, \c idrip, and \c
    imax. The value \c rhosurf defines the surface density. The
    parameter \c rhoenv defines the envelope. If \c rhoenv is smaller
    than \c rhosurf, then the envelope is ignored. The parameter \c
    icore defines how many zones will be in the core, and \c idec
    gives the number of points per decade in density in the crust.

    <hr>
    <b>Pairing specification</b>

    Parameter:
    - sfn1s0: neutron 1S0 gap model to be used
    - sfn3p2: neutron 3P2 gap model to be used
    - sfp1s0: proton 1S0 gap model to be used
    - sfl1s0: lambda hyperon (\f$ \Lambda \f$) 1S0 gap model to be used
    - fn1s0: scaling factor for neutron 1S0 gap
    - fn3p2: scaling factor for neutron 3P2 gap
    - fp1s0: scaling factor for proton 1S0 gap
    - fl1s0: scaling factor for lambda hyperon (\f$ \Lambda \f$) 1S0 gap

    sfn1s0:
    - 1: SFB
    - 2: CCDK
    - 3: WAP
    - 4: GC
    - 5: GIPSF
    - 201: Ioffe 1NS
    - 202: Ioffe 2NS
    - 203: Ioffe 3NS
    - 150: Three-parameter Gaussian

    sfn3p2:
    - 1: HGRR
    - 2: AO
    - 3: AO M1
    - 4: T72
    - 5: T72 M1
    - 6: BCLL92
    - 7: EEHJO96 NR
    - 8: EEHJO96 R
    - 101: Gap "a"
    - 102: Gap "b"
    - 103: Gap "c"
    - 150: three-parameter Gaussian
    - 201: Ioffe 1NT
    - 202: Ioffe 2NT
    - 203: Ioffe 3NT
    - >1000: Uniform

    sfp1s0:
    - 1: CCY MS
    - 2: CCY PS
    - 3: T73
    - 4: NS
    - 5: AO
    - 6: BCLL92
    - 7: CCDK
    - 21: T72
    - 22: AWP 2
    - 23: AWP 3
    - 201: Ioffe 1P
    - 202: Ioffe 2P
    - 203: Ioffe 3P
    - >1000: Uniform
    - 150: three-parameter Gaussian

    <hr>
    <b>Core conductivity</b>

    \verbatim embed:rst
    Based on [Baiko01tc]_, [Baym69si]_, [Gnedin95tc]_,
    [Shternin07eh]_, and [Flowers81tp]_.
    \endverbatim

    <hr>
    <b>Crust conductivity</b>

    \verbatim embed:rst
    Based on [Shternin06et]_, [Baiko95ta]_, [Potekhin99tp]_, ...
    \endverbatim

    <hr>
    <b>Opacities</b>

    The parameter \c iopacity gives no photon opacity and a value of 1
    gives normal photon opacity. The parameter \c Q_imp gives the
    impurity parameter for electron-impurity scattering.

    The parameter \c icon_core takes a value of 1 for the
    simple Flowers and Itoh formula
    \f[
    \lambda = 10^{23}
    \left( \frac{k_{F,n}}{1.6~\mathrm{fm}^{-1}} \right)
    \left( \frac{T}{10^8~\mathrm{K}} \right)
    \f]
    and a value of 2 uses the full calculation of Yakovlev et al.

    ICON_CRUST :

    This will essentially distinguish between the Itoh et al. and
    Yakovlev et al. calculations, both in the liquid and the crystal
    phases:

    Gamma > Gammac:
    1: e-phonon from Itoh et al. + e-impurity from Yakovlev & Urpin.
    2: e-phonon from Baiko & Yakovlev + e-impurity from Yakovlev & Urpin.
    3: e-phonon from Gnedin et al. (2001: appendix) + e-impurity from
    Yakovlev & Urpin.

    Gamma < Gammal:
    1: e-ion from Itoh et al.
    2: e-ion from Itoh et al.
    3: e-ion from Gnedin et al. (2001: appendix). Gammal <Gamma<Gammac:

    interpolate between the two previous cases (to avoid a
    discontinuity in \f$ \lambda \f$ in cases 1 & 2). [If you set
    Gammal = Gammac then, of course, there will be no interpolation !]

    If rho < 107 g cm-3 (“envelope”): none of the above, just use
    Potekhin et al. (1999). After all this the e-e scattering
    contribution (\ref Shternin06) is added.

    Gammac = gammacryst > gammaliq = Gammal are defined in the
    included file gamma_limits.inc.f

    <hr>
    <b>Effective masses</b>

    Dany's original version supported several different models for the
    core nucleon effective masses from the literature. In this
    version, the effective mass in the core must be provided in the
    core composition table. The neutron effective mass in the crust is
    determined by a simple function of the neutron Fermi momentum.

    <hr>
*/
class nscool {

public:
  /** \brief Object for computing critical temperatures
   */
  tcx atc;

  /** \brief Interpolation object
   */
  o2scl::interp_vec<std::vector<double>> iv;
  o2scl::interp2_direct<> id2;

  /** \brief Parameter for the envelope composition
   */
  double eta;

  /** \name Superfluid parameters
   */
  //@{
  /// Default 1 (SFB)
  int sfn1s0;
  /// Default 101 (Minimal gap "a")
  int sfn3p2;
  /// Default 3 (T73)
  int sfp1s0;
  /// Maximum critical temperature of neutron triplet superfluid
  double n3_tc;
  /// Fermi momentum at peak for neutron triplet superfluid
  double n3_kf;
  /// Fermi momentum width parameter for neutron triplet superfluid
  double n3_dk;
  /// Maximum critical temperature of proton singlet superfluid
  double p1_tc;
  /// Fermi momentum at peak for proton singlet superfluid
  double p1_kf;
  /// Fermi momentum width parameter for proton singlet superfluid
  double p1_dk;
  /// Maximum critical temperature of neutron singlet superfluid
  double n1_tc;
  /// Fermi momentum at peak for neutron singlet superfluid
  double n1_kf;
  /// Fermi momentum width parameter for neutron singlet superfluid
  double n1_dk;
  //@}

  /** \brief The NSCool debug parameter
   */
  int nscool_debug;

  /** \brief Fix the direct Urca process at a specified density
   */
  double fix_durca;
  /** \brief Direct Urca modulation parameter
   */
  double alpha_durca;
  /** \brief Direct Urca modulation parameter
   */
  double beta_durca;

  /// \name Main cooling curve output
  //@{
  /** \brief Time
   */
  std::vector<double> v_time;

  /** \brief Temperature
   */
  std::vector<double> v_tptr;

  /** \brief Photon luminosity
   */
  std::vector<double> v_lphot;

  /** \brief Neutrino luminosity
   */
  std::vector<double> v_lneut;

  /** \brief Heating
   */
  std::vector<double> v_lheat;
  //@}

  /// \name Data for the pair brehmsstrahlung rate
  //@{
  double pb_n2[56], pb_logt[56], pb_nalpha[56];
  std::vector<double> pb_logt1, pb_nalpha1, pb_n21;
  //@}

  /// \name Data for superfluid suppression factor
  //@{
  double sf_lgtau1[35], sf_lgtau2[35], sf_lgr[1225], sf_lgr2[1225];
  ubvector lgtau1_t, lgtau2_t;
  ubmatrix lgr_tt;
  //@}

  /// \name Hydrostatic input
  //@{
  /** \brief Crust EOS table

      This table should include (at least) the following columns:
      - <tt>"rho"</tt>: the energy density in units of
      \f$ \mathrm{g}/\mathrm{cm}^3 \f$
      - <tt>"P"</tt>: the pressure in units of
      \f$ \mathrm{dyne}/\mathrm{cm}^2 \f$
      - <tt>"n"</tt>: baryon number density in units of
      \f$ \mathrm{fm}^{-3} \f$
      - <tt>"A_cell"</tt>: Total number of nucleons in the W-S cell
      - <tt>"A_ion"</tt>: Total number of nucleons inside the nucleus
      - <tt>"Z"</tt>: Total number of protons inside the nucleus

      The table must be ordered so that the first row has the highest
      density and pressure and the last row has the smallest density
      and pressure. The table is limited to a maximum number of 500
      lines.

      This table is copied to the fortran arrays by \ref
      crust_comp() and \ref crust_eos() .

      The default crust is in <tt>crust_HZD_NV.o2</tt> .
  */
  o2scl::table<> nscool_crust;

  /** \brief Core EOS table

      This table should include (at least) the following columns:
      - <tt>"Rho"</tt>: the energy density in units of
      \f$ \mathrm{g}/\mathrm{cm}^3 \f$
      - <tt>"nbar"</tt>: baryon number density in units of
      \f$ \mathrm{fm}^{-3} \f$
      - <tt>"Ye"</tt>: Number of electrons per baryon
      - <tt>"Ymu"</tt>: Number of muons per baryon
      - <tt>"Yn"</tt>: Number of neutrons per baryon
      - <tt>"Yp"</tt>: Number of protons per baryon
      - <tt>"Yla"</tt>: Number of Lambda hyperons per baryon
      - <tt>"Ysm"</tt>: Number of Sigma minus hyperons per baryon
      - <tt>"Ys0"</tt>: Number of Sigma zero hyperons per baryon
      - <tt>"Ysp"</tt>: Number of Sigma plus hyperons per baryon
      - <tt>"mstp"</tt>: Proton reduced effective mass
      - <tt>"mstn"</tt>: Neutron reduced effective mass
      - <tt>"mstla"</tt>: Lambda hyperon reduced effective mass
      - <tt>"mstsm"</tt>: Sigma minus hyperon reduced effective mass
      - <tt>"msts0"</tt>: Sigma zero hyperon reduced effective mass
      - <tt>"mstsp"</tt>: Sigma plus hyperon reduced effective mass

      The table must go from high energy densities to low energy
      densities. This table is copied to the fortran arrays by \ref
      core_comp() .

      In Dany's original EOS files, the core EOS has a second colum
      for the pressure in units of \f$ \mathrm{dyne}/\mathrm{cm}^2 \f$
      which is not used by the code.

      Hyperons support is in progress at the moemnt.

      The default core EOS is in <tt>core_APR.o2</tt> .
  */
  o2scl::table<> nscool_core;

  /** \brief Stellar profile table

      This table should include (at least) the following columns:
      - <tt>"radius"</tt>: the radial coordinate (in \f$ \mathrm{m} \f$ )
      - <tt>"n_baryon"</tt>: baryon number density in units of
      \f$ \mathrm{fm}^{-3} \f$
      - <tt>"density"</tt>: the energy density in units of
      \f$ \mathrm{g}/\mathrm{cm}^3 \f$
      - <tt>"pressure"</tt>: the energy density in units of
      \f$ \mathrm{dyne}/\mathrm{cm}^2 \f$
      - <tt>"emass"</tt>: the enclosed gravitational mass in units of
      \f$ \mathrm{M}_{\odot} \f$
      - <tt>"phi"</tt>: the gravitational potential

      The table must be ordered from lower radii (the core) to larger
      radii (the surface). This table is used in \ref star_struct() .

      Dany's original tables had the first column labeled "step"
      and a final column labeled "bmass" that were ignored by
      the code.

      The default stellar profile is in <tt>tov_APR_14.o2</tt> .
  */
  o2scl::table<> nscool_tov;
  //@}

  /** \brief Output temperature and luminosity profiles
   */
  o2scl::table3d tl_prof;

  /** \brief Times to output

      This defaults to a 25-point grid which gives 2 time points every
      decade at late times.
  */
  std::vector<double> time_print;

  /// Desc
  std::vector<double> temp_ygrid;

  /** \brief Initial temperature at drip density relative to
      initial core temperature (default 0.8)
  */
  double T_fact_drip;

  /** \brief Initial temperature at surface relative to
      initial core temperature (default 0.5)
  */
  double T_fact_surf;

  /** \brief Flag for profile output (default 0.0)
   */
  double ptemp;

  /** Brief Electron
   */
  o2scl::fermion electron;

  /** Brief Electron with derivatives
   */
  o2scl::fermion_deriv electron_deriv;

  /** Brief Muon
   */
  o2scl::fermion muon;

  /** Brief Fermion thermodynamics
   */
  o2scl::fermion_rel fe;

  /** Brief Fermion thermodynamics with derivatives
   */
  o2scl::fermion_deriv_rel fdr;

  /** \brief Desc
   */
  bool sxrt_mode;

  /** \brief Desc
   */
  double sxrt_Tinit;

  size_t mdot_index;

  double gammacryst, gammaliq;

  nscool(std::string dir);

  double r_1s0_3p2B(double t1, double t2);

  /** \brief Desc
   */
  double cvelec(double t, double rho, double a, double z);

  /** \brief "Ion" specific heat?
   */
  double cvion(double t, double rho, double a, double z);

  /** \brief Total specific Heat
   */
  void cvtot(int i, double t, double rho, double aion, double zion, double &cv,
             double &cvneutron, double &cvproton, double &cvelectron,
             double &cvmuon, double &cvlambda, double &cvsminus,
             double &cvszero, double &cvsplus, double &cvquark, double &cvions,
             std::vector<double> &cve, std::vector<double> &cvm,
             std::vector<double> &cvn, std::vector<double> &cvp,
             std::vector<double> &cvla, std::vector<double> &cvsm,
             std::vector<double> &cvs0, std::vector<double> &cvsp,
             std::vector<double> &cvqu, std::vector<double> &dvqd,
             std::vector<double> &cvqs, double rhodrip, double rhocore,
             std::vector<double> &fhad, int istrange, std::vector<double> &tcn,
             std::vector<double> &tcp, std::vector<double> &tcla,
             std::vector<double> &tcsm, std::vector<double> &tcs0,
             std::vector<double> &tcsp, int isf);

  /** \brief Pressure in the crust
   */
  double pressure(double T, double Rho, double A, double Z);

  /** \brief Desc
   */
  void density(double T, double P, double A, double Z, double &Rho);

  /** \brief Desc
   */
  void opacity(double T, double Rho, double &A, double &Z, double &kappa,
               int iopacity);

  /** \brief con_core_bar_pairing_supr() - from subroutine of same name
   */
  void con_core_bar_pairing_supr(double yn, double yp, double &Rn1, double &Rn2,
                                 double &Rp1, double &Rp2, double &RC);

  /** \brief con_core_bar() - from subroutine con_core_bar(...)
   * This calculates the neutron thermal conductivity in the core from Baiko,
   * Haensel & Yakovlev, 2001, etc.
   */
  void con_core_bar(double Temp, double kf_e, double kf_mu, double kf_p,
                    double mst_p, double Tc_p, double kf_n, double mst_n,
                    double Tc_n, int isfn, double kf_la, double mst_la,
                    double Tc_la, double kf_sm, double mst_sm, double Tc_sm,
                    double kf_s0, double mst_s0, double Tc_s0, double kf_sp,
                    double mst_sp, double Tc_sp, double &sigma_bar,
                    double &lambda_bar, double debug, double &nu_e_s,
                    double &nu_e_l, int icontrol);

  /** \brief con_core_lep() - from subroutine con_core_lep(...)
   * This calculates thermal conductivity in the core from Shternin & Yakovlev,
   * PRD75, 103004, 2007 (electrons+muons).
   */
  void con_core_lep(double Temp, double kf_e, double kf_m, double kf_p,
                    double mst_p0, double Tc_p, double kf_sm, double mst_sm0,
                    double Tc_sm, double kf_sp, double mst_sp0, double Tc_sp,
                    double &sigma_lep, double &lambda_e, double &lambda_m,
                    double debug, double &nu_e_s, double &nu_e_l);

  /** \brief con_core() - from subroutine con_core(...)
   */
  void con_core(int icon_core, double debug, double Temp, double kf_e,
                double kf_mu, double kf_p, double mst_p, double Tc_p, // proton
                double kf_n, double mst_n, double Tc_n, int isfn,     // neutron
                double kf_la, double mst_la, double Tc_la,            // lambda
                double kf_sm, double mst_sm, double Tc_sm,            // sigma-
                double kf_s0, double mst_s0, double Tc_s0,            // sigma0
                double kf_sp, double mst_sp, double Tc_sp,            // sigma+
                double f_had, double &sigma, double &lambda, double &nu_e_s,
                double &nu_e_l);

  // A stub for con_crust_e_phonon_Itoh(...) which is "removed" in Fortran
  void con_crust_e_phonon_Itoh(double /*T*/, double /*rho*/, double /*A*/,
                               double /*A1*/, double /*Z*/, double &sigma_ph,
                               double &lambda_ph, double /*debug*/,
                               double &nu_e_s_ph, double &nu_e_l_ph);

  // Baiko & Yakovlev e-phonon conduction
  void con_crust_e_phonon_BY(double Temp, double rho, double A, double A1,
                             double Z, double &sigma_, double &lambda_,
                             double debug, double &nu_e_s, double &nu_e_l,
                             double rhodrip);

  // Potekhin, Baiko, Haensel & Yakovlev code:
  void con_env_e_phon_ion_PBHY(double T, double rho, double A, double A1,
                               double Z, double &sigma, double &lambda,
                               double debug, double &nu_e_s, double &nu_e_l);

  // OYAFORM(...)
  void OYAFORM(double BARD, int Index, double &Z, double &Anuc, double &A,
               double &xnuc, double &xnuct);

  // Gnedin, Yakovlev & Potekhin modifications:
  void con_e_phon_ion_GYP(double T, double rho, double A_in, double A1_in,
                          double Z_in, int ifs, double &sigma, double &lambda,
                          double debug, double &nu_e_s, double &nu_e_l,
                          double rhodrip);

  // Itoh etc. e-ion scattering in liquid
  void con_crust_e_ion_Itoh(double T, double rho, double A, double A1, double Z,
                            double &sigma, double &lambda, double &debug,
                            double &nu_e_s, double &nu_e_l);

  // e-impurity scattering from Yakovlev & UrPIn
  void con_crust_e_imp_YU(double T, double rho, double A, double A1, double Z,
                          double Q_imp, double &sigma, double &lambda,
                          double debug, double &nu_e_s, double &nu_e_l);

  // Electron-electron scattering => con_crust_ee
  void con_crust_ee(double T, double ne, double kfe, double &lambda,
                    double debug);

  // ------------------------------------------------------------------------
  // con_crust(...)
  // ------------------------------------------------------------------------
  void con_crust(int icon_crust, double debug, double T, double rho, double kfe,
                 double A, double A1, double Z, double Q_imp, double &sigma,
                 double &lambda, double &nu_e_s, double &nu_e_l,
                 double rhodrip);

  /**
   * @brief  Conductivity driver routine (translated from Fortran subroutine
   * `conduct`).
   *
   * @param i             Index into large arrays (like kfe[i], kfm[i], etc.).
   * @param T             Temperature [K].
   * @param rho           Density [g/cm^3].
   * @param A             "a" (nucleons per cell) from the Fortran code.
   * @param A1            "a1" (nucleons per nucleus) from the Fortran code.
   * @param Z             Proton number.
   * @param Q             Impurity parameter.
   * @param magfield      Magnetic field (unused here, but preserved).
   * @param[out] sigma    Returned electrical conductivity [cgs].
   * @param[out] lambda   Returned thermal conductivity [cgs-K].
   * @param debug         Debug level (if >=2, we print entering/exiting
   * messages).
   * @param[out] nu_e_s   e- scattering frequency (longitudinal, say).
   * @param[out] nu_e_l   e- scattering frequency (transverse, say).
   * @param icon_crust    Which crust model to use (1,2,3, etc.).
   * @param icon_core     Which core model to use (1,2, etc.).
   * @param rhodrip       Drip density threshold [g/cm^3].
   * @param rhocore       Core density threshold [g/cm^3].
   * @param kfe,kfm,kfn,kfp,kfla,kfsm,kfs0,kfsp  Large arrays with Fermi
   * momenta, etc.
   * @param mstn,mstp,mstla,mstsm,msts0,mstsp    Arrays with effective masses
   * (or similar).
   * @param tcn,tcp,tcla,tcsm,tcs0,tcsp         Arrays with critical
   * temperatures for pairing.
   * @param isf           Index dividing 1S0 vs. 3P2 neutron superfluid (in
   * original code).
   * @param fhad          Array with hadronic fraction (fhad[i]).
   * @param[in,out] istrange  On input used to check if matter is strange or
   * not. The Fortran sets `istrange=0` in this subroutine. We'll replicate that
   * behavior.
   *
   * The original Fortran sets `istrange=0` within the code, then chooses
   * between `con_core(...)` or `con_crust(...)`. If `rho >= rhocore`, we assume
   * "core" conductivity; otherwise "crust."
   *
   * This function calls external functions `con_core(...)` and `con_crust(...)`
   * which must be defined in your codebase (C++ equivalents of the original
   * Fortran).
   */
  void conduct(int i, double T, double rho, double A, double A1, double Z,
               double Q, double magfield, double &sigma, double &lambda,
               double debug, double &nu_e_s, double &nu_e_l, int icon_crust,
               int icon_core, double rhodrip, double rhocore,
               const std::vector<double> &kfe, const std::vector<double> &kfm,
               const std::vector<double> &kfn, const std::vector<double> &kfp,
               const std::vector<double> &kfla, const std::vector<double> &kfsm,
               const std::vector<double> &kfs0, const std::vector<double> &kfsp,
               const std::vector<double> &mstn, const std::vector<double> &mstp,
               const std::vector<double> &mstla,
               const std::vector<double> &mstsm,
               const std::vector<double> &msts0,
               const std::vector<double> &mstsp, const std::vector<double> &tcn,
               const std::vector<double> &tcp, const std::vector<double> &tcla,
               const std::vector<double> &tcsm, const std::vector<double> &tcs0,
               const std::vector<double> &tcsp, int isf,
               const std::vector<double> &fhad, int &istrange);

  // ------------------------------------------------------------------
  // numurca_nucl(...) => sets qmurca_nucl
  // ------------------------------------------------------------------
  void
  numurca_nucl(int i, double t, double &qmurca_nucl,
               const std::vector<double> &tcn, const std::vector<double> &tcp,
               int isf, const std::vector<double> &mstn,
               const std::vector<double> &mstp, const std::vector<double> &kfe,
               const std::vector<double> &kfm, const std::vector<double> &kfn,
               const std::vector<double> &kfp);

  // ------------------------------------------------------------------
  // numurca_q(...) => sets qmurca_q
  // ------------------------------------------------------------------
  void numurca_q(int i, double t, double rho, double &qmurca_q,
                 const std::vector<double> &kfqu,
                 const std::vector<double> &tcu,
                 const std::vector<double> &tcd);

  void numurca_hyp(int i, double t, double &qmurca_hyp);

  //@{
  /** \brief nubrem_crust_nn(...)
   */
  void nubrem_crust_nn(int i, double t, double vion, double &qbrem_nn,
                       const std::vector<double> &tcn, int isf,
                       const std::vector<double> &kfn,
                       const std::vector<double> &mstn);

  // ------------------------------------------------------------------
  // nubrem_nucl(...) => sets qbrem_nucl
  // ------------------------------------------------------------------
  void nubrem_nucl(int i, double t, double &qbrem_nucl,
                   const std::vector<double> &tcn,
                   const std::vector<double> &tcp, int isf,
                   const std::vector<double> &kfn,
                   const std::vector<double> &kfp,
                   const std::vector<double> &mstn,
                   const std::vector<double> &mstp);

  // ------------------------------------------------------------------
  // 1) neebrem(...) => calculates electron-electron neutrino pair
  // bremsstrahlung
  // ------------------------------------------------------------------
  void neebrem(double T, double mu, double &qeebrem);

  // ------------------------------------------------------------------
  // numurca_hyp(...), nubrem_hyp(...): stubs returning 0
  // ------------------------------------------------------------------
  void nubrem_hyp(int i, double t, double &qbrem_hyp);

  /** \brief \f$ n \rightarrow p+ \ell + {\bar{\nu_{\ell}}} \f$ Urca
    emissivity in \f$ \mathrm{erg}/\mathrm{cm}^3/\mathrm{s} \f$
  */
  double emissivity_durca_npl(double kfn, double kfp, double kfe, double ne,
                              double kfmu, double nmu, double rmn, double rmp,
                              double TK, double Tcn1s0, double Tcp1s0,
                              double Tcn3p2);

  // ------------------------------------------------------------------
  // Nudurca_q(...) => sets qdurca_q
  // ------------------------------------------------------------------
  void
  nudurca_q(int i, double t, double rho, double &qdurca_q,
            const std::vector<double> &tcu1, const std::vector<double> &tcu2,
            const std::vector<double> &tcu3, const std::vector<double> &tcd1,
            const std::vector<double> &tcd2, const std::vector<double> &tcd3,
            const std::vector<double> &tcs1, const std::vector<double> &tcs2,
            const std::vector<double> &tcs3, const std::vector<double> &kfe,
            const std::vector<double> &kfm, const std::vector<double> &kfqu,
            const std::vector<double> &kfqd, const std::vector<double> &kfqs,
            const std::vector<int> &idurca_quqd,
            const std::vector<int> &idurca_quqs);

  // ------------------------------------------------------------------
  // nudurca_h(...): Large direct Urca with hyperons
  // ------------------------------------------------------------------
  void nudurca_h(
      int i, double t, double rho, double &qdurca_np, double &qdurca_lap,
      double &qdurca_smn, double &qdurca_smla, double &qdurca_sms0,
      const std::vector<double> &tcn, const std::vector<double> &tcp,
      const std::vector<double> &tcla, int isf, const std::vector<double> &bar,
      const std::vector<double> &yelect, const std::vector<double> &ymuon,
      const std::vector<double> &mstp, const std::vector<double> &mstn,
      const std::vector<double> &mstla, const std::vector<double> &mstsm,
      const std::vector<double> &msts0, const std::vector<double> &mstsp,
      const std::vector<double> &durca_ctrl_e,
      const std::vector<double> &durca_ctrl_m,
      const std::vector<int> &idurca_lap, const std::vector<int> &idurca_smla,
      const std::vector<int> &idurca_smn, const std::vector<int> &idurca_sms0,
      const std::vector<int> &idurca_np);

  /** \brief Desc
   */
  void nu_1s0_pbf(double T, double Tc, double mst, double kf,
                  double &q_1s0_pbf);

  void nu_n3p2_B_pbf(double T, double Tc, double mst, double kf,
                     double &q_n3p2_pbf);

  /** \brief the energy loss rate per cubic centimeter in the crust
    of a neutron star from neutrino pair bremsstrahlung.

    From Kaminker et al, A&A 343 (1999), p. 1009, Equ. (40)
  */
  void nufast(int i, double t, double rho, double &qfast,
              const std::vector<double> &tcn, const std::vector<double> &tcp,
              int isf, const std::vector<double> &bar,
              const std::vector<double> &theta_k,
              const std::vector<double> &theta_p,
              const std::vector<double> &yelect, double rhoexo, double cexo,
              double pexo, const std::vector<double> &mstn,
              const std::vector<double> &mstp, const std::vector<double> &kfe);

  void npb_new(double temp, double rho, double &qnpb);

  /** \brief the energy loss rate per cubic centimeter in the crust
   * of a neutron star from neutrino pair bremsstrahlung.
   * checked on February 27, 1996 against figures of Itoh et al 1996
   */
  void npb(double t, double rho, double a, double z, double &qnpb);

  // ------------------------------------------------------------------
  // 4) npbpde(...) => partially degenerate electron neutrino pair brem
  // ------------------------------------------------------------------
  void npbpde(double t, double rho, double a, double z, double &qnpbpde);

  // ------------------------------------------------------------------
  // 5) npbl(...) => subroutine with data file => stub
  // ------------------------------------------------------------------
  void npbl(double t, double rho, double a, double z, double &qnpbl);

  // ------------------------------------------------------------------
  // 6) npbc(...) => subroutine with data file => stub
  // ------------------------------------------------------------------
  void npbc(double t, double rho, double a, double z, double &qnpbc);

  // ------------------------------------------------------------------
  // 7) npair(t,rho,a,z,qpair)
  // ------------------------------------------------------------------
  void npair(double t, double rho, double a, double z, double &qpair);

  // ------------------------------------------------------------------
  // 8) nphoto(t,rho,a,z,qphoto)
  // ------------------------------------------------------------------
  void nphoto(double t, double rho, double a, double z, double &qphoto);

  // ------------------------------------------------------------------
  // 9) nplasma(t,rho,a,z,qplasma)
  // ------------------------------------------------------------------
  void nplasma(double t, double rho, double a, double z, double &qplasma);

  /** \brief The energy loss rate per cubic centimeter in the crust
    of a neutron star from plasma neutrinos.

    from h.munakata, y.kohyama & n.itoh, ap.j.296(1985),p.197
  */
  void nplasma_old(double t, double rho, double a, double z, double &qplasma);

  /** \brief The energy loss rate per cubic centimeter in
      bubble phase of the crust.

      from L. Leinson, ApJ 415, p. 759, 1993
  */
  void nbub(int i, double t, double rho, double a, double z, double &qbubble,
            double rhocore, std::vector<double> tcn, int isf);

  /** \brief Desc
   * Calculate the energy loss rate per cubic centimeter in the crust
   * of a neutron star from synchrotron neutrinos.
   * From Bezchastnov, Haensel, Kaminker & Yakovlev,
   * A&A 328 (1997): p. 409
   */
  void nsynch(double t, double bfield, double kfe, double &qsynch);

  // ------------------------------------------------------------------
  // neutrino(...): main function
  // ------------------------------------------------------------------
  void neutrino(
      int i, double t, double rho, double a, double z, double &qtot,
      double &qeebrem, double &qnpb, double &qplasma, double &qsynch,
      double &qbubble, double &qpair, double &qphoto, double &qbrem_nn,
      double &qmurca_nucl, double &qbrem_nucl, double &qmurca_hyp,
      double &qbrem_hyp, double &qdurca_np, double &qdurca_lap,
      double &qdurca_smn, double &qdurca_smla, double &qdurca_sms0,
      double &qfast, double &qdurca_q, double &qmurca_q, double &qpbf_n1s0,
      double &qpbf_n3p2, double &qpbf_p1s0, double &qpbf_q, double debug,
      const std::vector<double> &nbfield2, double rhodrip, double rhocore,
      const std::vector<double> &mstp, const std::vector<double> &mstn,
      const std::vector<double> &mstla, const std::vector<double> &mstsm,
      const std::vector<double> &msts0, const std::vector<double> &mstsp,
      const std::vector<double> &kfe, const std::vector<double> &kfm,
      const std::vector<double> &kfp, const std::vector<double> &kfn,
      const std::vector<double> &kfqu, const std::vector<double> &kfqd,
      const std::vector<double> &kfqs, const std::vector<double> &bar,
      const std::vector<double> &yelect, const std::vector<double> &ymuon,
      const std::vector<double> &fhad, const std::vector<double> &theta_k,
      const std::vector<double> &theta_p, const std::vector<double> &v_ion,
      double rhoexo, double cexo, double pexo, double c_nu_str, double p_nu_str,
      double murca_increase, int inu_durca, int inu_eion, int inu_plasma,
      int inu_synch, int inu_n1s0_pbf, int inu_n3p2_pbf, int inu_p_pbf,
      int inu_bubble, int inu_photo, int inu_pair,
      const std::vector<int> &idurca_np, const std::vector<int> &idurca_lap,
      const std::vector<double> &durca_ctrl_e,
      const std::vector<double> &durca_ctrl_m,
      const std::vector<int> &idurca_smn, const std::vector<int> &idurca_smla,
      const std::vector<int> &idurca_sms0, const std::vector<int> &idurca_quqd,
      const std::vector<int> &idurca_quqs, const std::vector<double> &tcn,
      const std::vector<double> &tcp, const std::vector<double> &tcla,
      const std::vector<double> &tcu, const std::vector<double> &tcd,
      const std::vector<double> &tcs, const std::vector<double> &tcu1,
      const std::vector<double> &tcu2, const std::vector<double> &tcu3,
      const std::vector<double> &tcd1, const std::vector<double> &tcd2,
      const std::vector<double> &tcd3, const std::vector<double> &tcs1,
      const std::vector<double> &tcs2, const std::vector<double> &tcs3,
      int isf);

  /** \brief Load the default star, APR with M=1.4
   */
  void default_star(std::string dir = ".");

  /** \brief Compute the HHJ EOS
   */
  void hhj_eos(double mass);

  /// \name Functions called by the Fortran cooling code
  //@{
  /** \brief Specify initial temperature profile

      Called in <tt>NSCool.f</tt>.

      Uses \ref T_fact_surf and \ref T_fact_drip .
  */
  virtual void tptr_init(int ifteff, double tempini, double ephi_surf,
                         double ephi_drip, double ephi_core, double &tsurface,
                         double &tdrip, double &tcore, double &tb_acc0);

  /** \brief Compute effective temperature in envelope from
      boundary temperature

      This is a replacement for code originally in <tt>boundary.f</tt>
      and called in <tt>NSCool.f</tt>. The parameter \c Tb and the
      return value are both local temperatures (not redshifted).

      \verbatim embed:rst
      This function is based on Appendix A, section 3 in [Potekhin97it]_.
      \endverbatim

      The parameters <tt>bfield, Z, A, compactness, ifteff, istep, time,
      Ts1, Ts2, Rho, debug</tt> are currently unused (but might
      be used in future versions?).
  */
  virtual double Teff(double Tb, int ifteff, double eta_arg, double bfield,
                      int istep, double time, double Ts1, double Ts2, double Z,
                      double A, double Rho, int debug, double gs14,
                      double compactness);

  /** \brief Function for printing out iteration progress

      This function is only called if <tt>ptemp&gt;=1.0</tt> as
      specified in \ref cool_param() (the default). (This is slightly
      different than the original code which only outputs if
      <tt>ptemp=1.0</tt>.) This function's wrapper is called in
      <tt>NSCool.f</tt>. If <tt>ptemp&gt;=1.0</tt> then the
      temperature information is output to \ref tl_prof. If also
      <tt>ptemp&gt;=2.0</tt> then the temperature information is
      output to <tt>std::cout</tt>.

      The parameter \c time is the time in years, \c t_effective is
      the effective temperature at \f$ \infty \f$ in K (includes \f$
      \exp(\phi) \f$), \c imax is the maximum array index (always
      odd). The parameters \c w1 and \c w2 are the weighting factors
      for interpolation. The arrays \c otemp and \c temp are the
      previous and next temperature arrays, and \c olum and \c lum are
      the previous and new temperatures (both local values without any
      factors of \f$ \exp \phi \f$). The arrays \c rad and \c rrho are
      the radius (in cm) and mass(?) density (in \f$
      \mathrm{g}/\mathrm{cm}^3 \f$). The array \c ephi is \f$ \exp
      \phi \f$ and \c e2phi is \f$ \exp ( 2 \phi ) \f$ . The array \c
      dvol is the physical volume between \c i and \c i-1. (All of
      these arrays are zero-indexed in the original FORTRAN code).

  */
  virtual void
  print_temp(int istep, int itprint, double time, double t_effective, int imax,
             double w1, double w2, std::vector<double> otemp,
             std::vector<double> temp, std::vector<double> olum,
             std::vector<double> lum, std::vector<double> rad,
             std::vector<double> rrho, std::vector<double> ephi,
             std::vector<double> dvol, std::vector<double> e2phi,
             std::vector<double> tcn, std::vector<double> tcp,
             std::vector<double> qnu, std::vector<double> qeebrem,
             std::vector<double> qnpb, std::vector<double> qplasma,
             std::vector<double> qsynch, std::vector<double> qbubble,
             std::vector<double> qpair, std::vector<double> qphoto,
             std::vector<double> qbrem_nn, std::vector<double> qmurca_nucl,
             std::vector<double> qbrem_nucl, std::vector<double> qmurca_hyp,
             std::vector<double> qbrem_hyp, std::vector<double> qdurca_np,
             std::vector<double> qdurca_lap, std::vector<double> qdurca_smn,
             std::vector<double> qdurca_smla, std::vector<double> qdurca_sms0,
             std::vector<double> qfast, std::vector<double> qdurca_q,
             std::vector<double> qmurca_q, std::vector<double> qpbf_n1s0,
             std::vector<double> qpbf_p1s0, std::vector<double> qpbf_n3p2,
             std::vector<double> qpbf_q);

  /** \brief Output specific heats
   */
  virtual void print_cv(int itprint, int imax, std::vector<double> cv,
                        std::vector<double> cv_n, std::vector<double> cv_p,
                        std::vector<double> cv_e, std::vector<double> cv_m,
                        std::vector<double> cv_la, std::vector<double> cv_sm,
                        std::vector<double> cv_s0, std::vector<double> cv_sp,
                        std::vector<double> cv_q);

  /** \brief Set various numerical parameters

      The parameter \c time0 is the initial time (default 0) \c
      timemax is the maximum time in years (default \f$ 10^{10} \f$).

      \todo More docs here.

      \note If istep is greater than istepmax, the nscool code just
      silently exits without warning.

      This function's wrapper is called in <tt>NSCool.ff</tt>.
  */
  virtual void num_param(double &time0, double &timemax, int &istepmax,
                         int &itrial_max, int &itrial_opt, double &tcut,
                         double &dtime, double &dtlimit, double &scale_dt0,
                         double &scale_dt1, double &repeat, int &istart,
                         double &mratt, double &mratl, double &mrats,
                         double &tvar, double &svar, double &tcon);

  /** \brief Parameters at the boundary

      The parameter \c ifteff is 1 for the Te-Tb from \ref
      Gudmundsson83 and \c ifteff is 2 for that from \ref Nomoto87.
      These are old, the best is \c ifteff 3 from \ref Potekhin97. If
      \c ifteff is 0, it reads the Te-Tb from a file (probably not
      supported in this version). The value of \c eta determines the
      amount of light elements.

      The value eta is defined by
      \f[
      \eta = g_{s14}^2 \Delta M / M = P_{\mathrm{light}} / P_0
      \f]
      where \f$ P_0 \equiv 1.193 \times
      10^{34}~\mathrm{dyne}/\mathrm{cm}^2 \f$ where
      \f$ P_{\mathrm{light}} \f$ is the pressure at the
      bottom of the light element layer.

      If \c ifteff is 15, this simulates an accreting neutron
      star where \f$ T_b \f$ is held constant. Then \f$
      T_b \f$ is specified in \c tb_acc0 .

      [Dany:] Notice that light elements cannot be present at too high
      densities (e.g., C will burn by pycnonuclear reactions at about
      1010 g cm-3). So there is a maximum value that \f$ \eta \f$ can
      reach. The formula that Potekhin et al. give saturates when \f$
      \eta \f$ grows: thus a value as \f$ \eta=1 \f$ will give the
      maximum possible effect of a light element envelope (even if it
      is physically a wildly unrealistic high value).

      This function's wrapper is called in <tt>NSCool.f</tt>.
  */
  virtual void bound_param(int &ifteff, double &eta_arg, double &mag_coeff,
                           double &tb_acc0);

  /** \brief Specify several parameters

      The parameter \c pscreen controls screen output, \c debug is a
      generic debug variable. The parameters \c emnco, \c emncr, and
      \c emp are flags for the automatic computation of the
      effective masses in <tt>precool.f</tt> .

      The parameters \c pteff, \c ptemp, and \c pstar originally
      controlled whether or not iteration information was output to
      files. This file output is now replaced by \ref print_temp()
      and these parameters will be deprecated.

      This function's wrapper is called in <tt>NSCool.f</tt>.
  */
  virtual void cool_param(int &pscreen, double &debug, int &istep_debug,
                          double &pteff, double &ptemp_arg, double &pstar,
                          int &idump1, int &idump2, int &idump3,
                          double &tempmin, double &tempini, int &icvel_nodeg,
                          double &emnco, double &emncr, double &emp, double &p0,
                          int &itpmax, double *tprint);

  /**
   * @brief Calculates the concentrations Y's of all particles in the core using
   * the EOS table.
   *
   * @param version  Version string ("old", "new", "NEW", "QRK").
   * @param imax     (Output) Maximum index of the grid.
   * @param icore    (Output) Core index.
   * @param rrho     (Output) Density array (size 10001).
   * @param bar      (Output) Baryon density array (size 10001).
   * @param yneutr   (Output) Neutrino concentration array (size 10001).
   * @param yprot    (Output) Proton concentration array (size 10001).
   * @param yelect   (Output) Electron concentration array (size 10001).
   * @param ymuon    (Output) Muon concentration array (size 10001).
   * @param ylambda  (Output) Lambda concentration array (size 10001).
   * @param ysminus  (Output) S- concentration array (size 10001).
   * @param yszero   (Output) S0 concentration array (size 10001).
   * @param ysplus   (Output) S+ concentration array (size 10001).
   * @param yquarku  (Output) Up quark concentration array (size 10001).
   * @param yquarkd  (Output) Down quark concentration array (size 10001).
   * @param yquarks  (Output) Strange quark concentration array (size 10001).
   * @param theta_k  (Output) Theta_k array (size 10001).
   * @param theta_p  (Output) Theta_p array (size 10001).
   * @param fhad     (Output) Hadron fraction array (size 10001).
   * @param mstn     (Output) Neutron effective mass array (size 10001).
   * @param mstp     (Output) Proton effective mass array (size 10001).
   * @param mstla    (Output) Lambda effective mass array (size 10001).
   * @param mstsm    (Output) S- effective mass array (size 10001).
   * @param msts0    (Output) S0 effective mass array (size 10001).
   * @param mstsp    (Output) S+ effective mass array (size 10001).
   */
  virtual void
  core_comp(int &imax, int &icore, std::vector<double> &rrho,
            std::vector<double> &bar, std::vector<double> &yneutr,
            std::vector<double> &yprot, std::vector<double> &yelect,
            std::vector<double> &ymuon, std::vector<double> &ylambda,
            std::vector<double> &ysminus, std::vector<double> &yszero,
            std::vector<double> &ysplus, std::vector<double> &yquarku,
            std::vector<double> &yquarkd, std::vector<double> &yquarks,
            std::vector<double> &theta_k, std::vector<double> &theta_p,
            std::vector<double> &fhad, std::vector<double> &mstn,
            std::vector<double> &mstp, std::vector<double> &mstla,
            std::vector<double> &mstsm, std::vector<double> &msts0,
            std::vector<double> &mstsp);

  /** \brief Fix the settings for direct Urca

      AWS: alpha is the broadening parameter and beta is the
      fractional decrease of the direct Urca threshold

      SR: This function's wrapper is called in <tt>precool.f<tt>
   */
  virtual void urca_settings(double &durca, double &a_durca, double &b_durca);

  virtual void get_fermi_momenta(
      int &imax, int &icore, const std::vector<double> &rrho,
      const std::vector<double> &bar, const std::vector<double> yneutr,
      const std::vector<double> yprot, const std::vector<double> yelect,
      const std::vector<double> ymuon, const std::vector<double> ylambda,
      const std::vector<double> ysminus, const std::vector<double> yszero,
      const std::vector<double> ysplus, const std::vector<double> yquarku,
      const std::vector<double> yquarkd, const std::vector<double> yquarks,
      const std::vector<double> fhad, const std::vector<double> &theta_k,
      const std::vector<double> &theta_p, std::vector<double> &kfn,
      std::vector<double> &kfp, std::vector<double> &kfe,
      std::vector<double> &kfm, std::vector<double> &kfla,
      std::vector<double> &kfsm, std::vector<double> &kfs0,
      std::vector<double> &kfsp, std::vector<double> &kfqu,
      std::vector<double> &kfqd, std::vector<double> &kfqs,
      std::vector<int> &idurca_np, std::vector<int> &idurca_lap,
      std::vector<double> &durca_ctrl_e, std::vector<double> &durca_ctrl_m,
      std::vector<int> &idurca_smn, std::vector<int> &idurca_smla,
      std::vector<int> &idurca_sms0, std::vector<int> &idurca_quqd,
      std::vector<int> &idurca_quqs, std::vector<double> &durca_henon_e,
      std::vector<double> &durca_henon_m);

  /**
   * @brief Calculates the chemical composition, A, A', Z, and the electron &
   * neutron Y's in the crust.
   *
   * This function interpolates particle concentrations using the crust EOS
   * table and fills the provided arrays with the computed values.
   *
   * @param debug    Debug level (1 or higher for verbose output).
   * @param version  Version string ("old", "new", "NEW", "QRK").
   * @param imax     (Output) Maximum index of the grid.
   * @param icore    (Output) Core index.
   * @param rrho     (Input/Output) Density array (size 10001).
   * @param pres     (Input/Output) Pressure array (size 10001).
   * @param debar    (Output) Baryon differential volume array (size 10001).
   * @param dvol     (Input/Output) Volume array (size 10001).
   * @param bar      (Output) Baryon density array (size 10001).
   * @param a_cell   (Output) A_cell array (size 10001).
   * @param a_ion    (Output) A_ion array (size 10001).
   * @param z_ion    (Output) Z_ion array (size 10001).
   * @param v_ion    (Output) Ion volume fraction array (size 10001).
   * @param yelect   (Output) Electron concentration array (size 10001).
   * @param yneutr   (Output) Neutrino concentration array (size 10001).
   */
  virtual void
  crust_comp(int debug, const std::string &version, int &imax, int &icore,
             std::vector<double> &rrho, std::vector<double> &pres,
             std::vector<double> &debar, std::vector<double> &dvol,
             std::vector<double> &bar, std::vector<double> &A_cell,
             std::vector<double> &A_ion, std::vector<double> &Z_ion,
             std::vector<double> &v_ion, std::vector<double> &yelect,
             std::vector<double> &yneutr);

  /**
   * @brief Calculates the degenerate density (rhod) based on pressure
   * interpolation.
   *
   * This function interpolates the density (rhod) for degenerate matter using
   * logarithmic interpolation between provided pressure and density tables.
   * @param rrho  Reference to the rrho array (input).
   * @param pres  Reference to the pres array (input).
   * @param rhod  Reference to the rhod array (output).
   * @param imax  Maximum index of the grid.
   * @param ienv  Environment index.
   */
  virtual void crust_eos(const std::vector<double> &rrho,
                         const std::vector<double> &pres,
                         std::vector<double> &rhod, int imax, int ienv);

  /**
   * @brief Calculates the degenerate specific heat per cm³ over temperature
   * (Cv/T) for spin 1/2 fermions.
   *
   * This function computes Cv/T for degenerate fermions using their Fermi
   * momentum and effective mass. The calculation is based on the formula:
   *
   * \f[
   * Cv/T = \frac{\pi^2}{3} \cdot N_0 \cdot \frac{k_B^2}{\text{MeV} \cdot
   * (197)^3} \times 10^{39}
   * \f]
   *
   * where:
   * - \( N_0 = \frac{2 \cdot m \cdot p_f}{2 \cdot \pi^2} = \frac{m \cdot
   * p_f}{\pi^2} \)
   * - \( p_f \) is the Fermi momentum in MeV/c
   * - \( m \) is the effective mass in MeV/c²
   * - \( k_B \) is the Boltzmann constant in erg/K
   * - Conversion factors ensure Cv/T is in cgs units
   *
   * **Note**:
   * - For baryons, \( m \) should be the Landau effective mass (\( m^* \)).
   * - For leptons, \( m \) should be \( \sqrt{m^2 + p_f^2} \).
   * - The function returns 0.0 if the Fermi momentum \( p_f \) is zero.
   *
   * @param pf Fermi momentum in MeV/c.
   * @param m Effective mass in MeV/c².
   * @return Specific heat Cv/T in cgs units (erg cm⁻³ K⁻¹).
   * @throws std::invalid_argument if the effective mass is non-positive.
   */
  virtual double cvt_deg(double pf, double m);

  /**
   * @brief Calculates Cv/T for degenerate particles in both the core and crust.
   *
   * This function computes the specific heat per temperature (Cv/T) for various
   * fermions based on their Fermi momenta and effective masses. It handles both
   * core and crust regions.
   *
   * @param version    Version string (unused in current implementation).
   * @param emnco      Electron-neutron chemical potential correlation (unused).
   * @param emncr      Electron-neutron chemical potential correlation (unused).
   * @param emp        Electron-proton chemical potential (unused).
   * @param cve        Specific heat Cv/T for electrons (output).
   * @param cvm        Specific heat Cv/T for muons (output).
   * @param cvn        Specific heat Cv/T for neutrons (output).
   * @param cvp        Specific heat Cv/T for protons (output).
   * @param cvla       Specific heat Cv/T for lambdas (output).
   * @param cvsm       Specific heat Cv/T for S- (output).
   * @param cvs0       Specific heat Cv/T for S0 (output).
   * @param cvsp       Specific heat Cv/T for S+ (output).
   * @param cvqu       Specific heat Cv/T for up quarks (output).
   * @param cvqd       Specific heat Cv/T for down quarks (output).
   * @param cvqs       Specific heat Cv/T for strange quarks (output).
   * @param kfe        Fermi momentum for electrons (input).
   * @param kfm        Fermi momentum for muons (input).
   * @param kfn        Fermi momentum for neutrons (input).
   * @param kfp        Fermi momentum for protons (input).
   * @param kfla       Fermi momentum for lambdas (input).
   * @param kfsm       Fermi momentum for S- (input).
   * @param kfs0       Fermi momentum for S0 (input).
   * @param kfsp       Fermi momentum for S+ (input).
   * @param kfqu       Fermi momentum for up quarks (input).
   * @param kfqd       Fermi momentum for down quarks (input).
   * @param kfqs       Fermi momentum for strange quarks (input).
   * @param mstn       Effective mass for neutrons (input).
   * @param mstp       Effective mass for protons (input).
   * @param mstla      Effective mass for lambdas (input).
   * @param mstsm      Effective mass for S- (input).
   * @param msts0      Effective mass for S0 (input).
   * @param mstsp      Effective mass for S+ (input).
   * @param fhad       Hadron fraction (input).
   * @param imax       Maximum index of the grid.
   * @param idrip      Index of the neutron drip point.
   */
  virtual void get_spec_heat_degenerate(
      std::vector<double> &cve, std::vector<double> &cvm,
      std::vector<double> &cvn, std::vector<double> &cvp,
      std::vector<double> &cvla, std::vector<double> &cvsm,
      std::vector<double> &cvs0, std::vector<double> &cvsp,
      std::vector<double> &cvqu, std::vector<double> &cvqd,
      std::vector<double> &cvqs, const std::vector<double> kfe,
      const std::vector<double> kfm, const std::vector<double> kfn,
      const std::vector<double> kfp, const std::vector<double> kfla,
      const std::vector<double> kfsm, const std::vector<double> kfs0,
      const std::vector<double> kfsp, const std::vector<double> kfqu,
      const std::vector<double> kfqd, const std::vector<double> kfqs,
      const std::vector<double> mstn, const std::vector<double> mstp,
      const std::vector<double> mstla, const std::vector<double> mstsm,
      const std::vector<double> msts0, const std::vector<double> mstsp,
      const std::vector<double> fhad, int imax);

  /** \brief Data for superfluid suppression function

      This function's wrapper is called in <tt>neutrino_core.f</tt>.
  */
  virtual void sf_suppress_data(double *lgtau1, double *lgtau2, double *lgr);

  /** \brief Data for neutrino pair bremsstrahlung

      This function's wrapper is called in <tt>neutrino_crust.f</tt>.
  */
  virtual void pair_brem_data(double *logt, double *nalpha);

  /**
   * @brief Constructs the star grid based on density profiles and other
   * parameters.
   *
   * @param idec     Number of density zones per decade.
   * @param rhocore  Core density.
   * @param rhodrip  Density at neutron drip.
   * @param rhoenv   Envelope density threshold.
   * @param rhosurf  Surface density.
   * @param imax     (Output) Maximum index of the grid.
   * @param icore    (Output) Core index.
   * @param idrip    (Output) Neutron drip index.
   * @param ienv     (Output) Envelope boundary index.
   * @param rad      (Output) Radius array (size 10001).
   * @param rrho     (Output) Density array (size 10001).
   * @param pres     (Output) Pressure array (size 10001).
   * @param dvol     (Output) Volume array (size 10001).
   * @param emas     (Output) Energy density array (size 10001).
   * @param phi      (Output) Potential array (size 10001).
   */
  virtual void star_struct(int idec, double rhocore, double rhodrip,
                           double rhoenv, double rhosurf, int &imax, int &icore,
                           int &idrip, int &ienv, std::vector<double> &rad,
                           std::vector<double> &rrho, std::vector<double> &pres,
                           std::vector<double> &dvol, std::vector<double> &emas,
                           std::vector<double> &phi);

  virtual void get_Tc(int imax, int icore, int idrip, std::vector<double> &tcn,
                      std::vector<double> &tcp, std::vector<double> &tcla,
                      std::vector<double> &tcsm, std::vector<double> &tcs0,
                      std::vector<double> &tcsp, std::vector<double> &tcuu,
                      std::vector<double> &tcdd, std::vector<double> &tcss,
                      std::vector<double> &tcud, std::vector<double> &tcus,
                      std::vector<double> &tcds, std::vector<double> &tcu,
                      std::vector<double> &tcd, std::vector<double> &tcs,
                      int &sfl1s0, double fn1s0, double fn3p2, double fp1s0,
                      double fl1s0, int &isf, std::vector<double> kfn,
                      std::vector<double> kfp, std::vector<double> kfla,
                      std::vector<double> kfqu, std::vector<double> kfqd,
                      std::vector<double> kfqs, std::vector<double> bar,
                      std::vector<double> fhad, std::vector<double> yquarku,
                      std::vector<double> yquarkd, std::vector<double> yquarks);

  /** \brief Specification of nucleon gaps

      This function's wrapper is called in <tt>precool.f</tt>.
  */
  virtual void gaps(int &sfn1s0_arg, double &n1_tc_arg, double &n1_kf_arg,
                    double &n1_dk_arg, int &sfn3p2_arg, double &n3_tc_arg,
                    double &n3_kf_arg, double &n3_dk_arg, int &sfp1s0_arg,
                    double &p1_tc_arg, double &p1_kf_arg, double &p1_dk_arg);

  /**
   * @brief Calculates the baryon effective masses.
   *
   * This function calculates the baryon effective masses (in a simplistic
   * manner) in case they are not provided by the EOS table.
   *
   * @param kfn      Neutron Fermi momentum array (size isize + 1).
   * @param mstn     Neutron effective mass array (size isize + 1) - Output.
   * @param idrip    Index of the neutron drip point.
   * @param icore    Index of the core.
   */
  virtual void get_effective_masses(const std::vector<double> kfn,
                                    std::vector<double> &mstn, int idrip,
                                    int icore);

  /** \brief Main output

      This function stores the main cooling curve output into \ref
      v_time, \ref v_tptr, \ref v_lphot, \ref v_lneut, and \ref
      v_lheat .

      This function's wrapper is called in <tt>NSCool.f</tt>.

      This function also flips the stop flag if more than
      10000 steps are taken by the main cooling loop.
  */

  virtual void main_out(double time, double tptr, double lphot, double lneut,
                        double lheat, int stop);
  //@}

public:
  /** \brief Number of iterations to skip
      for the main output function (default 20)
  */
  int main_out_it;

  /** \brief Output the cooling curve to a file
   */
  void write_cool_curve(std::string fname = "cool_curve.o2");

  /** \brief Write temperature and luminosity profiles to a file

      This function copies the data into a table with a reorganized
      temperature grid (necessary because the cooling code doesn't
      know at what time the temperature will drop below the minimum).
  */
  void write_tl_prof(std::string fname = "tl_prof.o2");
};

// Global pointers to nscool class
extern std::vector<nscool *> nscool_ptrs;

#endif
