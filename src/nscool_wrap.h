/*
  -------------------------------------------------------------------
  
  Copyright (C) 2015-2020, Andrew W. Steiner and Satyajit Roy
  
  This file is part of nscool_wrap.
  
  nscool_wrap is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.
  
  nscool_wrap is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with O2scl. If not, see <http://www.gnu.org/licenses/>.

  -------------------------------------------------------------------
*/
#ifndef NSCOOL_WRAP_H
#define NSCOOL_WRAP_H

#include <vector>

#include <boost/numeric/ublas/vector.hpp>

#include <o2scl/table.h>
#include <o2scl/hdf_file.h>
#include <o2scl/hdf_io.h>
#include <o2scl/eos_had_base.h>
#include <o2scl/nstar_cold.h>
#include <o2scl/fermion_deriv_rel.h>
#include <o2scl/permutation.h>
#include <o2scl/lib_settings.h>

#include "emissivities.h"
#include "tc.h"


// Declaration for FORTRAN-defined main cooling subroutine
extern "C" void nscool_(int *irank, int *iret, double *neebrem_logt,
			double *neebrem_nalpha, double *neebrem_n2,
			double *sf_lgtau1, double *sf_logtau2,
			double *sf_lgr, double *sf_lgr2);

// Declaration for Fortran spline function
extern "C" void spline_(double *X, double *Y, int *IN,
			double *YP1, double *YP2, double *Y2);

// Declaration for Fortran spline2 function
extern "C" void spline2_(double *x1a, double *x2a, double *ya,
			 int *im, int *in, double *y2a);

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
    s=0.1;
    gamma=0.7;
    eps0=15.8/o2scl_const::hc_mev_fm;
    S0=32.0/o2scl_const::hc_mev_fm;
    n0=0.16;
    delta_P_K=0.0;
    delta_P_Gamma=0.0;
  }

  virtual int calc_temp_f_gen(double nB, double nQ, double nS,
                              double T, o2scl::thermo &th) {
    std::cerr << "Unimplemented." << std::endl;
    exit(-1);
    return 0;
  }
  /** \brief Equation of state as a function of density
   */
  virtual int calc_e(o2scl::fermion &ln, o2scl::fermion &lp,
		     o2scl::thermo &lth) {

    double barn=ln.n+lp.n;
      
    double xp;
    if (barn<=0.0) {
      xp=0.0;
      lth.ed=0.0;
      ln.mu=0.0;
      lp.mu=0.0;
      lth.pr=0.0;
      return 0;
    } else {
      xp=lp.n/barn;
    }

    double u=barn/n0;
    double sym=S0*pow(u,gamma);
    double symp=S0*gamma*pow(u,gamma-1.0);
      
    lth.ed=ln.m*ln.n+lp.m*lp.n+barn*(eps0*u*(u-2.0-s)/(1.0+s*u)+
				     sym*pow((1.0-2.0*xp),2.0));
    ln.mu=ln.m+(lth.ed-ln.m*ln.n-lp.m*lp.n)/barn+u*
      (eps0*(u-2.0-s)/(1.0+s*u)+eps0*u/(1.0+s*u)-eps0*u*(u-2.0-s)*s/
       pow(1.0+s*u,2.0)+pow(1-2.0*xp,2.0)*symp)+xp*4.0*(1-2.0*xp)*sym;
    lp.mu=lp.m+ln.mu-ln.m-4.0*(1.0-2.0*xp)*sym;
    lth.pr=-lth.ed+ln.mu*ln.n+lp.mu*lp.n;

    ln.kf=cbrt(3.0*o2scl_const::pi2*ln.n);
    lp.kf=cbrt(3.0*o2scl_const::pi2*lp.n);

    if (delta_P_Gamma>0.0 && lth.ed>1.5) {
      lth.pr+=(delta_P_K*pow(lth.ed,delta_P_Gamma)-
	       delta_P_K*pow(1.5,delta_P_Gamma));
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
    pointer list <tt>nscool_wrap_ptrs</tt>, change the input EOS
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
class nscool_wrap {
  
 public:

  /** \brief Object for computing critical temperatures
   */
  tc atc;
  
  /** \brief Object for computing emissivities
   */
  emissivities emis;
  
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
  //@}

  /// \name Data for superfluid suppression factor
  //@{
  double sf_lgtau1[35], sf_lgtau2[35], sf_lgr[1225], sf_lgr2[1225];
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
  
  nscool_wrap(std::string dir);

  /** \brief Desc
   */
  double cvelec(double t, double rho, double a, double z);
  
  /** \brief "Ion" specific heat?
   */
  double cvion(double t, double rho, double a, double z);
  /** \brief Pressure in the crust
   */
  double pressure(double T, double Rho, double A, double Z);
  
  /** \brief Desc
   */
  void density(double T, double P, double A, double Z, double &Rho);
  
  /** \brief Load the default star, APR with M=1.4
   */
  void default_star(std::string dir=".");
    
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
		      int istep, double time, double Ts1, double Ts2,
		      double Z, double A, double Rho, int debug,
		      double gs14, double compactness);
  
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
  virtual void print_temp(int istep, int itprint,
			  double time, double t_effective, int imax,
			  double w1, double w2, double *otemp, double *temp,
			  double *olum, double *lum, double *rad,
			  double *rrho, double *ephi, double *dvol,
			  double *e2phi, double *tcn, double *tcp,
			  double *qnu, double *qeebrem,
			  double *qnpb, double *qplasma,
			  double *qsynch, double *qbubble, double *qpair,
			  double *qphoto, double *qbrem_nn,
			  double *qmurca_nucl, double *qbrem_nucl,
			  double *qmurca_hyp, double *qbrem_hyp,
			  double *qdurca_np, double *qdurca_lap,
			  double *qdurca_smn, double *qdurca_smla,
			  double *qdurca_sms0, double *qfast,
			  double *qdurca_q, double *qmurca_q, 
			  double *qpbf_n1s0, double *qpbf_p1s0,
			  double *qpbf_n3p2, double *qpbf_q);
  
  /** \brief Output specific heats
   */
  virtual void print_cv(int itprint, int imax, double *cv,
			double *cv_n, double *cv_p,
			double *cv_e, double *cv_m, double *cv_la, 
			double *cv_sm, double *cv_s0, double *cv_sp,
			double *cv_q);
  
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
  virtual void bound_param(int &ifteff, double &eta_arg,
			   double &mag_coeff, double &tb_acc0);
  
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
			  double &tempmin, double &tempini,
			  int &icvel_nodeg, double &emnco, double &emncr,
			  double &emp, double &p0, int &itpmax,
			  double *tprint);
  
  /** \brief Specify core composition

      This function provides the core composition by
      copying the information in the \ref nscool_core table
      to the Fortran arrays.

      This function's wrapper is called in <tt>precool.f</tt>.
  */
  virtual void core_comp
    (double *rho_t, double *nbar_t, double *yelect_t, double *ymuon_t,
     double *yneutr_t, double *yprot_t, double *ylambda_t,
     double *ysminus_t, double *yszero_t, double *ysplus_t,
     double *mstp_t, double *mstn_t, double *mstla_t, double *mstsm_t,
     double *msts0_t, double *mstsp_t, int *ix);
  
  /** \brief Fix the settings for direct Urca

      AWS: alpha is the broadening parameter and beta is the
      fractional decrease of the direct Urca threshold

      SR: This function's wrapper is called in <tt>precool.f<tt>
   */
  virtual void urca_settings(double &durca, double &a_durca,
			     double &b_durca);
  
  /** \brief Specify crust composition
      
      This function provides the crust composition by
      copying the information in the \ref nscool_crust table
      to the Fortran arrays.

      This function's wrapper is called in <tt>precool.f</tt>.
  */
  virtual void crust_comp
    (double *Z_ion_t, double *A_ion_t, double *A_cell_t,
     double *bar_t, double *pres_t, double *rho_t,
     int *jmax);
  
  /** \brief Specify crust EOS
      
      This function provides the crust EOS by
      copying the information in the \ref nscool_crust table
      to the Fortran arrays.

      This function's wrapper is called in <tt>precool.f</tt>.
  */
  virtual void crust_eos(double *rho2, double *pres2, int *idata);

  /** \brief Data for superfluid suppression function

      This function's wrapper is called in <tt>neutrino_core.f</tt>.
  */
  virtual void sf_suppress_data(double *lgtau1, double *lgtau2,
				double *lgr);
  
  /** \brief Data for neutrino pair bremsstrahlung

      This function's wrapper is called in <tt>neutrino_crust.f</tt>.
  */
  virtual void pair_brem_data(double *logt, double *nalpha);

  /** \brief Specify the stellar structure
      
      This function provides the stellar profile by
      copying the information in the \ref nscool_crust table
      to the Fortran arrays. These arrays are unit-indexed 
      in the original Fortran code, but the pointers which
      are sent to this function are pointers to the first
      element, so the C-style arrays in this function are
      zero-indexed.

      See \ref nscool_tov for the proper units and column names for
      the various quantities.

      This function's wrapper is called in <tt>precool.f</tt>.
  */
  virtual void star_struct(int icore, double rhocore,
			   double *rad_t, double *bar_t, double *rho_t,
			   double *pres_t, double *emas_t, double *phi_t,
			   double *rad, int *jmax, int *jcore,
			   double *w1, double *w2);
  
  /** \brief Specification of nucleon gaps

      This function's wrapper is called in <tt>precool.f</tt>.
  */
  virtual void gaps(int &sfn1s0_arg, double &n1_tc_arg,
		    double &n1_kf_arg, double &n1_dk_arg,
		    int &sfn3p2_arg, double &n3_tc_arg,
		    double &n3_kf_arg, double &n3_dk_arg,
		    int &sfp1s0_arg, double &p1_tc_arg,
		    double &p1_kf_arg, double &p1_dk_arg);

  /** \brief Main output 

      This function stores the main cooling curve output into \ref
      v_time, \ref v_tptr, \ref v_lphot, \ref v_lneut, and \ref
      v_lheat .

      This function's wrapper is called in <tt>NSCool.f</tt>.
      
      This function also flips the stop flag if more than 
      10000 steps are taken by the main cooling loop.
  */
  virtual void main_out(double &time, double &tptr,
			double &lphot, double &lneut, double &lheat,
			int &stop);
  //@}
  
 public:

  /** \brief Number of iterations to skip
      for the main output function (default 20)
  */
  int main_out_it;
  
  /** \brief Execute main cooling calculation

      \todo Document in what cases <tt>iret</tt> can be nonzero
  */
  virtual int run(int irank); 

  /** \brief Output the cooling curve to a file
   */
  void write_cool_curve(std::string fname="cool_curve.o2");
  
  /** \brief Write temperature and luminosity profiles to a file
      
      This function copies the data into a table with a reorganized
      temperature grid (necessary because the cooling code doesn't
      know at what time the temperature will drop below the minimum).
  */
  void write_tl_prof(std::string fname="tl_prof.o2");

};

// Global pointers to nscool class
extern std::vector<nscool_wrap *> nscool_wrap_ptrs;

/// Set superfluid gaps
extern "C" void nscool_gaps_(int *irank, double *sfn1s0,
			     double *n1tc, double *n1kf, double *n1dk,
			     double *sfn3p2, double *n3tc, double *n3kf,
			     double *n3dk, double *sfp1s0,
			     double *p1tc, double *p1kf, double *p1dk); 

/// Main output function
extern "C" void nscool_main_out_(int *irank, double *time, double *tptr,
				 double *lphot, double *lneut, double *lheat,
				 int *stop); 

/// Obtain stellar structure
extern "C" void nscool_star_struct_(int *irank, int *icore,
   double *rhocore, double *rad_t, double *bar_t, double *rho_t,
   double *pres_t, double *emas_t, double *phi_t, double *rad,
   int *jmax, int *jcore, double *w1, double *w2); 

/// Obtain core composition
extern "C" void nscool_core_comp_
(int *irank, double *rho_t, double *nbar_t, double *yelect_t,
 double *ymuon_t, double *yneutr_t, double *yprot_t, double *ylambda_t,
 double *ysminus_t, double *yszero_t, double *ysplus_t,
 double *mstp_t, double *mstn_t, double *mstla_t, double *mstsm_t,
 double *msts0_t, double *mstsp_t, int *ix); 

/// Obtain crust composition
extern "C" void nscool_crust_comp_(int *irank, double *Z_ion_t,
				   double *A_ion_t, double *A_cell_t,
				   double *bar_t, double *pres_t,
				   double *rho_t, int *jmax); 

/// Obtain crust EOS
extern "C" void nscool_crust_eos_(int *irank, double *rho2, double *pres2,
				  int *idata); 

/// Set numerical parameters
extern "C" void nscool_num_param_
(int *irank, double *time0, double *timemax, int *istepmax,
 int *itrial_max, int *itrial_opt, double *tcut,
 double *dtime, double *dtlimit, double *scale_dt0,
 double *scale_dt1, double *repeat, int *istart,
 double *mratt, double *mratl, double *mrats,
 double *tvar, double *svar, double *tcon); 

/// Set cooling parameters
extern "C" void nscool_cool_param_
(int *irank, int *pscreen, double *debug, int *istep_debug,
 double *pteff, double *ptemp, double *pstar,
 int *idump1, int *idump2, int *idump3,
 double *tempmin, double *tempini,
 int *icvel_nodeg, double *emnco, double *emncr,
 double *emp, double *p0, int *itpmax, double *tprint); 

/// Set initial temperature profile
extern "C" void nscool_tptr_init_(int *irank, int *ifteff, double *tempini,
				  double *ephi_surf, double *ephi_drip,
				  double *ephi_core, double *tsurface,
				  double *tdrip, double *tcore,
				  double *tb_acc0); 

/// Set atmosphere parameters
extern "C" double nscool_teff_(int *irank, double *Tb, int *ifteff,
			       double *eta, double *bfield, int *istep,
			       double *time, double *Ts1, double *Ts2,
			       double *Z, double *A, double *Rho,
			       int *debug, double *gs14,
			       double *compactness); 

/// Set boundary parameters
extern "C" void nscool_bound_param_(int *irank, int *ifteff, double *eta,
				    double *mag_coeff, double *tb_acc0); 

/// Print out temperatures and luminosities
extern "C" void nscool_print_temp_
(int *irank, int *istep, int *itprint, double *time,
 double *t_effective, int *imax, double *w1, 
 double *w2, double *otemp, double *temp,
 double *olum, double *lum, double *rad,
 double *rrho, double *ephi, double *dvol,
 double *e2phi, double *tcn, double *tcp,
 double *qnu, double *qeebrem,
 double *qnpb, double *qplasma,
 double *qsynch, double *qbubble, double *qpair,
 double *qphoto, double *qbrem_nn,
 double *qmurca_nucl, double *qbrem_nucl,
 double *qmurca_hyp, double *qbrem_hyp,
 double *qdurca_np, double *qdurca_lap,
 double *qdurca_smn, double *qdurca_smla,
 double *qdurca_sms0, double *qfast,
 double *qdurca_q, double *qmurca_q, 
 double *qpbf_n1s0, double *qpbf_p1s0,
 double *qpbf_n3p2, double *qpbf_q);

/// Print specific heats
extern "C" void nscool_print_cv_
(int *irank, int *itprint,
 int *imax, double *cv, double *cv_n, double *cv_p, 
 double *cv_e, double *cv_m, double *cv_la, double *cv_sm, double *cv_s0, 
 double *cv_sp, double *cv_q); 

/// Handle Urca settings
extern "C" void nscool_urca_settings_(int *irank, double *fix_durca,
				      double *alpha_durca,
				      double *beta_durca); 

/// Compute critical temperatures
extern "C" void nscool_tc_new_(int *irank, int *imax, int *idrip, int *icore,
			       double *sfn1s0, double *n1tc,
			       double *n1kf, double *n1dk,
			       double *sfp1s0, double *p1tc,
			       double *p1kf, double *p1dk,
			       double *sfn3p2, double *n3tc,
			       double *n3kf, double *n3dk,
			       double *fn1s0, double *fp1s0, double *fn3p2,
			       double *kf_n, double *kf_p, double *kf_la,
			       double *kf_sm, double *kf_s0, double *kf_sp,
			       double *kf_u, double *kf_d, double *kf_s,
			       double *tc_n, double *tc_p, double *tc_la,
			       double *tc_uu, double *tc_dd, double *tc_ss,
			       double *tc_ud, double *tc_us, double *tc_ds,
			       double *tc_u, double *tc_d, double *tc_s,
			       int *isf);

extern "C" void nscool_density_(int *irank, double *T, double *P,
				double *A, double *Z, double *Rho); 

#endif
