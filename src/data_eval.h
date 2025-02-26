#ifndef DATA_EVAL_H
#define DATA_EVAL_H

#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <chrono>

#include <boost/numeric/ublas/vector.hpp>

#include <o2scl/nstar_cold.h>
#include <o2scl/tensor_grid.h>
#include <o2scl/eos_had_rmf.h>
#include <o2scl/eos_had_rmf_hyp.h>
#include <o2scl/nucleus_rmf.h>
#include <o2scl/table.h>
#include <o2scl/emulator.h>

#include <Python.h>
#include <o2scl/interpm_python.h>

#include "nscool_wrap.h"
#include "ins.h"
#include "sxrt.h"
#include "nsmr.h"

// uBlas vector type
typedef boost::numeric::ublas::vector<double> ubvector;

// uBlas matrix type
typedef boost::numeric::ublas::matrix<double> ubmatrix;

// Number of grid points for M-R, EOS, and cooling curve grids.
// Warning, right now, 100 is the only value which works because of
// the bookkeeping associated with the nuclear densities.
static const size_t ngrid=100;
// Number of data points not associated with a grid
static const size_t ndat_1d=169+6;
// Number of sets of data for each grid point
static const size_t ndat_2d=51;
// Number of points in the sxrt_lum tensor (3*6*30)
static const size_t n_sxrt=540;
// Total number of entries in dat[] array
static const size_t ndat=ndat_1d+ndat_2d*ngrid+n_sxrt;
//static const size_t ndat=2;

static const bool jan_emu=false;

// Point function type for MCMC class
typedef std::function<int(size_t,const ubvector &,double &,
			  std::array<double,ndat> &)> point_funct;

// Fill function type for MCMC class
typedef std::function<int(const ubvector &,double,std::vector<double> &,
			  std::array<double,ndat> &)> fill_funct;

extern bool compute_estimates;

/** \brief The interpolation estimate objects
 */
static const bool use_idw=false;

class data_eval {
  
public:

  /** \brief If true, use the C++ emulator
   */
  bool flag_emu_aws;

  /// C++ emulator
  //std::shared_ptr<o2scl::emulator_base<std::array<double,ndat>>> emu;
#ifdef O2SCL_SET_PYTHON
  std::shared_ptr<o2scl::interpm_python<> > ip_ins;
  std::shared_ptr<o2scl::interpm_python<> > ip_sxrt;
#endif
  /** \brief If true, treat an error as a point with a small
      likelihood (default true)
  */
  bool err_as_small_wgt;
  
  /// The INS cooling data
  ins_data id;

  double r14_max;
  
  /// The SXRT data
  sxrt_data sd;

  /// The NS mass and radius data
  nsmr_data nd;
  
  /// \name Desc
  //@{
  /** \brief If true, omit the isolated neutron star cooling calculation
   */
  bool no_ins;

  /** \brief If true, omit the SXRT calculation
   */
  bool no_sxrt;

  /** \brief If true, omit the SXRT calculation
   */
   bool no_qlmxb;

  /** \brief If true, omit the SXRT calculation
   */
  bool no_ligo;

  /** \brief If true, output more information about the INS likelihood
   */
  bool debug_ins;
  
  /** \brief If true, omit the nuclear structure calculation
   */
  bool no_nuclei;

  /** \brief If true, omit the prex calculation
   */
   bool no_prex;
  //@}

  /// \name Quantities specified in init()
  //@{
  /// Verbosity parameter (set in init())
  int verbose;

  /// Mapping from data names to indexes (set in init() )
  o2scl::vec_index dvi;
  
  /// Mapping from parameter names to indexes (set in init() )
  o2scl::vec_index pvi;
  
  // Number of parameters
  size_t nparam;
  //@}

  /// The sxrt luminosities
  o2scl::tensor_grid3<> ligo_data;

  /// \name EOS objects
  //@{
  /// The RMF EOS object
  o2scl::eos_had_rmf rmf;

  /// Pointer to the RMF EOS
  o2scl::eos_had_rmf *rmfp;

  /// The RMF EOS object
  o2scl::eos_had_rmf_hyp rmfh;
  //@}

  /// \name Neutron star objects
  //@{
  /// The cold neutron star object
  o2scl::nstar_cold nst;
  
  /// The NSCool object
  nscool_wrap nw;
  //@}

  /// \name Nuclear structure objects
  //@{
  /// The nuclear structure object
  o2scl::nucleus_rmf rn;

  /// The second nuclear structure object
  o2scl::nucleus_rmf rn2;
  //@}

  /// \name Output data
  //@{
  /// The neutron star cooling temperatures
  o2scl::tensor_grid3<> nscool_tptr;

  /// The neutron star cooling luminosities in erg/s
  o2scl::tensor_grid3<> nscool_lum;

  /// Desc
  o2scl::tensor_grid3<> nscool_lum_all;

  /// The sxrt luminosities
  o2scl::tensor_grid3<> sxrt_lum;

  /** \brief The lead 208 table 
   */
  std::shared_ptr<o2scl::table_units<> > nuc_profile1;

  /** \brief The zirconium 90 table 
   */
  std::shared_ptr<o2scl::table_units<> > nuc_profile2;

  /** \brief The EOS table
   */
  std::shared_ptr<o2scl::table_units<> > te;

  /** \brief The TOV table

      We copy the M vs. R table instead of keeping a pointer
      to it because we want to compute the full M-R curve
      first, but then use \ref nst to compute the profiles
      of fixed mass stars.
   */
  o2scl::table_units<> tmvsr;
  //@}

  /// \name Grids
  //@{
  /// The time grid for cooling curves
  std::vector<double> t_grid;

  /// The log eta grid for cooling curves
  std::vector<double> log_eta_grid;
  
  /// The mass grid for the cooling curves
  std::vector<double> mass_grid;

  /// The Mdot grid for the steady-state curves
  std::vector<double> log_mdot_grid;

  /// The mass grid for the M-R curve
  o2scl::uniform_grid<double> mvsr_mass_grid;

  /// The energy density grid for the EOS
  o2scl::uniform_grid<double> eden_grid;

  /// The baryon density grid for the EOS
  o2scl::uniform_grid<double> nb_grid;

  /// The radius grid for the nuclear structure profiles
  o2scl::uniform_grid<double> nucprof_grid;
  //@}

  /// \name Misc
  //@{
  /// The interpolator to interpolate cooling curve results
  o2scl::interp_vec<std::vector<double> > itp;

  /// The schwarzchild radius in km
  double schwarz_km;
  //@}

  /// Bool to use write_tl_prof function
  bool w_tl_prof;
  //@}

  data_eval();
  
  /** \brief Init the class 
   */
  void init(size_t nparam_, o2scl::vec_index &dvi_, o2scl::vec_index &pvi_,
	    size_t verbose_, ins_data &id_,
	    nsmr_data &nd_, sxrt_data &sd_);
  
  // ---------------------------------------------------------------------

  /// \name SXRT functions
  //@{
  
  /** \brief Return the difference between the total and target heating
      luminosity

      This function calls \ref acc_compute().
  */
  int match_lum(size_t nv, const ubvector &x, ubvector &y, double Q_heat,
		double mass, double rad, double Mdot);

  /** \brief At a fixed temperature, run the cooling code to determine the
      photon neutrino and heating luminosities
  */
  int acc_compute(double logT, double Q_heat, double mass,
		  double rad, double Mdot, double &lphot, double &lneut,
		  double &lheat);
  
  /** \brief Solve for steady-state at specified accretion rate
      using initial guess for core temperature

      Given the Mdot, Q_heat, mass, and rass, compute the steady state
      temperature, compute \c log10, and put it in \c logT.

      This function finds the root of \ref match_lum().
  */
  int ss_acc_solve(double Mdot, double Q_heat, double mass,
		   double rad, double &logT);

  /** \brief Compute steady-state curves

      This function calls \ref ss_acc_solve() for a range of
      accretion rates
   */
  int compute_ss(size_t nv, const ubvector &pars,
		 double &log_weight,std::array<double,ndat> &dat);

  /** \brief Calculate likelihood function
   */
  int sxrt_like(const ubvector &pars, double &log_weight,
                std::array<double,ndat> &dat);

  void no_sxrt_fun(std::array<double,ndat> &dat);
  void no_ins_fun(std::array<double,ndat> &dat);
  //@}
  
  // ---------------------------------------------------------------------

  /** \brief Isolated neutron star likelihood
   */
  int ins_like(size_t nv, const ubvector &pars, double &log_weight,
	       std::array<double,ndat> &dat);
  
  
  /** \brief Neutron star mass and radius likelihood
   */
  int nsmr_like(const ubvector &pars,std::array<double,ndat> &dat,
		std::vector<o2scl::table3d> &MR_data,double &log_weight);

  /** \brief Evaluate a Monte Carlo point
   */
  int point(size_t nv, const ubvector &pars, double &log_weight,
	    std::array<double,ndat> &dat);

  /** \brief Add auxillary quantities to 'line' so they can be
      stored in the table
  */
  int fill_line(const ubvector &pars, double log_weight,
		std::vector<double> &line, std::array<double,ndat> &dat);

};
#endif
