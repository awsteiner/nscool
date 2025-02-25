/*
  
  This code is separated into two classes for parallelization
  purposes. The is only one mcmc_wrapper instance but there are
  several data_eval instances, one for each OpenMP thread.

  ---------------------------------------------------------------

  Data comparison

  A. Go through all time estimates where both tc and t* are
  given and see what Potekhin recommends
  

  * 1. 1E0102:
  * 2. J0822: both tc and t* 
  * 3. J085201: 
  * 4. J104608:
  * 5. 1E1207: both tc and t*
  * 6. J141256: no t unc.
  * 7: J160103: 
  * 8: J1713: no t unc.
  * 9: J172054:
  * 10 J173203:
  * 11: J181852:
  * 12: J185238: both tc and t*
  * 13: J232327: 

  * 14: J0205: both tc and t*
  * 15: J0357: both tc and t*
  * 16: J0538: both tc and t*
  * 17: J061705: both tc and t*
  * 18: J0633+06: no t unc.
  * 19: J0633+17: no t unc.
  * 20: B0656: no t unc.
  * 21: B0833: both tc and t*
  * 22: B1055: no t unc.
  * 23: J1357: no t unc.
  * 24: B1706: no t unc.
  * 25: J1740: no t unc.
  * 26: J1741: no t unc.
  * 27: B1822: no t unc.
  * 28: B1823: no t unc.
  * 29: J1836: no t unc. (no utility, drop)
  * 30: B1951: both tc and t*
  * 31: J1957: no t unc.
  * 32: J2021: no t unc.
  * 33: B2334: both tc and t*

  * 34: J0726: no t unc.
  * 35: J1119: both tc and t*
  * 36: B1509: no t unc.
  * 37: J1718: no t unc.
  * 38: J1819: no t unc. (no utility, drop)

  * 39: J0420: no t unc. (no utility, drop)
  * 40: J0720: both tc and t* (Satyajit says drop?)
  * 41: J0806: no t unc. (no utility, drop)
  * 42: J1308: both tc and t* (Satyajit says drop?)
  * 43: J1605: 
  * 44: J1856: both tc and t*
  * 45: J2143: no t unc. (no utility, drop)

  * 46: J0007:, only L upper limit
  * 47: B0531:, only L upper limit (no utility, drop)
  * 48: B1727:, only L upper limit
  * 49: J2043:, only L upper limit (no utility, drop)
  * 50: B2224:, only L upper limit (no utility, drop)

  * Drop 51-55

  ---------------------------------------------------------------

  Todo list for Andrew:
  
  - Make sure the steady state curves converge reasonably often
  (so far so good)
  
  - Fix the upper/lower limits for 1808 and 1905
  
  - Check superfluidity models between INSs and SXRTs
  
  - Check H vs. C for Cas A
  
  - Ensure we're computing the symmetry energy
  (done I think)
  
  - Remove global variables

  - Fix duplication of string lists like ins_names, etc.
  
*/

#ifndef RS_H
#define RS_H

#include <string>
#include <vector>

#include <o2scl/mcmc_para.h>
#include <o2scl/misc.h>
#include <o2scl/prob_dens_mdim_amr.h>
#include <o2scl/interpm_idw.h>
#include <o2scl/emulator.h>
#include <o2scl/kde_python.h>
#include <o2scl/gmm_python.h>

#include "data_eval.h"
#include "ins.h"
#include "sxrt.h"
#include "nsmr.h"

/** \brief A specialized emulator for this code

    Note that this emulator has a data type of array<double,ndat>,
    because that is what the MCMC is using, but internally em1 uses a
    data type of vector<double>, since not all output fields are
    always emulated.
 */
class emulator_rs : public o2scl::emulator_unc<std::array<double,ndat>,
                                               std::array<double,ndat>,
                                               ubvector> {
  
public:

  /** \brief Desc
   */
  int mpi_rank;
  
  /** \brief Desc
   */
  int mpi_size;
  
  /** \brief The internal generic emulator
   */
  o2scl::emulator_interpm_idw_table<std::vector<double>,ubvector> em1;

  /** \brief The second internal generic emulator
   */
  //o2scl::emulator_interpm_krige_table
  //<std::vector<double>,ubvector,Eigen::MatrixXd,
  //o2scl_linalg::matrix_invert_det_eigen<Eigen::MatrixXd> > em2;
  
  /** \brief List of parameters and output quantities (including log_wgt)
   */
  std::vector<std::string> list;
  
  /** \brief The python emulator
   */
  //o2scl::emulator_python<std::vector<double>,ubvector> em3;
  
  /** \brief Pointer to the data_eval object for function evaluations
   */
  data_eval *dep;

  /** \brief Desc
   */
  double lw_10;

  /** \brief Desc
   */
  double r14_max;
  
  /** \brief List of filenames containing training tables
   */
  std::vector<std::string> files;
  
  /** \brief Number of parameters
   */
  size_t np;

  /** \brief Number of output quantities for em1
   */
  size_t nout;

  /** \brief Mapping of parameters to indices
   */
  o2scl::vec_index pvi;
  
  // Table from which to train
  o2scl::table_units<> table;

  /** \brief Mapping of output data to array indices
   */
  o2scl::vec_index dvi;

  /** \brief Random number generator
   */
  o2scl::rng<> r;

  emulator_rs();
  
  /** \brief Train the emulator with the data in \c tab_train
   */
  void train(o2scl::vec_index &pvii,
             o2scl::vec_index &dvii, data_eval *depi);
  
  /** \brief Evaluate the emulator, and or the full function if necessary
   */
  virtual int eval(size_t n, const ubvector &p, double &log_wgt,
                   std::array<double,ndat> &dat);
  
  /** \brief Evaluate the emulator, and or the full function if necessary
   */
  virtual int eval_unc(size_t n, const ubvector &p, double &log_wgt,
                       double &lw_unc, 
                       std::array<double,ndat> &dat,
                       std::array<double,ndat> &dat_unc);
  
};

/** \brief Main control class to perform an MCMC simulation
    of neutron star data
*/
class mcmc_wrapper {

protected:

public:

  /// The cooling NS data
  ins_data id;
  
  /// SXRTs
  sxrt_data sd;

  /// Neutron star mass and radius data
  nsmr_data nd;
  
  /// Number of parameters
  size_t nparam;

  /// No nuclei
  bool no_nuclei;

  /// No isolated NSs
  bool no_ins;

  /// Debug isolated NSs
  bool debug_ins;

  /// No SXRTs
  bool no_sxrt;
  
  /// MPI rank, set in constructor
  int mpi_rank;

  /// MPI size, set in constructor
  int mpi_size;

  /// Factor to decrease covariances by
  double covar_dec_factor;

  /// Factor to decrease variances by
  double var_dec_factor;
  
  /// Lower limit of parameter values, set in constructor
  ubvector low;

  /// Upper limit of parameter values, set in constructor
  ubvector high;
  
  /// Pointer to the \ref data_eval object for each thread
  data_eval *de;

  /// The number of data_eval objects
  size_t de_size;

  /** \brief Type of proposal distribution
   */
  std::string ptype;

  /// Factor to decrease step size by
  double step_fac;

  /// Maximum radius of a 1.4 solar mass NS
  double r14_max;

  /// The list of units, created in the constructor
  std::vector<std::string> param_units;

  /// The parameter list, created in the constructor
  o2scl::vec_index pvi;

  /// The data list, created in the constructor
  o2scl::vec_index dvi;

  /// Verbosity parameter
  int verbose;

  /// Number of OpenMP threads
  int n_threads;

  /// If true, write the temperature and luminosity profiles
  bool w_tl_prof;

  /** \brief New C++ emulator
   */
  bool flag_emu_aws;

  /** \brief Desc
   */
  emulator_rs ers;

  /// The MCMC object
  o2scl::mcmc_para_table<point_funct,fill_funct,
			 std::array<double,ndat>,
			 ubvector> mct;
  
  mcmc_wrapper();

  /** \brief Solve for the default steady-state heating curve
   */
  int steady_state(std::vector<std::string> &sv, bool itive_com);

  /** \brief Desc
   */
  int emu_aws(std::vector<std::string> &sv, bool itive_com);

  /** \brief Desc
   */
  int check_limits(std::vector<std::string> &sv, bool itive_com);
  
  /** \brief Desc
   */
  int clean(std::vector<std::string> &sv, bool itive_com);
  
  /** \brief Run the MCMC simulation
   */
  int mcmc(std::vector<std::string> &sv, bool itive_com);

  /** \brief Set initial point from last points in specified file
   */
  int initial_point_last(std::vector<std::string> &sv, bool itive_com);
  
  /** \brief Set initial point from first points in specified file
   */
  int initial_point_first(std::vector<std::string> &sv, bool itive_com);
  
  /** \brief Set initial point from best points in specified file
   */
  int initial_point_best(std::vector<std::string> &sv, bool itive_com);

  /** \brief Set initial point from random points in specified file
   */
  int initial_point_rand(std::vector<std::string> &sv, bool itive_com);
  
  /** \brief Set the number of threads
   */
  int threads(std::vector<std::string> &sv, bool itive_com);
  
  /** \brief Set initial point from best points in specified file
   */
  int make_gaussian(std::vector<std::string> &sv, bool itive_com);
  
  /** \brief Set initial point from best points in specified file
   */
  int make_gaussian2(std::vector<std::string> &sv, bool itive_com);
  
  /** \brief Desc
   */
  double gauss_min_func(size_t np, const ubvector &p, 
			o2scl::table_units<> &prev_res,
			o2scl::prob_dens_mdim_gaussian<> &mdim_gauss);
  
  /** \brief Desc
   */
  double gauss_min_func2(size_t np, const ubvector &p, 
			 o2scl::table_units<> &prev_res,
			 o2scl::prob_dens_mdim_gaussian<> &mdim_gauss);
  
  /** \brief Desc
   */
  int fit_gaussian(std::vector<std::string> &sv, bool itive_com);
  
  /** \brief Desc
   */
  int fit_gaussian2(std::vector<std::string> &sv, bool itive_com);

};

#endif
