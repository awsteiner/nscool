/*
  -------------------------------------------------------------------
  
  Copyright (C) 2015-2019, Andrew W. Steiner and Sophia Han
  
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
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <functional>

#ifndef NO_MPI
#include <mpi.h>
#endif

#include "nscool_wrap.h"

#include <o2scl/root_cern.h>
#include <o2scl/mroot_hybrids.h>
#include <o2scl/table_units.h>
#include <o2scl/hdf_file.h>
#include <o2scl/hdf_io.h>
#include <o2scl/hdf_eos_io.h>
#include <o2scl/nstar_cold.h>
#include <o2scl/eos_had_apr.h>
#include <o2scl/eos_had_skyrme.h>
#include <o2scl/eos_had_potential.h>
#include <o2scl/fermion.h>
#include <o2scl/cli_readline.h>
#include <o2scl/lib_settings.h>
#include <o2scl/mcmc_para.h>

using namespace std;
using namespace o2scl;
using namespace o2scl_hdf;
using namespace o2scl_const;

typedef boost::numeric::ublas::vector<double> ubvector;

/** \brief A Skyrme EOS modified at high densities
 */
class eos_had_skyrme2 : public eos_had_skyrme {

public:

  eos_had_skyrme2() {
    delta_P_K=0.0;
    delta_P_Gamma=0.0;
  }
    
  /** \brief The coefficient of the pressure modification
   */
  double delta_P_K;

  /** \brief The exponent of the pressure modification
   */
  double delta_P_Gamma;

  /** \brief Compute the EOS from the densities at zero temperature
   */
  virtual int calc_e(fermion &n, fermion &p, thermo &th) {
    eos_had_skyrme::calc_e(n,p,th);
    if (delta_P_Gamma>0.0 && th.ed>1.5) {
      th.pr+=(delta_P_K*pow(th.ed,delta_P_Gamma)-
	      delta_P_K*pow(1.5,delta_P_Gamma));
    }
    return 0;
  }
};

/** \brief An APR EOS with a Gibbs phase transition
 */
class apr_gibbs {
  
protected:

  /// \name Fermions
  //@{
  /// Compute zero-temperature thermodynamics
  fermion_zerot fzt;
  /// Neutron for low-density phase
  fermion n;
  /// Proton for low-density phase
  fermion p;
  /// Neutron for high-density phase
  fermion n2;
  /// Proton for high-density phase
  fermion p2;
  /// Electron for low-density phase
  fermion e;
  /// Muon for low-density phase
  fermion mu;
  //@}
  
  /// \name 'Thermo' objects
  //@{
  /// Baryon thermodynamics for low-density phase
  thermo hb;
  /// Leptonic thermodynamics for low-density phase
  thermo l;
  /// Baryon thermodynamics for high-density phase
  thermo hb2;
  /// Total thermodynamics
  thermo tot;
  /// Leptonic thermodynamics for high-density phase
  thermo l2;
  //@}

  /// \name Numerical methods
  //@{
  /// General solver
  mroot_hybrids<> solver;
  //@}

  /// Baryon density
  double nb;
  /// Volume fraction of low-density phase
  double chi;

  /// \name Phase specification
  //@{
  int phase;
  static const int low_phase=1;
  static const int mixed_phase=2;
  static const int high_phase=3;
  //@}

  /// Base APR EOS
  eos_had_apr apr;
  
  /// Table for output
  table_units<> at;
  
  /// HDF file for output
  hdf_file hf;

  /// For unit conversions, set in constructor
  convert_units &cu;

  /// Solve for neutron star matter (low-density phase)
  int nstar_low(size_t nv, const ubvector &x, ubvector &y) {
  
    n.n=x[0];
    p.n=nb-n.n;
  
    if (n.n<0.0 || p.n<0.0) {
      return 1;
    }

    apr.pion=eos_had_apr::ldp;
    apr.calc_e(n,p,hb);

    e.mu=n.mu-p.mu;
    mu.mu=e.mu;
    fzt.calc_mu_zerot(e);
    fzt.calc_mu_zerot(mu);
    l.ed=e.ed+mu.ed;
    l.pr=e.pr+mu.pr;
    l.en=e.en+mu.en;
  
    y[0]=p.n-e.n-mu.n;

    tot.pr=hb.pr+l.pr;
    tot.ed=hb.ed+l.ed;

    return 0;
  }

  /// Solve for neutron star matter (high-density phase)
  int nstar_high(size_t nv, const ubvector &x, ubvector &y) {

    n2.n=x[0];
    p2.n=nb-n2.n;
    
    if (n2.n<0.0 || p2.n<0.0) return 1;

    apr.pion=eos_had_apr::hdp;
    apr.calc_e(n2,p2,hb2);

    e.mu=n2.mu-p2.mu;
    mu.mu=e.mu;
    fzt.calc_mu_zerot(e);
    fzt.calc_mu_zerot(mu);
    l.ed=e.ed+mu.ed;
    l.pr=e.pr+mu.pr;
    l.en=e.en+mu.en;
  
    y[0]=p2.n-e.n-mu.n;

    tot.pr=hb2.pr+l.pr;
    tot.ed=hb2.ed+l.ed;
  
    return 0;
  }

  /// Solve for neutron star matter (mixed phase)
  int nstar_mixed(size_t nv, const ubvector &x, ubvector &y) {

    n.n=x[0];
    p.n=x[1];
    e.mu=x[2];
    n2.n=x[3];
    p2.n=x[4];
    mu.mu=e.mu;

    if (phase==low_phase) chi=1.0;
    else if (phase==high_phase) chi=0.0;
    else chi=x[5];

    if (n.n<0.0 || n2.n<0.0) return 1;
    if (p.n<0.0 || p2.n<0.0) return 1;

    apr.pion=eos_had_apr::ldp;
    apr.calc_e(n,p,hb);
  
    apr.pion=eos_had_apr::hdp;
    apr.calc_e(n2,p2,hb2);

    fzt.calc_mu_zerot(e);
    fzt.calc_mu_zerot(mu);
    l.ed=e.ed+mu.ed;
    l.pr=e.pr+mu.pr;
    l.en=e.en+mu.en;
  
    y[0]=nb-chi*(n.n+p.n)-(1.0-chi)*(n2.n+p2.n);
    y[1]=chi*p.n+(1.0-chi)*p2.n-e.n-mu.n;
    y[2]=n.mu-p.mu-e.mu;
    y[3]=p2.mu-p.mu;
    y[4]=n.mu-n2.mu;

    if (phase==mixed_phase) y[5]=hb.pr-hb2.pr;

    if (phase==low_phase) {
      tot.pr=hb.pr+l.pr;
      tot.ed=hb.ed+l.ed;
    } else if (phase==mixed_phase) {
      tot.pr=hb.pr+l.pr;
      tot.ed=hb.ed*chi+hb2.ed*(1.0-chi)+l.ed;
    } else {
      tot.pr=hb2.pr+l.pr;
      tot.ed=hb2.ed+l.ed;
    }

    return 0;
  }

public:

  apr_gibbs() : cu(o2scl_settings.get_convert_units()) {

    n.init(o2scl_settings.get_convert_units().convert
	   ("kg","1/fm",o2scl_mks::mass_neutron),2.0);
    p.init(o2scl_settings.get_convert_units().convert
	   ("kg","1/fm",o2scl_mks::mass_proton),2.0);
    n2.init(o2scl_settings.get_convert_units().convert
	    ("kg","1/fm",o2scl_mks::mass_neutron),2.0);
    p2.init(o2scl_settings.get_convert_units().convert
	    ("kg","1/fm",o2scl_mks::mass_proton),2.0);

    // Ensure that this works without GNU units
    o2scl_settings.get_convert_units().use_gnu_units=false;
    e.init(o2scl_settings.get_convert_units().convert
	   ("kg","1/fm",o2scl_mks::mass_electron),2.0);
    mu.init(o2scl_settings.get_convert_units().convert
	    ("kg","1/fm",o2scl_mks::mass_muon),2.0);

    n.non_interacting=false;
    p.non_interacting=false;
    n2.non_interacting=false;
    p2.non_interacting=false;
  }

  /// Main driver, computing the APR EOS and the associated M vs. R curve
  void run(table<> &nscool_core) {

    // Density grid
    double nbstart, nb_end, dnb;

    // Density at which to start looking for a mixed phase
    double nb_switch;

    // Variables and function values for solvers
    ubvector x(6), y(6);
  
    mm_funct f_nstar_mixed=std::bind
      (std::mem_fn<int(size_t,const ubvector &,ubvector &)>
       (&apr_gibbs::nstar_mixed),this,
       std::placeholders::_1,std::placeholders::_2,std::placeholders::_3);
    mm_funct f_nstar_low=std::bind
      (std::mem_fn<int(size_t,const ubvector &,ubvector &)>
       (&apr_gibbs::nstar_low),this,
       std::placeholders::_1,std::placeholders::_2,std::placeholders::_3);
    mm_funct f_nstar_high=std::bind
      (std::mem_fn<int(size_t,const ubvector &,ubvector &)>
       (&apr_gibbs::nstar_high),this,
       std::placeholders::_1,std::placeholders::_2,std::placeholders::_3);

    // Init density grid
    nbstart=0.005;
    dnb=0.002;
    nb_end=2.0;

    // Init solver tolerances
    solver.tol_abs=1.0e-10;
    solver.tol_rel=1.0e-12;

    // Initialize table
    nscool_core.clear_table();
    nscool_core.line_of_names("Rho Press nbar Ye Ymu Yn Yp mstp mstn");
    
    //--------------------------------------------------------------------
    // Init lepton fields to zero to start

    e.mu=0.0;
    e.n=0.0;
    e.kf=0.0;
    mu.mu=0.0;
    mu.n=0.0;
    mu.kf=0.0;

    //--------------------------------------------------------------------
    // Neutron star matter - Gibbs construction

    dnb=0.002;
    apr.pion=eos_had_apr::ldp;
    chi=1.0;
    at.clear_data();

    x[0]=nbstart/1.1;
    for(nb=nbstart;nb<=0.1701;nb+=dnb) {
      solver.msolve(1,x,f_nstar_low);
      double line[9]={cu.convert("1/fm^4","g/cm^3",tot.ed),
		      cu.convert("1/fm^4","dyne/cm^2",tot.pr),nb,
		      e.n/nb,mu.n/nb,(chi*n.n+(1.0-chi)*n2.n)/nb,
		      (chi*p.n+(1.0-chi)*p2.n)/nb,
		      (chi*p.ms+(1.0-chi)*p2.ms)/p.m,
		      (chi*n.ms+(1.0-chi)*n2.ms)/n.m};
      nscool_core.line_of_data(9,line);
    }
    nb-=dnb;
  
    phase=low_phase;
    x[0]=n.n;
    x[1]=p.n;
    x[2]=n.mu-p.mu;
    x[3]=n.n;
    x[4]=p.n;
  
    for(nb=0.17;nb<=nb_end;nb+=dnb) {
    
      if (phase!=mixed_phase) {
	solver.msolve(5,x,f_nstar_mixed);
	nstar_mixed(5,x,y);
      } else {
	solver.msolve(6,x,f_nstar_mixed);
      }
    
      if (hb.pr<hb2.pr && phase==low_phase) {
	cout << "Mixed phase begins at nb=" << nb << " fm^{-3}." << endl;
	phase=mixed_phase;
	x[5]=0.90;
	nb-=dnb;
      } else if (phase==mixed_phase && x[5]<0.0) {
	cout << "Mixed phase ends at nb=" << nb << " fm^{-3}." << endl;
	phase=high_phase;
	nb-=dnb;
      } else {
	double line[9]={cu.convert("1/fm^4","g/cm^3",tot.ed),
			cu.convert("1/fm^4","dyne/cm^2",tot.pr),nb,
			e.n/nb,mu.n/nb,(chi*n.n+(1.0-chi)*n2.n)/nb,
			(chi*p.n+(1.0-chi)*p2.n)/nb,
			(chi*p.ms+(1.0-chi)*p2.ms)/p.m,
			(chi*n.ms+(1.0-chi)*n2.ms)/n.m};
	nscool_core.line_of_data(9,line);
      }
    }
  
    return;
  }

};

/** \brief Class for computing steady-state accretion curves
 */
class sxrt_class : public nscool_wrap {

protected:

  /// \name Command-line parameter objects
  //@{
  o2scl::cli::parameter_string p_out_file;
  o2scl::cli::parameter_double p_eta;
  o2scl::cli::parameter_double p_mass;
  o2scl::cli::parameter_double p_T_fact_drip;
  o2scl::cli::parameter_double p_logT_init;
  o2scl::cli::parameter_double p_T_fact_surf;
  o2scl::cli::parameter_double p_Mdot_low;
  o2scl::cli::parameter_bool p_make_tables;
  o2scl::cli::parameter_int p_nscool_debug;
  o2scl::cli::parameter_double p_max_time;
  o2scl::cli::parameter_double p_fix_durca;
  o2scl::cli::parameter_double p_delta_P_K;
  o2scl::cli::parameter_double p_delta_P_Gamma;
  //@}

  /** \brief Table for the steady-state heating curve
   */
  o2scl::table_units<> ss_table;
  
  /** \brief The name of the output file
   */
  std::string out_file;

  /** \brief The neutron star mass
   */
  double mass;

  double schwarz_km;

  /** \brief Desc
   */
  double trad;

  /** \brief The lower limit of the Mdot loop
   */
  double Mdot_low;

  /** \brief The initial guess for log T
   */
  double logT_init;

  /** \brief A RMF EOS
   */
  eos_had_rmf rmf;

  /** \brief A potential model EOS
   */
  eos_had_potential pot;

  /** \brief The APR EOS used to compute effective masses
   */
  eos_had_apr apr;

  /** \brief The object for APR with a Gibbs phase transition
   */
  apr_gibbs ag;
  
  /** \brief The modified Skyrme EOS
   */
  eos_had_skyrme2 sk;
  
  /** \brief Object which performs unit conversions, set in constructor
   */
  convert_units &cu;

  /** \brief If true, throw an exception in the case of a convergence 
      failure
  */
  bool err_nonconv;
  
  /// The initial time
  double mpi_start_time;

  /// Maximum time for MCMC (default 28200)
  double max_time;
  
  /// Last time MCMC output was written to a file
  double last_output_time;

  /** \brief The MCMC auxillary data type
   */
  typedef std::array<double,7> mcmc_data_t;

  /// The MCMC point function type
  typedef std::function<int(size_t,const std::vector<double> &,double &,
			    mcmc_data_t &)> point_funct;

  /// The MCMC fill function type
  typedef std::function<int(const std::vector<double> &,double,
			    std::vector<double> &, mcmc_data_t &)> fill_funct;
  
  /// MCMC object
  mcmc_para_cli<point_funct,fill_funct,mcmc_data_t,std::vector<double> >
  mpc;

  /** \brief Desc
   */
  double cs2_max;
  
  /** \brief The most recent maximum mass from eos()
   */
  double ns_M_max;
  
  /** \brief The most recent maximum baryon density from eos()
   */
  double ns_nb_max;
  
  /** \brief The most recent maximum neutron density from eos()
   */
  double ns_np_max;
  
  /** \brief The most recent maximum proton density from eos()
   */
  double ns_nn_max;

  /** \brief The most recent proton density at saturation from eos()
   */
  double ns_np_sat;
  
  /** \brief The command-line object
   */
  cli_readline cl;
  
  /** \brief Neutron star structure
   */
  nstar_cold nc;

  /** \brief If true, cool for only \f$ 1.5 \times 10^{-12} \f$ s
      and set the initial temperature to \ref acc_Tinit
  */
  bool no_cooling;

  /** \brief If true, make tables (default false)
   */
  bool make_tables;

  /** \brief The coefficient of the pressure modification (default 0.0)
   */
  double delta_P_K;

  /** \brief The exponent of the pressure modification (default 1.0)
   */
  double delta_P_Gamma;

  /** \brief The deep crustal heating parameter in MeV (default 1.45)
   */
  double Q_heat;

  /** \brief Processor rank
   */
  int mpi_rank;

  /** \brief Number of processors
   */
  int mpi_nprocs;
  
  /** \brief Fill the MCMC data object
   */
  int fill_line(const std::vector<double> &pars, double log_weight, 
		std::vector<double> &line, mcmc_data_t &dat) {
    for(size_t i=0;i<dat.size();i++) line.push_back(dat[i]);
    return 0;
  }

public:

  sxrt_class() : nscool_wrap("./"),
		 cu(o2scl_settings.get_convert_units()) {
    no_cooling=false;
    out_file="sxrt.o2";
    mass=1.4;
    sfn3p2=101;
    sfp1s0=3;
    Mdot_low=3.0e-15;
    logT_init=8.0;
    make_tables=false;
    delta_P_K=0.0;
    delta_P_Gamma=1.0;
    //nc.include_muons=true;
    Q_heat=1.45;
    err_nonconv=true;
    mpi_start_time=0.0;
    nc.verbose=0;
    nc.def_tov.verbose=0;
    mpi_rank=0;
    mpi_nprocs=1;
    max_time=8*3600-600;
    schwarz_km=o2scl_mks::schwarzchild_radius/1.0e3;
  }

  /** \brief Run the MCMC
   */
  int mcmc(std::vector<std::string> &sv, bool itive_com) {

#ifndef NO_MPI
    MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&mpi_nprocs);
#endif
    
    // Set up MCMC point and line fill functions
    point_funct pf=std::bind
      (std::mem_fn<int(size_t,const std::vector<double> &,double &,
		       mcmc_data_t &)>(&sxrt_class::mcmc_point),this,
       std::placeholders::_1,std::placeholders::_2,std::placeholders::_3,
       std::placeholders::_4);
    fill_funct ff=std::bind
      (std::mem_fn<int(const std::vector<double> &,double,std::vector<double> &,
		       mcmc_data_t &)>(&sxrt_class::fill_line),this,
       std::placeholders::_1,std::placeholders::_2,std::placeholders::_3,
       std::placeholders::_4);

    // Parameter limits and initial point
    std::vector<double> low(14), high(14), init(14);
    low[0]=1.0;
    low[1]=1.0;
    low[2]=-5.0;
    low[3]=0.0;
    low[4]=-18.0;
    low[5]=0.16;
    low[6]=7.0;
    low[7]=cbrt(3.0*o2scl_const::pi2*0.16);
    low[8]=0.0;
    low[9]=7.0;
    low[10]=0.0;
    low[11]=0.0;
    low[12]=0.01;
    low[13]=1.0;

    high[0]=2.5;
    high[1]=2.5;
    high[2]=2.0;
    high[3]=2.0;
    high[4]=-7.0;
    high[5]=2.0;
    high[6]=10.0;
    high[7]=5.0;
    high[8]=5.0;
    high[9]=10.0;
    high[10]=5.0;
    high[11]=5.0;
    high[12]=0.75;
    high[13]=1.5;

    // Ensure we're using the parameterized gaps for the cooling
    // code
    sfp1s0=150;
    sfn3p2=150;
    
    vector<string> pnames={"M_1808","M_AqX1","K","Gamma","eta","nb_durca",
			   "log10_n3_tc","n3_kf","n3_dk",
			   "log10_p1_tc","p1_kf",
			   "p1_dk","alpha","Q"};
    vector<string> punits={"Msun","Msun","fm^{-4*(1-Gamma)}","",
			   "","1/fm^3","log10_K","1/fm","1/fm",
			   "log10_K","1/fm","1/fm","","MeV"};
			   
    /*
      for(size_t i=0;i<100;i++) {
      pnames.push_back(((std::string)"Ca_")+std::to_string(i));
      punits.push_back("K");
      }
      for(size_t i=0;i<100;i++) {
      pnames.push_back(((std::string)"Cb_")+std::to_string(i));
      punits.push_back("K");
      }
    */
    pnames.push_back("L_AqX1");
    punits.push_back("erg/s");
    pnames.push_back("L_1808");
    punits.push_back("erg/s");
    pnames.push_back("logT_AqX1");
    punits.push_back("log(K)");
    pnames.push_back("logT_1808");
    punits.push_back("log(K)");
    pnames.push_back("M_max");
    punits.push_back("Msun");
    pnames.push_back("nb_max");
    punits.push_back("1/fm^3");
    pnames.push_back("cs2_max");
    punits.push_back("");

    mpc.set_names_units(pnames,punits);
    
    mpc.aff_inv=true;
    mpc.n_walk=20;
    
#ifndef NO_MPI
    
    mpi_start_time=MPI_Wtime();
    last_output_time=mpi_start_time;
    
#endif
    
    if (true) {

      // Trying to find a better guess:
      
      double log_wgt;
      mcmc_data_t dat;

      init[0]=1.944383;
      init[1]=1.368001;
      init[2]=-1.96667;
      init[3]=0.03383324;
      init[4]=-11.13438;
      init[5]=0.5257649;
      init[6]=8.152829;
      init[7]=2.853505;
      init[8]=0.03641456;
      init[9]=7.537997;
      init[10]=0.6649047;
      init[11]=0.6108493;
      init[12]=0.09872011;
      init[13]=1.15724;

      /*
	low[7]=init[7]*(1.0-1.0e-4);
	low[8]=init[8]*(1.0-1.0e-4);
	low[10]=init[10]*(1.0-1.0e-4);
	low[11]=init[11]*(1.0-1.0e-4);
	
	high[7]=init[7]*(1.0+1.0e-4);
	high[8]=init[8]*(1.0+1.0e-4);
	high[10]=init[10]*(1.0+1.0e-4);
	high[11]=init[11]*(1.0+1.0e-4);
      */

      //mcmc_point(14,init,log_wgt,dat);
      //cout << log_wgt << endl;
      //      exit(-1);
    }

    if (true) {
      double lw;
      init[0]=1.737865;
      init[1]=1.006445;
      init[2]=-2.077553;
      init[3]=0.09086287;
      init[4]=-10.21136;
      init[5]=0.4140207;
      init[6]=7.235262;
      init[7]=2.925892;
      init[8]=0.3934482;
      init[9]=8.445619;
      init[10]=0.6124528;
      init[11]=0.04619732;
      init[12]=0.0294544;
      init[13]=1.014136;
      std::array<double,7> dat;
      for(size_t i=0;i<40;i++) {
	int ret=mcmc_point(14,init,lw,dat);
	cout << "X:" << endl;
	for(size_t j=0;j<14;j++) {
	  cout << init[j] << " ";
	}
	for(size_t j=0;j<7;j++) {
	  cout << dat[j] << " ";
	}
	cout << endl;
	init[2]+=0.1;
      }
      exit(-1);
    }
    
    mpc.step_fac=80.0;
    //mpc.step_fac=160.0;
    mpc.verbose=1;
    if (mpi_nprocs>1) mpc.verbose=0;
    mpc.user_seed=1;
    err_nonconv=false;
    vector<point_funct> vpf(1);
    vpf[0]=pf;
    vector<fill_funct> vff(1);
    vff[0]=ff;
    mpc.mcmc(14,low,high,vpf,vff);
    err_nonconv=true;
    
    return 0;
  }

  /** \brief Evaluate one MCMC point
   */
  int mcmc_point(size_t np, const std::vector<double> &pars, 
		 double &log_weight, mcmc_data_t &dat) {

    double mass_1808=pars[0];
    double mass_AqX1=pars[1];

    delta_P_K=pars[2];
    delta_P_Gamma=pars[3];

    double eta_AqX1=pars[4];
    double nb_durca=pars[5];
    fix_durca=nb_durca;

    n3_tc=pow(10.0,pars[6]);
    n3_kf=pars[7];
    n3_dk=pars[8];
    p1_tc=pow(10.0,pars[9]);
    p1_kf=pars[10];
    p1_dk=pars[11];

    if (n3_dk>n3_kf) {
      if (mpi_nprocs==1) {
	cout << "Neutron SF dk too large." << endl;
      }
      return 7;
    }
    if (p1_dk>p1_kf) {
      if (mpi_nprocs==1) {
	cout << "Proton SC dk too large." << endl;
      }
      return 8;
    }

    // AWS: alpha is the broadening parameter and beta is the
    // fractional decrease of the direct Urca threshold
    alpha_durca=pars[12];

    Q_heat=pars[13];
    
    std::vector<std::string> sv(3);

    // Check that the direct Urca does not start below
    // saturation
    if (nb_durca*(1.0-alpha_durca)<0.16) {
      if (mpi_nprocs==1) {
	cout << "Urca density too small" << endl;
      }
      return 3;
    }
    
    double logT_1808, logT_AqX1;

    // ----------------------------------------------------
    // 1808 section

    eta=0.0;
    mass=mass_1808;

    sv[0]="eos";
    //sv[1]="hhj";
    sv[1]="skyrme";
    sv[2]="SLy4";
    int ret=eos(sv,false);
    if (ret!=0) {
      if (mpi_nprocs==1) {
	cout << "EOS function failed." << endl;
      }
      return 9;
    }

    // Check that maximum mass is sufficiently large
    if (ns_M_max<2.0) {
      if (mpi_nprocs==1) {
	cout << "Maximum mass, " << ns_M_max << ", too small." << endl;
      }
      return 10;
    }
    // Check that masses are less than maximum
    if (mass_1808>ns_M_max || mass_AqX1>ns_M_max) {
      if (mpi_nprocs==1) {
	cout << "Mass greater than maximum." << endl;
      }
      return 2;
    }
    // Check for causality
    if (ns_nb_max>nc.acausal) {
      if (mpi_nprocs==1) {
	cout << "EOS acausal." << endl;
      }
      return 1;
    }

    if (p1_kf<cbrt(3.0*o2scl_const::pi2*ns_np_sat)) {
      if (mpi_nprocs==1) {
	cout << "Proton SC kf smaller than saturation in 1808." << endl;
      }
      return 4;
    }

    if (n3_kf>cbrt(3.0*o2scl_const::pi2*ns_nn_max)) {
      if (mpi_nprocs==1) {
	cout << "Neutron SF kf greater than max for 1808: " << n3_kf << " "
	     << ns_nn_max << endl;
      }
      return 5;
    }
    if (p1_kf>cbrt(3.0*o2scl_const::pi2*ns_np_max)) {
      if (mpi_nprocs==1) {
	cout << "Proton SC kf greater than max for 1808: " << p1_kf << " "
	     << ns_np_max << endl;
      }
      return 6;
    }

    double lphot,lneut,lheat;
    {
      no_cooling=true;
      time_print[0]=1.1e-12;
      
      logT_1808=logT_init;
      double Mdot=1.0e-11;
      int ret1=solve(Mdot,logT_1808);
      if (ret1!=0) {
	if (mpi_nprocs==1) {
	  cout << "Solver failed for 1808." << endl;
	}
	return 14;
      }
      acc_compute(logT_1808,lphot,lneut,lheat);
    }
    double Lphot_1808=log10(lphot);
    double wgt_1808=1.0/(1.0+exp((Lphot_1808-log10(5.0e30))/0.5));
    
    // ----------------------------------------------------
    // AqX1 section

    eta=pow(10.0,eta_AqX1);
    mass=mass_AqX1;
    
    eos(sv,false);
    if (ret!=0) {
      if (mpi_nprocs==1) {
	cout << "EOS function failed." << endl;
      }
      return 9;
    }

    if (p1_kf<cbrt(3.0*o2scl_const::pi2*ns_np_sat)) {
      if (mpi_nprocs==1) {
	cout << "Proton SC kf smaller than saturation in AqX1." << endl;
      }
      return 11;
    }
    if (n3_kf>cbrt(3.0*o2scl_const::pi2*ns_nn_max)) {
      if (mpi_nprocs==1) {
	cout << "Neutron SF kf greater than max for AqX1: " << n3_kf << " "
	     << ns_nn_max << endl;
      }
      return 12;
    }
    if (p1_kf>cbrt(3.0*o2scl_const::pi2*ns_np_max)) {
      if (mpi_nprocs==1) {
	cout << "Proton SC kf greater than max for AqX1: " << p1_kf << " "
	     << ns_np_max << endl;
      }
      return 13;
    }

    {
      no_cooling=true;
      time_print[0]=1.1e-12;
      
      logT_AqX1=logT_init+0.5;
      double Mdot=4.0e-10;
      int ret2=solve(Mdot,logT_AqX1);
      if (ret2!=0) {
	if (mpi_nprocs==1) {
	  cout << "Solver failed for AqX1." << endl;
	}
	return 15;
      }
      acc_compute(logT_AqX1,lphot,lneut,lheat);
    }
    double Lphot_AqX1=log10(lphot);
    double wgt_AqX1=exp(-pow((Lphot_AqX1-log10(5.3e33))/0.5,2.0)/2.0);
    
    // ----------------------------------------------------
    // Final computation of weight

    if (mpi_nprocs==1) {
      cout << "Lphot_AqX1,Lphot_1808: " << Lphot_AqX1 << " " << Lphot_1808
	   << endl;
      
      cout << "wgt_AqX1,wgt_1808: " << wgt_AqX1 << " " << wgt_1808 << endl;
    }
    
    double weight=wgt_AqX1*wgt_1808;

    log_weight=log(weight);

    if (mpi_nprocs==1) {
      cout << "wgt,log(wgt): " << weight << " " << log_weight << endl;
    }

    // ----------------------------------------------------
    // Add data 

    dat[0]=Lphot_AqX1;
    dat[1]=Lphot_1808;
    dat[2]=logT_AqX1;
    dat[3]=logT_1808;
    dat[4]=ns_M_max;
    dat[5]=ns_nb_max;
    dat[6]=cs2_max;
    
    if (false) {
      cout << "init[0]=" << pars[0] << ";" << endl;
      cout << "init[1]=" << pars[1] << ";" << endl;
      cout << "init[2]=" << pars[2] << ";" << endl;
      cout << "init[3]=" << pars[3] << ";" << endl;
      cout << "init[4]=" << pars[4] << ";" << endl;
      cout << "init[5]=" << pars[5] << ";" << endl;
      cout << "init[6]=" << pars[6] << ";" << endl;
      cout << "init[7]=" << pars[7] << ";" << endl;
      cout << "init[8]=" << pars[8] << ";" << endl;
      cout << "init[9]=" << pars[9] << ";" << endl;
      cout << "init[10]=" << pars[10] << ";" << endl;
      cout << "init[11]=" << pars[11] << ";" << endl;
      cout << "init[12]=" << pars[12] << ";" << endl;
      cout << "init[13]=" << pars[13] << ";" << endl;
    }
    
    //char ch;
    //std::cin >> ch;
    
    return 0;
  }
  
  /** \brief Main output function for NSCool
   */
  virtual void main_out(double &time, double &tptr,
			double &lphot, double &lneut, double &lheat) {

    if (no_cooling==false) {
      std::cout.width(4);
      std::cout << v_time.size() << " "
		<< time << " " << tptr << " " << lphot << " "
		<< lneut << " " << lheat << std::endl;
    }

    v_time.push_back(time);
    v_tptr.push_back(tptr);
    v_lphot.push_back(lphot);
    v_lneut.push_back(lneut);
    v_lheat.push_back(lheat);
    
    return;
  }

  /** \brief The overloaded function from \ref nscool_wrap::cool_param()
   */
  virtual void cool_param(int &pscreen, double &debug, int &istep_debug,
			  double &pteff, double &ptemp, double &pstar,
			  int &idump1, int &idump2, int &idump3,
			  double &tempmin, double &tempini,
			  int &icvel_nodeg, double &emnco, double &emncr,
			  double &emp, double &p0, int &itpmax,
			  double *tprint) {

    nscool_wrap::cool_param(pscreen,debug,istep_debug,pteff,ptemp,
			    pstar,idump1,idump2,idump3,tempmin,
			    tempini,icvel_nodeg,emnco,emncr,emp,p0,
			    itpmax,tprint);
    
    if (no_cooling) {
      tempini=acc_Tinit;
    }

    return;
  }

  /** \brief The overloaded function from \ref nscool_wrap::num_param()
   */
  virtual void num_param(double &time0, double &timemax, int &istepmax,
			 int &itrial_max, int &itrial_opt, double &tcut,
			 double &dtime, double &dtlimit, double &scale_dt0,
			 double &scale_dt1, double &repeat, int &istart,
			 double &mratt, double &mratl, double &mrats,
			 double &tvar, double &svar, double &tcon) {

    nscool_wrap::num_param(time0,timemax,istepmax,itrial_max,itrial_opt,
			   tcut,dtime,dtlimit,scale_dt0,scale_dt1,
			   repeat,istart,mratt,mratl,mrats,tvar,svar,
			   tcon);
    if (no_cooling) {
      timemax=1.5e-12;
    }
    
    return;
  }
  
protected:
  
  /** \brief Initial core temperature
   */
  double acc_Tinit;

  /** \brief Accretion rate
   */
  double acc_Mdot;
  
  /** \brief Solve for steady-state at specified accretion rate
      using initial guess for core temperature
  */
  int solve(double Mdot, double &logT) {
    acc_Mdot=Mdot;
    mroot_hybrids<> mh;
    mm_funct func=std::bind(std::mem_fn<int(size_t,const ubvector &,
					    ubvector &)>
			    (&sxrt_class::match_lum),this,
			    std::placeholders::_1,std::placeholders::_2,
			    std::placeholders::_3);
    ubvector x(1), y(1);
    x[0]=logT;
    // Ignore solver convergence errors
    mh.err_nonconv=false;
    mh.def_jac.err_nonconv=false;
    // First, ensure initial guess is valid
    int ret=match_lum(1,x,y);
    // If so, then call solver
    if (ret==0) {
      ret=mh.msolve(1,x,func);
      logT=x[0];
    }
    // If it failed, try a different initial guess
    if (ret!=0) {
      cout << "Solver recovery:" << endl;
      cout << "logT_init: " << logT_init << endl;
      for(double init=6.0;ret!=0 && init<8.501;init+=0.5) {
	x[0]=init;
	// Try to get an initial guess from this value of log T
	ret=match_lum(1,x,y);
	cout << "ret: " << init << " " << ret << endl;
	// If it worked, call the solver
	if (ret==0) {
	  mh.verbose=1;
	  ret=mh.msolve(1,x,func);
	}
	cout << "ret: " << init << " " << ret << endl;
      }
    }
    return ret;
  }
  
  /** \brief Return the difference between the total
      and target heating luminosity
  */
  int match_lum(size_t nv, const ubvector &x, ubvector &y) {
    double logT=x[0];
    double lphot, lneut, lheat;
    int ret=acc_compute(logT,lphot,lneut,lheat);
    y[0]=(lphot+lneut-lheat)/lheat;
    return ret;
  }

  /** \brief At a fixed temperature, run the cooling code 
      to determine the photon neutrino and heating luminosities
  */
  int acc_compute(double logT, double &lphot, double &lneut,
		  double &lheat) {
    acc_Tinit=pow(10.0,logT);
    int ret=run(0);
    if (ret!=0) {
      cout << "Failed in acc_compute()." << endl;
      if (err_nonconv) {
	O2SCL_ERR("Failed in acc_compute().",exc_efailed);
      }
    }
    lphot=v_lphot[v_lphot.size()-1];
    lneut=v_lneut[v_lneut.size()-1];
    lheat=Q_heat*acc_Mdot/1.0e-10*6.03e33*sqrt(1.0-schwarz_km*mass/trad);
    return ret;
  }
  
public:
  
  /** \brief Set the superfluid/superconducting gaps
   */
  int gaps_cmd(std::vector<std::string> &sv, bool itive_com) {
    sfn1s0=std::stoi(sv[1]);
    sfn3p2=std::stoi(sv[2]);
    sfp1s0=std::stoi(sv[3]);
    size_t index=4;
    if (sfn1s0==150) {
      n1_tc=std::stod(sv[index]);
      index++;
      n1_kf=std::stod(sv[index]);
      index++;
      n1_dk=std::stod(sv[index]);
      index++;
    }
    if (sfn3p2==150) {
      n3_tc=std::stod(sv[index]);
      index++;
      n3_kf=std::stod(sv[index]);
      index++;
      n3_dk=std::stod(sv[index]);
      index++;
    }
    if (sfp1s0==150) {
      p1_tc=std::stod(sv[index]);
      index++;
      p1_kf=std::stod(sv[index]);
      index++;
      p1_dk=std::stod(sv[index]);
      index++;
    }
    return 0;
  }

  /** \brief Reformat the EOS and TOV tables for use in Dany's code
   */
  int reformat(std::vector<std::string> &sv, bool itive_com) {

    ofstream fout;

    fout.open(sv[1]);
    fout.setf(ios::scientific);
    fout << "      6    "
	 << nscool_core.get_nlines()-1 << "    "
	 << nscool_core.get_nlines()-1 << "    Itext Imax Icore" << std::endl;
    fout << std::endl;
    fout << "      EOS comment" << std::endl;
    fout << std::endl;
    fout << "   Rho        Press       nbar       Ye         "
	 << "Ymu        Yn         Yp         Yla        "
	 << "Ysm        Ys0        Ysp       mstp       mstn       "
	 << "mstla      mstsm      msts0      mstsp" << std::endl;
    fout << "  g/cm3     dyne/cm2     #/fm3     [A_cell]    "
	 << "[A_ion]     [Z]" << std::endl;
    fout.precision(4);
    for(size_t i=0;i<nscool_core.get_nlines();i++) {
      fout << " ";
      fout << nscool_core.get("Rho",i) << " ";
      fout << nscool_core.get("Press",i) << " ";
      fout << nscool_core.get("nbar",i) << " ";
      fout << nscool_core.get("Ye",i) << " ";
      fout << nscool_core.get("Ymu",i) << " ";
      fout << nscool_core.get("Yn",i) << " ";
      fout << nscool_core.get("Yp",i) << " ";
      fout << 0.0 << " ";
      fout << 0.0 << " ";
      fout << 0.0 << " ";
      fout << 0.0 << " ";
      fout << nscool_core.get("mstp",i) << " ";
      fout << nscool_core.get("mstn",i) << " ";
      fout << 0.0 << " ";
      fout << 0.0 << " ";
      fout << 0.0 << " ";
      fout << 0.0 << std::endl;
    }
    fout.close();

    fout.open(sv[2]);
    fout << "       6     "
	 << nscool_tov.get_nlines()-1 << "      "
	 << 0 << "      " << 0 << std::endl;
    fout << std::endl;
    fout << "       TOV Comment." << std::endl;
    fout << std::endl;
    fout << "  step        radius       baryon#       "
	 << "density        pressure       encl. mass            "
	 << "phi     encl. bar. mass " << std::endl;
    fout << "                (m)        (#/fm3)       "
	 << "(g/cm3)        (dyn/cm2)      (sol. mass)           "
	 << "          (sol. mass)   " << std::endl;
    fout << endl;
    for(size_t i=0;i<nscool_tov.get_nlines();i++) {
      fout << "   ";
      fout.width(3);
      fout.setf(ios::right);
      fout << ((int)(nscool_tov.get("step",i)+1.0e-8)) << "     ";
      fout.unsetf(ios::right);
      fout.precision(8);
      fout << nscool_tov.get("radius",i) << "   ";
      fout.setf(ios::scientific);
      fout.precision(6);
      fout << nscool_tov.get("n_baryon",i) << "   ";
      fout << nscool_tov.get("density",i) << "   ";
      fout << nscool_tov.get("pressure",i) << "   ";
      fout.precision(8);
      fout << nscool_tov.get("emass",i) << "   ";
      fout.precision(6);
      fout << nscool_tov.get("phi",i) << "   ";
      fout.precision(8);
      fout << nscool_tov.get("bmass",i) << std::endl;
      fout.unsetf(ios::scientific);
    }
    fout.close();
    
    return 0;
  }

  /** \brief Compute the EOS and TOV profile
   */
  int eos(std::vector<std::string> &sv, bool itive_com) {

    string type=sv[1], name="";
    if (sv.size()>2) name=sv[2];

    convert_units &cu=o2scl_settings.get_convert_units();
    eos_had_base *eos_had_ptr=0;

    if (type=="skyrme") {

      sk.delta_P_K=delta_P_K;
      sk.delta_P_Gamma=delta_P_Gamma;

      if (mpi_nprocs==1) {
	cout << "Using K=" << sk.delta_P_K << " and Gamma="
	     << sk.delta_P_Gamma << endl;
      }
      
      skyrme_load(sk,name);
      eos_had_ptr=&sk;

      // ------------------------------------------------------------
      // Construct the EOS from an nstar_cold object
      
      // Automatically construct beta-equilibrium EOS
      nc.err_nonconv=err_nonconv;
      nc.def_tov.err_nonconv=err_nonconv;
      nc.set_eos(sk);
      int eos_ret=nc.calc_eos();
      if (eos_ret!=0) {
	if (mpi_nprocs==1) {
	  cout << "EOS calculation in class nstar_cold failed." << endl;
	}
	return 16;
      }
      if (mpi_nprocs==1) {
	cout << "acausal: " << nc.acausal << endl;
      }
      shared_ptr<o2scl::table_units<> > nc_eos=nc.get_eos_results();

      // Check for increasing pressure
      double pr_bad=-1.0;
      double ed_bad=-1.0;
      size_t i_bad=0;
      for(size_t i=0;i<nc_eos->get_nlines()-1;i++) {
	if (pr_bad<=0.0 && nc_eos->get("pr",i+1)<nc_eos->get("pr",i)) {
	  pr_bad=nc_eos->get("pr",i);
	  ed_bad=nc_eos->get("ed",i);
	  i_bad=i;
	  if (mpi_nprocs==1) {
	    cout << i_bad << " " << ed_bad << " " << pr_bad << endl;
	  }
	}
      }
      if (pr_bad>0.0 && ed_bad<1.0) {
	if (mpi_nprocs==1) {
	  cout << "Pressure decreasing at a low density" << endl;
	}
	return 10;
      }
      if (pr_bad>0.0) {
	for(size_t i=i_bad+1;i<nc_eos->get_nlines();i++) {
	  nc_eos->set("pr",i,pr_bad+0.01*((double)(i-i_bad)));
	}
      }
      
      nc.def_eos_tov.read_table(*nc_eos,"ed","pr","nb");

      ns_np_sat=nc_eos->interp("nb",0.16,"np");

      // Prepare to compute effective masses
      fermion n(o2scl::o2scl_settings.get_convert_units().convert
		("kg","1/fm",o2scl_mks::mass_neutron),2.0);
      fermion p(o2scl::o2scl_settings.get_convert_units().convert
		("kg","1/fm",o2scl_mks::mass_proton),2.0);
      n.non_interacting=false;
      p.non_interacting=false;
      thermo th;
      
      // ------------------------------------------------------------
      // Create a core table of the correct size with the correct units

      // Clear table for new columns
      nscool_core.clear_table();
      nscool_core.line_of_names("Rho Press nbar Ye Ymu Yn Yp mstp mstn");
      nscool_core.line_of_names("Yla Ysm Ys0 Ysp mstla mstsm msts0 mstsp");
      
      // Baryon density grid
      double nb_min=0.09;
      double nb_max=nc_eos->max("nb");
      double dnb=(nb_max-nb_min)/178.0;
      
      // Fill core table
      for(double nb=nb_max;nb>nb_min-dnb/10.0;nb-=dnb) {
	n.n=nc_eos->interp("nb",nb,"nn");
	p.n=nc_eos->interp("nb",nb,"np");
	sk.calc_e(n,p,th);
	double Ymu=0.0;
	//if (nc_eos->is_column("nmu")) nc_eos->interp("nb",nb,"nmu")/nb;
	//if (Ymu<0.0) Ymu=0.0;
	double line[17]={cu.convert("1/fm^4","g/cm^3",
				    nc_eos->interp("nb",nb,"ed")),
			 cu.convert("1/fm^4","dyne/cm^2",
				    nc_eos->interp("nb",nb,"pr")),nb,
			 nc_eos->interp("nb",nb,"ne")/nb,
			 Ymu,
			 nc_eos->interp("nb",nb,"nn")/nb,
			 nc_eos->interp("nb",nb,"np")/nb,
			 n.ms/n.m,p.ms/p.m,
			 0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0};
	nscool_core.line_of_data(17,line);
      }
      
      if (make_tables) {
	o2scl_hdf::hdf_file hf2;
	hf2.open_or_create("core.o2");
	hdf_output(hf2,nscool_core,"core");
	hf2.close();
      }

      // ------------------------------------------------------------
      // Use the nstar_cold object to construct the profile

      // Construct neutron star profile
      nc.def_tov.calc_gpot=true;
      int mvsr_ret=nc.calc_nstar();
      if (mvsr_ret!=0) {
	if (mpi_nprocs==1) {
	  cout << "Mass-radius calculation in class nstar_cold failed." << endl;
	}
	return 17;
      }
      shared_ptr<o2scl::table_units<> > nc_prof=nc.get_tov_results();

      if (make_tables) {
	o2scl_hdf::hdf_file hf2;
	hf2.open_or_create("mvsr.o2");
	hdf_output(hf2,*nc_prof,"mvsr");
	hf2.close();
      }
      
      ns_M_max=nc_prof->max("gm");
      if (mpi_nprocs==1) {
	cout << "M_max: " << ns_M_max << endl;
      }
      // Remove all rows past maximum mass
      nc_prof->set_nlines(nc_prof->lookup("gm",ns_M_max)+1);
      ns_nb_max=nc_prof->get("nb",nc_prof->lookup("gm",ns_M_max));
      if (fix_durca>0.0) {
	if (mpi_nprocs==1) {
	  cout << "fix_durca, M_dUrca: " << fix_durca
	       << " " << nc_prof->interp("nb",fix_durca,"gm") << endl;
	}
	//for(size_t i=0;i<nc_prof->get_nlines();i++) {
	//cout << nc_prof->get("nb",i) << " " << nc_prof->get("gm",i) << endl;
	//}
      }
      if (mpi_nprocs==1) {
	cout << "nb_M_max: " << ns_nb_max << endl;
      }
      ns_nn_max=nc_eos->interp("nb",ns_nb_max,"nn");
      ns_np_max=nc_eos->interp("nb",ns_nb_max,"np");
      if (nc_prof->get("pr",nc_prof->lookup("gm",ns_M_max))<pr_bad) {
	if (mpi_nprocs==1) {
	  cout << "Pressure bad below central pressure of maximum mass"
	       << endl;
	}
	return 11;
      }

      cs2_max=nc_eos->interp("ed",nc_prof->max("ed"),"cs2");

      double mass2=mass;
      if (mass2<0.0) mass2=mass+nc_prof->max("gm");
      if (mass2>ns_M_max) {
	if (mpi_nprocs==1) {
	  cout << "Mass " << mass2 << " larger than maximum mass: "
	       << ns_M_max << endl;
	}
	return 12;
      }
      if (mpi_nprocs==1) {
	cout << "Set mass to " << mass2 << endl;
      }
      int mass_ret=nc.fixed(mass2);
      if (mass_ret!=0) {
	if (mpi_nprocs==1) {
	  cout << "Fixed-mass calculation in class nstar_cold failed." << endl;
	}
	return 17;
      }
      
      // ------------------------------------------------------------
      // Create a table with the right size and the right units

      // Clear table for new columns
      nscool_tov.clear_table();
      nscool_tov.line_of_names(((string)"step radius n_baryon density ")+
			       "pressure emass phi bmass");
      
      // Radial grid
      double r_max=nc_prof->max("r");
      trad=r_max;
      
      // Fill TOV table
      size_t ix=0;
      for(size_t ix=0;ix<153;ix++) {
	// A grid which focuses most of the points on the outer part
	// of the star
	double r=r_max*2.0*(1.0-pow(2.0,-((double)ix)/152.0));
	double line[8]={((double)ix),r*1.0e3,nc_prof->interp("r",r,"nb"),
			cu.convert("1/fm^4","g/cm^3",
				   nc_prof->interp("r",r,"ed")),
			cu.convert("1/fm^4","dyne/cm^2",
				   nc_prof->interp("r",r,"pr")),
			nc_prof->interp("r",r,"gm"),
			nc_prof->interp("r",r,"gp"),
			nc_prof->interp("r",r,"bm")};
	nscool_tov.line_of_data(8,line);
      }
      
      if (make_tables) {
	o2scl_hdf::hdf_file hf2;
	hf2.open_or_create("tov.o2");
	hdf_output(hf2,nscool_tov,"tov");
	hf2.close();
      }

    } else if (type=="apr") {

      eos_had_ptr=&apr;

      // ------------------------------------------------------------
      // Compute EOS
      
      table<> tab_eos;
      ag.run(tab_eos);

      // ------------------------------------------------------------
      // Reinterpolate to generate core table

      table<> nscool_core_old=nscool_core;
      
      // Clear table for new columns
      nscool_core.clear_table();
      nscool_core.line_of_names("Rho Press nbar Ye Ymu Yn Yp mstp mstn");
      
      // Baryon density grid
      double nb_min=0.09;
      double nb_max=tab_eos.max("nbar");
      double dnb=(nb_max-nb_min)/178.0;

      nscool_core_old.set_interp_type(itp_linear);
      // Fill core table
      for(double nb=nb_max;nb>nb_min-dnb/10.0;nb-=dnb) {
	double line[9]={nscool_core_old.interp("nbar",nb,"Rho"),
			nscool_core_old.interp("nbar",nb,"Press"),
			nb,
			nscool_core_old.interp("nbar",nb,"Ye"),
			nscool_core_old.interp("nbar",nb,"Ymu"),
			nscool_core_old.interp("nbar",nb,"Yn"),
			nscool_core_old.interp("nbar",nb,"Yp"),
			nscool_core_old.interp("nbar",nb,"mstp"),
			nscool_core_old.interp("nbar",nb,"mstn")};
	nscool_core.line_of_data(9,line);
      }
      
      if (make_tables) {
	o2scl_hdf::hdf_file hf2;
	hf2.open_or_create("core.o2");
	hdf_output(hf2,nscool_core,"core");
	hf2.close();
      }
      
      // ------------------------------------------------------------
      // Run TOV solver

      tov_solve ts;
      eos_tov_interp te;
      te.default_low_dens_eos();
      ts.verbose=0;
      table_units<> core2;//=nscool_core;
      core2.set_unit("Rho","g/cm^3");
      core2.set_unit("Press","dyne/cm^2");
      core2.set_unit("nbar","1/fm^3");
      core2.sort_table("nbar");
      te.read_table(core2,"Rho","Press","nbar");
      ts.set_units("g/cm^3","dyne/cm^2","1/fm^3");
      ts.set_eos(te);
      ts.calc_gpot=true;

      ts.max();
      shared_ptr<o2scl::table_units<> > nc_tov=ts.get_results();
      
      double mass2=mass;
      if (mass2<0.0) mass2=mass+nc_tov->max("gm");
      if (mpi_nprocs==1) {
	cout << "Set mass to " << mass2 << endl;
      }
      ts.fixed(mass2);

      // Radial grid
      double r_max=nc_tov->max("r");
      
      // Clear table for new columns
      nscool_tov.clear_table();
      nscool_tov.line_of_names(((string)"step radius n_baryon density ")+
			       "pressure emass phi bmass");

      // Fill TOV table
      size_t ix=0;
      for(size_t ix=0;ix<153;ix++) {
	// A grid which focuses most of the points on the outer part
	// of the star
	double r=r_max*2.0*(1.0-pow(2.0,-((double)ix)/152.0));
	double line[8]={((double)ix),r*1.0e3,nc_tov->interp("r",r,"nb"),
			nc_tov->interp("r",r,"ed"),
			nc_tov->interp("r",r,"pr"),
			nc_tov->interp("r",r,"gm"),
			nc_tov->interp("r",r,"gp"),
			nc_tov->interp("r",r,"bm")};
	nscool_tov.line_of_data(8,line);
      }

      if (make_tables) {
	o2scl_hdf::hdf_file hf2;
	hf2.open_or_create("tov.o2");
	hdf_output(hf2,nscool_tov,"tov");
	hf2.close();
      }
      
    } else if (type=="hhj") {

      //nc.include_muons=true;
      
      eos_had_hhj hhj;
      hhj.delta_P_K=delta_P_K;
      hhj.delta_P_Gamma=delta_P_Gamma;

      // ------------------------------------------------------------
      // Construct the EOS from an nstar_cold object
      
      // Automatically construct beta-equilibrium EOS
      nc.verbose=0;
      nc.def_tov.verbose=0;
      nc.err_nonconv=err_nonconv;
      nc.def_tov.err_nonconv=err_nonconv;
      nc.set_eos(hhj);
      int eos_ret=nc.calc_eos();
      if (eos_ret!=0) {
        if (mpi_nprocs==1) {
          cout << "EOS calculation in class nstar_cold failed." << endl;
        }
        return 16;
      }
      if (mpi_nprocs==1) {
        cout << "acausal: " << nc.acausal << endl;
      }
      shared_ptr<o2scl::table_units<> > nc_eos=nc.get_eos_results();
      
      // Check for increasing pressure
      double pr_bad=-1.0;
      double ed_bad=-1.0;
      size_t i_bad=0;
      for(size_t i=0;i<nc_eos->get_nlines()-1;i++) {
        if (pr_bad<=0.0 && nc_eos->get("pr",i+1)<nc_eos->get("pr",i)) {
          pr_bad=nc_eos->get("pr",i);
          ed_bad=nc_eos->get("ed",i);
          i_bad=i;
          if (mpi_nprocs==1) {
            cout << i_bad << " " << ed_bad << " " << pr_bad << endl;
          }
        }
      }
      if (pr_bad>0.0 && ed_bad<1.0) {
        if (mpi_nprocs==1) {
          cout << "Pressure decreasing at a low density" << endl;
        }
        return 10;
      }
      if (pr_bad>0.0) {
        for(size_t i=i_bad+1;i<nc_eos->get_nlines();i++) {
          nc_eos->set("pr",i,pr_bad+0.01*((double)(i-i_bad)));
        }
      }

      nc.def_eos_tov.read_table(*nc_eos,"ed","pr","nb");
      
      ns_np_sat=nc_eos->interp("nb",0.16,"np");

      // Prepare to compute effective masses
      fermion n(o2scl::o2scl_settings.get_convert_units().convert
		("kg","1/fm",o2scl_mks::mass_neutron),2.0);
      fermion p(o2scl::o2scl_settings.get_convert_units().convert
		("kg","1/fm",o2scl_mks::mass_proton),2.0);
      n.non_interacting=false;
      p.non_interacting=false;
      thermo th;
      
      // ------------------------------------------------------------
      // Create a core table of the correct size with the correct units

      // Clear table for new columns
      nscool_core.clear_table();
      nscool_core.line_of_names("Rho Press nbar Ye Ymu Yn Yp mstp mstn");
      
      // Baryon density grid
      double nb_min=0.09;
      double nb_max=nc_eos->max("nb");
      double dnb=(nb_max-nb_min)/178.0;
      
      // Fill core table
      for(double nb=nb_max;nb>nb_min-dnb/10.0;nb-=dnb) {
	n.n=nc_eos->interp("nb",nb,"nn");
	p.n=nc_eos->interp("nb",nb,"np");
	hhj.calc_e(n,p,th);
	double Ymu=0.0;
	//if (nc_eos->is_column("nmu")) nc_eos->interp("nb",nb,"nmu")/nb;
	//if (Ymu<0.0) Ymu=0.0;
	double line[9]={cu.convert("1/fm^4","g/cm^3",
				   nc_eos->interp("nb",nb,"ed")),
			cu.convert("1/fm^4","dyne/cm^2",
				   nc_eos->interp("nb",nb,"pr")),nb,
			nc_eos->interp("nb",nb,"ne")/nb,
			Ymu,
			nc_eos->interp("nb",nb,"nn")/nb,
			nc_eos->interp("nb",nb,"np")/nb,
			n.ms/n.m,p.ms/p.m};
	nscool_core.line_of_data(9,line);
      }

      if (make_tables) {
	o2scl_hdf::hdf_file hf2;
	hf2.open_or_create("core.o2");
	hdf_output(hf2,nscool_core,"core");
	hf2.close();
      }

      // ------------------------------------------------------------
      // Use the nstar_cold object to construct the profile

      // Construct neutron star profile
      nc.def_tov.calc_gpot=true;
      int mvsr_ret=nc.calc_nstar();
      if (mvsr_ret!=0) {
        if (mpi_nprocs==1) {
          cout << "Mass-radius calculation in class nstar_cold failed." << endl;
        }
        return 17;
      }
      shared_ptr<o2scl::table_units<> > nc_prof=nc.get_tov_results();

      ns_M_max=nc_prof->max("gm");
      if (mpi_nprocs==1) {
        cout << "M_max: " << ns_M_max << endl;
	cout << "R(M_max): "
	     << nc_prof->get("r",nc_prof->lookup("gm",ns_M_max))
	     << endl;
      }
      // Remove all rows past maximum mass
      nc_prof->set_nlines(nc_prof->lookup("gm",ns_M_max)+1);
      ns_nb_max=nc_prof->get("nb",nc_prof->lookup("gm",ns_M_max));
      if (fix_durca>0.0) {
        if (mpi_nprocs==1) {
          cout << "fix_durca, M_dUrca: " << fix_durca
               << " " << nc_prof->interp("nb",fix_durca,"gm") << endl;
        }
        //for(size_t i=0;i<nc_prof->get_nlines();i++) {
        //cout << nc_prof->get("nb",i) << " " << nc_prof->get("gm",i) << endl;
        //}
      }
      ns_nn_max=nc_eos->interp("nb",ns_nb_max,"nn");
      ns_np_max=nc_eos->interp("nb",ns_nb_max,"np");
      if (nc_prof->get("pr",nc_prof->lookup("gm",ns_M_max))<pr_bad) {
        if (mpi_nprocs==1) {
          cout << "Pressure bad below central pressure of maximum mass"
               << endl;
        }
        return 11;
      }

      cs2_max=nc_eos->interp("ed",nc_prof->max("ed"),"cs2");
      
      double mass2=mass;
      if (mass2<0.0) mass2=mass+nc_prof->max("gm");
      if (mass2>ns_M_max) {
        if (mpi_nprocs==1) {
          cout << "Mass " << mass2 << " larger than maximum mass: "
               << ns_M_max << endl;
        }
        return 12;
      }
      if (mpi_nprocs==1) {
        cout << "Set mass to " << mass2 << endl;
      }
      int mass_ret=nc.fixed(mass2);
      if (mass_ret!=0) {
        if (mpi_nprocs==1) {
          cout << "Fixed-mass calculation in class nstar_cold failed." << endl;
        }
        return 17;
      }
      
      // ------------------------------------------------------------
      // Create a table with the right size and the right units

      // Clear table for new columns
      nscool_tov.clear_table();
      nscool_tov.line_of_names(((string)"step radius n_baryon density ")+
			       "pressure emass phi bmass");
      
      // Radial grid
      double r_max=nc_prof->max("r");
      trad=r_max;
      
      // Fill TOV table
      size_t ix=0;
      for(size_t ix=0;ix<153;ix++) {
	// A grid which focuses most of the points on the outer part
	// of the star
	double r=r_max*2.0*(1.0-pow(2.0,-((double)ix)/152.0));
	double line[8]={((double)ix),r*1.0e3,nc_prof->interp("r",r,"nb"),
			cu.convert("1/fm^4","g/cm^3",
				   nc_prof->interp("r",r,"ed")),
			cu.convert("1/fm^4","dyne/cm^2",
				   nc_prof->interp("r",r,"pr")),
			nc_prof->interp("r",r,"gm"),
			nc_prof->interp("r",r,"gp"),
			nc_prof->interp("r",r,"bm")};
	nscool_tov.line_of_data(8,line);
      }

      if (make_tables) {
	o2scl_hdf::hdf_file hf2;
	hf2.open_or_create("tov.o2");
	hdf_output(hf2,nscool_tov,"tov");
	hf2.close();
      }
      
    } else if (type=="pot") {

      // ------------------------------------------------------------
      // Construct the EOS from an nstar_cold object
      
      // Automatically construct beta-equilibrium EOS
      nc.verbose=0;
      nc.def_tov.verbose=0;
      nc.set_eos(pot);
      nc.err_nonconv=false;
      nc.calc_eos();
      shared_ptr<o2scl::table_units<> > nc_eos=nc.get_eos_results();
      nc.def_eos_tov.read_table(*nc_eos,"ed","pr","nb");

      // Prepare to compute effective masses
      fermion n(o2scl::o2scl_settings.get_convert_units().convert
		("kg","1/fm",o2scl_mks::mass_neutron),2.0);
      fermion p(o2scl::o2scl_settings.get_convert_units().convert
		("kg","1/fm",o2scl_mks::mass_proton),2.0);
      n.non_interacting=false;
      p.non_interacting=false;
      thermo th;
      
      // ------------------------------------------------------------
      // Create a core table of the correct size with the correct units

      // Clear table for new columns
      nscool_core.clear_table();
      nscool_core.line_of_names("Rho Press nbar Ye Ymu Yn Yp mstp mstn");
      
      // Baryon density grid
      double nb_min=0.09;
      double nb_max=nc_eos->max("nb");
      double dnb=(nb_max-nb_min)/178.0;
      
      // Fill core table
      for(double nb=nb_max;nb>nb_min-dnb/10.0;nb-=dnb) {
	n.n=nc_eos->interp("nb",nb,"nn");
	p.n=nc_eos->interp("nb",nb,"np");
	pot.calc_e(n,p,th);
	double line[9]={cu.convert("1/fm^4","g/cm^3",
				   nc_eos->interp("nb",nb,"ed")),
			cu.convert("1/fm^4","dyne/cm^2",
				   nc_eos->interp("nb",nb,"pr")),nb,
			nc_eos->interp("nb",nb,"ne")/nb,
			0.0,
			nc_eos->interp("nb",nb,"nn")/nb,
			nc_eos->interp("nb",nb,"np")/nb,
			n.ms/n.m,p.ms/p.m};
	nscool_core.line_of_data(9,line);
      }
      
      // ------------------------------------------------------------
      // Use the nstar_cold object to construct the profile

      // Construct neutron star profile
      nc.def_tov.calc_gpot=true;
      nc.calc_nstar();
      shared_ptr<o2scl::table_units<> > nc_prof=nc.get_tov_results();

      double mass2=mass;
      if (mass2<0.0) mass2=mass+nc_prof->max("gm");
      cout << "Set mass to " << mass2 << endl;
      nc.fixed(mass2);
      
      // ------------------------------------------------------------
      // Create a table with the right size and the right units

      // Clear table for new columns
      nscool_tov.clear_table();
      nscool_tov.line_of_names(((string)"step radius n_baryon density ")+
			       "pressure emass phi bmass");
      
      // Radial grid
      double r_max=nc_prof->max("r");
      
      // Fill TOV table
      size_t ix=0;
      for(size_t ix=0;ix<153;ix++) {
	// A grid which focuses most of the points on the outer part
	// of the star
	double r=r_max*2.0*(1.0-pow(2.0,-((double)ix)/152.0));
	double line[8]={((double)ix),r*1.0e3,nc_prof->interp("r",r,"nb"),
			cu.convert("1/fm^4","g/cm^3",
				   nc_prof->interp("r",r,"ed")),
			cu.convert("1/fm^4","dyne/cm^2",
				   nc_prof->interp("r",r,"pr")),
			nc_prof->interp("r",r,"gm"),
			nc_prof->interp("r",r,"gp"),
			nc_prof->interp("r",r,"bm")};
	nscool_tov.line_of_data(8,line);
      }

      eos_had_ptr=&pot;

    } else if (type=="rmf") {

      // Automatically construct beta-equilibrium EOS
      nc.verbose=0;
      nc.def_tov.verbose=0;
      nc.set_eos(rmf);
      nc.err_nonconv=false;
      nc.calc_eos();
      shared_ptr<o2scl::table_units<> > nc_eos=nc.get_eos_results();
      nc.def_eos_tov.read_table(*nc_eos,"ed","pr","nb");

      // Prepare to compute effective masses
      fermion n(o2scl::o2scl_settings.get_convert_units().convert
		("kg","1/fm",o2scl_mks::mass_neutron),2.0);
      fermion p(o2scl::o2scl_settings.get_convert_units().convert
		("kg","1/fm",o2scl_mks::mass_proton),2.0);
      n.non_interacting=false;
      p.non_interacting=false;
      thermo th;
      
      // Clear table for new columns
      nscool_core.clear_table();
      nscool_core.line_of_names("Rho Press nbar Ye Ymu Yn Yp mstp mstn");
      
      // Baryon density grid
      double nb_min=0.09;
      double nb_max=nc_eos->max("nb");
      double dnb=(nb_max-nb_min)/178.0;
      
      // Fill core table
      for(double nb=nb_max;nb>nb_min-dnb/10.0;nb-=dnb) {
	n.n=nc_eos->interp("nb",nb,"nn");
	p.n=nc_eos->interp("nb",nb,"np");
	rmf.calc_e(n,p,th);
	double line[9]={cu.convert("1/fm^4","g/cm^3",
				   nc_eos->interp("nb",nb,"ed")),
			cu.convert("1/fm^4","dyne/cm^2",
				   nc_eos->interp("nb",nb,"pr")),nb,
			nc_eos->interp("nb",nb,"ne")/nb,
			0.0,
			nc_eos->interp("nb",nb,"nn")/nb,
			nc_eos->interp("nb",nb,"np")/nb,
			n.ms/n.m,p.ms/p.m};
	nscool_core.line_of_data(9,line);
      }
      
      // Construct neutron star profile
      nc.def_tov.calc_gpot=true;
      nc.calc_nstar();
      shared_ptr<o2scl::table_units<> > nc_prof=nc.get_tov_results();

      double mass2=mass;
      if (mass2<0.0) mass2=mass+nc_prof->max("gm");
      cout << "Set mass to " << mass2 << endl;
      nc.fixed(mass2);
      
      // Clear table for new columns
      nscool_tov.clear_table();
      nscool_tov.line_of_names(((string)"step radius n_baryon density ")+
			       "pressure emass phi bmass");
      
      // Radial grid
      double r_max=nc_prof->max("r");
      
      // Fill TOV table
      size_t ix=0;
      for(size_t ix=0;ix<153;ix++) {
	// A grid which focuses most of the points on the outer part
	// of the star
	double r=r_max*2.0*(1.0-pow(2.0,-((double)ix)/152.0));
	double line[8]={((double)ix),r*1.0e3,nc_prof->interp("r",r,"nb"),
			cu.convert("1/fm^4","g/cm^3",
				   nc_prof->interp("r",r,"ed")),
			cu.convert("1/fm^4","dyne/cm^2",
				   nc_prof->interp("r",r,"pr")),
			nc_prof->interp("r",r,"gm"),
			nc_prof->interp("r",r,"gp"),
			nc_prof->interp("r",r,"bm")};
	nscool_tov.line_of_data(8,line);
      }

      eos_had_ptr=&rmf;
    }
    
    return 0;
  }
  
  /** \brief Solve for steady-state cooling at one accretion rate
   */
  int steady_state_one(std::vector<std::string> &sv, bool itive_com) {

    sfn1s0=1;
    sfp1s0=150;
    sfn3p2=150;
    n3_tc=1.055e9;
    n3_kf=1.88628;
    n3_dk=0.486867;
    p1_tc=4.99188e9;
    p1_kf=1.22489;
    p1_dk=0.53767;
    
    no_cooling=true;
    time_print[0]=1.1e-12;
    ptemp=1.0;

    if (sv.size()<2) {
      cerr << "Need Mdot." << endl;
      return 2;
    }
    
    double logT=logT_init;
    double Mdot=o2scl::stod(sv[1]);
    solve(Mdot,logT);
    double lphot,lneut,lheat;
    acc_compute(logT,lphot,lneut,lheat);
    cout << "H3." << endl;

    write_tl_prof();
    cout << "H4." << endl;

    hdf_file hf;
    hf.open("tl_prof.o2");
    o2scl::table3d tl_prof2;
    std::string name;
    hdf_input(hf,tl_prof2,name);
    hf.close();

    cout << "H5." << endl;
    cout << Mdot << " " << pow(10.0,logT) << " "
	 << lphot << " " << lneut << " " << lheat << endl;
    
    for(size_t i=0;i<tl_prof2.get_nslices();i++) {
      cout << tl_prof2.get_slice_name(i) << " ";
    }
    cout << endl;
    for(size_t j=0;j<tl_prof2.get_nx();j++) {
      for(size_t i=0;i<tl_prof2.get_nslices();i++) {
	cout << tl_prof2.get(j,0,tl_prof2.get_slice_name(i)) << " ";
      }
      cout << endl;
    }

    return 0;
  }
  
  /** \brief Solve for the steady-state heating curve
   */
  int steady_state(std::vector<std::string> &sv, bool itive_com) {

    no_cooling=true;
    
    table_units<> &t=ss_table;
    t.clear_table();
    t.line_of_names("Mdot T_core L_phot_inf L_neut_inf L_heat_inf");
    t.set_unit("Mdot","Msun/yr");
    t.set_unit("T_core","K");
    t.set_unit("L_phot_inf","erg/s");
    t.set_unit("L_neut_inf","erg/s");
    t.set_unit("L_heat_inf","erg/s");

    if (make_tables) {
      o2scl_hdf::hdf_file hf2;
      hf2.open_or_create("core.o2");
      hdf_output(hf2,nscool_core,"core2");
      hf2.close();
      hf2.open_or_create("tov.o2");
      hdf_output(hf2,nscool_tov,"tov2");
      hf2.close();
    }

    double logT=logT_init;
    //if (type=="skyrme" && name=="QMC3" && mass<2.5) logT=7.0;
    cout << "M_dot T_core L_phot L_neut L_heat" << endl;
    for(double Mdot=6.0e-12;Mdot<1.0e-8;Mdot*=1.2) {
      solve(Mdot,logT);
      double lphot,lneut,lheat;
      acc_compute(logT,lphot,lneut,lheat);
      double line[5]={Mdot,pow(10.0,logT),lphot,lneut,lheat};
      cout << line[0] << " " << line[1] << " "
	   << line[2] << " " << line[3] << " " << line[4] << endl;
      t.line_of_data(5,line);
    }
    logT=logT_init;
    //if (type=="skyrme" && name=="QMC3" && mass<2.5) logT=7.0;
    for(double Mdot=6.0e-12/1.2;Mdot>Mdot_low;Mdot/=1.2) {
      solve(Mdot,logT);
      double lphot,lneut,lheat;
      acc_compute(logT,lphot,lneut,lheat);
      double line[5]={Mdot,pow(10.0,logT),lphot,lneut,lheat};
      cout << line[0] << " " << line[1] << " "
	   << line[2] << " " << line[3] << " " << line[4] << endl;
      t.line_of_data(5,line);
    }
    t.sort_table("Mdot");
    t.add_constant("mass",mass);
    
    if (true) {
      o2scl_hdf::hdf_file hf;
      hf.open_or_create(out_file);
      hdf_output(hf,t,"sxrt");
      hf.close();
    }

    return 0;
  }

  /** \brief Compute a cooling assuming an isolated star
   */
  int cool(std::vector<std::string> &sv, bool itive_com) {

    no_cooling=false;

    // Run the cooling code
    cout << "Here4." << endl;
    run(0);
    cout << "Here4b." << endl;
    
    write_cool_curve();
    write_tl_prof();

    return 0;
  }

  /** \brief Set-up the command-line parsing object and run
   */
  int interface(int argc, char *argv[]) {

    static const int nopt=7;
    comm_option_s options[nopt]={
      {'s',"steady-state","Steady-state heating curve",0,0,"","",
       new comm_option_mfptr<sxrt_class>(this,&sxrt_class::steady_state),
       cli::comm_option_both},
      {'o',"steady-state-one","Steady-state heating at one Mdot",1,1,
       "<Mdot>","",
       new comm_option_mfptr<sxrt_class>(this,&sxrt_class::steady_state_one),
       cli::comm_option_both},
      {'e',"eos","Choose EOS",1,2,"<type> [name]","",
       new comm_option_mfptr<sxrt_class>(this,&sxrt_class::eos),
       cli::comm_option_both},
      {0,"reformat","Reformat EOS and TOV profile for NSCool",2,2,"","",
       new comm_option_mfptr<sxrt_class>(this,&sxrt_class::reformat),
       cli::comm_option_both},
      {'c',"cool","Simple isolated cooling",0,0,"<type> [name]","",
       new comm_option_mfptr<sxrt_class>(this,&sxrt_class::cool),
       cli::comm_option_both},
      {'g',"gaps","Set superconducting/superfluid gaps",3,12,
       "<n1 type> <n3 type> <p1 type> [gap params]","",
       new comm_option_mfptr<sxrt_class>(this,&sxrt_class::gaps_cmd),
       cli::comm_option_both},
      {'m',"mcmc","Perform an MCMC simulation for 1808 and AqX1",0,0,"","",
       new comm_option_mfptr<sxrt_class>(this,&sxrt_class::mcmc),
       cli::comm_option_both}
    };
    cl.set_comm_option_vec(nopt,options);

    p_out_file.str=&out_file;
    p_out_file.help=((string)"Output filename (default 'sxrt.o2').");
    cl.par_list.insert(make_pair("out-file",&p_out_file));

    p_eta.d=&eta;
    p_eta.help=((string)"Determine envelope composition (default 0).");
    cl.par_list.insert(make_pair("eta",&p_eta));

    p_Mdot_low.d=&Mdot_low;
    p_Mdot_low.help=((string)"Lower limit for Mdot/M (default 3.0e-15).");
    cl.par_list.insert(make_pair("Mdot_low",&p_Mdot_low));

    p_mass.d=&mass;
    p_mass.help=((string)"Neutron star gravitational mass (default 1.4).");
    cl.par_list.insert(make_pair("mass",&p_mass));
    
    p_T_fact_drip.d=&T_fact_drip;
    p_T_fact_drip.help=((string)"Temperature factor for drip density ")+
      "(default 0.8).";
    cl.par_list.insert(make_pair("T_fact_drip",&p_T_fact_drip));
    
    p_logT_init.d=&logT_init;
    p_logT_init.help=((string)"Initial log10 temperature ")+
      "(default 8.0).";
    cl.par_list.insert(make_pair("logT_init",&p_logT_init));
    
    p_T_fact_surf.d=&T_fact_surf;
    p_T_fact_surf.help=((string)"Temperature factor for surface ")+
      "(default 0.5).";
    cl.par_list.insert(make_pair("T_fact_surf",&p_T_fact_surf));
    
    p_make_tables.b=&make_tables;
    p_make_tables.help=((string)"If true, make tables for ")+
      "use in NSCool (default 0).";
    cl.par_list.insert(make_pair("make_tables",&p_make_tables));

    p_fix_durca.d=&fix_durca;
    p_fix_durca.help=((string)"If greater than zero, set as the ")
      +"density for direct Urca (default 0.0)";
    cl.par_list.insert(make_pair("fix_durca",&p_fix_durca));

    p_delta_P_K.d=&delta_P_K;
    p_delta_P_K.help=((string)"Coefficient for high-density ")+
      "change in Skyrme pressure (default 1.0).";
    cl.par_list.insert(make_pair("delta_P_K",&p_delta_P_K));

    p_delta_P_Gamma.d=&delta_P_Gamma;
    p_delta_P_Gamma.help=((string)"Exponent for high-density ")+
      "change in Skyrme pressure (default 0.0).";
    cl.par_list.insert(make_pair("delta_P_Gamma",&p_delta_P_Gamma));

    p_nscool_debug.i=&nscool_debug;
    p_nscool_debug.help="NSCool debug parameter (default 0)";
    cl.par_list.insert(make_pair("nscool_debug",&p_nscool_debug));

    p_max_time.d=&max_time;
    p_max_time.help="Maximum time for MCMC (default 28200)";
    cl.par_list.insert(make_pair("max_time",&p_max_time));

    cl.prompt="sxrt> ";
    cl.run_auto(argc,argv);
    
    return 0;
  }

};

int main(int argc, char *argv[]) {

#ifndef NO_MPI
  MPI_Init(&argc,&argv);
#endif
  
  cout.setf(ios::scientific);

  // Set global pointer
  sxrt_class nw;
  nscool_wrap_ptrs.resize(1);
  nscool_wrap_ptrs[0]=&nw;

  nw.interface(argc,argv);

#ifndef NO_MPI
  MPI_Finalize();
#endif

  return 0;
}
