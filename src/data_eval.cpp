#include <o2scl/hdf_eos_io.h>
#include <o2scl/interpm_idw.h>
#include <o2scl/mcmc_para.h>

#include "data_eval.h"

using namespace std;
using namespace o2scl;
using namespace o2scl_hdf;
using namespace o2scl_const;

data_eval::data_eval() : nw("./") {

  // Tell the nuclear structure object which EOS to use
  rn.set_eos(rmf);

  // Tell the nuclear structure object which EOS to use
  rn2.set_eos(rmf);

  // Tell the cooling code to use parameterized gaps instead of internal gap
  // functions

  nw.sfn1s0=1;
  nw.sfp1s0=150;
  nw.sfn3p2=150;
  nw.ptemp=1.0;

  // Tell the neutron star object which EOS to use and A few other settings
  nst.include_muons=true;
  nst.def_tov.calc_gpot=true;

  // Set up the cooling curve data objects and associated grid
  std::vector<size_t> sz={3,6,100};
  nscool_tptr.resize(3,sz);
  nscool_lum.resize(3,sz);
  // Set up the sxrt curve data object
  std::vector<size_t> sz2={3,6,30};
  sxrt_lum.resize(3,sz2);

  // --------------------------------------------------------------
  // Set up the grid for the cooling curves
    
  // Time grid
  for(int i=0;i<100;i++) {
    t_grid.push_back(1.0e2*pow(1.0e5,((double)i)/99.0));
  }

  // Eta grid
  log_eta_grid={-17.0,-12.0,-7.0};

  // Mass grid
  mass_grid={1.0,1.2,1.4,1.6,1.8,2.0};

  // Log mdot grid
  o2scl::uniform_grid<double>
    log_mdot_temp=o2scl::uniform_grid_end<double>(-12,-9,29);
  for(size_t j=0;j<30;j++) {
    log_mdot_grid.push_back(log_mdot_temp[j]);
  }

  // Combined grids
  std::vector<std::vector<double> > tensor_grid={log_eta_grid,mass_grid,
    t_grid};
    
  nscool_tptr.set_grid(tensor_grid);
  nscool_lum.set_grid(tensor_grid);

  std::vector<std::vector<double> > tensor_grid2={log_eta_grid,mass_grid,
    log_mdot_grid};
  sxrt_lum.set_grid(tensor_grid2);

  // --------------------------------------------------------------

  // Set for linear interpolation
  itp.set_type(o2scl::itp_linear);

  // Omega, rho, and nucleon masses
  rmf.mw=3.965499e+00;
  rmf.mr=3.866679e+00;
  rmf.mnuc=939.0/o2scl_const::hc_mev_fm;
    
  // Set convergence error flags for eos_had_rmf object
  rmf.err_nonconv=false;
  rmf.def_mroot.err_nonconv=false;
  rmf.def_mroot.def_jac.err_nonconv=false;
  rmf.def_sat_mroot.err_nonconv=false;
  rmf.def_sat_mroot.def_jac.err_nonconv=false;

  // Set convergence error flags for nstar_cold object
  nst.err_nonconv=false;
  nst.def_tov.err_nonconv=false;
  nst.def_tov.def_solver.err_nonconv=false;
  nst.def_tov.def_solver.def_jac.err_nonconv=false;

  // Set convergence error flags for nucleus_rmf object
  rn.err_nonconv=false;
  rn.generic_ode=true;
  rn2.err_nonconv=false;
  rn2.generic_ode=true;

  // This initial value doesn't matter because it is set by the MCMC function 
  verbose=1;

  // Delta mass is 0.024
  mvsr_mass_grid=o2scl::uniform_grid_end<double>(0.08,2.456,ngrid-1);
  // Delta energy density is 0.08
  eden_grid=o2scl::uniform_grid_end<double>(0.32,8.24,ngrid-1);
  // Delta baryon density is 0.012
  nb_grid=o2scl::uniform_grid_end<double>(0.04,1.228,ngrid-1);
  // Delta r is 0.12
  nucprof_grid=o2scl::uniform_grid_end<double>(0.04,11.92,ngrid-1);

  //fit_cool_tptr=false;

  rmfp=&rmf;
  nst.set_eos(*rmfp);

  schwarz_km=o2scl_const::schwarzchild_radius_f<double>()/1.0e3;

  err_as_small_wgt=true;

  no_nuclei=false;
  no_ins=false;
  no_sxrt=false;
  no_qlmxb=false;
  no_ligo=false;
  debug_ins=false;
  
  flag_emu_aws=false;
}

void data_eval::init(size_t nparam_, o2scl::vec_index &dvi_,
                     o2scl::vec_index &pvi_,
                     size_t verbose_, ins_data &id_,
                     nsmr_data &nd_, sxrt_data &sd_) {
  
  nparam=nparam_;
  dvi=dvi_;
  pvi=pvi_;
  verbose=verbose_;
  id=id_;
  nd=nd_;
  sd=sd_;

  return;
}

int data_eval::point(size_t nv, const ubvector &pars, double &log_weight,
                     std::array<double,ndat> &dat) {

  bool check_timing=false;

  std::chrono::time_point<std::chrono::high_resolution_clock> start1;
  std::chrono::time_point<std::chrono::high_resolution_clock> start;

  if (check_timing){
    start1 = std::chrono::high_resolution_clock::now(); 
    start = std::chrono::high_resolution_clock::now();
  }
                     
  bool check_cooling=false;
  
  log_weight=0.0;
  
  if (verbose>1) {
    std::cout << "Parameters: " << std::endl;
    for(size_t i=0;i<pars.size();i++) {
      std::cout << "pars[" << pvi[i] << "]=" << pars[i] << ";" << std::endl;
    } 
  }
  
  // --------------------------------------------------------------
  // Add 'blank' entries to dat array 
  
  for(size_t k=0;k<ndat;k++) dat[k]=1.0e99;
  
  // --------------------------------------------------------------
  // Set emulated and log_wgt_unc
  
  dat[dvi["emulated"]]=0.0;
  dat[dvi["log_wgt_unc"]]=0.0;
  
  // --------------------------------------------------------------
  // Copy parameters
  
  nw.n3_tc=pow(10.0,pars[pvi["log10_Tcn"]]);
  nw.n3_kf=pars[pvi["k_maxn"]];
  nw.n3_dk=pars[pvi["del_kn"]];
  nw.p1_tc=pow(10.0,pars[pvi["log10_Tcp"]]);
  nw.p1_kf=pars[pvi["k_maxp"]];
  nw.p1_dk=pars[pvi["del_kp"]];
  
  rmfp->b=pars[pvi["b"]];
  rmfp->c=pars[pvi["c"]];
  rmfp->cr=pars[pvi["c_r"]];
  rmfp->ms=pars[pvi["m_s"]];
  rmfp->zeta=pars[pvi["zeta"]];
  rmfp->cs=pars[pvi["c_s"]];
  rmfp->cw=pars[pvi["c_w"]];
  rmfp->b1=pars[pvi["b1"]];
  
  rmfp->xi=pars[pvi["xi"]];
  rmfp->a1=pars[pvi["a1"]];
  rmfp->a2=pars[pvi["a2"]];
  rmfp->a3=pars[pvi["a3"]];
  rmfp->a4=pars[pvi["a4"]];
  rmfp->a5=pars[pvi["a5"]];
  rmfp->a6=pars[pvi["a6"]];
  rmfp->b2=pars[pvi["b2"]];
  rmfp->b3=pars[pvi["b3"]];
  
  // alpha
  nw.alpha_durca=pars[pvi["alpha"]];
  
  // --------------------------------------------------------------
  // Saturation
  
  bool check_RMF_EOS=false;
  if (check_RMF_EOS) {
    cout.precision(10);
    cout << rmfp->mnuc << endl;
    cout << rmfp->ms << endl;
    cout << rmfp->mr << endl;
    cout << rmfp->mw << endl;
    cout << rmfp->cs << endl;
    cout << rmfp->cw << endl;
    cout << rmfp->cr << endl;
    cout << rmfp->b << endl;
    cout << rmfp->c << endl;
    cout << rmfp->zeta << endl;
    cout << rmfp->xi << endl;
    cout << rmfp->a1 << endl;
    cout << rmfp->a2 << endl;
    cout << rmfp->a3 << endl;
    cout << rmfp->a4 << endl;
    cout << rmfp->a5 << endl;
    cout << rmfp->a6 << endl;
    cout << rmfp->b1 << endl;
    cout << rmfp->b2 << endl;
    cout << rmfp->b3 << endl;
    cout.precision(6);
  }
  
  int mpi_rank = 0, mpi_size=0;
#ifndef NO_MPI    
  MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
  //if (mpi_rank==0) {
    //cout << "System free:" << endl; int sys_ret=system("free"); cout <<
    //"System done." << endl;
  //}
#endif
  
  rmfp->mnuc=939.0/o2scl_const::hc_mev_fm;
  rmfp->def_neutron.mu=rmfp->def_neutron.m;
  rmfp->def_proton.mu=rmfp->def_proton.m;
  rmfp->set_fields(0.2,0.1,0.01);
  rmfp->set_n_and_p(rmfp->def_neutron,rmfp->def_proton);
  //rmfp->verbose=2;
  int sat_ret=rmfp->saturation();
  if (sat_ret!=0) {
    dat[dvi["sat_fail"]]=1.0;
    if (verbose>0) {
      std::cout << "Fail: Saturation failed." << std::endl;
    }
    if (err_as_small_wgt) {
      log_weight=-802.0;
      return 0;
    }
    return 2;
  } else {
    dat[dvi["sat_fail"]]=0.0;
  }
  if (verbose>1) {
    std::cout << "Saturation density: " << rmfp->n0 << std::endl;
    std::cout << "Binding energy: " << rmfp->eoa*o2scl_const::hc_mev_fm
              << std::endl;
    std::cout << "Effective mass: " << rmfp->msom << std::endl;
    std::cout << "Symmetry energy: " << rmfp->esym*o2scl_const::hc_mev_fm
              << std::endl;
    std::cout << "Compressibility: " << rmfp->comp*o2scl_const::hc_mev_fm
              << std::endl;
    std::cout << "L: " << rmfp->fesym_slope(rmfp->n0)*o2scl_const::hc_mev_fm
              << std::endl;
  }
  
  // Saturation density
  dat[dvi["n0"]]=rmfp->n0;
  //Binding energy
  dat[dvi["EoA"]]=rmfp->eoa*o2scl_const::hc_mev_fm;
  // Effective energy
  dat[dvi["msom"]]=rmfp->msom;
  // Symmetry energy
  dat[dvi["S"]]=rmfp->esym*o2scl_const::hc_mev_fm;
  // Compressibility
  dat[dvi["K"]]=rmfp->comp*o2scl_const::hc_mev_fm;
  // Slope of symmetry energy
  dat[dvi["L"]]=rmfp->fesym_slope(rmfp->n0)*o2scl_const::hc_mev_fm;
  // Symmetry energy (alternate method)
  dat[dvi["S2"]]=rmfp->fesym_diff(rmfp->n0)*o2scl_const::hc_mev_fm;
  // Value of eta
  dat[dvi["eta"]]=rmfp->feta(rmfp->n0);
  
  if (dat[dvi["S2"]]<20.0 || dat[dvi["S2"]]>45.0) {
    if (verbose>0) {
      std::cout << "Fail: S2 out of range." << std::endl;
    }
    if (err_as_small_wgt) {
      log_weight=-859.0;
      //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
      return 0;
    }
    return 34;
  }
  
  if (dat[dvi["S"]]>36.0 || dat[dvi["L"]]>120.0) {
    if (verbose>0) {
      std::cout << "Fail: S or L out of range." << std::endl;
    }
    if (err_as_small_wgt) {
      log_weight=-860.0;
      //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
      return 0;
    }
    return 34;
  }
  
  
  if (dat[dvi["n0"]]<0.14 || dat[dvi["n0"]]>0.17) {
    if (verbose>0) {
        std::cout << "Fail: n0 out of range." << std::endl;
    }
    if (err_as_small_wgt) {
      log_weight=-835.0;
      //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
      return 0;
    }
    return 35;
  }

  if (dat[dvi["EoA"]]<-17.0 || dat[dvi["EoA"]]>-15.0) {
    if (verbose>0) {
      std::cout << "Fail: E/A out of range." << std::endl;
    }
    if (err_as_small_wgt) {
      log_weight=-836.0;
      //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
      return 0;
    }
    return 36;
  }

  if (dat[dvi["K"]]<220.0 || dat[dvi["K"]]>260.0) {
    if (verbose>0) {
      std::cout << "Fail: K out of range." << std::endl;
    }
    if (err_as_small_wgt) {
      log_weight=-834.0;
      //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
      return 0;
    }
    return 34;
  }

  if (check_timing) {
    // Record end time
    auto end = std::chrono::high_resolution_clock::now();
    // Compute the duration
    std::chrono::duration<double> elapsed = end - start;
    // Output the elapsed time in seconds
    std::cout << "Execution time saturation: " << elapsed.count() << " seconds\n";
    // Start nuclei 
    start = std::chrono::high_resolution_clock::now();
  }
  
  // --------------------------------------------------------------
  // Nuclei Record start time

  double L_Pb208=0.0, L_Ca40=0.0, L_Zr90=0.0;
  int nuc_ret;
  
  if (verbose<=1) {
    rn.set_verbose(0);
    rn2.set_verbose(0);
  } else {
    rn.set_verbose(1);
    rn2.set_verbose(1);
  }
  
  // Pb 208
  dat[dvi["nuc_fail"]]=0.0;
  
  if (no_nuclei) {
    dat[dvi["lw_nuc"]]=0.0;
    dat[dvi["lw_prex"]]=0.0;
  }
  
  if (!no_nuclei) {
    
    nuc_ret=rn.run_nucleus(82,126,0,0);
    
    nuc_profile1=rn.get_profiles();
    if (nuc_ret!=0) {
      dat[dvi["nuc_fail"]]=1.0;
      if (verbose>0) {
        std::cout << "Fail: Failed to compute Pb 208." << std::endl;
      }
      if (err_as_small_wgt) {
        log_weight=-807.0;
        //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
        return 0;
      }
      return 7;
    }
    if (nuc_profile1->get_nlines()<5) {
      dat[dvi["nuc_fail"]]=1.0;
      if (verbose>0) {
        std::cout << "Fail: Lead 208 table too small." << std::endl;
      }
      if (err_as_small_wgt) {
        log_weight=-828.0;
        //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
        return 0;
      }
      return 28;
    }
    
    // Added PREX constraint on rnrp for Pb-208
    
    L_Pb208=exp(-((rn.etot+7.87)*(rn.etot+7.87))
                /(2.0*0.16*0.16))*
      exp(-((rn.r_charge-5.50)*
            (rn.r_charge-5.50))/(2.0*0.12*0.12));

    if (!no_prex) {
      L_Pb208*=exp(-((rn.rnrp-0.283)*
      (rn.rnrp-0.283))/(2.0*0.071*0.071));
      dat[dvi["lw_prex"]]=-((rn.rnrp-0.283)*
                          (rn.rnrp-0.283))/(2.0*0.071*0.071);
    } else {
      dat[dvi["lw_prex"]]=0.0;
    }
    dat[dvi["lw_nuc"]]=-((rn.etot+7.87)*(rn.etot+7.87))
      /(2.0*0.16*0.16)-((rn.r_charge-5.50)*
                        (rn.r_charge-5.50))/(2.0*0.12*0.12);
    
    dat[dvi["etot_Pb208"]]=rn.etot;
    dat[dvi["rcharge_Pb208"]]=rn.r_charge;
    dat[dvi["rnrp_Pb208"]]=rn.rnrp;
    if (verbose>0) {
      std::cout << "Pb 208: " << rn.etot << " " << rn.r_charge << " "
                << rn.rnrp << std::endl;
    }
    
    log_weight+=log(L_Pb208);
    
    for(size_t i=0;i<ngrid;i++) {
      dat[dvi[((std::string)"Pb_rhop_")+o2scl::szttos(i)]]=
        nuc_profile1->get("rhop",i*3);
      dat[dvi[((std::string)"Pb_rhon_")+o2scl::szttos(i)]]=
        nuc_profile1->get("rhon",i*3);
    }
    
  } else {
    
    dat[dvi["etot_Pb208"]]=0.0;
    dat[dvi["rcharge_Pb208"]]=0.0;
    dat[dvi["rnrp_Pb208"]]=0.0;
    for(size_t i=0;i<ngrid;i++) {
      dat[dvi[((std::string)"Pb_rhop_")+o2scl::szttos(i)]]=0.0;
      dat[dvi[((std::string)"Pb_rhon_")+o2scl::szttos(i)]]=0.0;
    }
    
  }
  
  // Zr 90
  
  if (!no_nuclei) {
    
    nuc_ret=rn2.run_nucleus(40,50,0,0);
    
    nuc_profile2=rn2.get_profiles();
    if (nuc_ret!=0) {
      dat[dvi["nuc_fail"]]=1.0;
      if (verbose>0) {
        std::cout << "Fail: Failed to compute Zr 90." << std::endl;
      }
      if (err_as_small_wgt) {
        log_weight=-809.0;
        //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
        return 0;
      }
      return 9;
    }
    if (nuc_profile2->get_nlines()<5) {
      dat[dvi["nuc_fail"]]=1.0;
      if (verbose>0) {
        std::cout << "Fail: Zr 90 table too small." << std::endl;
      }
      if (err_as_small_wgt) {
        log_weight=-829.0;
        //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
        return 0;
      }
      return 29;
    }
    
    L_Zr90=exp(-((rn2.etot+8.71)*(rn2.etot+8.71))
               /(2.0*0.16*0.16))*
      exp(-((rn2.r_charge-4.27)*
            (rn2.r_charge-4.27))/(2.0*0.12*0.12));
    
    dat[dvi["etot_Zr90"]]=rn2.etot;
    dat[dvi["rcharge_Zr90"]]=rn2.r_charge;
    dat[dvi["rnrp_Zr90"]]=rn2.rnrp;
    if (verbose>0) {
      std::cout << "Zr 90: " << rn2.etot << " " << rn2.r_charge << " "
                << rn2.rnrp << std::endl;
    }
    
    log_weight+=log(L_Zr90);
    dat[dvi["lw_nuc"]]+=log(L_Zr90);
    
    for(size_t i=0;i<ngrid;i++) {
      dat[dvi[((std::string)"Zr_rhop_")+o2scl::szttos(i)]]=
        nuc_profile2->get("rhop",i*3);
      dat[dvi[((std::string)"Zr_rhon_")+o2scl::szttos(i)]]=
        nuc_profile2->get("rhon",i*3);
    }
    
  } else {
    
    dat[dvi["etot_Zr90"]]=0.0;
    dat[dvi["rcharge_Zr90"]]=0.0;
    dat[dvi["rnrp_Zr90"]]=0.0;
    for(size_t i=0;i<ngrid;i++) {
      dat[dvi[((std::string)"Zr_rhop_")+o2scl::szttos(i)]]=0.0;
      dat[dvi[((std::string)"Zr_rhon_")+o2scl::szttos(i)]]=0.0;
    }
    
  }
  
  
  // Ca 40
  
  if (false) {
    
    nuc_ret=rn.run_nucleus(20,20,0,0);
    nuc_profile1=rn.get_profiles();
    
    if (nuc_ret!=0) {
      dat[dvi["nuc_fail"]]=1.0;
      if (verbose>0) {
        std::cout << "Fail: Failed to compute Ca 40." << std::endl;
      }
      if (err_as_small_wgt) {
        log_weight=-808.0;
        //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
        return 0;
      }
      return 8;
    }
    
    L_Ca40=exp(-((rn.etot+8.45)*(rn.etot+8.45))
               /(2.0*0.16*0.16))*
      exp(-((rn.r_charge-3.48)*
            (rn.r_charge-3.48))/(2.0*0.12*0.12));
    
    dat[dvi["etot_Ca40"]]=rn.etot;
    dat[dvi["rcharge_Ca40"]]=rn.r_charge;
    dat[dvi["rnrp_Ca40"]]=rn.rnrp;
    if (verbose>0) {
      std::cout << "Ca 40: " << rn.etot << " " << rn.r_charge << " "
                << rn.rnrp << std::endl;
    }
    
    log_weight+=log(L_Ca40);
    
    for(size_t i=0;i<ngrid;i++) {
      dat[dvi[((std::string)"Ca_rhop_")+o2scl::szttos(i)]]=
        nuc_profile1->get("rhop",i*3);
      dat[dvi[((std::string)"Ca_rhon_")+o2scl::szttos(i)]]=
        nuc_profile1->get("rhon",i*3);
    }
    
  } else {
    
    L_Ca40=1.0;
    
    dat[dvi["etot_Ca40"]]=0.0;
    dat[dvi["rcharge_Ca40"]]=0.0;
    dat[dvi["rnrp_Ca40"]]=0.0;
    
    for(size_t i=0;i<ngrid;i++) {
      dat[dvi[((std::string)"Ca_rhop_")+o2scl::szttos(i)]]=0.0;
      dat[dvi[((std::string)"Ca_rhon_")+o2scl::szttos(i)]]=0.0;
    }
    
  }
  
  // --------------------------------------------------------------
  // Output nuclear structure likelihoods
  
  if (verbose>1 && !no_nuclei) {
    std::cout << "L_Pb208, L_Zr90, L_Ca40: "
              << L_Pb208 << " " << L_Zr90 << " " << L_Ca40 << std::endl;
  }

  if (check_timing) {
    // Record end time
    auto end = std::chrono::high_resolution_clock::now();
    // Compute the duration
    std::chrono::duration<double> elapsed = end - start;
    // Output the elapsed time in seconds
    std::cout << "Execution time nuclei: " << elapsed.count() << " seconds\n";
    // Begin timing copute EOS
    start = std::chrono::high_resolution_clock::now();
  }
  
  // --------------------------------------------------------------
  // Compute EOS

  rmfp->set_n_and_p(rmfp->def_neutron,rmfp->def_proton);
  
  nst.verbose=0;
  nst.def_tov.verbose=0;
  int ce_ret=0;
  dat[dvi["eos_fail"]]=0.0;
  ce_ret=nst.calc_eos();
  
  rmfp->set_thermo(rmfp->def_thermo);
  
  if (ce_ret!=0) {
    dat[dvi["eos_fail"]]=1.0;
    if (verbose>0) {
      std::cout << "Fail: Object nstar_cold failed to compute EOS."
                << std::endl;
    }
    if (err_as_small_wgt) {
      log_weight=-803.0;
      //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
      return 0;
    }
    return 3;
  }
  
  te=nst.get_eos_results();
  
  if (te->get_nlines()<5) {
    dat[dvi["eos_fail"]]=1.0;
    if (verbose>0) {
      std::cout << "Fail: EOS table too small." << std::endl;
    }
    if (err_as_small_wgt) {
      log_weight=-827.0;
      //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
      return 0;
    }
    return 27;
  }
  
  if (false) {
    for(size_t j=0;j<te->get_nlines();j++) {
      std::cout << "EOS: " << j << " " << te->get("ed",j) << " "
                << te->get("pr",j) << std::endl;
    }
  }
  
  // Proton fractions
  for(size_t i=0;i<ngrid;i++) {
    dat[dvi[((std::string)"xp_")+o2scl::szttos(i)]]=
      te->interp("nb",nb_grid[i],"np")/nb_grid[i];
    dat[dvi[((std::string)"Esym_")+o2scl::szttos(i)]]=
      rmfp->fesym(nb_grid[i])*o2scl_const::hc_mev_fm;
    dat[dvi[((std::string)"Esym2_")+o2scl::szttos(i)]]=
      rmfp->fesym_diff(nb_grid[i])*o2scl_const::hc_mev_fm;
    dat[dvi[((std::string)"cs2_")+o2scl::szttos(i)]]=
      te->interp("nb",nb_grid[i],"cs2"); // check this!
    if (false) {
      std::cout << i << " " << nb_grid[i] << " "
                << te->interp("nb",nb_grid[i],"np") << " "
                << te->interp("nb",nb_grid[i],"np")/nb_grid[i]
                << std::endl;
      //std::cout << dat[dvi[((string)"Esym_")+o2scl::szttos(i)]] << std::endl;
      //std::cout << dat[dvi[((string)"Esym2_")+o2scl::szttos(i)]] << std::endl;
    }
  }
  //exit(-1);

  if (check_timing){
    // Record end time
    auto end = std::chrono::high_resolution_clock::now();
    // Compute the duration
    std::chrono::duration<double> elapsed = end - start;
    // Output the elapsed time in seconds
    std::cout << "Execution time calc-EOS: " << elapsed.count() << " seconds\n";
    // start timing MR-curve
    start = std::chrono::high_resolution_clock::now();
  }
  
  // --------------------------------------------------------------
  // Compute mass-radius curve 
  

  for(size_t k=0;k<te->get_nlines();k++) {
    if (!std::isfinite(te->get("ed",k)) ||
        !std::isfinite(te->get("pr",k)) ||
        !std::isfinite(te->get("nb",k))) {
      dat[dvi["eos_fail"]]=1.0;
      if (verbose>0) {
        std::cout << "Fail: EOS not finite." << std::endl;
      }
      if (err_as_small_wgt) {
        log_weight=-820.0;
        cout << "log weight" << log_weight << endl;
        for(size_t ij=0;ij<ndat;ij++) dat[ij]=0.0;
        return 0;
      }
      return 20;
    }
  }
  
  if (te->get("pr",0)>te->get("pr",te->get_nlines()-1)) {
    dat[dvi["eos_fail"]]=1.0;
    if (verbose>0) {
      std::cout << "Fail: Core table decreasing." << std::endl;
    }
    if (err_as_small_wgt) {
      log_weight=-819.0;
      //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
      return 0;
    }
    return 19;
  }
  
  nst.def_eos_tov.err_nonconv=false;
  int cn_ret=0;
  //nst.verbose=2; nst.def_eos_tov.err_nonconv=true; nst.def_eos_tov.verbose=1;
  //nst.err_nonconv=true;
  nst.def_tov.ang_vel=true;
  cn_ret=nst.calc_nstar();
  dat[dvi["tov_fail"]]=0.0;
  
  if (cn_ret!=0) {
    dat[dvi["tov_fail"]]=1.0;
    if (verbose>0) {
      std::cout << "Fail: Object nstar_cold failed to compute M-R curve."
                << std::endl;
    }
    if (err_as_small_wgt) {
      log_weight=-804.0;
      //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
      return 0;
    }
    return 4;
  }
  
  tmvsr=*nst.get_tov_results();
  tmvsr.set_interp_type(o2scl::itp_linear);

  if (check_timing) {
    // Record end time
    auto end = std::chrono::high_resolution_clock::now();
    // Compute the duration
    std::chrono::duration<double> elapsed = end - start;
    // Output the elapsed time in seconds
    std::cout << "Execution time TOV: " << elapsed.count() << " seconds\n";
    start = std::chrono::high_resolution_clock::now();
  }
  
  // --------------------------------------------------------------
  // Compute EOS
  

  if (tmvsr.get_nlines()<5) {
    dat[dvi["tov_fail"]]=1.0;
    if (verbose>0) {
      std::cout << "Fail: TOV table too small." << std::endl;
    }
    if (err_as_small_wgt) {
      log_weight=-830.0;
      //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
      return 0;
    }
    return 30;
  }
  
  if (false) {
    for(size_t j=0;j<tmvsr.get_nlines();j++) {
      std::cout << "M-R: " << j << " " << tmvsr.get("gm",j) << " "
                << tmvsr.get("r",j) << std::endl;
    }
  }
  
  bool mvsr_finite=true;
  for(size_t j=0;j<tmvsr.get_nlines();j++) {
    if (!std::isfinite(tmvsr.get("gm",j))) mvsr_finite=false;
  }
  if (mvsr_finite==false) {
    dat[dvi["tov_fail"]]=1.0;
    if (verbose>0) {
      std::cout << "Fail: TOV table has non-finite masses." << std::endl;
    }
    if (err_as_small_wgt) {
      log_weight=-833.0;
      //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
      return 0;
    }
    return 33;
  }
  
  double Mns_max=tmvsr.max("gm");
  size_t max_row=tmvsr.lookup("gm",Mns_max);
  
  dat[dvi["Mns_max"]]=Mns_max;
  if (verbose>0) {
    std::cout << "Mns_max: " << Mns_max << std::endl;
  }
  
  // Remove all rows above maximum mass
  tmvsr.set_nlines(max_row+1);
  
  // Return an error if the maximum mass is too small
  if (tmvsr.max("gm")<2.0) {
    if (verbose>0) {
      std::cout << "Fail: Maximum mass (" << tmvsr.max("gm")
                << ") too small." << std::endl;
    }
    if (err_as_small_wgt) {
      log_weight=-801.0;
      //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
      return 0;
    }
    return 1;
  }
  
  // Compute radius, central baryon density, and central energy density of
  // maximum mass star
  dat[dvi["Rns_max"]]=tmvsr.get("r",max_row);
  dat[dvi["nb_max"]]=tmvsr.get("nb",max_row);
  dat[dvi["e_max"]]=tmvsr.get("ed",max_row);
  
  // --------------------------------------------------------------
  // Check causality
  
  double nb_max=tmvsr.get("nb",max_row);
  if (tmvsr.is_column("cs2")) tmvsr.delete_column("cs2");
  
  if (tmvsr.get("ed",0)==tmvsr.get("ed",tmvsr.get_nlines()-1)) {
    dat[dvi["eos_fail"]]=1.0;
    std::cout << "Fail: M-R table problem with ed "
              << tmvsr.get("ed",0) << std::endl;
    if (err_as_small_wgt) {
      log_weight=-822.0;
      //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
      return 0;
    }
    return 22;
  }
  if (tmvsr.get("pr",0)==tmvsr.get("pr",tmvsr.get_nlines()-1)) {
    dat[dvi["tov_fail"]]=1.0;
    std::cout << "Fail: M-R table problem with pr "
              << tmvsr.get("pr",0) << std::endl;
    if (err_as_small_wgt) {
      log_weight=-823.0;
      //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
      return 0;
    }
    return 23;
  }
  tmvsr.deriv("ed","pr","cs2");
  for(size_t i=0;i<tmvsr.get_nlines();i++) {
    if (tmvsr.get("nb",i)<nb_max && tmvsr.get("cs2",i)>1.0) {
      if (verbose>0) {
        std::cout << "Fail: acausal." << std::endl;
      }
      if (err_as_small_wgt) {
        log_weight=-813.0;
        //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
        return 0;
      }
      return 13;
    }
  }
  
  te->set_interp_type(o2scl::itp_linear);
  
  double cs2_max=te->interp("ed",tmvsr.max("ed"),"cs2");
  
  if (te->get("nb",0)==te->get("nb",te->get_nlines()-1)) {
    dat[dvi["eos_fail"]]=1.0;
    std::cout << "Fail: EOS table problem with nb "
              << te->get("nb",0) << std::endl;
    if (err_as_small_wgt) {
      log_weight=-824.0;
      //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
      return 0;
    }
    return 24;
  }
  if (te->get("kfn",0)==te->get("kfn",te->get_nlines()-1)) {
    dat[dvi["eos_fail"]]=1.0;
    std::cout << "Fail: EOS table problem with kfn "
              << te->get("kfn",0) << std::endl;
    if (err_as_small_wgt) {
      log_weight=-825.0;
      //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
      return 0;
    }
    return 25;
  }
  if (te->get("kfp",0)==te->get("kfp",te->get_nlines()-1)) {
    dat[dvi["eos_fail"]]=1.0;
    std::cout << "Fail: EOS table problem with kfp "
              << te->get("kfp",0) << std::endl;
    if (err_as_small_wgt) {
      log_weight=-826.0;
      //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
      return 0;
    }
    return 26;
  }

  if (check_timing) {
    // Record end time
    auto end = std::chrono::high_resolution_clock::now();
    // Compute the duration
    std::chrono::duration<double> elapsed = end - start;
    // Output the elapsed time in seconds
    std::cout << "Execution time EOS: " << elapsed.count() << " seconds\n";
    start = std::chrono::high_resolution_clock::now();
  }
  
  // --------------------------------------------------------------
  // Compute ligo probability

  // We linearly interpolate the log of the probability instead of taking the
  // log of the probability since the latter may lead to a negative probability.

  if (!no_ligo) {
    if (ligo_data.get_rank()==0) {
      o2scl_hdf::hdf_file hfx;
      hfx.open("../ligo/gw170817_kde.o2");
      std::string name="kde_log_prob";
      o2scl_hdf::hdf_input(hfx,ligo_data,name);
      hfx.close();
    }

    dat[dvi["ligo_fail"]]=0.0;

    // If the point is outside of the range specified in the data file, reject
    // it so that we don't do anomalous extrapolations
    if (pars[pvi["M_chirp"]]<ligo_data.get_grid(0,0) ||
        pars[pvi["M_chirp"]]>ligo_data.get_grid(0,ligo_data.get_size(0)-1)) {
      dat[dvi["ligo_fail"]]=1.0;
      if (verbose>0) {
        cout.precision(10);
        cout << "low,M_chirp,high: " << ligo_data.get_grid(0,0) << " "
             << pars[pvi["M_chirp"]] << " "
             << ligo_data.get_grid(0,ligo_data.get_size(0)-1) << endl;
        cout.precision(6);
        cout << "M_chirp out of LIGO data file range." << endl;
      }
      if (err_as_small_wgt) {
        log_weight=-854.0;
        //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
        return 0;
      }
      return 16;
    }
    if (pars[pvi["q"]]<ligo_data.get_grid(1,0) ||
        pars[pvi["q"]]>ligo_data.get_grid(1,ligo_data.get_size(1)-1)) {
      dat[dvi["ligo_fail"]]=1.0;
      if (verbose>0) {
        cout.precision(10);
        cout << "low,q,high: " << ligo_data.get_grid(1,0) << " "
             << pars[pvi["q"]] << " "
             << ligo_data.get_grid(1,ligo_data.get_size(1)-1) << endl;
        cout.precision(6);
        cout << "q out of LIGO data file range." << endl;
      }
      if (err_as_small_wgt) {
        log_weight=-855.0;
        //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
        return 0;
      }
      return 16;
    }

      double M_chirp=pars[pvi["M_chirp"]];
      double q=pars[pvi["q"]];
      double m1=M_chirp*pow(1.0+q,0.2)/pow(q,0.6);
      double m2=M_chirp*pow(q,0.4)*pow(1.0+q,0.2);

      // I's in Msun*km^2
      double I1=tmvsr.interp("gm",m1,"rjw")/3.0/schwarz_km;
      double I2=tmvsr.interp("gm",m2,"rjw")/3.0/schwarz_km;
      // To compute I_bar, divide by G^2*M^3
      double G=schwarz_km/2.0;
      double I_bar1=I1/G/G/m1/m1/m1;
      double I_bar2=I2/G/G/m2/m2/m2;

      // Jim's fit from Steiner, Lattimer, and Brown (2016)
      double b0=-30.5395;
      double b1=38.3931;
      double b2=-16.3071;
      double b3=3.36972;
      double b4=-0.26105;

      double li=log(I_bar1);
      double li2=li*li;
      double li3=li*li2;
      double li4=li*li3;
      double li5=li*li4;
      double li6=li*li5;

      double Lambda1=exp(b0+b1*li+b2*li2+b3*li3+b4*li4);

      li=log(I_bar2);
      li2=li*li;
      li3=li*li2;
      li4=li*li3;
      li5=li*li4;
      li6=li*li5;
      double Lambda2=exp(b0+b1*li+b2*li2+b3*li3+b4*li4);

      double Lambdat=16.0/13.0*((m1+12.0*m2)*pow(m1,4.0)*Lambda1+
                                (m2+12.0*m1)*pow(m2,4.0)*Lambda2)/
        pow(m1+m2,5.0);
      cout << "Lambdat: " << Lambdat << endl;
      //exit(-1);

    if (Lambdat<ligo_data.get_grid(2,0) ||
        Lambdat>ligo_data.get_grid(2,ligo_data.get_size(2)-1)) {
      dat[dvi["ligo_fail"]]=1.0;
      if (verbose>0) {
        cout.precision(10);
        cout << "low,Lambdat,high: " << ligo_data.get_grid(2,0) << " "
             << Lambdat << " "
             << ligo_data.get_grid(2,ligo_data.get_size(2)-1) << endl;
        cout.precision(6);
        cout << "Lambdat out of LIGO data file range." << endl;
      }
      if (err_as_small_wgt) {
        log_weight=-856.0;
        //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
        return 0;
      }
      return 16;
    }
    double lw_ligo=ligo_data.interp_linear(pars[pvi["M_chirp"]],
                                           pars[pvi["q"]],
                                           Lambdat);
    log_weight+=lw_ligo;
    dat[dvi["Lambdat"]]=Lambdat;
    dat[dvi["lw_ligo"]]=lw_ligo;
    
    if (verbose>0) {
      cout << "log weight from LIGO, log weight after: "
           << lw_ligo << " " << log_weight << endl;
    }
  } else {
    double lw_ligo=0.0;
    dat[dvi["lw_ligo"]]=lw_ligo;
  }

  if (check_timing) {
    // Record end time
    auto end = std::chrono::high_resolution_clock::now();
    // Compute the duration
    std::chrono::duration<double> elapsed = end - start;
    // Output the elapsed time in seconds
    std::cout << "Execution time LIGO: " << elapsed.count() << " seconds\n";
    start = std::chrono::high_resolution_clock::now();
  }
  
  // --------------------------------------------------------------
  // Check Fermi momenta
  
  if (true) {
    double kfn_min=te->interp("nb",0.16,"kfn");
    dat[dvi["kfn_min"]]=kfn_min;
    double kfp_min=te->interp("nb",0.16,"kfp");
    dat[dvi["kfp_min"]]=kfp_min;
    double kfn_max=te->interp("nb",nb_max,"kfn");
    dat[dvi["kfn_max"]]=kfn_max;
    double kfp_max=te->interp("nb",nb_max,"kfp");
    dat[dvi["kfp_max"]]=kfp_max;
    if (true) {
      if (nw.n3_kf<kfn_min || nw.n3_kf>kfn_max) {
        if (verbose>0) {
          std::cout << "Neutron Fermi momentum out of bounds. " << std::endl;
          std::cout << "kfn_min, kfn, kfn_max: "
                    << kfn_min << " " << nw.n3_kf << " " << kfn_max
                    << std::endl;
        }
        if (err_as_small_wgt) {
          log_weight=-816.0;
          //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
          return 0;
        }
        return 16;
      }
      if (nw.p1_kf<kfp_min || nw.p1_kf>kfp_max) {
        if (verbose>0) {
          std::cout << "Proton Fermi momentum out of bounds." << std::endl;
          std::cout << "kfp_min, kfp, kfp_max: "
                    << kfp_min << " " << nw.p1_kf << " " << kfp_max
                    << std::endl;
        }
        if (err_as_small_wgt) {
          log_weight=-817.0;
          //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
          return 0;
        }
        return 17;
      }
      if (nw.n3_dk>kfn_max || nw.p1_dk>kfp_max) {
        std::cout << "Delta k_F out of bounds. " << nw.n3_dk << " "
                  << kfn_max << " " << nw.p1_dk << " " << kfp_max
                  << std::endl;
        if (err_as_small_wgt) {
          log_weight=-818.0;
          //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
          return 0;
        }
        return 18;
      }
    }
  }
  // --------------------------------------------------------------
  // Check R1.4 to be within constraints

  double R14x=tmvsr.interp("gm",1.4,"r");
  if (verbose>0) {
    cout << "R14: " << R14x << endl;
  }
  
  // AWS 10/21/2020: changed from 11km to 9km SR 02/26/2025: Ignoring these
  // constraints for nomma runs if (R14x>16.0 || R14x<9.0) {
  // dat[dvi["tov_fail"]]=1.0; if (verbose>0) { std::cout << "Fail. Radius of
  // 1.4 unphysical." << std::endl;
  //   }
  //   if (err_as_small_wgt) { log_weight=-837.0; //for(size_t k=0;k<ndat;k++)
  //     dat[k]=0.0; return 0;
  //   }
  //   return 37;
  // }
  //
  // if (tmvsr.interp("gm",1.4,"r")>r14_max) { if (err_as_small_wgt) {
  //   log_weight=-843.0; //for(size_t k=0;k<ndat;k++) dat[k]=0.0; return 0;
  //   }
  //   return 43;
  // }

  if (check_timing) {
    // Record end time
    auto end = std::chrono::high_resolution_clock::now();
    // Compute the duration
    std::chrono::duration<double> elapsed = end - start;
    // Output the elapsed time in seconds
    std::cout << "Execution time Fermi mom and radius: " << elapsed.count() << " seconds\n";
    start = std::chrono::high_resolution_clock::now();
  }
  
  // --------------------------------------------------------------
  // Compute the probability from the mass-radius data
  if (!no_qlmxb) {
    double lw_qlmxb=0.0;
    int ns_ret=nsmr_like(pars,dat,nd.MR_data,lw_qlmxb);
    dat[dvi["lw_qlmxb"]]=lw_qlmxb;
    log_weight+=lw_qlmxb;
    
    dat[dvi["nsmr_fail"]]=0.0;
    if (ns_ret!=0 || log_weight<-800.0) {
      dat[dvi["nsmr_fail"]]=1.0;
      if (err_as_small_wgt) {
        log_weight=-843.0;
        //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
        return 0;
      }
      return 43;
    }
  } else {
    double lw_qlmxb=0.0;
    dat[dvi["lw_qlmxb"]]=lw_qlmxb;
  }

  if (check_timing) {
    // Record end time
    auto end = std::chrono::high_resolution_clock::now();
    // Compute the duration
    std::chrono::duration<double> elapsed = end - start;
    // Output the elapsed time in seconds
    std::cout << "Execution time NSMR: " << elapsed.count() << " seconds\n";
    start = std::chrono::high_resolution_clock::now();
  }
  
  // --------------------------------------------------------------
  // Fill dat[] array with EOS and M-R curve

  for(size_t i=0;i<ngrid;i++) {
    if (mvsr_mass_grid[i]<Mns_max) {
      dat[dvi[((std::string)"R_")+o2scl::szttos(i)]]=
        tmvsr.interp("gm",mvsr_mass_grid[i],"r");
    } else {
      dat[dvi[((std::string)"R_")+o2scl::szttos(i)]]=0.0;
    }
    dat[dvi[((std::string)"P_")+o2scl::szttos(i)]]=
      te->interp("ed",eden_grid[i],"pr");
    if (mvsr_mass_grid[i]<Mns_max) {
      dat[dvi[((std::string)"PM_")+o2scl::szttos(i)]]=
        tmvsr.interp("gm",mvsr_mass_grid[i],"pr");
    } else {
      dat[dvi[((std::string)"PM_")+o2scl::szttos(i)]]=0.0;
    }
    dat[dvi[((std::string)"Enb_")+o2scl::szttos(i)]]=
      te->interp("nb",nb_grid[i],"ed");
    dat[dvi[((std::string)"Pnb_")+o2scl::szttos(i)]]=
      te->interp("nb",nb_grid[i],"pr");
  }
  // --------------------------------------------------------------
  // For the EOS calculation, provide thermo and fermion objects
  o2scl::thermo th;
  o2scl::fermion n(o2scl::o2scl_settings.get_convert_units().convert
                   ("kg","1/fm",o2scl_const::mass_neutron_f<double>()),2.0);
  o2scl::fermion p(o2scl::o2scl_settings.get_convert_units().convert
                   ("kg","1/fm",o2scl_const::mass_proton_f<double>()),2.0);
  
  // --------------------------------------------------------------
  // Provide the beta-equilibrium EOS to the nscool_wrap object
  
  nw.nscool_core.clear();
  nw.nscool_core.set_nlines(te->get_nlines()-1);
  nw.nscool_core.line_of_names(((std::string)"Rho Press nbar Ye ")+
                               "Ymu Yn Yp Yla Ysm Ys0 Ysp mstp mstn "+
                               "mstla mstsm msts0 mstsp");
  int temp_index=0;
  
  // This copies all but the lowest density point to the nscool_core table. I'm
  // not sure why the lowest density point (which is n_B=0.05 fm^{-3} as
  // determined by the o2scl::nstar_cold object) is omitted, but this seems to
  // work for now.
  
  for(int i=te->get_nlines()-1; i>0; i--) {
    
    //if (te->get("nb",i)>0.085) {
    
    n.n=te->get("nn",i);
    p.n=te->get("np",i);
    rmfp->calc_e(n,p,th);
    
    double nb=te->get("nb",i);
    double ed=nst.def_eos_tov.ed_from_nb(nb);
    double pr=nst.def_eos_tov.pr_from_nb(nb);
    nw.nscool_core.set
      ("Rho",temp_index,o2scl::o2scl_settings.get_convert_units().convert
       ("Msun/km^3","g/cm^3",ed));
    nw.nscool_core.set
      ("Press",temp_index,o2scl::o2scl_settings.get_convert_units().convert
       ("Msun/km^3","dyne/cm^2",pr));
    nw.nscool_core.set("nbar",temp_index,nb);
    nw.nscool_core.set("Ye",temp_index,te->get("ne",i)/te->get("nb",i));
    nw.nscool_core.set("Ymu",temp_index,te->get("nmu",i)/te->get("nb",i));
    nw.nscool_core.set("Yn",temp_index,te->get("nn",i)/te->get("nb",i));
    nw.nscool_core.set("Yp",temp_index,te->get("np",i)/te->get("nb",i));
    nw.nscool_core.set("Yla",temp_index,0.0);
    nw.nscool_core.set("Ysm",temp_index,0.0);
    nw.nscool_core.set("Ys0",temp_index,0.0);
    nw.nscool_core.set("Ysp",temp_index,0.0);
    nw.nscool_core.set("mstp",temp_index,p.ms/p.m);
    nw.nscool_core.set("mstn",temp_index,n.ms/n.m);
    nw.nscool_core.set("mstla",temp_index,1.0);
    nw.nscool_core.set("mstsm",temp_index,1.0);
    nw.nscool_core.set("msts0",temp_index,1.0);
    nw.nscool_core.set("mstsp",temp_index,1.0);
    
    //}
    
    temp_index++;
  }
  
  // --------------------------------------------------------------
  // Update the crust
  
  if (true) {
    for(size_t j=0;j<nw.nscool_crust.get_nlines();j++) {
      double nb=nw.nscool_crust.get("n",j);
      double ed=nst.def_eos_tov.ed_from_nb(nb);
      double pr=nst.def_eos_tov.pr_from_nb(nb);
      //std::cout << nb << " " << nw.nscool_crust.get("rho",j) << " " <<
      //nw.nscool_crust.get("P",j) << " ";
      nw.nscool_crust.set
        ("rho",j,o2scl::o2scl_settings.get_convert_units().convert
         ("Msun/km^3","g/cm^3",ed));
      nw.nscool_crust.set
        ("P",j,o2scl::o2scl_settings.get_convert_units().convert
         ("Msun/km^3","dyne/cm^2",pr));
      //std::cout << nw.nscool_crust.get("rho",j) << " " <<
      //nw.nscool_crust.get("P",j) << std::endl;
    }
  }

  if (check_timing) {
    // Record end time
    auto end = std::chrono::high_resolution_clock::now();
    // Compute the duration
    std::chrono::duration<double> elapsed = end - start;
    // Output the elapsed time in seconds
    std::cout << "Execution time set-Nscool: " << elapsed.count() << " seconds\n";
    start = std::chrono::high_resolution_clock::now();
  }

#ifndef NO_MPI    
  MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);

  // Ensure that multiple MPI ranks are not writing to the filesystem at the
  // same time
  int tag=0, buffer=0;
  if (mpi_size>1 && mpi_rank>=1) {
    MPI_Recv(&buffer,1,MPI_INT,mpi_rank-1,
             tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  }
#endif

  if (!no_ins && !no_sxrt) {
  
  if(!flag_emu_aws){    
    //size_t total_tasks=log_eta_grid.size()*mass_grid.size(); size_t
    //current_tasks=0;

    // Setup task list as a two sets of triplets, source first, destination
    // second vector<size_t> tasks;

  // --------------------------------------------------------------
  // Compute the cooling curves
  
    nw.sxrt_mode=false;
    nw.time_print[0]=1.0e-10;
    //nw.ptemp=2.0;
    
  for(size_t imass=0;imass<6;imass++) {
    //tasks.push_back(imass);
    double ns_mass=mass_grid[imass];
    // ---------------------------------------------------------
	  // Compute the neutron star structure and store the EOS in a table for the
	  // NScool object

	  int f_ret=nst.fixed(ns_mass);
	  if (f_ret!=0) {
	    if (verbose>0) {
	      std::cout << "Fail: Object nstar_cold failed to "
		      << "compute profile of a "
		      << ns_mass << " neutron star." << std::endl;
	    }
	    log_weight=-846.0;
	    return 0;
	  }

	  // Note that this is different from class variable 'tmvsr', which stores the
	  // full M-R curve. 
	  std::shared_ptr<o2scl::table_units<> > tov_tab=nst.get_tov_results();
	
	  if (tov_tab->get_nlines()<5) {
	    if (verbose>0) {
	      std::cout << "Fail: TOV profile too small." << std::endl;
	    }
	    log_weight=-858.0;
	    return 0;
	  }
	
	  nw.nscool_tov.clear();
	  nw.nscool_tov.line_of_names
	    (((std::string)"step radius n_baryon density ")+
	   "pressure emass phi bmass");
	  nw.nscool_tov.set_nlines(tov_tab->get_nlines());

	  for(size_t i=0; i<tov_tab->get_nlines(); i++) {
	    nw.nscool_tov.set("step",i,i);
	    nw.nscool_tov.set("radius",i,tov_tab->get("r",i)*1000);
	    nw.nscool_tov.set("n_baryon",i,tov_tab->get("nb",i));
	    nw.nscool_tov.set("density",i,tov_tab->get("ed",i)*3.5176758e+14);
	    nw.nscool_tov.set("pressure",i,tov_tab->get("pr",i)*3.1615293e+34);
	    nw.nscool_tov.set("emass",i, tov_tab->get("gm",i));
	    nw.nscool_tov.set("phi",i,tov_tab->get("gp",i));
	    nw.nscool_tov.set("bmass",i,tov_tab->get("bm",i));
	  }

    for(size_t ieta=0;ieta<3;ieta++) {
       
	    nw.eta=1.0e-17;
	    if (ieta==1) nw.eta=1.0e-12;
	    if (ieta==2) nw.eta=1.0e-7;

	  if (verbose>1) {
	    std::cout << "ieta: " << ieta << " imass: " << imass 
		    << " eta: " << nw.eta << " mass: " << ns_mass
		    << std::endl;
          std::cout << "tsz" << "  "
                    << "time" << "         " << "tptr" << "         " 
                    << "lphot" << "        " << "lneut" << "        " 
                    << "dtime" << std::endl;
	  }
	// --------------------------------------------------------------
	// Compute the cooling curve 

	// Control the number of timesteps between output. A large number effectively
	// specifies no output.
	if (verbose>1) {
	  nw.main_out_it=20;
	} else {
	  nw.main_out_it=500;
	}
	
#ifndef NO_OPENMP
	size_t i_thread=omp_get_thread_num();
#else
	size_t i_thread=0;
#endif
	//nw.nscool_debug=0.0;
	
	if (false) {
	  o2scl_hdf::hdf_file hf;
	  hf.open_or_create("temp.o2");
	  hdf_output(hf,nw.nscool_core,"core");
	  hdf_output(hf,nw.nscool_crust,"crust");
	  hdf_output(hf,nw.nscool_tov,"tov");
	  hf.close();
	}
	
	int nw_ret=nw.run(i_thread);

  if (nw_ret!=0) {
	  if (verbose>0) {
	    std::cout << "Fail: Neutron star cooling calculation failed "
		      << "nw_ret: " << nw_ret << std::endl;
	  }
	  log_weight=-806.0;
	  return 0;
	}
	
	if (nw.v_time.size()<5) {
	  if (verbose>0) {
	    std::cout << "Fail: Cooling data not present." << std::endl;
	  }
	  log_weight=-832.0;
	  return 0;
	}
	
  // Write specific heats into output file

  const ubvector &r_grid=nw.tl_prof.get_x_data();
  //computing specific heat at 1.0e4 years
  size_t j=17;
  double int_cv=0;
  for (size_t i=1;i<=r_grid.size();i+=2){
    size_t io2=(i-1)/2;
    // Get the volume information 
    double vol=nw.tl_prof.get(i,j,"vol");
    // Get the specific heat information
    double cv=nw.tl_prof.get(i,j,"cv");
    // Compute the integrated specific heats
    int_cv+=cv*vol;
  }
  // get the redshift and surface temp
  double ephi_surf=nw.tl_prof.get(r_grid.size()-1,j,"ephi");
  double T_surf=(ephi_surf*nw.tl_prof.get(r_grid.size()-1,j,"Tinf"))/1e8;
  int_cv/=T_surf;
  dat[dvi["cv_"+o2scl::szttos(imass)]]=int_cv;
	
	// Interpolate the temperature and luminosity from the cooling calculation
	// into the tensor_grid objects
	for(int k=0;k<100;k++) {
          itp.set(nw.v_time.size(),nw.v_time,nw.v_tptr);
	  double interp_tptr=itp.eval(t_grid[k]);
	  nscool_tptr.set(ieta,imass,k,interp_tptr);
          itp.set(nw.v_time.size(),nw.v_time,nw.v_lphot);
	  double interp_lum=itp.eval(t_grid[k]);
	  nscool_lum.set(ieta,imass,k,interp_lum);
	}
  // ---------------------------------------------------
  // Do SXRT calculations here to save time Q
  double Q_heat=pars[pvi["Q"]];

  if (verbose>1) {
    cout << "eta          M            Mdot         "
         << "log_T        L_phot" << endl;
  }	

  double R=tmvsr.interp("gm",ns_mass,"r");
  double logT=8.0;
  for(size_t k=0;k<30;k++) {
    nw.mdot_index=k;
	  double Mdot=pow(10.0,log_mdot_grid[k]);
	  int ret1=ss_acc_solve(Mdot,Q_heat,ns_mass,R,logT); 
	  if (ret1!=0) {
	    if (verbose>0) {
	      cout << "SXRT curve failed." << endl;
	    }
	    log_weight=-838.0;
	    //for(size_t ik=0;ik<ndat;ik++) dat[ik]=0.0;
	    return 0;
	  }
	  double lphot=0.0, lneut=0.0, lheat=0.0;
	  int iret2=acc_compute(logT,Q_heat,ns_mass,R,Mdot,
			  lphot,lneut,lheat);

	  if (verbose>1) {
	    cout << nw.eta << " " << ns_mass << " " << Mdot << " "
	    << logT << " " << lphot << endl;
	  }

	  sxrt_lum.set(ieta,imass,k,lphot);
	  if (iret2!=0) {
	    if (verbose>0) {
	      cout << "SXRT curve failed (2)." << endl;
	    }
	    log_weight=-839.0;
	    //for(size_t ik=0;ik<ndat;ik++) dat[ik]=0.0;
	    return 0;
	  }
  }

  nw.sxrt_mode=false;    
  // -----------------------------------------------
      }
    }
    
  } else {
  // -----------------------------------------------
  // Use ML emulators to generate cooling and accretion curves

    ubvector ex(25), ey(1800), ez(540);
    for(size_t g=0;g<25;g++){
      ex[g]=pars[g];
    }
    
    ip_ins->eval(ex,ey);

    size_t index=0;
    for(size_t ieta=0;ieta<3;ieta++) {
      for(size_t imass=0;imass<6;imass++) {
        for(size_t i=0;i<ngrid;i++) {
          nscool_lum.set(ieta,imass,i,ey[index]);
          index+=1;
        }
      }
    }

    ip_sxrt->eval(ex,ez);

    index=0;
    for(size_t ieta=0;ieta<3;ieta++) {
      for(size_t imass=0;imass<6;imass++) {
        for(size_t i=0;i<30;i++) {
          sxrt_lum.set(ieta,imass,i,ez[index]);
          index+=1;
        }
      }
    }
  }
    }

#ifndef NO_MPI
  // Send a message to the next MPI rank
  if (mpi_size>1 && mpi_rank<mpi_size-1) {
    MPI_Send(&buffer,1,MPI_INT,mpi_rank+1,
             tag,MPI_COMM_WORLD);
  }
#endif
  
  // --------------------------------------------------------------
  // Copy cooling curves to dat[] array
  
  for(size_t ieta=0;ieta<3;ieta++) {
    for(size_t imass=0;imass<6;imass++) {
      for(size_t i=0;i<ngrid;i++) {
        dat[dvi[((std::string)"cT_")+o2scl::szttos(ieta)+"_"+
                o2scl::szttos(imass)+"_"+o2scl::szttos(i)]]=
          nscool_tptr.get(ieta,imass,i);

        dat[dvi[((std::string)"cL_")+o2scl::szttos(ieta)+"_"+
                o2scl::szttos(imass)+"_"+o2scl::szttos(i)]]=
          nscool_lum.get(ieta,imass,i);
      }
      for(size_t i=0;i<30;i++) {
        dat[dvi[((string)"sL_")+o2scl::szttos(ieta)+"_"+
                o2scl::szttos(imass)+"_"+o2scl::szttos(i)]]=
          sxrt_lum.get(ieta,imass,i);
      }
    }
  }

  if (check_timing) {
    // Record end time
    auto end = std::chrono::high_resolution_clock::now();
    // Compute the duration
    std::chrono::duration<double> elapsed = end - start;
    // Output the elapsed time in seconds
    std::cout << "Execution time Cooling: " << elapsed.count() << " seconds\n";
    start = std::chrono::high_resolution_clock::now();
  }
  // --------------------------------------------------------------
  // INS likelihood calculation

  dat[dvi["ins_fail"]]=0.0;
  if (no_ins==false) {
    double lw_ins=0.0;
    int il_ret=ins_like(nv,pars,lw_ins,dat);
    dat[dvi["lw_ins"]]=lw_ins;
    log_weight+=lw_ins;
    if (il_ret!=0 || log_weight<-800) {
      dat[dvi["ins_fail"]]=1.0;
      if (err_as_small_wgt) {
        log_weight=-841.0;
        //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
        return 0;
      }
      return 41;
    }
  } else {
    no_ins_fun(dat);
    dat[dvi["lw_ins"]]=0.0;
  }
  // --------------------------------------------------------------
  // SXRT likelihood calculation

  dat[dvi["sxrt_fail"]]=0.0;
  if (no_sxrt==false) {
    double lw_sxrt=0.0;
    int sl_ret=sxrt_like(pars,lw_sxrt,dat);
    dat[dvi["lw_sxrt"]]=lw_sxrt;
    log_weight+=lw_sxrt;
    if (sl_ret!=0 || log_weight<-800.0) {
      dat[dvi["sxrt_fail"]]=1.0;
      if (err_as_small_wgt) {
        log_weight=-842.0;
        //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
        return 0;
      }
      return 42;
    }
  } else {
    no_sxrt_fun(dat);
    dat[dvi["lw_sxrt"]]=0.0;
  }

  if (check_timing) {
    // Record end time
    auto end = std::chrono::high_resolution_clock::now();
    // Compute the duration
    std::chrono::duration<double> elapsed = end - start;
    // Output the elapsed time in seconds
    std::cout << "Execution time INS and SXRT: " << elapsed.count() << " seconds\n";
  }
  
  // ---------------------------------------------------------------
  
  dat[dvi["cs2_max"]]=cs2_max;
  
  // --------------------------------------------------------------
  // Output final log likelihood
  
  if (verbose>0) {
    std::cout << "Final log weight: " << log_weight << std::endl;
  }
  
  dat[dvi["prob_fail"]]=0.0;
  if (!std::isfinite(log_weight)) {
    dat[dvi["prob_fail"]]=1.0;
    if (verbose>0) {
      std::cout << "Fail: log(wgt) is not finite." << std::endl;
    }
    if (err_as_small_wgt) {
      log_weight=-812.0;
      //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
      return 0;
    }
    return 12;
  }
  
  dat[dvi["other_fail"]]=0.0;
  for(size_t i=0;i<ndat;i++) {
    dat[dvi["other_fail"]]=0.0;
    if (!std::isfinite(dat[i])) {
      if (verbose>0) {
        std::cout << "Fail: dat[" << i << "] is not finite." << std::endl;
      }
      if (err_as_small_wgt) {
        log_weight=-815.0;
        //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
        return 0;
      }
      return 15;
    }
  }
  
  // --------------------------------------------------------------
  // Test accuracy of estimators
  
  if (false) {
    ubvector vout(1+ndat), verr(1+ndat);
    std::vector<size_t> index;
    std::vector<double> extrap;
    interpm_idw<> esti_idw;
    if (log_weight>-799.0) {
      std::cout << "Going to interpm_idw." << std::endl;
      esti_idw.eval_unc_tl_index(pars,vout,verr,index,extrap);
      std::cout << "intermp_idw estimate: "
                << vout[0] << " " << verr[0] << " " << exp(log_weight)
                << std::endl;
    }
  }
  
  // --------------------------------------------------------------
  
  if (check_cooling) {
    std::cout << "Check cooling is true, so exiting." << std::endl;
    exit(-1);
  }
  
  bool debug_dat=false;
  if (debug_dat) {
    std::vector<std::string> svin, svout;
    for(size_t ij=0;ij<ndat;ij++) {
      svin.push_back(o2scl::szttos(ij)+" "+o2scl::dtos(dat[ij]));
    }
    o2scl::screenify(svin.size(),svin,svout);
    for(size_t ij=0;ij<svout.size();ij++) {
      std::cout << svout[ij] << std::endl;
    }
    exit(-1);
  }
  
  if (verbose>1) {
    std::cout << "Success." << std::endl;
    /*
      std::cout << "Parameters: " << std::endl; for(size_t
      i=0;i<pars.size();i++) { std::cout << "pars[" << i << "]=" << pars[i] <<
      ";" << std::endl;
      }
      exit(-1);
    */
  }
  //return -10; exit(-1);
  if (check_timing) {
    // Record end time
    auto end = std::chrono::high_resolution_clock::now();
    // Compute the duration
    std::chrono::duration<double> elapsed = end - start1;
    // Output the elapsed time in seconds
    std::cout << "Execution time point function: " << elapsed.count() << " seconds\n";
  }
  
  return 0;
  // End of point() function
}
  
int data_eval::fill_line(const ubvector &pars, double log_weight,
                         std::vector<double> &line,
                         std::array<double,ndat> &dat) {
  for(size_t i=0;i<ndat;i++) {
    line.push_back(dat[i]);
  }
  //if (line[pvi.size()+5+dvi["emulated"]]>0.5) return -20;
  return 0;
}
