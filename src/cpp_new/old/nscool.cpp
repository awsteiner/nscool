#include "nscool.h"

#ifdef O2SCL_MPI
#include "mpi.h"
#endif

std::vector<nscool *> nscool_ptrs;

nscool::nscool(std::string dir) {

  o2scl_hdf::hdf_file hf;
  std::string name;

  int mpi_rank=0, mpi_size=1;

#ifdef O2SCL_MPI
  // Get MPI rank, etc.
  MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);

  // Ensure that multiple threads aren't writing to the
  // filesystem at the same time
  int tag=0, buffer=0;
  if (mpi_size>1 && mpi_rank>=1) {
    MPI_Recv(&buffer,1,MPI_INT,mpi_rank-1,
	     tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  }
#endif

  if (false) {
    std::cout << "In nscool, rank " << mpi_rank
	      << " reading data files." << std::endl;
  }
    
  default_star(dir);

#ifdef O2SCL_MPI
  if (mpi_size>1 && mpi_rank<mpi_size-1) {
    MPI_Send(&buffer,1,MPI_INT,mpi_rank+1,
	     tag,MPI_COMM_WORLD);
  }
#endif  
  istrange=0;
  idt=1;
  htot=0.0;
  contraction=0.0;
  gammacryst=210.0;
  gammaliq=180.0;
  version = "NEW";
  // Read Numerical Parameters: nscool_num_param---------------------------
  time0=0.0;
  timemax=2.0e10;
  istepmax=1000000;
  itrial_max=20;
  itrial_opt=12;
  tcut=2.0;
  dtime=1.0e-12;
  dtlimit=3.15e15;
  scale_dt0=1.2;
  scale_dt1=1.5;
  repeat=0.2;
  istart=2;
  mratt=1.0e-12;
  mratl=1.0e-10;
  mrats=1.0e-10;
  tvar=1.20;
  svar=1.05;
  tcon=1.0e12;
  // ---------------------------------------------------------------------
  dtime=dtime*year;
  time0=time0*year;
  double odtime=dtime;
  double scale_dt=scale_dt1;
  // Initialize the cooling calculation: nscool_cool_param----------------
  pscreen=1;
  debug=1.0;
  istep_debug=0;
  pteff=0.0;
  ptemp_arg=1.0;
  pstar=0.0;
  idump1=1;
  idump2=111;
  idump3=421;
  tempmin=1.0e4;
  tempini=1.0e10;
  // If this is true, then the function 'cvelec()' is used
  // to handle the electron specific heat rather than
  // the simple expression for degenerate electrons
  icvel_nodeg=0;
  emnco=5.0;
  emncr=5.0;
  emp=3.0;
  p0=0.1;

  ptemp=0.0;
  // Setup time_print
  time_print.resize(25);
  time_print[0]=1.0e-10;
  time_print[1]=1.0e-4;
  time_print[2]=3.0e-4;
  time_print[3]=1.0e-3;
  time_print[4]=3.0e-3;
  time_print[5]=1.0e-2;
  time_print[6]=3.0e-2;
  time_print[7]=1.0e-1;
  time_print[8]=3.0e-1;
  time_print[9]=1.0e+0;
  time_print[10]=3.0e+0;
  time_print[11]=1.0e+1;
  time_print[12]=3.0e+1;
  time_print[13]=1.0e+2;
  time_print[14]=3.0e+2;
  time_print[15]=1.0e+3;
  time_print[16]=3.0e+3;
  time_print[17]=1.0e+4;
  time_print[18]=3.0e+4;
  time_print[19]=1.0e+5;
  time_print[20]=3.0e+5;
  time_print[21]=1.0e+6;
  time_print[22]=3.0e+6;
  time_print[23]=1.0e+7;
  time_print[24]=3.0e+7;
  T_fact_drip=0.8;
  T_fact_surf=0.5;
  eta=0.0;

  temp_ygrid.clear();
  for (size_t i=0;i<time_print.size();i++) {
    temp_ygrid.push_back(time_print[i]);
  }

  // The array 'tprint' the cooling code is actually zero-indexed,
  // but Dany's code ignores the first value.
  itpmax=time_print.size();
  for(int i=1;i<=itpmax;i++) {
    tprint[i]=time_print[i-1];
  }

  if (sxrt_mode) {
    tempini=sxrt_Tinit;
    timemax=1.5e-12;
  }
  //READ OUTER BOUNDARY PARAMETERS: Nscool_bound_param---------------------
  ifteff=3;
  eta_arg=eta;
  mag_coeff=3.0;
  tb_acc0=0.0;
  // ----------------------------------------------------------------------
  eta=eta/0.443;
  eta_0=eta;
  /* READ PAIRING PARAMETERS:
    This is the SFB neutrino singlet gap, the T73 proton singlet
    gap, and neutrino triplet gap "a" from the minimal cooling
    paper.
  */
  sfn1s0=1;
  sfn3p2=101;
  sfp1s0=3;
  sfl1s0=0;
  fn1s0=1;
  fn3p2=1;
  fp1s0=1;
  fl1s0=1;
  sfquark=0;
  // READ CONDUCTIVITY PARAMETERS: ----------------------------------------
  iopacity=0;
  icon_crust=3;
  icon_core=2;
  iconnothing2=0;
  iconnothing3=0;
  iconnothing4=0;
  iconnothing5=0;
  connothing1=0.0;
  connothing2=0.0;
  qimp=1.0e-10;
  connothing4=0.0;
  connothing5=0.0;
  // READ HEATING PARAMETERS: ---------------------------------------------
  i_heat_deep_crust=0;
  i_heat_deposit=0;
  t_dep=0.0;
  del_t_dep=0.0;
  total_heat=0.0;
  i_dep=0;
  i_heat_convert=0;
  MeV_neutron=0.0;
  i_heat_vortex_creep=0;
  j_44=0.0;
  i_heat_joule=0;
  i_heat_field_decay=0;
  i_heat_cold1=0;
  i_heat_cold2=0;
  i_heat_cold3=0;
  i_heat_cold4=0;
  heat_cold1=0.0;
  heat_cold2=0.0;
  heat_cold3=0.0;
  heat_cold4=0.0;
  // READ MAGNETIC FIELD PARAMETERS:--------------------------------------
  ifield = 0;
  i_gr_field = 1;
  bfield = 0.0e12;
  i0 = 113;
  i1 = 289;
  nothing = 0.1;
  start_b_defusion = 0.0;
  i_joule_heat = 0;
  i_conb = 0;
  i_eleconb = 0;
  // READ ACCRETION PARAMETERS:-------------------------------------------
  i_acc = 0;
  m_dot0 = 0.0;
  t_acc0 = 0.0;
  t_acc1 = 0.0;
  t_acc2 = 0.0;
  alpha_acc = 0.0;
  time_step_min = 0.0;
  eta_Edd = 0.0;
  X_Edd = 0.0;
  t_acc0 *= 3.15576e7;
  t_acc1 *= 3.15576e7;
  t_acc2 *= 3.15576e7;
  time_step_min *= 3.15576e7;
  eta = std::max(1.0e-50, eta);
  eta0 = std::max(1.0e-50, eta0);
  // READ STAR STRUCTURE LAYOUT: -----------------------------------------
  rhocore=1.6e14;
  rhodrip=4.0e11;
  rhoenv=1.0e8;
  rhosurf=1.0e10;
  icore=111;
  idec=60;

  n1_tc=0.0;
  n1_kf=0.0;
  n1_dk=0.0;
  mdot_index=0;

  nscool_debug=0;

  fix_durca=0.0;
  alpha_durca=1.0e-8;
  beta_durca=1.0;

  main_out_it=20;
    
  // Read data for pair bremsstrahlung
  pair_brem_data(pb_logt,pb_nalpha);
  int pb_n=56;
  double pb_bound1=1.0e30;
  double pb_bound2=1.0e30;
  spline(pb_logt,pb_nalpha,&pb_n,&pb_bound1,&pb_bound2,pb_n2);
    
  // Data for superfluid suppression
  sf_suppress_data(sf_lgtau1,sf_lgtau2,sf_lgr);
  int sf_n1=35;
  int sf_n2=35;
  spline2(sf_lgtau1,sf_lgtau2,sf_lgr,&sf_n1,&sf_n2,sf_lgr2);
    
  // Lepton inits
  electron.init(o2scl::o2scl_settings.get_convert_units().convert
		("kg","1/fm",o2scl_const::mass_electron_f<double>()),2.0);
  electron_deriv.init(o2scl::o2scl_settings.get_convert_units().convert
		      ("kg","1/fm",o2scl_const::mass_electron_f<double>()),2.0);
  muon.init(o2scl::o2scl_settings.get_convert_units().convert
	    ("kg","1/fm",o2scl_const::mass_muon_f<double>()),2.0);

  sxrt_mode=false;
  sxrt_Tinit=0.0;

}

void nscool::sf_suppress_data(double *lgtau1, double *lgtau2,
				   double *lgr) {
  
#include "../sf_suppression.h"
    
  return;
}

void nscool::pair_brem_data(double *logt, double *nalpha) {

  nalpha[0]=2.119400e+02;
  nalpha[1]=2.119314e+02;
  nalpha[2]=2.118338e+02;
  nalpha[3]=2.116714e+02;
  nalpha[4]=2.114446e+02;
  nalpha[5]=2.111548e+02;
  nalpha[6]=2.108042e+02;
  nalpha[7]=2.103965e+02;
  nalpha[8]=2.099369e+02;
  nalpha[9]=2.094313e+02;
  nalpha[10]=2.088862e+02;
  nalpha[11]=2.025300e+02;
  nalpha[12]=1.965045e+02;
  nalpha[13]=1.913841e+02;
  nalpha[14]=1.870169e+02;
  nalpha[15]=1.832339e+02;
  nalpha[16]=1.799095e+02;
  nalpha[17]=1.769527e+02;
  nalpha[18]=1.742953e+02;
  nalpha[19]=1.718859e+02;
  nalpha[20]=1.555852e+02;
  nalpha[21]=1.458835e+02;
  nalpha[22]=1.389667e+02;
  nalpha[23]=1.335816e+02;
  nalpha[24]=1.291637e+02;
  nalpha[25]=1.254108e+02;
  nalpha[26]=1.221419e+02;
  nalpha[27]=1.192400e+02;
  nalpha[28]=1.166251e+02;
  nalpha[29]=9.846525e+01;
  nalpha[30]=8.600176e+01;
  nalpha[31]=7.553085e+01;
  nalpha[32]=6.634075e+01;
  nalpha[33]=5.830211e+01;
  nalpha[34]=5.135016e+01;
  nalpha[35]=4.539033e+01;
  nalpha[36]=4.030437e+01;
  nalpha[37]=3.596899e+01;
  nalpha[38]=1.500019e+01;
  nalpha[39]=8.665605e+00;
  nalpha[40]=5.942522e+00;
  nalpha[41]=4.494431e+00;
  nalpha[42]=3.613857e+00;
  nalpha[43]=3.027656e+00;
  nalpha[44]=2.611401e+00;
  nalpha[45]=2.301280e+00;
  nalpha[46]=2.061534e+00;
  nalpha[47]=1.063250e+00;
  nalpha[48]=7.494363e-01;
  nalpha[49]=5.912537e-01;
  nalpha[50]=4.941738e-01;
  nalpha[51]=4.277401e-01;
  nalpha[52]=3.790197e-01;
  nalpha[53]=3.415363e-01;
  nalpha[54]=3.116674e-01;
  nalpha[55]=0.000000e-00;

  logt[0]=log10(1.000000e-10)+9.0;
  logt[1]=log10(1.160093e-04)+9.0;
  logt[2]=log10(2.320186e-04)+9.0;
  logt[3]=log10(3.480278e-04)+9.0;
  logt[4]=log10(4.640371e-04)+9.0;
  logt[5]=log10(5.800464e-04)+9.0;
  logt[6]=log10(6.960557e-04)+9.0;
  logt[7]=log10(8.120650e-04)+9.0;
  logt[8]=log10(9.280742e-04)+9.0;
  logt[9]=log10(1.044084e-03)+9.0;
  logt[10]=log10(1.160093e-03)+9.0;
  logt[11]=log10(2.320186e-03)+9.0;
  logt[12]=log10(3.480278e-03)+9.0;
  logt[13]=log10(4.640371e-03)+9.0;
  logt[14]=log10(5.800464e-03)+9.0;
  logt[15]=log10(6.960557e-03)+9.0;
  logt[16]=log10(8.120650e-03)+9.0;
  logt[17]=log10(9.280742e-03)+9.0;
  logt[18]=log10(1.044084e-02)+9.0;
  logt[19]=log10(1.160093e-02)+9.0;
  logt[20]=log10(2.320186e-02)+9.0;
  logt[21]=log10(3.480278e-02)+9.0;
  logt[22]=log10(4.640371e-02)+9.0;
  logt[23]=log10(5.800464e-02)+9.0;
  logt[24]=log10(6.960557e-02)+9.0;
  logt[25]=log10(8.120650e-02)+9.0;
  logt[26]=log10(9.280742e-02)+9.0;
  logt[27]=log10(1.044084e-01)+9.0;
  logt[28]=log10(1.160093e-01)+9.0;
  logt[29]=log10(2.320186e-01)+9.0;
  logt[30]=log10(3.480278e-01)+9.0;
  logt[31]=log10(4.640371e-01)+9.0;
  logt[32]=log10(5.800464e-01)+9.0;
  logt[33]=log10(6.960557e-01)+9.0;
  logt[34]=log10(8.120650e-01)+9.0;
  logt[35]=log10(9.280742e-01)+9.0;
  logt[36]=log10(1.044084e+00)+9.0;
  logt[37]=log10(1.160093e+00)+9.0;
  logt[38]=log10(2.320186e+00)+9.0;
  logt[39]=log10(3.480278e+00)+9.0;
  logt[40]=log10(4.640371e+00)+9.0;
  logt[41]=log10(5.800464e+00)+9.0;
  logt[42]=log10(6.960557e+00)+9.0;
  logt[43]=log10(8.120650e+00)+9.0;
  logt[44]=log10(9.280742e+00)+9.0;
  logt[45]=log10(1.044084e+01)+9.0;
  logt[46]=log10(1.160093e+01)+9.0;
  logt[47]=log10(2.320186e+01)+9.0;
  logt[48]=log10(3.480278e+01)+9.0;
  logt[49]=log10(4.640371e+01)+9.0;
  logt[50]=log10(5.800464e+01)+9.0;
  logt[51]=log10(6.960557e+01)+9.0;
  logt[52]=log10(8.120650e+01)+9.0;
  logt[53]=log10(9.280742e+01)+9.0;
  logt[54]=log10(1.044084e+02)+9.0;
  logt[55]=log10(1.000000e+10)+9.0;
    
  return;
}

void nscool::default_star(std::string dir) {
    
  o2scl_hdf::hdf_file hf;
  std::string name;
    
  // Read default crust EOS
  hf.open(dir+"/crust_HZD_NV.o2");
  hdf_input(hf,nscool_crust,name);
  hf.close();
    
  // Read default core EOS
  hf.open(dir+"/core_APR.o2");
  hdf_input(hf,nscool_core,name);
  hf.close();
    
  // Read default stellar profile
  hf.open(dir+"/tov_APR_14.o2");
  hdf_input(hf,nscool_tov,name);
  hf.close();
    
  return;
}

void nscool::hhj_eos(double mass) {

  eos_had_hhj hhj;
  o2scl::nstar_cold nc;
  o2scl::convert_units<double> &cu=o2scl::o2scl_settings.get_convert_units();

  // ------------------------------------------------------------
  // Construct the EOS from an nstar_cold object
      
  // Automatically construct beta-equilibrium EOS
  nc.verbose=0;
  nc.def_tov.verbose=0;
  nc.set_eos(hhj);
  nc.err_nonconv=false;
  nc.calc_eos();
  std::shared_ptr<o2scl::table_units<> > nc_eos=nc.get_eos_results();
  nc.def_eos_tov.read_table(*nc_eos,"ed","pr","nb");

  // Prepare to compute effective masses
  o2scl::fermion n(o2scl::o2scl_settings.get_convert_units().convert
		   ("kg","1/fm",o2scl_const::mass_neutron_f<double>()),2.0);
  o2scl::fermion p(o2scl::o2scl_settings.get_convert_units().convert
		   ("kg","1/fm",o2scl_const::mass_proton_f<double>()),2.0);
  n.non_interacting=false;
  p.non_interacting=false;
  o2scl::thermo th;
      
  // ------------------------------------------------------------
  // Create a core table of the correct size with the correct units

  // Clear table for new columns
  nscool_core.clear();
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
    hhj.calc_e(n,p,th);
    double line[17]={cu.convert("1/fm^4","g/cm^3",
				nc_eos->interp("nb",nb,"ed")),
		     cu.convert("1/fm^4","dyne/cm^2",
				nc_eos->interp("nb",nb,"pr")),nb,
		     nc_eos->interp("nb",nb,"ne")/nb,
		     nc_eos->interp("nb",nb,"nmu")/nb,
		     nc_eos->interp("nb",nb,"nn")/nb,
		     nc_eos->interp("nb",nb,"np")/nb,
		     n.ms/n.m,p.ms/p.m,0.0,0.0,0.0,0.0,
		     1.0,1.0,1.0,1.0};
    nscool_core.line_of_data(17,line);
  }
      
  // ------------------------------------------------------------
  // Use the nstar_cold object to construct the profile

  // Construct neutron star profile
  nc.def_tov.calc_gpot=true;
  nc.calc_nstar();
  std::shared_ptr<o2scl::table_units<> > nc_prof=nc.get_tov_results();
  std::cout << "M_max: " << nc_prof->max("gm") << std::endl;
  std::cout << "R(M_max): "
	    << nc_prof->get("r",nc_prof->lookup("gm",nc_prof->max("gm")))
	    << std::endl;

  double mass2=mass;
  if (mass2<0.0) mass2=mass+nc_prof->max("gm");
  std::cout << "Set mass to " << mass2 << std::endl;
  nc.fixed(mass2);
      
  // ------------------------------------------------------------
  // Create a table with the right size and the right units

  // Clear table for new columns
  nscool_tov.clear();
  nscool_tov.line_of_names(((std::string)"step radius n_baryon density ")+
			   "pressure emass phi bmass");
      
  // Radial grid
  double r_max=nc_prof->max("r");
      
  // Fill TOV table
  for(size_t ix=0;ix<153;ix++) {
    // A grid which focuses most of the points on the outer part
    // of the star
    double r=r_max*2.0*(1.0-pow(2.0,-((double)ix)/152.0));
    std::vector<double> line={((double)ix),r*1.0e3,
      nc_prof->interp("r",r,"nb"),
      cu.convert("1/fm^4","g/cm^3",
                 nc_prof->interp("r",r,"ed")),
      cu.convert("1/fm^4","dyne/cm^2",
                 nc_prof->interp("r",r,"pr")),
      nc_prof->interp("r",r,"gm"),
      nc_prof->interp("r",r,"gp"),
      nc_prof->interp("r",r,"bm")};
    nscool_tov.line_of_data(8,line);
  }

  return;
}

void nscool::tptr_init
(int ifteff, double tempini, double ephi_surf,
 double ephi_drip, double ephi_core, double &tsurface,
 double &tdrip, double &tcore, double &tb_acc0) {
  
  if (ifteff!=15) {
    if (tempini>0.0) {
      tsurface=T_fact_surf*ephi_surf*tempini;
      tdrip=T_fact_drip*ephi_drip*tempini;
      tcore=1.0*ephi_core*tempini;
    } else {
      tsurface=1.0e9;
      tdrip=2.0e10;
      tcore=1.0e11;
    }
  } else {
    tb_acc0*=ephi_surf;
    tsurface=tb_acc0;
    tdrip=tb_acc0;
    tcore=tb_acc0;
  }
  return;
}

double nscool::Teff
(double Tb, int ifteff, double eta_arg, double bfield,
 int istep, double time, double Ts1, double Ts2,
 double Z, double A, double Rho, int debug,
 double gs14, double compactness) {

  // The internal temperature in units of 10^9 K
  double Tb9=Tb/1.0e9;

  // T_{*} in units of 10^{6} K
  double Ts=sqrt(7.0e0*Tb9*sqrt(gs14));

  // zeta
  double z=Tb9-Ts/1.0e3;

  // The effective temperature to the fourth power for an Iron
  // envelope in units of 10^{6} K
  double t4_iron=gs14*(pow(7.0e0*z,2.25)+pow(z/3.0e0,1.25));

  // The effective temperature to the fourth power for a
  // fully accreted envelope in units of 10^{6} K
  double t4_wacc=gs14*pow(18.1e0*Tb9,2.42);

  // For a partially accreted envelope
  double t4_acc;
  if (eta_arg>1.0e-30) {
    double a=(1.2e0+pow(5.3e-6/eta_arg,0.38))*pow(Tb9,5.0/3.0);
    t4_acc=(a*t4_iron+t4_wacc)/(a+1.0e0);
  } else {
    t4_acc=t4_iron;
  }

  // Return the final effective temperature in Kelvin
  return pow(t4_acc,0.25)*1.0e6;
}

void nscool::print_temp(int istep, int itprint,
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
			     double *qpbf_n3p2, double *qpbf_q) {
  
  // T_eff, as passed by the Fortran code in t_effective,
  // is not currently stored in tl_prof.o2. The variable
  // istep is also not currently stored anywhere.

  if (fabs(time_print[itprint]-time)/fabs(time)>1.0e-6) {
    O2SCL_ERR("Temperature print sanity check.",o2scl::exc_einval);
  }

  if (sxrt_mode) {
    itprint=mdot_index;
  }
    
  if (itprint==0) {
    tl_prof.clear();
    std::vector<double> r_grid;
    for(int i=imax;i>=1;i-=2) {
      r_grid.push_back(rad[imax+1-i]/1.0e5);
    }
    if(!sxrt_mode) {
    tl_prof.set_xy("r",r_grid.size(),r_grid,
		   "t",temp_ygrid.size(),temp_ygrid);
       } else {
         tl_prof.set_xy("r",r_grid.size(),r_grid,
		   "mdot",temp_ygrid.size(),temp_ygrid);
       }
    
    tl_prof.line_of_names(((std::string)"rho ephi vol Tinf Linf tcn tcp qnu ")+
			  "qeebrem qnpb qplasma qsynch qbubble qpair "+
			  "qphoto qbrem_nn qmurca_nucl qbrem_nucl "+
			  "qmurca_hyp qbrem_hyp qdurca_np qdurca_lap "+
			  "qdurca_smn qdurca_smla qdurca_sms0 qfast "+
			  "qdurca_q qmurca_q qpbf_n1s0 qpbf_p1s0 "+
			  "qpbf_n3p2 qpbf_q qmax "+
			  "cv cv_n cv_p cv_e cv_m cv_la "+
			  "cv_sm cv_s0 cv_sp cv_q");
    tl_prof.add_constant("it_last",itprint);
  }

  tl_prof.set_constant("it_last",itprint);

  if (ptemp>=2.0) {
    std::cout << "Time: " << time << " years, T_eff: "
	      << t_effective << " K" << std::endl;

  }
  
  for(int i=imax;i>=1;i-=2) {
    int io2=i/2;
    
    double logtemp=w1*log(otemp[i])+w2*log(temp[i]);
    double temperature=exp(logtemp);
    double lumino=0.0;
    if (i!=1) {
      double loglum=w1*log(fabs(olum[i-1]))+w2*log(fabs(lum[i-1]));
      lumino=exp(loglum);
      if (lum[i-1]<0.0) lumino*=-1.0;
    }
    tl_prof.set(io2,itprint,"rho",rrho[i]);
    tl_prof.set(io2,itprint,"ephi",ephi[i]);
    tl_prof.set(io2,itprint,"vol",dvol[i]+dvol[i+1]);
    tl_prof.set(io2,itprint,"Tinf",temperature/ephi[i]);
    tl_prof.set(io2,itprint,"Linf",lumino/e2phi[i-1]);
    tl_prof.set(io2,itprint,"tcn",tcn[i]);
    tl_prof.set(io2,itprint,"tcp",tcp[i]);
    tl_prof.set(io2,itprint,"qnu",qnu[i]);
    tl_prof.set(io2,itprint,"qeebrem",qeebrem[i]);
    tl_prof.set(io2,itprint,"qnpb",qnpb[i]);
    tl_prof.set(io2,itprint,"qplasma",qplasma[i]);
    tl_prof.set(io2,itprint,"qsynch",qsynch[i]);
    tl_prof.set(io2,itprint,"qbubble",qbubble[i]);
    tl_prof.set(io2,itprint,"qpair",qpair[i]);
    tl_prof.set(io2,itprint,"qphoto",qphoto[i]);
    tl_prof.set(io2,itprint,"qbrem_nn",qbrem_nn[i]);
    tl_prof.set(io2,itprint,"qmurca_nucl",qmurca_nucl[i]);
    tl_prof.set(io2,itprint,"qbrem_nucl",qbrem_nucl[i]);
    tl_prof.set(io2,itprint,"qmurca_hyp",qmurca_hyp[i]);
    tl_prof.set(io2,itprint,"qbrem_hyp",qbrem_hyp[i]);
    tl_prof.set(io2,itprint,"qdurca_np",qdurca_np[i]);
    tl_prof.set(io2,itprint,"qdurca_lap",qdurca_lap[i]);
    tl_prof.set(io2,itprint,"qdurca_smn",qdurca_smn[i]);
    tl_prof.set(io2,itprint,"qdurca_smla",qdurca_smla[i]);
    tl_prof.set(io2,itprint,"qdurca_sms0",qdurca_sms0[i]);
    tl_prof.set(io2,itprint,"qfast",qfast[i]);
    tl_prof.set(io2,itprint,"qdurca_q",qdurca_q[i]);
    tl_prof.set(io2,itprint,"qmurca_q",qmurca_q[i]);
    tl_prof.set(io2,itprint,"qpbf_n1s0",qpbf_n1s0[i]);
    tl_prof.set(io2,itprint,"qpbf_p1s0",qpbf_p1s0[i]);
    tl_prof.set(io2,itprint,"qpbf_n3p2",qpbf_n3p2[i]);
    tl_prof.set(io2,itprint,"qpbf_q",qpbf_q[i]);

    {
      std::vector<double> qvec;
      qvec.push_back(fabs(qeebrem[i]));
      qvec.push_back(fabs(qnpb[i]));
      qvec.push_back(fabs(qplasma[i]));
      qvec.push_back(fabs(qsynch[i]));
      qvec.push_back(fabs(qbubble[i]));
      qvec.push_back(fabs(qpair[i]));
      qvec.push_back(fabs(qphoto[i]));
      qvec.push_back(fabs(qbrem_nn[i]));
      qvec.push_back(fabs(qmurca_nucl[i]));
      qvec.push_back(fabs(qbrem_nucl[i]));
      qvec.push_back(fabs(qmurca_hyp[i]));
      qvec.push_back(fabs(qbrem_hyp[i]));
      qvec.push_back(fabs(qdurca_np[i]));
      qvec.push_back(fabs(qdurca_lap[i]));
      qvec.push_back(fabs(qdurca_smn[i]));
      qvec.push_back(fabs(qdurca_smla[i]));
      qvec.push_back(fabs(qdurca_sms0[i]));
      qvec.push_back(fabs(qfast[i]));
      qvec.push_back(fabs(qdurca_q[i]));
      qvec.push_back(fabs(qmurca_q[i]));
      qvec.push_back(fabs(qpbf_n1s0[i]));
      qvec.push_back(fabs(qpbf_p1s0[i]));
      qvec.push_back(fabs(qpbf_n3p2[i]));
      qvec.push_back(fabs(qpbf_q[i]));
      o2scl::permutation order(qvec.size());
      o2scl::vector_sort_index(qvec.size(),qvec,order);
      if (qvec[order[qvec.size()-1]]>0.0) {
	tl_prof.set(io2,itprint,"qmax",order[qvec.size()-1]+1);
      } else {
	tl_prof.set(io2,itprint,"qmax",-order[qvec.size()-1]+1);
      }
    }

    {
      double total=qeebrem[i]+qnpb[i]+qplasma[i]+qsynch[i]+
	qbubble[i]+qpair[i]+qphoto[i]+qbrem_nn[i]+
	qmurca_nucl[i]+qbrem_nucl[i]+qmurca_hyp[i]+qbrem_hyp[i]+
	qdurca_np[i]+qdurca_lap[i]+qdurca_smn[i]+qdurca_smla[i]+
	qdurca_sms0[i]+qfast[i]+qdurca_q[i]+qmurca_q[i]+
	qpbf_n1s0[i]+qpbf_p1s0[i]+qpbf_n3p2[i]+qpbf_q[i];
      if (fabs(qnu[i]-total)/fabs(qnu[i])>1.0e-6) {
	std::cout << "Emissivity problem " << qnu[i] << " " << total
		  << std::endl;
	exit(-1);
      }
    }
      
    if (ptemp>=2.0) {
      std::cout << i << " "
		<< tl_prof.get_grid_x(io2) << " "
		<< rrho[i] << " "
		<< ephi[i] << " "
		<< dvol[i]+dvol[i+1] << " " 
		<< temperature/ephi[i] << " "
		<< lumino/e2phi[i-1] << " "
		<< qnu[i] << " "
		<< qeebrem[i] << " "
		<< qnpb[i] << " "
		<< qplasma[i] << " "
		<< qsynch[i] << " "
		<< qbubble[i] << " "
		<< qpair[i] << " "
		<< qphoto[i] << " "
		<< qbrem_nn[i] << " "
		<< qmurca_nucl[i] << " "
		<< qbrem_nucl[i] << " "
		<< qmurca_hyp[i] << " "
		<< qbrem_hyp[i] << " "
		<< qdurca_np[i] << " "
		<< qdurca_lap[i] << " "
		<< qdurca_smn[i] << " "
		<< qdurca_smla[i] << " "
		<< qdurca_sms0[i] << " "
		<< qfast[i] << " "
		<< qdurca_q[i] << " "
		<< qmurca_q[i] << " "
		<< qpbf_n1s0[i] << " "
		<< qpbf_p1s0[i] << " "
		<< qpbf_n3p2[i] << " "
		<< qpbf_q[i] << std::endl;
    }
  }
  if (ptemp>=2.0) {
    std::cout << std::endl;
  }
  return;
}

void nscool::print_cv(int itprint, int imax, double *cv,
			   double *cv_n, double *cv_p,
			   double *cv_e, double *cv_m, double *cv_la, 
			   double *cv_sm, double *cv_s0, double *cv_sp,
			   double *cv_q) {
  
  if (sxrt_mode) {
    itprint=mdot_index;
  }
  
  if (ptemp>=1.0) {
    for(int i=imax;i>=1;i-=2) {
      int io2=i/2;
      tl_prof.set(io2,itprint,"cv",cv[i]);
      tl_prof.set(io2,itprint,"cv_n",cv_n[i]);
      tl_prof.set(io2,itprint,"cv_p",cv_p[i]);
      tl_prof.set(io2,itprint,"cv_e",cv_e[i]);
      tl_prof.set(io2,itprint,"cv_m",cv_m[i]);
      tl_prof.set(io2,itprint,"cv_la",cv_la[i]);
      tl_prof.set(io2,itprint,"cv_sm",cv_sm[i]);
      tl_prof.set(io2,itprint,"cv_s0",cv_s0[i]);
      tl_prof.set(io2,itprint,"cv_sp",cv_sp[i]);
      tl_prof.set(io2,itprint,"cv_q",cv_q[i]);
      if (ptemp>=2.0) {
	std::cout << i << " " << cv_n[i] << " " << cv_p[i] << " "
		  << cv_e[i] << " " << cv_m[i] << " "
		  << cv_la[i] << " " << cv_sm[i] << " "
		  << cv_s0[i] << " " << cv_sp[i] << " "
		  << cv_q[i] << std::endl;
      }
    }
  }

  return;
}

void nscool::num_param(double &time0, double &timemax, int &istepmax,
			    int &itrial_max, int &itrial_opt, double &tcut,
			    double &dtime, double &dtlimit, double &scale_dt0,
			    double &scale_dt1, double &repeat, int &istart,
			    double &mratt, double &mratl, double &mrats,
			    double &tvar, double &svar, double &tcon) {
  time0=0.0;
  timemax=2.0e10;
  istepmax=1000000;
  itrial_max=20;
  itrial_opt=12;
  tcut=2.0;
  dtime=1.0e-12;
  dtlimit=3.15e15;
  scale_dt0=1.2;
  scale_dt1=1.5;
  repeat=0.2;
  istart=2;
  mratt=1.0e-12;
  mratl=1.0e-10;
  mrats=1.0e-10;
  tvar=1.20;
  svar=1.05;
  tcon=1.0e12;

  if (sxrt_mode) {
    timemax=1.5e-12;
  }
    
  return;
}

void nscool::bound_param(int &ifteff, double &eta_arg,
			      double &mag_coeff, double &tb_acc0) {
  ifteff=3;
  eta_arg=eta;
  mag_coeff=3.0;
  tb_acc0=0.0;
  return;
}
  
void nscool::cool_param(int &pscreen, double &debug, int &istep_debug,
			     double &pteff, double &ptemp_arg, double &pstar,
			     int &idump1, int &idump2, int &idump3,
			     double &tempmin, double &tempini,
			     int &icvel_nodeg, double &emnco, double &emncr,
			     double &emp, double &p0, int &itpmax,
			     double *tprint) {

  pscreen=1;
  debug=((double)(nscool_debug));
  //debug=1;
  istep_debug=0;
  pteff=0.0;
  ptemp_arg=1.0;
  pstar=0.0;
  idump1=1;
  idump2=111;
  idump3=421;
  tempmin=1.0e4;
  tempini=1.0e10;
  // If this is true, then the function 'cvelec()' is used
  // to handle the electron specific heat rather than
  // the simple expression for degenerate electrons
  icvel_nodeg=0;
  emnco=5.0;
  emncr=5.0;
  emp=3.0;
  p0=0.1;

  // The array 'tprint' the cooling code is actually zero-indexed,
  // but Dany's code ignores the first value.
  itpmax=time_print.size();
  for(int i=1;i<=itpmax;i++) {
    tprint[i]=time_print[i-1];
  }

  if (sxrt_mode) {
    tempini=sxrt_Tinit;
  }
  return;
}

void nscool::urca_settings(double &durca, double &a_durca,
				double &b_durca) {
  durca=fix_durca;
  a_durca=alpha_durca;
  b_durca=beta_durca;
  return;
}

void nscool::crust_eos(double *rho2, double *pres2, int *idata) {

  *idata=((int)(nscool_crust.get_nlines()));
  for(size_t i=0;i<nscool_crust.get_nlines();i++) {
    rho2[i]=nscool_crust.get("rho",(*idata)-1-i);
    pres2[i]=nscool_crust.get("P",(*idata)-1-i);
  }

  return;
}

void nscool::star_struct(int icore, double rhocore,
			      double *rad_t, double *bar_t, double *rho_t,
			      double *pres_t, double *emas_t, double *phi_t,
			      double *rad, int *jmax, int *jcore,
			      double *w1, double *w2) {

  *jmax=nscool_tov.get_nlines();
  if (*jmax>9999) {
    O2SCL_ERR("Table too large.",o2scl::exc_einval);
  }
  *jcore=0;
  for(size_t j=0;j<nscool_tov.get_nlines();j++) {
    rad_t[j]=nscool_tov.get("radius",j)*100.0;
    bar_t[j]=nscool_tov.get("n_baryon",j);
    // This is energy density
    rho_t[j]=nscool_tov.get("density",j);
    pres_t[j]=nscool_tov.get("pressure",j);
    emas_t[j]=nscool_tov.get("emass",j);
    phi_t[j]=nscool_tov.get("phi",j);
    if (rho_t[j]<rhocore && (*jcore)==0) {
      *jcore=j;
    }
  }
  if (rad_t[1]<rad_t[0]) {
    O2SCL_ERR("Structure table should be increasing in radius",
	      o2scl::exc_einval);
  }
  if (*jcore==0) {
    O2SCL_ERR2("Variable 'jcore' not set in ",
	       "nscool::star_struct().",o2scl::exc_einval);
  }

  /*
    double drho=rho_t[(*jcore)-1]-rho_t[(*jcore)];
    *w1=(rhocore-rho_t[(*jcore)])/drho;
    *w2=1.0-(*w1);
    double rad_core=(*w1)*rad_t[(*jcore)-1]+(*w2)*rad_t[(*jcore)];
    for(size_t i=0;i<=icore;i++) {
    rad[i]=cbrt(((float)i)/((float)icore))*rad_core;
    }
  */
    
  return;
}

void nscool::write_cool_curve(std::string fname) {
  o2scl_hdf::hdf_file hf;
  hf.open_or_create(fname);
  o2scl::table_units<> t;
  t.line_of_names("t T L_neut L_phot L_heat");
  t.set_unit("t","yr");
  t.set_unit("T","K");
  t.set_unit("L_neut","erg/s");
  t.set_unit("L_phot","erg/s");
  t.set_unit("L_heat","erg/s");
  for(size_t i=0;i<v_time.size();i++) {
    std::vector<double> line={v_time[i],v_tptr[i],v_lneut[i],v_lphot[i],
		    v_lheat[i]};
    t.line_of_data(5,line);
  }
  hdf_output(hf,t,"cool_curve");
  hf.close();
  return;
}

void nscool::write_tl_prof(std::string fname) {

  if (!tl_prof.is_constant("it_last")) {
    std::cerr << "No table to write in write_tl_prof()."
	      << std::endl;
    return;
  }
    
  o2scl_hdf::hdf_file hf;
  hf.open_or_create(fname);
    
  // Create new table omitting temperatures not stored
  int num_times=1+((int)(tl_prof.get_constant("it_last")+1.0e-6));
  
  //std::cout << num_times << " " << temp_ygrid.size() << std::endl;
    
  // If the time grid hasn't been filled, then restructure
  // the table 
  if (num_times!=((int)temp_ygrid.size())) {
    o2scl::table3d tl_prof2;
    const ubvector &r_grid=tl_prof.get_x_data();
    const ubvector &t_grid=tl_prof.get_y_data();
    ubvector t_grid2(num_times);
    o2scl::vector_copy(num_times,t_grid,t_grid2);
    tl_prof2.set_xy("r",r_grid.size(),r_grid,"t",num_times,t_grid2);
    tl_prof2.line_of_names(((std::string)"rho ephi vol Tinf Linf qnu ")+
			   "qeebrem qnpb qplasma qsynch qbubble qpair "+
			   "qphoto qbrem_nn qmurca_nucl qbrem_nucl "+
			   "qmurca_hyp qbrem_hyp qdurca_np qdurca_lap "+
			   "qdurca_smn qdurca_smla qdurca_sms0 qfast "+
			   "qdurca_q qmurca_q qpbf_n1s0 qpbf_p1s0 "+
			   "qpbf_n3p2 qpbf_q qmax cv "+
			   "cv_n cv_p cv_e cv_m cv_la "+
			   "cv_sm cv_s0 cv_sp cv_q");
    for(size_t i=0;i<r_grid.size();i++) {
      for(size_t j=0;j<((size_t)num_times);j++) {
	tl_prof2.set(i,j,"rho",tl_prof.get(i,j,"rho"));
	tl_prof2.set(i,j,"ephi",tl_prof.get(i,j,"ephi"));
	tl_prof2.set(i,j,"vol",tl_prof.get(i,j,"vol"));
	tl_prof2.set(i,j,"Tinf",tl_prof.get(i,j,"Tinf"));
	tl_prof2.set(i,j,"Linf",tl_prof.get(i,j,"Linf"));
	tl_prof2.set(i,j,"qnu",tl_prof.get(i,j,"qnu"));
	tl_prof2.set(i,j,"qeebrem",tl_prof.get(i,j,"qeebrem"));
	tl_prof2.set(i,j,"qnpb",tl_prof.get(i,j,"qnpb"));
	tl_prof2.set(i,j,"qplasma",tl_prof.get(i,j,"qplasma"));
	tl_prof2.set(i,j,"qsynch",tl_prof.get(i,j,"qsynch"));
	tl_prof2.set(i,j,"qbubble",tl_prof.get(i,j,"qbubble"));
	tl_prof2.set(i,j,"qpair",tl_prof.get(i,j,"qpair"));
	tl_prof2.set(i,j,"qphoto",tl_prof.get(i,j,"qphoto"));
	tl_prof2.set(i,j,"qbrem_nn",tl_prof.get(i,j,"qbrem_nn"));
	tl_prof2.set(i,j,"qmurca_nucl",tl_prof.get(i,j,"qmurca_nucl"));
	tl_prof2.set(i,j,"qbrem_nucl",tl_prof.get(i,j,"qbrem_nucl"));
	tl_prof2.set(i,j,"qmurca_hyp",tl_prof.get(i,j,"qmurca_hyp"));
	tl_prof2.set(i,j,"qbrem_hyp",tl_prof.get(i,j,"qbrem_hyp"));
	tl_prof2.set(i,j,"qdurca_np",tl_prof.get(i,j,"qdurca_np"));
	tl_prof2.set(i,j,"qdurca_lap",tl_prof.get(i,j,"qdurca_lap"));
	tl_prof2.set(i,j,"qdurca_smn",tl_prof.get(i,j,"qdurca_smn"));
	tl_prof2.set(i,j,"qdurca_smla",tl_prof.get(i,j,"qdurca_smla"));
	tl_prof2.set(i,j,"qdurca_sms0",tl_prof.get(i,j,"qdurca_sms0"));
	tl_prof2.set(i,j,"qfast",tl_prof.get(i,j,"qfast"));
	tl_prof2.set(i,j,"qdurca_q",tl_prof.get(i,j,"qdurca_q"));
	tl_prof2.set(i,j,"qmurca_q",tl_prof.get(i,j,"qmurca_q"));
	tl_prof2.set(i,j,"qpbf_n1s0",tl_prof.get(i,j,"qpbf_n1s0"));
	tl_prof2.set(i,j,"qpbf_p1s0",tl_prof.get(i,j,"qpbf_p1s0"));
	tl_prof2.set(i,j,"qpbf_n3p2",tl_prof.get(i,j,"qpbf_n3p2"));
	tl_prof2.set(i,j,"qpbf_q",tl_prof.get(i,j,"qpbf_q"));
	tl_prof2.set(i,j,"qmax",tl_prof.get(i,j,"qmax"));
	tl_prof2.set(i,j,"cv",tl_prof.get(i,j,"cv"));
	tl_prof2.set(i,j,"cv_n",tl_prof.get(i,j,"cv_n"));
	tl_prof2.set(i,j,"cv_p",tl_prof.get(i,j,"cv_p"));
	tl_prof2.set(i,j,"cv_e",tl_prof.get(i,j,"cv_e"));
	tl_prof2.set(i,j,"cv_m",tl_prof.get(i,j,"cv_m"));
	tl_prof2.set(i,j,"cv_la",tl_prof.get(i,j,"cv_la"));
	tl_prof2.set(i,j,"cv_sm",tl_prof.get(i,j,"cv_sm"));
	tl_prof2.set(i,j,"cv_s0",tl_prof.get(i,j,"cv_s0"));
	tl_prof2.set(i,j,"cv_sp",tl_prof.get(i,j,"cv_sp"));
	tl_prof2.set(i,j,"cv_q",tl_prof.get(i,j,"cv_q"));
      }
    }

    tl_prof2.set_interp_type(o2scl::itp_nearest_neigh);
    o2scl::table3d t3dug=tl_prof2.slice_to_uniform_grid
      ("qmax",100,false,100,true);
    tl_prof2.set_interp_type(o2scl::itp_linear);
    t3dug.set_interp_type(o2scl::itp_linear);
    for(size_t k=0;k<tl_prof2.get_nslices();k++) {
      std::string sl_name=tl_prof2.get_slice_name(k);
      if (sl_name!="qmax") {
	t3dug.add_slice_from_table(tl_prof2,sl_name,sl_name);
      }
    }
 
    hdf_output(hf,((const o2scl::table3d &)(tl_prof2)),"tl_prof");
    hdf_output(hf,((const o2scl::table3d &)(t3dug)),"tl_prof_ug");

  } else {

    if (false){
      const ubvector &r_grid=tl_prof.get_x_data();
      const ubvector &t_grid=tl_prof.get_y_data();
      for (size_t i=0;i<r_grid.size();i++) {
        for (size_t j=0;j<t_grid.size();j++) {
          std::cout << i << " " << j << " " 
          << tl_prof.get(i,j,"rho") << std::endl;
        }
      }
    }

    tl_prof.set_interp_type(o2scl::itp_nearest_neigh);
    o2scl::table3d t3dug=tl_prof.slice_to_uniform_grid
      ("qmax",100,false,100,true);
    tl_prof.set_interp_type(o2scl::itp_linear);
    t3dug.set_interp_type(o2scl::itp_linear);
    for(size_t k=0;k<tl_prof.get_nslices();k++) {
      std::string sl_name=tl_prof.get_slice_name(k);
      if (sl_name!="qmax") {
	t3dug.add_slice_from_table(tl_prof,sl_name,sl_name);
      }
    }
      
    hdf_output(hf,((const o2scl::table3d &)(tl_prof)),"tl_prof");
    hdf_output(hf,((const o2scl::table3d &)(t3dug)),"tl_prof_ug");
      
  }
    
  hf.close();
  return;
}

void nscool::gaps(int &sfn1s0_arg, double &n1_tc_arg,
		       double &n1_kf_arg, double &n1_dk_arg,
		       int &sfn3p2_arg, double &n3_tc_arg,
		       double &n3_kf_arg, double &n3_dk_arg,
		       int &sfp1s0_arg, double &p1_tc_arg,
		       double &p1_kf_arg, double &p1_dk_arg) {

  // If sfn3p2 and sfp1s0 are 150, then the double parameters
  // specify the Gaussian
  sfn3p2_arg=sfn3p2;
  sfp1s0_arg=sfp1s0;
  sfn1s0_arg=sfn1s0;
  if (sfn3p2_arg==150) {
    n3_tc_arg=n3_tc;
    n3_kf_arg=n3_kf;
    n3_dk_arg=n3_dk;
  }
  if (sfp1s0_arg==150) {
    p1_tc_arg=p1_tc;
    p1_kf_arg=p1_kf;
    p1_dk_arg=p1_dk;
  }
  if (sfn1s0_arg==150) {
    n1_tc_arg=n1_tc;
    n1_kf_arg=n1_kf;
    n1_dk_arg=n1_dk;
  }
  return;
};

double nscool::cvt_deg(double pf, double m) {
  if (pf == 0.0) {
    return 0.0; } else {
    double N0 = 2.0*m*pf/(2.0*o2scl_const::pi2*o2scl_const::pi);
    double cvt = o2scl_const::pi2/3.0*N0;
    cvt *= kb*kb/(MeV*197.0*197.0*197.0)*1.0e39; // Converts to cgs units
    return cvt; }
}

void nscool::get_degenerate_density(std::vector<double> rrho, 
          std::vector<double> pres, std::vector<double> rhod, int imax, int ienv) {
  double rho2[isize + 1]; double pres2[isize + 1]; int idata;
  //nscool_crust_eos(irank, rho2, pres2, idata);
  idata=((int)(nscool_crust.get_nlines()));
  for(size_t i=0;i<nscool_crust.get_nlines();i++) {
    rho2[i]=nscool_crust.get("rho",(idata)-1-i);
    pres2[i]=nscool_crust.get("P",(idata)-1-i);
  }

  for (int i=0; i<ienv; ++i) {
    rhod[i] = rrho[i]; }
  for (int i=imax; i>=ienv; --i) {
    int j = 0;
    while (true) {
      ++j;
      if ((pres[i]>=pres2[j]) && (pres[i]<=pres2[j+1])) {
        double x = (log(pres[i])-log(pres2[j]))/(log(pres2[j+1])-log(pres2[j]));
        double y = (log(pres2[j+1])-log(pres[i]))/(log(pres2[j+1])-log(pres2[j]));
        double lrhod = y*log(rho2[j])+x*log(rho2[j+1]);
        rhod[i] = std::max(exp(lrhod), rrho[i]);
        if ((i<imax) && (rrho[i+1] == rhod[i+1])) {
          rhod[i] = rrho[i];
        }
      } else {
        break; // Exit loop if condition not met
      }
    }
  }
}

void nscool::get_Tc(int imax, int icore, int idrip,
            std::vector<double> tcn, std::vector<double> tcp, std::vector<double> tcla, 
            std::vector<double> tcsm, std::vector<double> tcs0, std::vector<double> tcsp,
            std::vector<double> tcuu, std::vector<double> tcdd, std::vector<double> tcss, 
            std::vector<double> tcud, std::vector<double> tcus, std::vector<double> tcds,
            std::vector<double> tcu, std::vector<double> tcd, std::vector<double> tcs,
            int sfn1s0, int sfn3p2, int sfp1s0, int sfl1s0,
            double fn1s0, double fn3p2, double fp1s0, double fl1s0,
            double &kfmax_n3p2, double &delkf_n3p2, double &tcmax_n3p2, int &isf,
            std::vector<double> kfn, std::vector<double> kfp, std::vector<double> kfla, 
            std::vector<double> kfqu, std::vector<double> kfqd, std::vector<double> kfqs,
            std::vector<double> bar, std::vector<double> fhad, std::vector<double> yquarku, 
            std::vector<double> yquarkd, std::vector<double> yquarks) {
  // AWS: determine gap parameters
  //gaps(irank,sfn1s0,dinput_n1tc,dinput_n1kf,
   //       dinput_n1dk,sfn3p2,dinput_n3tc,dinput_n3kf,
   //       dinput_n3dk,sfp1s0,dinput_p1tc,dinput_p1kf,dinput_p1dk);
  // Just to be safe:
  for (int i=0; i<=imax; ++i) {
    tcn[i]=tcp[i]=tcla[i]=tcuu[i]=tcdd[i]=tcss[i]=
    tcud[i]=tcus[i]=tcds[i]=tcu[i]=tcd[i]=tcs[i]=1.0;
  }
  // 1s0 neutron superfluidity
  if (sfn1s0 == 1) {
    for (int i=0; i<=idrip; ++i) {
      tcn[i] = std::max(1.0, atc.cpp_tcn1_sfb(kfn[i]))*fn1s0; }
  } else if (sfn1s0 == 2) {
    for (int i=0; i<=idrip; ++i) {
      tcn[i] = std::max(1.0, atc.cpp_tcn1_ccdk(kfn[i]))*fn1s0; }
  } else if (sfn1s0 == 3) {
    for (int i=0; i<=idrip; ++i) {
      tcn[i] = std::max(1.0, atc.cpp_tcn1_wap(kfn[i]))*fn1s0; }
  } else if (sfn1s0 == 4) {
    for (int i=0; i<=idrip; ++i) {
      tcn[i] = std::max(1.0, atc.cpp_tcn1_gc(kfn[i]))*fn1s0; }
  } else if (sfn1s0 == 5) {
    for (int i=0; i<=idrip; ++i) {
      tcn[i] = std::max(1.0, atc.cpp_tcn1_gipsf(kfn[i]))*fn1s0; }
  } else if (sfn1s0 == 201) {
    for (int i=0; i<=idrip; ++i) {
      tcn[i] = std::max(1.0, atc.cpp_tc_Ioffe_1ns(kfn[i]))*fn1s0; }
  } else if (sfn1s0 == 202) {
    for (int i=0; i<=idrip; ++i) {
      tcn[i] = std::max(1.0, atc.cpp_tc_Ioffe_2ns(kfn[i]))*fn1s0; }
  } else if (sfn1s0 == 203) {
    for (int i=0; i<=idrip; ++i) {
      tcn[i] = std::max(1.0, atc.cpp_tc_Ioffe_3ns(kfn[i]))*fn1s0; }
  } else if (sfn1s0 == 150) {
    double tcmax_n1s0 = dinput_n1tc;
    double kfmax_n1s0 = dinput_n1kf;
    double delkf_n1s0 = dinput_n1dk;
    for (int i=0; i<=idrip; ++i) {
      double temp = tcmax_n1s0*exp(-pow((kfn[i]-kfmax_n1s0)/delkf_n1s0,2))*fn1s0;
      if (temp >= tcn[i]) {
        tcn[i] = temp;
        if (isf == i-1) {
          isf = i; } } } }
  // 3p2 neutron superfluidity
  isf = -1;

  auto fill_tcn = [idrip,kfn,fn3p2] (funct func,std::vector<double> tcn,int isf) {
    for (int i=0; i<=idrip; ++i) {
      double temp = func(kfn[i])*fn3p2;
      if (temp >= tcn[i]) {
        tcn[i] = temp;
        isf = i;
      } } };

  auto fill_tcn_ioffe = [idrip,kfn,fn3p2] (funct func,std::vector<double> tcn,int isf) {
    for (int i=0; i<=idrip; ++i) {
      double temp = std::max(1.0, func(kfn[i]))*fn3p2;
      if (temp >= tcn[i]) {
        tcn[i] = temp;
        isf = i;
      } } };

  if (sfn3p2 == 1) fill_tcn(atc.cpp_tcn3_hgrr,tcn,isf);
  else if (sfn3p2 == 2) fill_tcn(atc.cpp_tcn3_ao,tcn,isf);
  else if (sfn3p2 == 3) fill_tcn(atc.cpp_tcn3_ao_m1,tcn,isf);
  else if (sfn3p2 == 4) fill_tcn(atc.cpp_tcn3_t72,tcn,isf);
  else if (sfn3p2 == 5) fill_tcn(atc.cpp_tcn3_t72_m1,tcn,isf);
  else if (sfn3p2 == 6) fill_tcn(atc.cpp_tcn3_bcll92,tcn,isf);
  else if (sfn3p2 == 7) fill_tcn(atc.cpp_tcn3_eehjo96_nr,tcn,isf);
  else if (sfn3p2 == 8) fill_tcn(atc.cpp_tcn3_eehjo96_r,tcn,isf);
  else if (sfn3p2 >= 100 && sfn3p2 < 200) {
    if (sfn3p2 == 101) {
      kfmax_n3p2 = 1.8;
      delkf_n3p2 = 0.5;
      tcmax_n3p2 = 1.0e9;
    } else if (sfn3p2 == 102) {
      kfmax_n3p2 = 2.0;
      delkf_n3p2 = 0.5;
      tcmax_n3p2 = 3.0e9;
    } else if (sfn3p2 == 103) {
      kfmax_n3p2 = 2.5;
      delkf_n3p2 = 0.7;
      tcmax_n3p2 = 1.0e10;
    } else if (sfn3p2 == 150) {
      kfmax_n3p2 = dinput_n3tc;
      delkf_n3p2 = dinput_n3kf;
      tcmax_n3p2 = dinput_n3dk;
    }
    for (int i=0; i<=idrip; ++i) {
      double temp = tcmax_n3p2*exp(-pow((kfn[i]-kfmax_n3p2)/delkf_n3p2,2))*fn3p2;
      if (temp >= tcn[i]) {
        tcn[i] = temp;
        if (isf == i-1) {
          isf = i;
        }
      }
    }
  } else if (sfn3p2 == 201) fill_tcn_ioffe(cpp_tc_Ioffe_1nt,tcn,isf);
  else if (sfn3p2 == 202) fill_tcn_ioffe(cpp_tc_Ioffe_2nt,tcn,isf);
  else if (sfn3p2 == 203) fill_tcn_ioffe(cpp_tc_Ioffe_3nt,tcn,isf);
  // For uniform Tc gap
  else if (sfn3p2 >= 1000) {
    for (int i = 0; i <= icore; ++i) {
      tcn[i] = sfn3p2;
    }
    isf = icore;
  }

  if (sfp1s0 == 1) {
    for (int i = 0; i <= icore; ++i) {
        tcp[i] = std::max(1.0, atc.cpp_tcp1_ccy_ms(kfp[i]))*fp1s0; }
  } else if (sfp1s0 == 2) {
    for (int i = 0; i <= icore; ++i) {
        tcp[i] = std::max(1.0, atc.cpp_tcp1_ccy_ps(kfp[i]))*fp1s0; }
  } else if (sfp1s0 == 3) {
    for (int i = 0; i <= icore; ++i) {
        tcp[i] = std::max(1.0, atc.cpp_tcp1_t73(kfp[i]))*fp1s0; }
  } else if (sfp1s0 == 4) {
    for (int i = 0; i <= icore; ++i) {
        tcp[i] = std::max(1.0, atc.cpp_tcp1_ns(kfp[i]))*fp1s0; }
  } else if (sfp1s0 == 5) {
    for (int i = 0; i <= icore; ++i) {
        tcp[i] = std::max(1.0, atc.cpp_tcp1_ao(kfp[i]))*fp1s0; }
  } else if (sfp1s0 == 6) {
    for (int i = 0; i <= icore; ++i) {
        tcp[i] = std::max(1.0, atc.cpp_tcp1_bcll92(kfp[i]))*fp1s0; }
  } else if (sfp1s0 == 7) {
    for (int i = 0; i <= icore; ++i) {
        tcp[i] = std::max(1.0, atc.cpp_tcp1_ccdk(kfp[i]))*fp1s0; }
  } else if (sfp1s0 == 21) {
    for (int i = 0; i <= icore; ++i) {
        tcp[i] = std::max(1.0, atc.cpp_tcn1_t72(kfp[i]))*fp1s0; }
  } else if (sfp1s0 == 22) {
    for (int i = 0; i <= icore; ++i) {
        tcp[i] = std::max(1.0, atc.cpp_tcn1_awp_2(kfp[i]))*fp1s0; }
  } else if (sfp1s0 == 23) {
    for (int i = 0; i <= icore; ++i) {
        tcp[i] = std::max(1.0, atc.cpp_tcn1_awp_3(kfp[i]))*fp1s0; }
  } else if (sfp1s0 == 201) {
    for (int i = 0; i <= icore; ++i) {
        tcp[i] = std::max(1.0, atc.cpp_tc_Ioffe_1p(kfp[i]))*fp1s0; }
  } else if (sfp1s0 == 202) {
    for (int i = 0; i <= icore; ++i) {
        tcp[i] = std::max(1.0, atc.cpp_tc_Ioffe_2p(kfp[i]))*fp1s0; }
  } else if (sfp1s0 == 203) {
    for (int i = 0; i <= icore; ++i) {
        tcp[i] = std::max(1.0, atc.cpp_tc_Ioffe_3p(kfp[i]))*fp1s0; }
  } else if (sfp1s0 >= 1e3) {
    for (int i = 0; i <= icore; ++i) {
        tcp[i] = sfp1s0; }
  } else if (sfp1s0 == 150) {
    double tcmax_p1s0 = dinput_p1tc;
    double kfmax_p1s0 = dinput_p1kf;
    double delkf_p1s0 = dinput_p1dk;
    for (int i = 0; i <= icore; ++i) {
        tcp[i] = dinput_p1tc*exp(-(kfp[i]-dinput_p1kf)*(kfp[i]-dinput_p1kf)/
        (delkf_p1s0*delkf_p1s0))*fp1s0;
    }
  }

  // 1s0 Lambda superfluidity
  if (sfl1s0 == 1) {
    for (int i=0; i<=icore; ++i) {
        tcla[i] = std::max(1.0, cpp_tcla1_bb(kfla[i],bar[i]))*fl1s0;
    }
  }
  double gap;
  for (int i=0;i<=icore;++i) {
    // uu:
    double pf_uu=kfqu[i]*0.197;
    gap=exp(-pow(pf_uu-pf0_uu,2)/pow(dpf_uu,2));
    tcuu[i]=1.1604e13*gap_uu*gap;
    if (yquarku[i]==0.0) tcuu[i]=0.0;
    
    // dd:
    double pf_dd=kfqd[i]*0.197;
    gap=exp(-pow(pf_dd-pf0_dd,2)/pow(dpf_dd,2));
    tcdd[i] = 1.1604e13*gap_dd*gap;
    if (yquarkd[i]==0.0) tcdd[i]=0.0;
    
    // ss:
    double pf_ss=kfqs[i]*0.197;
    gap=exp(-pow(pf_ss-pf0_ss,2)/pow(dpf_ss,2));
    tcss[i]=1.1604e13*gap_ss*gap;
    if (yquarks[i]==0.0) tcss[i]=0.0;
    // ud:
    double pf_ud=(kfqu[i]+kfqd[i])/2.0*0.197;
    gap=exp(-pow(pf_ud-pf0_ud,2)/pow(dpf_ud,2));
    tcud[i]=1.1604e13*gap_ud*gap;
    if ((yquarku[i]==0.0) || (yquarkd[i]==0.0)) tcud[i]=0.0;
    // us:
    double pf_us=(kfqu[i]+kfqs[i])/2.0*0.197;
    gap=exp(-pow(pf_us-pf0_us,2)/pow(dpf_us,2));
    tcus[i]=1.1604e13*gap_us*gap;
    if ((yquarku[i]==0.0) || (yquarks[i]==0.0)) tcus[i]=0.0;
    // ds:
    double pf_ds=(kfqd[i]+kfqs[i])/2.0*0.197;
    gap=exp(-pow(pf_ds-pf0_ds,2)/pow(dpf_ds,2));
    tcds[i]=1.1604e13*gap_ds*gap;
    if ((yquarkd[i]==0.0) || (yquarks[i]==0.0)) tcds[i]=0.0;
    // Take only the maximum Tc:
    tcu[i] = std::max({tcuu[i], tcud[i], tcus[i]});
    tcd[i] = std::max({tcdd[i], tcud[i], tcds[i]});
    tcs[i] = std::max({tcss[i], tcus[i], tcds[i]});
    
    if (fhad[i] == 1.0) {
      tcu[i]=tcd[i]=tcs[i] = 1.0;
    }
  }
// Just in case the above formulas give Tc<0 instead of Tc=0
// at the edges where Tc is almost 0
  for (int i = 0; i <= idrip; ++i) {
    tcn[i] = std::abs(tcn[i]);
    tcp[i] = std::abs(tcp[i]);
    tcla[i] = std::abs(tcla[i]);
    tcuu[i] = std::abs(tcuu[i]);
    tcdd[i] = std::abs(tcdd[i]);
    tcss[i] = std::abs(tcss[i]);
    tcud[i] = std::abs(tcud[i]);
    tcus[i] = std::abs(tcus[i]);
    tcds[i] = std::abs(tcds[i]);
    tcu[i] = std::abs(tcu[i]);
    tcd[i] = std::abs(tcd[i]);
    tcs[i] = std::abs(tcs[i]);
  }
}

void nscool::get_spec_heat_degenerate(
    std::vector<double>& cve, std::vector<double>& cvm, std::vector<double>& cvn, std::vector<double>& cvp,
    std::vector<double>& cvla, std::vector<double>& cvsm, std::vector<double>& cvs0, std::vector<double>& cvsp,
    std::vector<double>& cvqu, std::vector<double>& cvqd, std::vector<double>& cvqs,
    const std::vector<double>& kfe, const std::vector<double>& kfm, const std::vector<double>& kfn,
    const std::vector<double>& kfp, const std::vector<double>& kfla, const std::vector<double>& kfsm,
    const std::vector<double>& kfs0, const std::vector<double>& kfsp, const std::vector<double>& kfqu,
    const std::vector<double>& kfqd, const std::vector<double>& kfqs,
    const std::vector<double>& mstp, const std::vector<double>& mstn, const std::vector<double>& mstla,
    const std::vector<double>& mstsm, const std::vector<double>& msts0, const std::vector<double>& mstsp,
    const std::vector<double>& fhad, int imax) {
    
  for (int j = 0; j <= imax; ++j) {
    double pfe = kfe[j]*197.0;
    double me = std::sqrt(0.511*0.511+pfe*pfe);
    double pfm = kfm[j]*197.0;
    double mm = std::sqrt(105.0*105.0+pfm*pfm);
    double pfn = kfn[j]*197.0;
    double mn = 939.56*mstn[j];   
    double pfp = kfp[j]*197.0;
    double mp = 938.27*mstp[j];
    double pfla = kfla[j]*197.0;
    double mla = 1116.0*mstla[j];    
    double pfsm = kfsm[j]*197.0;
    double msm = 1193.0*mstsm[j];    
    double pfs0 = kfs0[j]*197.0;
    double ms0 = 1193.0*msts0[j];  
    double pfsp = kfsp[j]*197.0;
    double msp = 1193.0 * mstsp[j]; 
    double pfqu = kfqu[j]*197.0;
    double mqu = std::sqrt(5.0*5.0+pfqu*pfqu);
    double pfqd = kfqd[j] * 197.0;
    double mqd = std::sqrt(8.0*8.0+pfqd*pfqd);
    double pfqs = kfqs[j]*197.0;
    double mqs = std::sqrt(strange_mass*strange_mass+pfqs*pfqs);
        
    cve[j] = cvt_deg(pfe, me);
    cvm[j] = cvt_deg(pfm, mm);
    cvn[j] = cvt_deg(pfn, mn)*fhad[j];
    cvp[j] = cvt_deg(pfp, mp)*fhad[j];
    cvla[j] = cvt_deg(pfla, mla)*fhad[j];
    cvsm[j] = cvt_deg(pfsm, msm)*fhad[j];
    cvs0[j] = cvt_deg(pfs0, ms0)*fhad[j];
    cvsp[j] = cvt_deg(pfsp, msp)*fhad[j];
    cvqu[j] = cvt_deg(pfqu, mqu)*(1.0-fhad[j]);
    cvqd[j] = cvt_deg(pfqd, mqd)*(1.0-fhad[j]);
    cvqs[j] = cvt_deg(pfqs, mqs)*(1.0-fhad[j]);
  }
}

void nscool::get_effective_masses(const std::string version,
                          double emnco, double emncr, double emp,
                          const std::vector<double>& kfn, const std::vector<double>& kfp,
                          std::vector<double>& mstn, std::vector<double>& mstp, std::vector<double>& mstla,
                          std::vector<double>& mstsm, std::vector<double>& msts0, std::vector<double>& mstsp,
                          int idrip, int icore) {
  for (int i = icore + 1; i <= idrip; ++i) {
    mstn[i] = std::min(1.0, 1.09 - 0.11 * kfn[i]);
  }
}

void nscool::get_fermi_momenta(int imax, int icore,
                       std::vector<double>& rrho, std::vector<double>& bar,
                       std::vector<double>& yneutr, std::vector<double>& yprot,
                       std::vector<double>& yelect, std::vector<double>& ymuon,
                       std::vector<double>& ylambda,
                       std::vector<double>& ysminus, std::vector<double>& yszero,
                       std::vector<double>& ysplus,
                       std::vector<double>& yquarku, std::vector<double>& yquarkd,
                       std::vector<double>& yquarks, std::vector<double>& fhad,
                       std::vector<double>& theta_k, std::vector<double>& theta_p,
                       std::vector<double>& kfn, std::vector<double>& kfp,
                       std::vector<double>& kfe, std::vector<double>& kfm,
                       std::vector<double>& kfla, std::vector<double>& kfsm,
                       std::vector<double>& kfs0, std::vector<double>& kfsp,
                       std::vector<double>& kfqu, std::vector<double>& kfqd,
                       std::vector<double>& kfqs,
                       std::vector<int>& idurca_np, std::vector<int>& idurca_lap,
                       std::vector<int>& idurca_smn, std::vector<int>& idurca_smla,
                       std::vector<int>& idurca_sms0,
                       std::vector<int>& idurca_quqd, std::vector<int>& idurca_quqs,
                       std::vector<double>& durca_ctrl_e, std::vector<double>& durca_ctrl_m,
                       std::vector<double>& durca_henon_e, std::vector<double>& durca_henon_m) {
  const double coeff = 3.0*o2scl_const::pi2*o2scl_const::pi;
  // Calculate the fermi momenta in the core:
  for (int j=0; j<=icore; ++j) {
    double nn, np, nla, nsm, ns0, nsp, nqu, nqd, nqs;
    if (fhad[j] != 0.0) {
      nn = std::max(0.0,yneutr[j]*bar[j])/fhad[j];
      np = std::max(0.0,yprot[j]*bar[j])/fhad[j];
      nla = std::max(0.0,ylambda[j]*bar[j])/fhad[j];
      nsm = std::max(0.0,ysminus[j]*bar[j])/fhad[j];
      ns0 = std::max(0.0,yszero[j]*bar[j])/fhad[j];
      nsp = std::max(0.0,ysplus[j]*bar[j])/fhad[j];
    } else {
      nn = np = nla = nsm = ns0 = nsp = 0.0;
    }

    if (fhad[j]!=1.0) {
      nqu = std::max(0.0, yquarku[j]*bar[j])/(1.0-fhad[j]);
      nqd = std::max(0.0, yquarkd[j]*bar[j])/(1.0-fhad[j]);
      nqs = std::max(0.0, yquarks[j]*bar[j])/(1.0-fhad[j]);
    } else {
      nqu = nqd = nqs = 0.0;
    }

    kfn[j] = pow(3.0*o2scl_const::pi2*nn, 1.0/3.0);
    kfp[j] = pow(3.0*o2scl_const::pi2*np, 1.0/3.0);
    kfla[j] = pow(3.0*o2scl_const::pi2*nla, 1.0/3.0);
    kfsm[j] = pow(3.0*o2scl_const::pi2*nsm, 1.0/3.0);
    kfs0[j] = pow(3.0*o2scl_const::pi2*ns0, 1.0/3.0);
    kfsp[j] = pow(3.0*o2scl_const::pi2*nsp, 1.0/3.0);
    kfqu[j] = pow(3.0*o2scl_const::pi2*nqu, 1.0/3.0);
    kfqd[j] = pow(3.0*o2scl_const::pi2*nqd, 1.0/3.0);
    kfqs[j] = pow(3.0*o2scl_const::pi2*nqs, 1.0/3.0);

    double ne = std::abs(yelect[j]*bar[j]);
    double nm = std::abs(ymuon[j]*bar[j]);
    kfe[j] = pow(3.0*o2scl_const::pi2*ne, 1.0/3.0);
    if (yelect[j]<=0.0) kfe[j] = -kfe[j];
    kfm[j] = pow(3.0*o2scl_const::pi2*nm, 1.0/3.0);
    if (ymuon[j] <= 0.0) kfm[j] = -kfm[j];

    double charge_l = -ne-nm;
    double charge_h = np+nsp-nsm;
    double charge_q = (2.0/3.0*nqu-1.0/3.0*nqd-1.0/3.0*nqs);
    double charge = charge_l+charge_h+charge_q;
    double baryon_h = nn+np+nla+nsm+ns0+nsp;
    double baryon_q = 1.0/3.0*(nqu+nqd+nqs);
    double baryon = baryon_h+baryon_q;

    double barrel = baryon/bar[j];
    if (std::abs(charge)>=1.0e-2) {
      std::cout << "Charge neutrality violated at:" << std::endl;
      std::cout << j << " Rho= " << rrho[j] << ": charge/fm3= " << charge << std::endl;
      icore = 0;
      return;
    }
    if (std::abs(barrel)-1.0>=1.0e-2) {
      std::cout << "Baryons do not sum up to baryon density at:" << std::endl;
      std::cout << j << " Rho= " << rrho[j] << ": sum(baryons)/baryon#= " << baryon << std::endl;
      icore = 0;
      return;
    }
  }

  // Calculate the fermi momenta in the crust:
  for (int j = icore+1; j <= imax; ++j) {
    double ne = yelect[j]*bar[j];
    double nn = yneutr[j]*bar[j];
        
    kfe[j] = pow(3.0*o2scl_const::pi2*ne, 1.0/3.0);
    kfm[j] = 0.0;
    kfn[j] = pow(3.0*o2scl_const::pi2*nn, 1.0/3.0);
    kfp[j] = 0.0;
    kfla[j] = kfsm[j] = kfs0[j] = kfsp[j] = kfqu[j] = kfqd[j] = kfqs[j] = 0.0;
  }

  if (false) {
    // Dany's method for neutron-proton direct Urca
    for (int j = 0; j <= icore; ++j) {
      if ((kfp[j] < kfn[j]+kfe[j]) &&
          (kfn[j] < kfp[j]+kfe[j]) &&
          (kfe[j] < kfp[j]+kfn[j])) {
        idurca_np[j] = 1;
        if ((kfp[j] < kfn[j]+kfm[j]) &&
            (kfn[j] < kfp[j]+kfm[j]) &&
            (kfm[j] < kfp[j]+kfn[j])) {
            idurca_np[j] = 2;
        }
      } else {
        idurca_np[j] = 0;
      }
    }
  } else {
    // New method for neutron-proton direct Urca
    double fix_durca = 0.0;
    double alpha_durca_frac = 1.0e-8;
    double beta_durca_frac = 1.0;

    // AWS: Get direct Urca settings
    // AWS: alpha is the broadening parameter and beta is the
    // AWS: fractional decrease of the direct Urca threshold
    //nscool_urca_settings(irank, fix_durca, alpha_durca_frac, beta_durca_frac);
    alpha_durca_frac = alpha_durca;

    // AWS: Compute triangle squared areas
    for (int j = 0; j <= icore; ++j) {
      double s = (kfp[j]+kfn[j]+kfe[j])/2.0;
      durca_henon_e[j] = s*(s-kfp[j])*(s-kfe[j])*(s-kfn[j]);
      s = (kfp[j]+kfn[j]+kfm[j])/2.0;
      durca_henon_m[j] = s*(s-kfp[j])*(s-kfm[j])*(s-kfn[j]);
    }

    // AWS: Use linear interpolation to compute direct Urca
    // AWS: density thresholds
    double nb_durca_e, nb_durca_m;
    if (fix_durca == 0.0) {
      nb_durca_e = 0.0;
      nb_durca_m = 0.0;
      for (int j = 0; j < icore; ++j) {
        if (nb_durca_e == 0.0 && durca_henon_e[j]*durca_henon_e[j+1]<0.0) {
          double nb_low = bar[j];
          double nb_high = bar[j+1];
          nb_durca_e = nb_low-(nb_high-nb_low)*durca_henon_e[j]/
            (durca_henon_e[j+1]-durca_henon_e[j]);
        }
        if (nb_durca_m == 0.0 && durca_henon_m[j]*durca_henon_m[j+1]<0.0) {
          double nb_low = bar[j];
          double nb_high = bar[j+1];
          nb_durca_m = nb_low-(nb_high-nb_low)*durca_henon_m[j]/
            (durca_henon_m[j+1]-durca_henon_m[j]);
        }
      }
    } else {
      nb_durca_e = fix_durca;
      nb_durca_m = fix_durca;
    }

    // AWS: Now compute direct Urca control functions from threshold
    // AWS: densities
    // SH: Add two possible modifications
    // early onset (step function); broadening
    for (int j = 0; j <= icore; ++j) {
      idurca_np[j] = 0;
      durca_ctrl_e[j] = 0.0;
      durca_ctrl_m[j] = 0.0;

      if (nb_durca_e > 0.0) {
        if (bar[j] >= (1.0+alpha_durca_frac)*beta_durca_frac*nb_durca_e) {
          idurca_np[j] = 1;
          durca_ctrl_e[j] = 1.0;
        } else if (bar[j] >= (1.0-alpha_durca_frac)*beta_durca_frac*nb_durca_e) {
          idurca_np[j] = 3;
          durca_ctrl_e[j] = 0.5+1.0/(2.0*alpha_durca_frac)*
              (bar[j]-nb_durca_e)/nb_durca_e;
          if (nb_durca_m > 0.0) {
            if (bar[j] >= (1.0+alpha_durca_frac)*beta_durca_frac*nb_durca_m) {
              idurca_np[j] = 2;
              durca_ctrl_m[j] = 1.0;
            } else if (bar[j] >= (1.0-alpha_durca_frac)*beta_durca_frac*nb_durca_m) {
              idurca_np[j] = 4;
              durca_ctrl_m[j] = 0.5+1.0/(2.0*alpha_durca_frac)*
                  (bar[j]-nb_durca_m)/nb_durca_m;
            }
          }
        }
      }
    }
  }
  // Dany's method for hyperon and quark direct Urca
  auto solve_hqdur = [kfe,kfm] (std::vector<double> vec1, 
                      std::vector<double> vec2,int j){
    if ((vec1[j] < vec2[j] + kfe[j]) &&
        (vec2[j] < vec1[j] + kfe[j]) &&
        (kfe[j] < vec1[j] + vec2[j])) {
      return 1;
      if ((vec1[j] < vec2[j] + kfm[j]) &&
          (vec2[j] < vec1[j] + kfm[j]) &&
          (kfm[j] < vec1[j] + vec2[j])) 
        return 2;
    } else 
      return 0; };
      
  for (int j = 0; j <= icore; ++j) {
    idurca_lap[j] = solve_hqdur(kfp,kfla,j);
    idurca_smn[j] = solve_hqdur(kfsm,kfn,j);
    idurca_smla[j] = solve_hqdur(kfsm,kfla,j);
    idurca_sms0[j] = solve_hqdur(kfsm,kfs0,j);
    idurca_quqd[j] = solve_hqdur(kfqu,kfqd,j);
    idurca_quqs[j] = solve_hqdur(kfqu,kfqs,j);
  }
}

void nscool::get_crust_chemistry(int debug, const std::string version, int imax, int icore,
                         std::vector<double> rrho, std::vector<double> pres, 
                         std::vector<double> debar, std::vector<double> dvol, 
                         std::vector<double> bar, std::vector<double> a_cell, 
                         std::vector<double> a_ion, std::vector<double> z_ion, 
                         std::vector<double> v_ion, std::vector<double> yelect, 
                         std::vector<double> yneutr) {

  std::vector<double> rho_t(500), pres_t(500), bar_t(500),
      A_cell_t(500), A_ion_t(500), Z_ion_t(500);
  if (debug>1.0) {
    std::cout << 'Entering subroutine get_crust_chemistry' << std::endl;
  }
  int jmax=((int)(nscool_crust.get_nlines()));
  if (nscool_crust.get_nlines()>=500) {
    O2SCL_ERR("Crust table too large in nscool_wrap::crust_comp().",
	      o2scl::exc_einval);
  }
  for(size_t i=0;i<nscool_crust.get_nlines();i++) {
    size_t i2=(jmax)-1-i;
    rho_t[i]=nscool_crust.get("rho",i2);
    pres_t[i]=nscool_crust.get("P",i2);
    bar_t[i]=nscool_crust.get("n",i2);
    A_cell_t[i]=nscool_crust.get("A_cell",i2);
    A_ion_t[i]=nscool_crust.get("A_ion",i2);
    Z_ion_t[i]=nscool_crust.get("Z",i2);
  }
  if (rho_t[1]<rho_t[0]) {
    O2SCL_ERR("Crust composition table should be decreasing",
	      o2scl::exc_einval);
  }
  int jdrip=0; int jget_drip=0;
  for (int j=jmax;j>=1;--j) {
    if (jget_drip == 0) {
      if (A_cell_t[j] != A_ion_t[j]) {
        jdrip=j;
        jget_drip=1;
      }
    }
  }

  // Make sure that rho_t and bar_t at jmax are smaller than in core:
  int jjmax = jmax;
  for (int j=jmax; j>=1; --j) {
    if (rho_t[j] >= rrho[icore]) {
      jjmax = j-1;
    }
  }
  jmax = jjmax;

  // Adjust properties based on conditions
  jmax = jmax+1;
  rho_t[jmax] = rrho[icore];
  bar_t[jmax] = bar[icore];
  pres_t[jmax] = pres[icore];

  A_cell_t[jmax] = Z_ion_t[jmax]/yelect[icore];
  A_ion_t[jmax] = A_ion_t[jmax-1];

  int j = jmax;
  std::vector<double> a_ion, a_cell, z_ion;
  for (int i=icore+1; i<=imax; ++i) {
    while (rrho[i] <= rho_t[j]) {
      --j;
    }
    double dd = (rho_t[j+1]-rho_t[j]);
    double w2 = (rrho[i]-rho_t[j])/dd;
    double w1 = 1.0-w2;
    if (A_ion_t[j+1] == A_cell_t[j+1]) {
      a_ion[i] = A_ion_t[j+1];
      a_cell[i] = A_cell_t[j+1];
      z_ion[i] = Z_ion_t[j+1];
    } else {
      // New version: A & Z constant, only A_cell changes
      a_cell[i] = w1*A_cell_t[j]+w2*A_cell_t[j+1];
      a_ion[i] = A_ion_t[j+1];
      z_ion[i] = Z_ion_t[j+1];
    }
    // Get the baryon number density
    bar[i] = w1*bar_t[j]+w2*bar_t[j+1];
    // Calculate the fraction of volume occupied by ions
    double r1 = 1.1; // Scale parameter, in fm
    double vion = (4.0/3.0)*3.14159*pow(r1,3)*a_ion[i];
    vion *= 1.0e-39; // in cm^3
    double nion = rrho[i]/(1.66e-24*a_ion[i]); // ion density per cm^3
    v_ion[i] = nion*vion;
    v_ion[i] = std::min(1.0, v_ion[i]); // Make sure it's < 1
    ++j;
  }
  // Calculate the Y's of the e & n:
  for (int i = icore + 1; i <= imax; ++i) {
    yelect[i] = z_ion[i]/a_cell[i];
    yneutr[i] = (a_cell[i]-a_ion[i])/a_cell[i];
  }
  // Clean up the core, just in case:
  for (int i = 0; i <= icore; ++i) {
    z_ion[i] = 0.0;
    a_ion[i] = 0.0;
    a_cell[i] = 0.0;
    v_ion[i] = 0.0;
  }

  std::vector<double> debar(imax + 1);
  debar[0] = 0.0;
  for (int i=1; i<=imax; ++i) {
    debar[i] = (bar[i-1]+bar[i])/2.0*dvol[i];
  }

  // Exiting subroutine get_crust_chemistry
  if (debug >= 1) {
    std::cout << "Exiting subroutine get_crust_chemistry" << std::endl;
  }
  return;
}

void nscool::get_core_chemistry(const std::string version, int imax, int icore, 
                        const std::vector<double>& rrho, std::vector<double>& bar, 
                        std::vector<double>& yneutr, std::vector<double>& yprot, 
                        std::vector<double>& yelect, std::vector<double>& ymuon, 
                        std::vector<double>& ylambda, std::vector<double>& ysminus, 
                        std::vector<double>& yszero, std::vector<double>& ysplus, 
                        std::vector<double>& yquarku, std::vector<double>& yquarkd, 
                        std::vector<double>& yquarks, std::vector<double>& fhad, 
                        std::vector<double>& theta_k, std::vector<double>& theta_p,
                        std::vector<double>& mstp, std::vector<double>& mstn,
                        std::vector<double>& mstla, std::vector<double>& mstsm,
                        std::vector<double>& msts0, std::vector<double>& mstsp) {
  std::vector<double> rho_t(500), nbar_t(500);
  std::vector<double> yneutr_t(500), yprot_t(500), yelect_t(500),
        ymuon_t(500), ylambda_t(500), ysminus_t(500), yszero_t(500), ysplus_t(500),
        yquarku_t(500), yquarkd_t(500), yquarks_t(500), fhad_t(500);
  std::vector<double> mstp_t(isize), mstn_t(isize), mstla_t(isize),
        mstsm_t(isize), msts0_t(isize), mstsp_t(isize);
  std::vector<double> theta_k_t(500), theta_p_t(500);

  int it = 99999;
  int i0 = 99999;
  int ix=nscool_core.get_nlines();
  for(size_t i=0;i<nscool_core.get_nlines();i++) {
    // rho is energy density
    rho_t[i]=nscool_core.get("Rho",i);
    nbar_t[i]=nscool_core.get("nbar",i);
    yelect_t[i]=nscool_core.get("Ye",i);
    ymuon_t[i]=nscool_core.get("Ymu",i);
    yneutr_t[i]=nscool_core.get("Yn",i);
    yprot_t[i]=nscool_core.get("Yp",i);
    ylambda_t[i]=nscool_core.get("Yla",i);
    ysminus_t[i]=nscool_core.get("Ysm",i);
    yszero_t[i]=nscool_core.get("Ys0",i);
    ysplus_t[i]=nscool_core.get("Ysp",i);
      
    double diff=yprot_t[i]+ysplus_t[i]-ysminus_t[i]-
      yelect_t[i]-ymuon_t[i];
    if (fabs(diff)>1.0e-5) {
      std::cerr << "Charge problem in nscool_wrap::core_comp()."
		<< std::endl;
      std::cout << i << " " << yprot_t[i] << " " << yelect_t[i] << " "
		<< ymuon_t[i] << " " << ysplus_t[i] << " "
		<< ysminus_t[i] << " " [i] << std::endl;
      (ix)=0;
      return;
    }
    if (fabs(yprot_t[i]+yneutr_t[i]+ylambda_t[i]+ysminus_t[i]+
	     yszero_t[i]+ysplus_t[i]-1.0)>1.0e-5) {
      std::cerr << "Baryon problem in core_comp." << std::endl;
      std::cout << i << " " << yneutr_t[i] << " "
		<< yprot_t[i] << " " << ylambda_t[i] << " "
		<< ysminus_t[i] << " " << yszero_t[i] << " "
		<< ysplus_t[i] << std::endl;
      (ix)=0;
      return;
    }
      
    // These are the reduced effective masses, i.e. m^{*}/m
    mstp_t[i]=nscool_core.get("mstp",i);
    mstn_t[i]=nscool_core.get("mstn",i);
    mstla_t[i]=nscool_core.get("mstla",i);
    mstsm_t[i]=nscool_core.get("mstsm",i);
    msts0_t[i]=nscool_core.get("msts0",i);
    mstsp_t[i]=nscool_core.get("mstsp",i);
  }
    
  if (rho_t[1]>rho_t[0]) {
    O2SCL_ERR("Core composition table should be decreasing",
	      o2scl::exc_einval);
  }
    
  if (ix == 0) {
    icore = 0;
    return;
  }

  for (int i = 1; i <= ix; ++i) {
    theta_k_t[i]=theta_p_t[i]=yquarku_t[i]=yquarkd_t[i]=yquarks_t[i]=ylambda_t[i]=0.0;
    ysminus_t[i]=yszero_t[i]=ysplus_t[i]=mstla_t[i]=mstsm_t[i]=msts0_t[i]=mstsp_t[i]=0.0;
  }

  double xnut = 9.2819e32;

  for (int i0 = 0; i0 <= icore; ++i0) {
    int i1, i2;
    if (rrho[i0]>=rho_t[1]) {
      i1 = 1;
      i2 = 2;
    } else if (rrho[i0]<=rho_t[ix]) {
      i1 = ix-1;
      i2 = ix;
    } else {
      int i = i1-1;
      while (true) {
        ++i;
        if ((rrho[i0]>=rho_t[i+1]) && (rrho[i0]<=rho_t[i])) {
          i1 = i;
          i2 = i+1;
          break;
        }
      }
    }

    double x1 = (log(rho_t[i2])-log(rrho[i0]))/(log(rho_t[i2])-log(rho_t[i1]));
    double x2 = (log(rrho[i0])-log(rho_t[i1]))/(log(rho_t[i2])-log(rho_t[i1]));

    bar[i0] = x1*nbar_t[i1]+x2*nbar_t[i2];
    yelect[i0] = x1*yelect_t[i1]+x2*yelect_t[i2];
    ymuon[i0] = x1*ymuon_t[i1]+x2*ymuon_t[i2];
    yneutr[i0] = x1*yneutr_t[i1]+x2*yneutr_t[i2];
    yprot[i0] = x1*yprot_t[i1]+x2*yprot_t[i2];
    ylambda[i0] = x1*ylambda_t[i1]+x2*ylambda_t[i2];
    ysminus[i0] = x1*ysminus_t[i1]+x2*ysminus_t[i2];
    yszero[i0] = x1*yszero_t[i1]+x2*yszero_t[i2];
    ysplus[i0] = x1*ysplus_t[i1]+x2*ysplus_t[i2];
    yquarku[i0] = x1*yquarku_t[i1]+x2*yquarku_t[i2];
    yquarkd[i0] = x1*yquarkd_t[i1]+x2*yquarkd_t[i2];
    yquarks[i0] = x1*yquarks_t[i1]+x2*yquarks_t[i2];
    theta_k[i0] = x1*theta_k_t[i1]+x2*theta_k_t[i2];
    theta_p[i0] = x1*theta_p_t[i1]+x2*theta_p_t[i2];

    if ((version == "old") || (version == "new") || (version == "NEW")) {
      fhad[i0] = 1.0;
    } else if (version == "QRK") {
      fhad[i0] = x1*fhad_t[i1]+x2*fhad_t[i2];
      if (fhad[i0] > 1.0) fhad[i0] = 1.0;
      if (fhad[i0] < 0.0) fhad[i0] = 0.0;
    }

    // Check for consistency
    double bnuc = yneutr[i0]+yprot[i0];
    double bhyp = ylambda[i0]+ysminus[i0]+yszero[i0]+ysplus[i0];
    double bqua = (1.0/3.0)*(yquarku[i0]+yquarkd[i0]+yquarks[i0]);
    double btot = bnuc+bhyp+bqua;
    double qlep = -yelect[i0]-ymuon[i0];
    double qnuc = yprot[i0];
    double qhyp = ysplus[i0]-ysminus[i0];
    double qqua = (1.0/3.0)*(2.0*yquarku[i0]-yquarkd[i0]-yquarks[i0]);
    double qtot = qlep+qnuc+qhyp+qqua;

    if (std::abs(btot-1.0) > 1.0e-2) {
      std::cout << "i, rho, Btot, Qtot = " << i0 << " " << rrho[i0] 
                << " " << btot << " " << qtot << std::endl;
      std::cout << "Btot not equal to 1 !" << i0 << " " << yneutr[i0] 
                << " " << yprot[i0] << " " << ylambda[i0] << " " 
                << ysminus[i0] << " " << yszero[i0] << " " << ysplus[i0] 
                << " " << bnuc << " " << bhyp << " " << bqua << " " 
                << btot << std::endl;
      icore = 0;
      return;
    }

    if (std::abs(qtot) > 1.0e-2) {
      std::cout << "i, rho, Btot, Qtot = " << i0 << " " << rrho[i0] 
                << " " << btot << " " << qtot << std::endl;
      std::cout << "Qtot not equal to 0 !" << i0 << " " << qlep << " " 
                << qnuc << " " << qhyp << " " << qqua << " " << qtot 
                << std::endl;
      std::cout << yneutr[i0] << " " << yprot[i0] << " " << yelect[i0] 
                << " " << ymuon[i0] << std::endl;
      icore = 0;
      return;
    }

    // Get the baryon effective masses if version is NEW
    if (version == "NEW") {
      mstp[i0] = x1*mstp_t[i1]+x2*mstp_t[i2];
      mstn[i0] = x1*mstn_t[i1]+x2*mstn_t[i2];
      mstla[i0] = x1*mstla_t[i1]+x2*mstla_t[i2];
      mstsm[i0] = x1*mstsm_t[i1]+x2*mstsm_t[i2];
      msts0[i0] = x1*msts0_t[i1]+x2*msts0_t[i2];
      mstsp[i0] = x1*mstsp_t[i1]+x2*mstsp_t[i2];
    }
  }
  // Clean up Y's in the crust, just in case:
  // (yelect & yneutr will be calculated in "get_crust_chemistry")
  for (int i0=icore+1;i0<=imax;++i0) {
    yelect[i0]=ymuon[i0]=yneutr[i0]=yprot[i0]=ylambda[i0]=ysminus[i0]=yszero[i0]=ysplus[i0] = 0.0;
    yquarku[i0]=yquarkd[i0]=yquarks[i0]=theta_k[i0]=theta_p[i0]=fhad[i0] = 1.0; // Also set to 1.0 just in case
  }
  return;
}

void grid(int idec, double rhocore, double rhodrip, double rhoenv, double rhosurf,
          int &imax, int &icore, int &idrip, int &ienv, std::vector<double> &rad, 
          std::vector<double> &rrho, std::vector<double> &pres, std::vector<double> &dvol, 
          std::vector<double> &emas, std::vector<double> &phi) {
  std::vector<double> rad_t(10000), bar_t(10000), rho_t(10000), pres_t(10000), emas_t(10000), phi_t(10000);
  // DP: Define zone indices: icore, idrip & isurf
  icore=2*((icore-1)/2)+1; // Makes sure icore is odd
  int idel1=static_cast<int>(std::log10(rhocore/rhodrip)*static_cast<double>(idec));
  idel1=2*(idel1/2);// Makes sure idel1 is even
  idrip=icore+idel1;
  int idel2=static_cast<int>(std::log10(rhodrip/rhosurf)*static_cast<double>(idec));
  idel2=2*(idel2/2); // Makes sure idel2 is even
  int isurf=idrip+idel2;
  int jmax=nscool_tov.get_nlines();
  if (jmax>9999) {
    O2SCL_ERR("Table too large.",o2scl::exc_einval);
  }
  int jcore=0;
  for(size_t j=0;j<nscool_tov.get_nlines();j++) {
    rad_t[j]=nscool_tov.get("radius",j)*100.0;
    bar_t[j]=nscool_tov.get("n_baryon",j);
    // This is energy density
    rho_t[j]=nscool_tov.get("density",j);
    pres_t[j]=nscool_tov.get("pressure",j);
    emas_t[j]=nscool_tov.get("emass",j);
    phi_t[j]=nscool_tov.get("phi",j);
    if (rho_t[j]<rhocore && (jcore)==0) {
      jcore=j;
    }
  }
  if (rad_t[1]<rad_t[0]) {
    O2SCL_ERR("Structure table should be increasing in radius",
	      o2scl::exc_einval);
  }
  if (jcore==0) {
    O2SCL_ERR2("Variable 'jcore' not set in ",
	       "nscool_wrap::star_struct().",o2scl::exc_einval);
  }
  // Calculate core radius
  double drho=rho_t[jcore]-rho_t[jcore+1];
  double w1=(rhocore-rho_t[jcore+1])/drho;
  double w2=1.0-w1;
  double rad_core=w1*rad_t[jcore]+w2*rad_t[jcore+1];

  // Define Star grid: CORE
  for (int i=0;i<=icore;++i) {
    rad[i]=std::pow(static_cast<double>(i)/static_cast<double>(icore),1.0/3.0)*
    rad_core;
  }

  // Initialize arrays
  double rrho[icore+1],emas[icore+1],phi[icore+1],pres[icore+1],dvol[icore+1];

  rrho[0] = rho_t[0];
  emas[0] = 0.0;
  phi[0] = phi_t[0];
  pres[0] = pres_t[0];
  dvol[0] = 0.0;

  int j = 0;
  for (int i=1;i<=icore;++i) {
    while (rad_t[j]<rad[i]) {
      ++j;
    }
    double delrad=rad_t[j]-rad_t[j-1];
    w1=(rad_t[j]-rad[i])/delrad;
    w2=1.0-w1;
    rrho[i]=w1*rho_t[j-1]+w2*rho_t[j];
    emas[i]=w1*emas_t[j-1]+w2*emas_t[j];
    phi[i]=w1*phi_t[j-1]+w2*phi_t[j];
    pres[i]=w1*pres_t[j-1]+w2*pres_t[j];
    if (i==1) {
      dvol[i]=4.0/3.0*M_PI*std::pow(rad[i],3);
    } else {
      dvol[i]=4.0*M_PI*((rad[i-1]+rad[i])/2.0)*((rad[i]-rad[i-1])/
                std::sqrt(1.0-2.92e5*(emas[i-1]+emas[i])/
                (rad[i-1]+rad[i])));
    }
  }

  rrho[icore]=rhocore;
  double dlogrho=std::log10(rhocore/rhodrip);
  double dlrho=dlogrho/static_cast<double>(idrip-icore);
  for (int i=icore+1;i<=idrip;++i) {
    double lrho=std::log10(rhocore)-static_cast<double>(i-icore)*dlrho;
    rrho[i]=std::pow(10.0,lrho);
  }
  dlogrho=std::log10(rhodrip/rhosurf);
  dlrho=dlogrho/static_cast<double>(isurf-idrip);
  for (int i=idrip+1;i<=isurf;++i) {
    double lrho=std::log10(rhodrip)-static_cast<double>(i-idrip)*dlrho;
    rrho[i]=std::pow(10.0,lrho);
  }

  int j=0;
  for (int i=icore+1;i<=isurf;++i) {
    while (rho_t[j]>rrho[i]) {
      ++j;
    }
    double dellrho=std::log10(rho_t[j-1])-std::log10(rho_t[j]);
    w1=(std::log10(rrho[i])-std::log10(rho_t[j]))/dellrho;
    w2=1.0-w1;
    rad[i]=w1*rad_t[j-1]+w2*rad_t[j];
    emas[i]=w1*emas_t[j-1]+w2*emas_t[j];
    phi[i]=w1*phi_t[j-1]+w2*phi_t[j];
    pres[i]=w1*pres_t[j-1]+w2*pres_t[j];
    dvol[i]=4.0*M_PI*((rad[i-1]+rad[i])/2.0)*((rad[i]-rad[i-1])/
            std::sqrt(1.0-2.92e5*(emas[i-1]+emas[i])/
            (rad[i-1]+rad[i])));
    --j;
  }

  dvol[isurf+1]=dvol[isurf];
  rrho[idrip]=rhodrip;
  rrho[isurf]=rhosurf;
  // Find the envelope boundary:
  int ienv=isurf+2;
  for (int i=isurf;i>=idrip;i-=2) {
    if (rrho[i]<rhoenv) {
      ienv=i;
    }
  }
  int imax=isurf;
  return;
}

void nscool::cooling(int iret,std::vector<double> neebrem_logt, 
            std::vector<double>neebrem_nalpha, std::vector<double> neebrem_n2,
            std::vector<double> sf_lgtau1,std::vector<double> sf_lgtau2,
            std::vector<std::vector<double>> sf_lgr, 
            std::vector<std::vector<double>> sf_lgr2){

  if (debug >= 1) {
    std::cout << "Initializing" << std::endl;
  }
  // Get the time independent pieces of physics: ------------------------------
  grid(idec,rhocore,rhodrip,rhoenv,rhosurf,
        imax,icore,idrip,ienv,rad,rrho,pres,dvol,emas,phi);

  get_core_chemistry(version,imax,icore,rrho,
         bar,yneutr,yprot,yelect,ymuon,ylambda,ysminus,yszero,
         ysplus,yquarku,yquarkd,yquarks,theta_k,theta_p,fhad,
         mstn,mstp,mstla,mstsm,msts0,mstsp);

  if (icore==0) {
    iret=2;
    std::cout << 'Problem in get_core_chemistry().' << std::endl;
    return;
  }

  get_crust_chemistry(debug,version,imax,icore,
    rrho,pres,debar,dvol,bar,a_cell,a_ion,z_ion,v_ion,
    yelect,yneutr);
  get_fermi_momenta(imax,icore,rrho,bar,yneutr,
    yprot,yelect,ymuon,ylambda,ysminus,yszero,ysplus,yquarku,
    yquarkd,yquarks,fhad,theta_k,theta_p,
    kfn,kfp,kfe,kfm,kfla,kfsm,kfs0,kfsp,kfqu,kfqd,kfqs,
    idurca_np,idurca_lap,idurca_smn,idurca_smla,idurca_sms0,idurca_quqd,idurca_quqs,
    durca_ctrl_e,durca_ctrl_m,durca_henon_e,durca_henon_m);

  if (icore==0) {
    iret=4;
    std::cout << 'Problem in get_fermi_momenta().' << std::endl;
    return;
  }

  get_effective_masses(version,emnco,emncr,emp,
    kfn,kfp,mstn,mstp,mstla,mstsm,msts0,mstsp,idrip,icore);
  get_spec_heat_degenerate(cve,cvm,cvn,cvp,cvla,
    cvsm,cvs0,cvsp,cvqu,cvqd,cvqs,
    kfe,kfm,kfn,kfp,kfla,kfsm,kfs0,kfsp,kfqu,kfqd,kfqs,
    mstn,mstp,mstla,mstsm,msts0,mstsp,fhad,imax);
  get_Tc(imax,icore,idrip,
    tcn,tcp,tcla,tcsm,tcs0,tcsp,
    tcuu,tcdd,tcss,tcud,tcus,tcds,
    tcu,tcd,tcs,
    sfn1s0,sfn3p2,sfp1s0,sfl1s0,
    fn1s0,fn3p2,fp1s0,fl1s0,
    kfmax_n3p2,delkf_n3p2,tcmax_n3p2,isf,
    kfn,kfp,kfla,kfqu,kfqd,kfqs,bar,fhad,yquarku,yquarkd,yquarks);
  get_degenerate_density(rrho,pres,rhod,imax,ienv);
  // Calculate the T-independent coefficients: -------------------------------------
  if (debug >= 1) 
    std::cout << "Calculating T-independent coeff." << std::endl;

  for (int i=0;i<imax;++i) {
    ephi[i] = std::exp(phi[i]);
    e2phi[i] = ephi[i]*ephi[i];
    a2ephin[i] = std::pow(4.0*pi*rad[i]*rad[i],2)*ephi[i];
  }

  dephi[0]=0.0;
  for (int i=1;i<imax-1;++i) {
    dephi[i]=(ephi[i+1]-ephi[i-1])/(rad[i+1]-rad[i-1]);
  }
  dephi[imax-1]=dephi[imax-2];

  double radius=rad[imax-1];
  double root=std::sqrt(1.0-2.0*g*msol*emas[imax-1]/(radius*c*c));
  double factor=root/(4.0*pi*radius*radius)/(6.022e23)*1.0e39;
  double constant=4.0*pi*g*msol*emas[imax-1]*4.0/3.0*5.67e-5*e2phi[imax-1]/
                      pres[imax-1]/root;

  double gs=g*msol*emas[imax-1]/(radius*radius)/root;
  double gs14=gs/1.0e14;

  double compactness=2.0*g*msol*emas[imax-1]/(radius*c*c);
  // -----------------------------------------------------------------------------
  for (int i=imax;i>=1;i-=2) {
    nbfield2[i]=0.0;
  }
  // Calculate the initial Temp and Lum profiles ---------------------------------
  if (debug >= 1) 
    std::cout << "Calculating Initial T profile" << std::endl;
  // nscool_tptr_init ------------------------------------------------------------
  if (ifteff!=15) {
    if (tempini>0.0) {
      tsurface=T_fact_surf*ephi[imax]*tempini;
      tdrip=T_fact_drip*ephi[idrip]*tempini;
      tcore=1.0*ephi[0]*tempini;
    } else {
      tsurface=1.0e9;
      tdrip=2.0e10;
      tcore=1.0e11;
    }
  } else {
    tb_acc0*=ephi[imax];
    tsurface=tb_acc0;
    tdrip=tb_acc0;
    tcore=tb_acc0;
  }
  // -----------------------------------------------------------------------------
  double dec=1.1;
  for (int i=0;i<=icore;++i) {
    temp[i]=tcore;
  }
  for (int i=icore+1;i<=idrip;++i) {
    double w1=(std::log10(rad[idrip])-std::log10(rad[i]))/
          (std::log10(rad[idrip])-std::log10(rad[icore]));
    double w2=1.0-w1;
    double lt=w1*std::log10(tcore)+w2*std::log10(tdrip);
    temp[i]=std::pow(10.0,lt);
  }
  for (int i=idrip+1;i<=imax;++i) {
    double w1=(std::log10(rad[imax])-std::log10(rad[i]))/
      (std::log10(rad[imax])-std::log10(rad[idrip]));
    double w2=1.0-w1;
    double lt=w1*std::log10(tdrip)+w2*std::log10(tsurface);
    temp[i]=std::pow(10.0,lt);
  }
  dtemp[0]=0.0;
  for (int i=2;i<=imax-1;i+=2) {
    dtemp[i]=(temp[i+1]-temp[i-1])/(debar[i]+debar[i+1]);
  }
  // Calculate the inner envelope profile: ---------------------------------------
  if (debug >= 1) 
    std::cout << "Calculating envelope profile" << std::endl;

  for (int i=imax-1;i>=ienv+1;i-=2) {
    double x=(std::log(rrho[i+1])-std::log(rrho[i]))/
      (std::log(rrho[i+1])-std::log(rrho[i-1]));
    double y=(std::log(rrho[i])-std::log(rrho[i-1]))/
      (std::log(rrho[i+1])-std::log(rrho[i-1]));
    double ltemp=y*std::log(temp[i+1])+x*std::log(temp[i-1]);
    temp[i]=std::exp(ltemp);
  }

  for (int i=ienv;i<=imax;++i) {
    if (temp[i]<tcon) {
      density(temp[i]/ephi[i],pres[i],a_ion[i],z_ion[i],rrho[i]);
      rrho[i]=std::min(rrho[i],rhod[i]);
      bar[i]=6.022e-16*rrho[i];
      double dr=debar[i]/rrho[i]*factor; 
      rad[i+1]=rad[i]+dr;
    }
  }
  // ----------------------------------------------------------------------------
  if (debug>=1.0)
    std::cout << "Calculating initial L profile" << std::endl;


  for (int i=imax;i>=1;i-=2) {
    conduct(i,temp[i]/ephi[i],rrho[i],a_cell[i],a_ion[i],z_ion[i],qimp,
      nbfield2[i],sig,lambda[i],debug,nu_e_s,nu_e_l,icon_crust,icon_core,
      rhodrip,rhocore,kfe,kfm,kfn,kfp,kfla,kfsm,kfs0,kfsp,mstn,mstp,
      mstla,mstsm,msts0,mstsp,tcn,tcp,tcla,tcsm,tcs0,tcsp,isf,fhad,istrange);

    opacity(temp[i]/ephi[i],rrho[i],a_cell[i],z_ion[i],kappa[i],iopacity);

    double acd=7.56e-15*c/(3.0*kappa[i]*rrho[i]);
    fp[i]=(lambda[i]+4.0*acd*std::pow(temp[i]/ephi[i],3))*bar[i]/lsol;
  }

  lum[0]=0.0;
  for (int i=2;i<=imax-1;i+=2) {
    lum[i]=-(fp[i+1]+fp[i-1])/2.0*a2ephin[i]*dtemp[i];
    if (lum[i]==0.0) {
      lum[i]=1.0e-3;
    }
  }

  for (int i=1;i<=imax-2;i+=2) {
    dlum[i]=(nlum[i+1]-nlum[i-1])/(debar[i]+debar[i+1]);
  }
  // ---------------------------------------------------------------------------
  dlum[imax]=0.0;
  // ---------------------------------------------------------------------------
  for (int i=0;i<=imax;++i) {
    otemp[i]=temp[i];
    olum[i-1]=lum[i-1];
    orad[i]=rad[i];
    orrho[i]=rrho[i];
    rrho1[i]=rrho[i];
    obar[i]=bar[i];
    bar1[i]=bar[i];
  }
  // -----------------------------------------------------------------------
  // COOLING
  // -----------------------------------------------------------------------
  // starting from line 1063 again
  double time = time0;  // Initialize the time
  int icycle = 0;  // Initialize the counter for accretion cycles
  int itprint = 1;  // To print out only at the required times
  // THIS IS THE MAIN TIME LOOP:
  // -----------------------------------------------------------------------
  for (int istep=1;istep<=istepmax; istep++) {
    double debug=0.0;
    if (istep>=istep_debug) 
      debug = 1.0;
    if (debug>=1.0) 
      std::cout << "Going: istep=" << istep << std::endl;
    int itrial=0;
    double ratiot=1.e-2;
    double ratiol=1.e-2;
    if (dtime/year<=1.0e-20) {
      iret=1;
      std::cout << dtime/year << " NSCool failed. dtime too small." << std::endl;
      return;
    }

    if (debug>=1.0) {
      std::cout << "Guessing NLum & NTemp" << std::endl;
    }
    const double coeff_int = 0.8;
    for (int i=1;i<=imax;i+=2) {
      ntemp[i] = temp[i]+coeff_int*(temp[i]-otemp[i])*dtime/odtime;
    }

    dtemp[0]=0.0;
    for (int i=2;i<=imax-1;i+=2) {
      dtemp[i]=(ntemp[i+1]-ntemp[i-1])/(debar[i]+debar[i+1]);
    }

    nlum[0]=0.0;
    for (int i=2;i<=imax-1;i+=2) {
      nlum[i]=lum[i]+coeff_int*(lum[i]-olum[i])*dtime/odtime;
    }

    for (int i=1;i<=imax-2;i+=2) {
      dlum[i]=(nlum[i+1]-nlum[i-1])/(debar[i]+debar[i+1]);
    }

    dlum[imax] = 0.0;
    // Here needs to implement a goto loop
    itrial++;
    if (itrial==itrial_max+1) {
      double tcut = std::sqrt(scale_dt0);
      if (time<=1.0e5) {
        tcut=std::sqrt(scale_dt1);
      }
      dtime/=tcut;
      if (debug>0.4 && debug<0.6) {
        std::cout << "Exceeded iterations." << std::endl;
      }
      itrial=0;
    }

    for (int i=0;i<=imax;i++) {
      rad[i]=orad[i];
      rrho[i]=orrho[i];
      rrho1[i]=rrho[i];
    }

    double oteffective = teffective;
    // Calculate the new density in inner envelope at ntemp
    if (debug >= 1.0) 
      std::cout << "Calculating new density at NTemp" << std::endl;
    for (int i=imax-1;i>=ienv+1;i-=2) {
      double x=(std::log(rrho[i+1])-std::log(rrho[i]))/
              (std::log(rrho[i+1])-std::log(rrho[i-1]));
      double y=(std::log(rrho[i])-std::log(rrho[i-1])) /
              (std::log(rrho[i+1])-std::log(rrho[i-1]));
      double ltemp=y*std::log(ntemp[i+1])+x*std::log(ntemp[i-1]);
      ntemp[i]=std::exp(ltemp);
    }

    for (int i=ienv;i<=imax;i++) {
      if (ntemp[i]<tcon) {
        density(ntemp[i]/ephi[i],pres[i],a_cell[i],z_ion[i],rrho[i]);
        rrho[i]=std::min(rrho[i], rhod[i]);
        bar[i]=1.0e-39*6.022e23*rrho[i];
        double dr=debar[i]/rrho[i]*factor;
        rad[i+1]=rad[i]+dr;
        a2ephin[i]=std::pow(4.0*pi*rad[i]*rad[i],2)*ephi[i];
      }
    }
    // Calculate the physical parameters at ntemp
    if (debug >= 1.0) 
      std::cout << "Calculating physics at NTemp" << std::endl;
    for (int i=1;i<=imax;i+=2) {
      double t=ntemp[i]/ephi[i];
      double d=rrho[i];
      double a=a_cell[i];
      double a1=a_ion[i];
      double z=z_ion[i];
      neutrinot(i, t, d, a, z, qnu[i], qeebrem[i], qnpb[i], qplasma[i],
                 qsynch[i], qbubble[i], qpair[i], qphoto[i], qbrem_nn[i],
                 qmurca_nucl[i], qbrem_nucl[i], qmurca_hyp[i], qbrem_hyp[i],
                 qdurca_np[i], qdurca_lap[i], qdurca_smn[i], qdurca_smla[i],
                 qdurca_sms0[i], qfast[i], qdurca_q[i], qmurca_q[i],
                 qpbf_n1s0[i], qpbf_n3p2[i], qpbf_p1s0[i], qpbf_q[i],
                 debug, naa, nbfield2, rhodrip, rhocore, mstp, mstn, mstla,
                 mstsm, msts0, mstsp, kfe, kfm, kfp, kfn, kfqu, kfqd, kfqs,
                 bar, yelect, ymuon, fhad, theta_k, theta_p, v_ion,
                 rhoexo, cexo, pexo, c_nu_str, p_nu_str,
                 murca_increase, inu_durca, inu_eion, inu_plasma, inu_synch,
                 inu_n1s0_pbf, inu_n3p2_pbf, inu_p_pbf,
                 inu_bubble, inu_photo, inu_pair,
                 idurca_np, idurca_lap, durca_ctrl_e, durca_ctrl_m,
                 idurca_smn, idurca_smla, idurca_sms0,
                 idurca_quqd, idurca_quqs, tcn, tcp, tcla, tcu, tcd, tcs,
                 tcu1, tcu2, tcu3, tcd1, tcd2, tcd3, tcs1, tcs2, tcs3, isf,
                 neebrem_logt, neebrem_nalpha, neebrem_n2,
                 sf_lgtau1, sf_lgtau2, sf_lgr, sf_lgr2);

      qqq[i]=qnu[i]-heat[i];
      specheat(i,t,d,a,z,cv[i],
           cv_n[i],cv_p[i],cv_e[i],cv_m[i],
           cv_l[i],cv_sm[i],cv_s0[i],cv_sp[i],
           cv_q[i],cv_ion[i],cve,cvm,cvn,cvp,cvla,
          cvsm,cvs0,cvsp,cvqu,cvqd,cvqs,rhodrip,rhocore,fhad,istrange,
          tcn,tcp,tcla,tcsm,tcs0,tcsp,isf);
      if (debug >= 1.0) 
      std::cout << "Calling conduct: " << i << " " << d << " " << t << std::endl;
      conduct(i,t,d,a,a1,z,qimp,nbfield2[i],
           sig,lambda1[i],debug,
           nu_e_s,nu_e_l,icon_crust,icon_core,rhodrip,rhocore,
           kfe,kfm,kfn,kfp,kfla,kfsm,kfs0,kfsp,mstn,mstp,
           mstla,mstsm,msts0,mstsp,tcn,tcp,tcla,tcsm,tcs0,tcsp,isf,
           fhad,istrange);
      if (debug >= 1.0) 
        std::cout << "done" << std::endl;
      opacity(t,d,a,z,kappa[i],iopacity);
      double acd=7.56-15*c/(3*kappa[i]*d);
      fp[i]=(lambda[i]+4.*acd*std::pow(t,3))*bar[i]/lsol;
      fq[i]=bar1[i]/cv[i]*lsol;
      fr[i]=e2phi[i]*qqq[i]/cv[i]-
           ephi[i]*pres[i]/cv[i]*
           (std::log(rrho[i])-std::log(orrho[i]))/dtime*contraction;
      
    }

    // Calculate the physical parameters at (1-tinc)*ntemp
    if (debug >= 1.0) 
      std::cout << "Calculating physics at NTemp" << std::endl;
    for (int i=1;i<=imax;i+=2) {
      double t=ntemp1[i]/ephi[i];
      double d=rrho1[i];
      double a=a_cell[i];
      double a1=a_ion[i];
      double z=z_ion[i];
      double qn00,qn01,qn02,qn03,qn04,qn05,qn06,qn07,qn08,qn09,q10,qn11,
                qn12,qn13,qn14,qn15,qn16,qn17,qn18,qn19,q20,qn21,qn22,qn23,
                xx1,xx2,xx3,xx4,xx5,xx6,xx7,xx8,xx9,xx0;
      neutrino(i, t, d, a, z, qnu1[i], qn00,qn01,qn02,qn03,qn04,qn05,
                 qn06,qn07,qn08,qn09,q10,qn11,qn12,qn13,qn14,qn15,qn16,qn17,
                 qn18,qn19,q20,qn21,qn22,qn23,
                 debug, naa, nbfield2, rhodrip, rhocore, mstp, mstn, mstla,
                 mstsm, msts0, mstsp, kfe, kfm, kfp, kfn, kfqu, kfqd, kfqs,
                 bar, yelect, ymuon, fhad, theta_k, theta_p, v_ion,
                 rhoexo, cexo, pexo, c_nu_str, p_nu_str,
                 murca_increase, inu_durca, inu_eion, inu_plasma, inu_synch,
                 inu_n1s0_pbf, inu_n3p2_pbf, inu_p_pbf,
                 inu_bubble, inu_photo, inu_pair,
                 idurca_np, idurca_lap, durca_ctrl_e, durca_ctrl_m,
                 idurca_smn, idurca_smla, idurca_sms0,
                 idurca_quqd, idurca_quqs, tcn, tcp, tcla, tcu, tcd, tcs,
                 tcu1, tcu2, tcu3, tcd1, tcd2, tcd3, tcs1, tcs2, tcs3, isf,
                 neebrem_logt, neebrem_nalpha, neebrem_n2,
                 sf_lgtau1, sf_lgtau2, sf_lgr, sf_lgr2);

      qqq1[i]=qnu1[i]-heat1[i];
      specheat(i,t,d,a,z,cv1[i],
           xx1,xx2,xx3,xx4,xx5,xx6,xx7,xx8,xx9,xx0,cve,cvm,cvn,cvp,cvla,
          cvsm,cvs0,cvsp,cvqu,cvqd,cvqs,rhodrip,rhocore,fhad,istrange,
          tcn,tcp,tcla,tcsm,tcs0,tcsp,isf);
      if (debug >= 1.0) 
      std::cout << "Calling conduct: " << i << " " << d << " " << t << std::endl;
      conduct(i,t,d,a,a1,z,qimp,nbfield2[i],
           sig,lambda1[i],debug,
           nu_e_s,nu_e_l,icon_crust,icon_core,rhodrip,rhocore,
           kfe,kfm,kfn,kfp,kfla,kfsm,kfs0,kfsp,mstn,mstp,
           mstla,mstsm,msts0,mstsp,tcn,tcp,tcla,tcsm,tcs0,tcsp,isf,
           fhad,istrange);
      if (debug >= 1.0) 
      std::cout << "done" << std::endl;
      opacity(t,d,a,z,kappa1[i],iopacity);
      double acd1=7.56-15*c/(3*kappa1[i]*d);
      fp1[i]=(lambda1[i]+4.*acd1*std::pow(t,3))*bar1[i]/lsol;
      fq1[i]=bar1[i]/cv[i]*lsol;
      fr1[i]=e2phi[i]*qqq1[i]/cv1[i]-
             ephi[i]*pres[i]/cv1[i]*
             (std::log(rrho1[i])-std::log(orrho[i]))/dtime*contraction;
      
    }

    // Calculate the derivatives of fp,fq & fr
    if (debug>=1.0) 
      std::cout << "Calculating derivatives of FP, FQ, FR" << std::endl;
    for (int i=1;i<=imax;i+=2) {
        double t=ntemp[i];
        double t1=ntemp1[i];
        dfp[i]=(fp[i]-fp1[i])/(t-t1);
        dfq[i]=(fq[i]-fq1[i])/(t-t1);
        dfr[i]=(fr[i]-fr1[i])/(t-t1);
    }

    if (debug>=1.0) 
        std::cout << "Calculating FF" << std::endl;

    ff[0]=0.0;
    for (int i=2;i<=imax-1;i+=2) {
      ff[i]=nlum[i]+0.5*(fp[i-1]+fp[i+1])*a2ephin[i]*dtemp[i];
      ff[i-1]=fr[i-1]+fq[i-1]*dlum[i-1]+
              (ntemp[i-1]-temp[i-1])/dtime;
    }

    ff[imax]=0.0;
    // Matrix inversion for Newton-Raphson method

    if (debug >= 1.0) 
      std::cout << "Newton-Raphson" << std::endl;

    for (int i=2;i<=imax-1;i+=2) {
      fa[i]=0.5*dfp[i+1]*a2ephin[i]*dtemp[i]+
          0.5*(fp[i+1]+fp[i-1])*a2ephin[i]/
                (debar[i]+debar[i+1]);
      fb[i]=0.5*dfp[i-1]*a2ephin[i]*dtemp[i]-
          0.5*(fp[i+1]+fp[i-1])*a2ephin[i]/
              (debar[i]+debar[i+1]);
      fc[i]=1.0;
    }

    for (int i=1;i<=imax-2;i+=2) {
      fa[i]=fq[i]/(debar[i]+debar[i+1]);
      fb[i]=-fq[i]/(debar[i]+debar[i+1]);
      fc[i]=dfr[i]+dfq[i]*dlum[i]+1.0/dtime;
    }

    fk[1]=-ff[1]/fc[1];
    fj[1]=fa[1]/fc[1];
    for (int i=2;i<=imax-1;++i) {
      fk[i]=-(ff[i]+fb[i]*fk[i-1])/(fc[i]-fb[i]*fj[i-1]);
      fj[i]=fa[i]/(fc[i]-fb[i]*fj[i-1]);
    }

    if (debug >= 1.0) 
      std::cout << "Boundary Condition" << std::endl;

    double ntp;

    if (ifteff != 15) {
      double epsilon=1.0e-8;
      double precision=1.0e-12;
      double coeff=4.0*pi*radius*radius*5.67e-5*e2phi[imax]/lsol;
      double lhs=nlum[imax-1]+fk[imax-1]+fj[imax-1]*ntemp[imax];
      ntp=ntemp[imax];
      double tp0_keep=ntp;
        
      double tp0 = ntp;
      double teff0 = Teff(tp0/ephi[imax],ifteff,eta,
                          0.0, istep,time,z_ion[imax],
                          a_ion[imax],rrho[imax],debug,gs14,compactness);
      if (debug==-50.0) 
        std::cout << "Tb0, Te0 = " << tp0 << ", " << teff0 << std::endl;
        
      double tp1=(1.0+epsilon)*tp0;
      double teff1=Teff(tp1/ephi[imax],ifteff,eta,
                          0.0, istep,time,z_ion[imax],
                          a_ion[imax],rrho[imax],debug,gs14,compactness);
      if (debug==-50.0) 
        std::cout << "Tb1, Te1 = " << tp1 << ", " << teff1 << std::endl;
        
      double derivative=coeff*(std::pow(teff1,4)-std::pow(teff0,4))/(epsilon*tp0);
      derivative=-fj[imax-1]-derivative;

      if (debug==-50.0) 
        std::cout << "Derivative = " << derivative << std::endl;
        
      double function=lhs-fj[imax-1]*tp0-coeff*std::pow(teff0,4);
      if (debug==-50.0)
        std::cout << "Function = " << function << std::endl;
        
      ntp=tp0-function/derivative;
      if (debug==-50.0) {
        std::cout << "Del(Tp)/Tp = " << std::abs(tp0-ntp)/tp0 << std::endl;
        std::cout << "------> New Tb = " << ntp << std::endl;
      }
        
      if ((ntp<=0.0) || (ntp>1.0e12)) { // In case the method diverges
        tcut=std::sqrt(scale_dt0);
        if (time<=1.0e5) {
          tcut = std::sqrt(scale_dt1);
        }
        dtime = dtime/tcut;
        if (debug>0.4 && debug<0.6) 
          std::cout << "BC problem." << std::endl;
        itrial=0; // goto statement needs fix
        }
        
        if (std::abs(tp0-ntp)/tp0>precision) {
          tp0=ntp; // goto statement needs fix
        }
    } else {
        ntp=tb_acc0; // Fixed T_b for accretion
      }
      // Get ntemp & nlum
      if (debug >= 1.0) 
        std::cout << "Getting NTemp & NLum" << std::endl;
      delt[imax]=ntp-ntemp[imax];
      for (int i=imax-2;i>=1;i-=2) {
        dell[i+1]=fk[i+1]-fj[i+1]*delt[i+2];
        dell[i+1]=std::copysign(std::min(2.0e3*std::abs(nlum[i+1]),
                  std::abs(dell[i+1])),dell[i+1]);
        delt[i]=fk[i]-fj[i]*dell[i+1];
        delt[i]=std::copysign(std::min(0.5*ntemp[i],
              std::abs(delt[i])),delt[i]);
      }
      dell[0]=0.0; // This is the inner boundary condition !

      if (debug>0.4 && debug<0.6) {
        std::cout << "itrial,imax,delt,dell" << std::endl;
        std::cout << " " << itrial << " " << imax << " " 
        << delt[imax-2] << " " << dell[imax-1] << std::endl;
      }

      for (int i=0;i<imax; i+=2) {
        nlum[i]=nlum[i]+dell[i];
        ntemp[i+1]=ntemp[i+1]+delt[i+1];
      }

      dtemp[0]=0.0;
      for (int i=2;i<imax-1;i++) {
        dtemp[i]=(ntemp[i+1]-ntemp[i-1])/(debar[i]+debar[i+1]);
      }
      for (int i=1;i<imax-2;i+=2) {
        dlum[i]=(nlum[i+1]-nlum[i-1])/(debar[i]+debar[i+1]);
      }
      dlum[imax]=0.0; 
      //----------------------------------------------------------------
      // Analyze the results to see if it has converged
      // ---------------------------------------------------------------
      if (debug>=1.0) 
        std::cout << "Analyzing Results" << std::endl;

      double ratiot=0.0;
      double ratiol=0.0;
      int iratt=0;
      int iratl=0;

      for (int i=1;i<imax-2;i+=2) {
        double ratl=std::abs(dell[i+1])/(std::abs(nlum[i+1])+1.0e-12);
        if (ratl>ratiol) {
          ratiol=ratl;
          iratl=i+1;
        }
        double ratt=std::abs(delt[i])/(ntemp[i]+1.0e-30);
        if (ratt>ratiot) {
          ratiot=ratt;
          iratt=i;
        }
      }
      // This was used when doing dipolar magnetic field evolution.
      // Not used any more.
      double ratios=0.0;
      int irats=1;
      // ---------------------------------------------------------------
      if (pscreen==3) {
        if (ifield==2) {
          std::cout << "   dT/T=" << ratiot << "(" << iratt << ")"
                << "   dL/L=" << ratiol << "(" << iratl << ")"
                << "   dS/S=" << ratios << "(" << irats << ")" << std::endl;
        } else {
          std::cout << "   dT/T=" << ratiot << "(" << iratt << ")"
                << "   dL/L=" << ratiol << "(" << iratl << ")" << std::endl;
        }
      }
      // ---------------------------------------------------------------
      if (debug>0.4 && debug<0.6) {
        std::cout << "ratio,irat,mrat   "
              << ratiot << " " << iratt << " " << mratt << "   "
              << ratiol << " " << iratl << " " << mratl << std::endl;
      }
      // ---------------------------------------------------------------
      // Decide if converged or not:
      if (ratiot<mratt && ratiol<mratl && ratios<mrats) {
        continue;// Converged! Continue to the next time step.
      } else {
      // Not converged! Go back for another iteration.
        goto 2000;
      }

      // ---------------------------------------------------------------

      double luminosity=nlum[imax-1]/(ephi[imax-1]*ephi[imax-1]);
      double sign_l=std::abs(nlum[imax-1])/nlum[imax-1];
      double teffective=sign_l*std::pow(std::abs(luminosity)/
            (4.0*pi*radius*radius*5.67e-5),0.25)
            *std::pow(lsol,0.25)*ephi[imax-1];

      // c PHILOSOPHY OF TIME STEP CONTROL:
      //
      // (Time of step just finished is "time+dtime", not just time !)
      // The new "dtime" will be "scale_dt*dtime" with "scale_dt" calculated below.
      // Allows for 2 different "scale_dt": at early time, while relaxing from initial
      // conditions, accuracy is not important and one can allow for larger timestep:

      if (debug >= 1) 
        std::cout << "Calculating New Time Step" << std::endl;

      double scale_dt = scale_dt0;
      if (time <= 1.0e5) {
        scale_dt = scale_dt1;
      }

      // TEMP variation: "max_dtemp" is the max. relative variation of T in the
      // star and "icht" the zone where "max_dtemp" is obtained

      int icht = 0;
      double max_dtemp = 0.0;
      for (int i = 1; i <= imax - 2; i += 2) {
        double mdt = std::abs(temp[i] - ntemp[i]) / ntemp[i];
        if (mdt > max_dtemp) {
          max_dtemp = mdt;
          icht = i;
        }
      }
      // ---------------------------------------------------------------
      // c Check if "max_dtemp" is not too large. If "1+max_dtemp" exceeds "tvar",
      // then "scale_dt" is reduced correspondingly:
      double chtemp = 0.0;
      if ((tvar - 1.0) < max_dtemp) {
        scale_dt = scale_dt * (tvar - 1.0) / max_dtemp;
        if (time < 1.0e5) {
          scale_dt = std::min(scale_dt, scale_dt1);
        } else {
          scale_dt = std::min(scale_dt, scale_dt0);
        }
        chtemp = 1.0;
      }
      // ----------------------------------------------------------------
      // c If "scale_dt" has been reduced too much, i.e. "max_dtemp" too large,
      // then the present time-step is recalculated with a smaller "dtime"
      // unless one is still at the first few time-steps:

      if ((scale_dt < repeat) && (istep > istart)) {
        dtime = repeat * dtime;
        // --------------------------------------------------------------
        if (pscreen >= 2) {
          std::cout << std::endl;
          std::cout << "dtime too large, do it again" << std::endl;
          std::cout << "time=" << (time + dtime) / year << std::endl;
          std::cout << "dtime=" << dtime / year << std::endl;
          std::cout << "dtime ratio=" << dtime / odtime << std::endl;
          std::cout << std::endl;
        }
        // ----------------------------------------------------------------
        if (debug > 0.4 && debug < 0.6) 
          std::cout << "Temperature changed too much." << std::endl;
          itrial-0; // another goto statement needs fix
      }

      // In case convergence is reached in too many trials, scale_dt is reduced
      double chtrial;
      if ((itrial > itrial_opt) && (istep > istart)) {
        chtrial = 1.0; // this means the time-step is controlled by needing too many iterations
        double olddt = scale_dt;
        if (time < 1.0e5) {
          scale_dt = scale_dt / std::pow(scale_dt1, static_cast<double>(itrial - itrial_opt) / 2.0);
          scale_dt = std::min(scale_dt, scale_dt1);
        } else {
          scale_dt = scale_dt / std::pow(scale_dt0, (1.0 + static_cast<double>(itrial - itrial_opt)) / 2.0);
          scale_dt = std::min(scale_dt, scale_dt0);
        }
      } else {
        chtrial = 0.0;
      }
      // Before setting the next time and time-step, stuff are printed out and updated:
      // -------------------------------------------------------------------
      // End of iterations
      // -------------------------------------------------------------------
      for (int i = 1; i <= imax; i += 2) {
        otemp[i] = temp[i];
        temp[i] = ntemp[i];
        orrho[i] = rrho[i];
        orad[i] = rad[i];
        obar[i] = bar[i];
      }

      for (int i = 2; i <= imax - 1; i += 2) {
        olum[i] = lum[i];
        lum[i] = nlum[i];
        orrho[i] = rrho[i];
        orad[i] = rad[i];
        obar[i] = bar[i];
      }
      // -------------------------------------------------------------------
      //period=nperiod;
      // -------------------------------------------------------------------
      // Crystalization density:
      // -------------------------------------------------------------------
      if (debug>=1) 
        std::cout << "Calculating Crystalization Densities" << std::endl;

      for (int i=imax;i>=icore+2;i-=2) {
        gamma[i]=2.273e5*pow(z_ion[i],2)*pow(rrho[i]/a_cell[i],1.0/3.0)/
               (temp[i]/ephi[i]);
        if (gamma[i]<gammaliq) {
          cryst[i]=-1.0;
        } else if ((gamma[i]>=gammaliq) && (gamma[i]<=gammacryst)) {
          cryst[i]=0.0;
        } else {
          cryst[i]=1.0;
        }
      }
      for (int i=icore;i>= 0;i-=2)
        gamma[i]=1.0;
    
      int iliq=0;
      int icryst=0;

      for (int i=imax;i>=icore+2;i-=2) {
        if (cryst[i]==0.0) {
          iliq=i;
          continue;
        }
      }
      for (int i=imax;i>=icore+2;i-=2) {
        if (cryst[i]==1.0) {
          icryst=i;
          continue;
        }
      }

      // ----------------------------------------------------------------
      // Calculate the neutrino luminosity and heating:
      // ----------------------------------------------------------------
      if (debug>=1) 
        std::cout << "Calculating Total Neutrino Luminosity and Heating" << std::endl;
      double lnu=0.0;
      double lnu0=0.0;
      double lh=0.0;
      double lh0=0.0;
      for (int i=1;i<=imax;i+=2) {
        lnu+=e2phi[i]*qnu[i]*(dvol[i]+dvol[i+1]);
        lnu0+=ephi[i]*qnu[i]*(dvol[i]+dvol[i+1]);   // without energy red-shift
        lh+=e2phi[i]*heat[i]*(dvol[i]+dvol[i+1]);
        lh0+=ephi[i]*heat[i]*(dvol[i]+dvol[i+1]);   // without energy red-shift
      }
      lnu/=lsol;
      lh/=lsol;
      htot+=lh*dtime;
      // ---------------------------------------------------------------
      // CALCULATE THE INTEGRATED NEUTRINO LUMINOSITIES:
      // ---------------------------------------------------------------
      // Note: lnu_tot, calculated from qnu(i), is the garanteed total
      // neutrino luminosity. The other ones are only informative.
      double lmurca_nucl=0.0;
      double lbrem_nucl=0.0;
      double lplasma=0.0;
      double lnpb=0.0;
      double lpbf_n1S0=0.0;
      double lpbf_n3P2=0.0;
      double lpbf_p1S0=0.0;
      for (int i=1;i<=imax;i+=2) {
        double e2p=e2phi[i];
        lmurca_nucl+=qmurca_nucl[i]*(dvol[i]+dvol[i+1])*e2p;
        lbrem_nucl+=qbrem_nucl[i]*(dvol[i]+dvol[i+1])*e2p;
        lplasma+=qplasma[i]*(dvol[i]+dvol[i+1])*e2p;
        lnpb+=qnpb[i]*(dvol[i]+dvol[i+1])*e2p;
        lpbf_n1S0+=qpbf_n1s0[i]*(dvol[i]+dvol[i+1])*e2p;
        lpbf_n3P2+=qpbf_n3p2[i]*(dvol[i]+dvol[i+1])*e2p;
        lpbf_p1S0+=qpbf_p1s0[i]*(dvol[i]+dvol[i+1])*e2p;
      }
      // --------------------------------------------------------------
      // CALCULATE THE INTEGRATED SPECIFIC HEATS:
      // --------------------------------------------------------------
      // cv_tot_all, calculated from cv(i), is the garanteed total
      // specific heat. The other ones are only informative.
      double cv_core=0.0;
      double cv_crust=0.0;
      double cv_tot_all=0.0;
      double cv_tot_ion=0.0;
      double cv_tot_neu=0.0;
      double cv_tot_pro=0.0;
      double cv_tot_ele=0.0;
      double cv_tot_muo=0.0;
      double cv_tot_lam=0.0;
      double cv_tot_sim=0.0;
      double cv_tot_si0=0.0;
      double cv_tot_sip=0.0;
      double cv_tot_qrk=0.0;
      double cv_phot=0.0;
      for (int i=1;i<=imax;i+=2) {
        cv_tot_all+=cv[i]*(dvol[i] + dvol[i+1]);
        cv_tot_ion+= cv_ion[i]*(dvol[i]+dvol[i+1]);
        cv_tot_neu+= cv_n[i]*(dvol[i]+dvol[i+1]);
        cv_tot_pro+= cv_p[i]*(dvol[i]+dvol[i+1]);
        cv_tot_ele+= cv_e[i]*(dvol[i]+dvol[i+1]);
        cv_tot_muo+= cv_m[i]*(dvol[i]+dvol[i+1]);
        cv_tot_lam+= cv_l[i]*(dvol[i]+dvol[i+1]);
        cv_tot_sim+= cv_sm[i]*(dvol[i]+dvol[i+1]);
        cv_tot_si0+= cv_s0[i]*(dvol[i]+dvol[i+1]);
        cv_tot_sip+= cv_sp[i]*(dvol[i]+dvol[i+1]);
        cv_tot_qrk+= cv_q[i]*(dvol[i]+dvol[i+1]);
        cv_phot+=4.0*7.56e-15*pow((ntemp[i]/ephi[i]),3)*(dvol[i]+dvol[i+1]);
      }
      for (int i=1;i<=icore+2;i++) {
        cv_core+=cv[i]*(dvol[i]+dvol[i+1]);
      }
      for (int i=icore+2;i<=imax;i++) {
        cv_crust+=cv[i]*(dvol[i]+dvol[i+1]);
      }
      // -------------------------------------------------------------------
      // TEMP
      // -------------------------------------------------------------------
      if (debug>0.4 && debug<0.6) 
        std::cout << "time " << time+dtime/year << std::endl;

      if (ptemp>=1.0) {
        // AWS: Determine if we should output temperature at this iteration
        if (((time+dtime)/year >= tprint[itprint-1]) && (itprint<=itpmax)) {
          itprint++;

          // AWS: Compute weights and effective temperature
          double w1=((time+dtime)-tprint[itprint-2]*year)/dtime;
          double w2=1.0-w1;
          double logtemp=w1*log(sign_l*oteffective)+
                         w2*log(sign_l*teffective);
          double t_effective=sign_l*exp(logtemp);

          print_temp(istep,itprint-2,
                          tprint[itprint-2],t_effective,imax,w1,w2,otemp,
                          temp, olum, lum, rad, rrho, ephi, dvol, e2phi, tcn, tcp,
                          qnu, qeebrem, qnpb, qplasma, qsynch, qbubble, qpair,
                          qphoto, qbrem_nn, qmurca_nucl, qbrem_nucl, qmurca_hyp,
                          qbrem_hyp, qdurca_np, qdurca_lap, qdurca_smn,
                          qdurca_smla, qdurca_sms0, qfast, qdurca_q, qmurca_q,
                          qpbf_n1s0, qpbf_p1s0, qpbf_n3p2, qpbf_q);

          print_cv(itprint-2, imax, cv, cv_n,
                        cv_p, cv_e, cv_m, cv_l, cv_sm, cv_s0, cv_sp, cv_q);

        }
      }
      // ------------------------------------------------------------------
      // TEFF
      // ------------------------------------------------------------------
      if (pteff >= 1.0) {
        // skipping because Its not used
        cv_core = 0.0;
        cv_crust = 0.0;
        for (int i=0;i<icore;i+=2) {
          cv_core+=(dvol[i]+dvol[i+1])*cv[i];
        }
        for (int i=icore+2;i<imax;i+=2) {
          cv_crust+=(dvol[i]+dvol[i+1])*cv[i];
        }
        // -----------------------------------------------------------------
        if (pteff == 1.0) {
          std::cout << std::fixed << std::setprecision(3)
                  << std::setw(8) << istep
                  << std::scientific << std::setprecision(3)
                  << std::setw(12) << (time + dtime) / year
                  << std::setw(12) << teffective
                  << std::setw(12) << lum[imax - 1] * lsol
                  << std::setw(12) << lnu * lsol
                  << std::setw(12) << lh * lsol
                  << std::endl;
        } else {
          std::cout << "WARNING: Not Teff print out defined !" << std::endl;
          exit(-1);
        }
      }
      // -------------------------------------------------------------------
      nscool_main_out((time+dtime)/year,teffective,nlum[imax-1]*lsol,lnu*lsol,dtime/year,istop);

      if (istop > 0) {
        iret = 3;
        std::cout << dtime << " NSCool failed. Stepsize vanishing." << std::endl;
        continue; 
      }
      // -------------------------------------------------------------------
      // CALCULATE THE NEW TIME STEP
      // -------------------------------------------------------------------
      time=time+dtime;
      odtime=dtime;
      dtime=std::min(scale_dt*dtime,dtlimit);
      // -------------------------------------------------------------------
      // For accretion scenarios: time step must moreover be shortened 
      // dramatically when a new outburst is approaching (to make sure it is  
      // much shorter than the outburst duration, or rise time, or any relevant 
      // time scale which has to be resolved by the code):
      // -------------------------------------------------------------------
      // Transient FRED ("Fast rise and exponential decay")
      if (i_acc == 1) {
        double time_step_cut = 100.0;

        if (((time+3.0*dtime)>=t_acc0) && (icycle == 0)) {
          double timeleft=t_acc0-time;
          dtime=std::max(timeleft/3.0,t_acc2/time_step_cut);
        }

        if (time>t_acc0) {
          if (delt_acc/t_acc2 <= 10.0) {
            scale_dt0 = 1.05;
          } else {
            scale_dt0 = 1.2;
          }
          double t_next = t_acc0+static_cast<double>(icycle + 1)*t_acc1;
          if ((time+3.0*dtime)>=t_next) {
            double timeleft=t_next-time;
            dtime = std::max(timeleft/3.0,t_acc2/time_step_cut);
          }
        }

        if (time>=t_acc0) 
          dtlimit=t_acc1/20.0;
    
      }
      // Transient STEP----------------------------------------------------
      if (i_acc==2) {
        double timeleft,t_next,t_end,day;
        if (((time+2.0*dtime)>=t_acc0) && (time<t_acc0)) {
          timeleft=t_acc0-time;
          dtime=std::max(timeleft/3.0,time_step_min);
          // Print messages if needed
        }

        if (time > t_acc0) {
          t_next=t_acc0+static_cast<double>(icycle-1)*t_acc1+t_acc1;
          t_end=t_acc0+static_cast<double>(icycle-1)*t_acc1+t_acc2;

          if ((time+dtime>t_end) && (time+2.0*dtime>=t_next)) {
            timeleft=t_next-time;
            dtime=std::max(timeleft/3.0,time_step_min);
          } else if ((time<=t_end) && (time+2.0*dtime>=t_end)) {
            timeleft=t_end-time;
            dtime=std::max(timeleft/3.0,time_step_min);
          }
        }

        if (time>=t_acc0) {
          dtlimit=t_acc1/20.0;
        }
      }
      // Heat deposition--------------------------------------------------
      if (i_heat_deposit==1) {
        double t_slow=std::max(dtime,1000.0*del_t_dep);

        if (std::abs(t_dep-(time+dtime))<t_slow) {
          double timeleft=std::abs(t_dep-(time+dtime));
          dtime = std::max(timeleft/30.0,del_t_dep/100.0);
        }
      }

      if (time/year>=timemax) {
        return;
      }

      if ((sign_l*teffective)<tempmin) {
        return;
      }
    }
  };