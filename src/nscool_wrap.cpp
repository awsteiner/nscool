#include "nscool_wrap.h"

#ifdef O2SCL_MPI
#include "mpi.h"
#endif

std::vector<nscool_wrap *> nscool_wrap_ptrs;

nscool_wrap::nscool_wrap(std::string dir) {

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
    std::cout << "In nscool_wrap, rank " << mpi_rank
	      << " reading data files." << std::endl;
  }
    
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

#ifdef O2SCL_MPI
  if (mpi_size>1 && mpi_rank<mpi_size-1) {
    MPI_Send(&buffer,1,MPI_INT,mpi_rank+1,
	     tag,MPI_COMM_WORLD);
  }
#endif
    
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

  /*
    This is the SFB neutrino singlet gap, the T73 proton singlet
    gap, and neutrino triplet gap "a" from the minimal cooling
    paper.
  */
  sfn1s0=1;
  sfn3p2=101;
  sfp1s0=3;

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
  spline_(pb_logt,pb_nalpha,&pb_n,&pb_bound1,&pb_bound2,pb_n2);
    
  // Data for superfluid suppression
  sf_suppress_data(sf_lgtau1,sf_lgtau2,sf_lgr);
  int sf_n1=35;
  int sf_n2=35;
  spline2_(sf_lgtau1,sf_lgtau2,sf_lgr,&sf_n1,&sf_n2,sf_lgr2);
    
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

double nscool_wrap::cvelec(double t, double rho, double a, double z) {
  double hb=1.054588e-27;
  double kb=1.380662e-16;
  double c=2.997924e10;
  double na=6.022045e23;
  double me=9.109e-28;
  double pi=3.14159265;
    
  double ne=na*rho*z/a;
  double pf=hb*cbrt(3.0*pi*pi*ne);
  double ef=sqrt((me*me*c*c*c*c)+(pf*c*c))-me*c*c;
  double tf=ef/kb;
  double xe=pf/(me*c);
  double ae=xe*xe/sqrt(1.+xe*xe);
  double cvt=ne*kb*kb*pi*pi/(me*c*c)/ae;
    
  double t0=tf/50.0;
    
  if (t<0.5*t0) {
    return cvt*t;
  }

  electron_deriv.n=ne;
  fdr.calc_density(electron_deriv,t);
  return electron_deriv.dsdT;
}

double nscool_wrap::cvion(double t, double rho, double a, double z) {

  double rhodrip=4.3e11;
  double bcv=0.95043;
  double ccv=0.18956;
  double dcv=-0.81487;
  double hcv=3225.0;
  double cte=141.7;
  double cv0[15]={0.0,2.956,2.829,2.633,2.389,2.118,1.840,1.572,
		  1.323,1.102,0.909,0.745,0.609,0.496,0.404};
    
  double gamma=2.273e5*z*z*cbrt(rho/a)/t;
  double gamma14=pow(gamma,0.25);
  double a1;
  if (rho>=rhodrip) {
    a1=3.0*z;
  } else {
    a1=a;
  }
  double nionkb=1.38e-16*6.022e23*rho/a;
  double delta=1.0/t*z*sqrt(rho/(a1*a))*6.022e23;
  if (gamma<=0.1) {
    return 1.5*nionkb;
  } else if (gamma<=0.2) {
    double cv1=1.5*nionkb;
    double cv2=nionkb*(0.75*bcv*gamma14+1.25*ccv/gamma14+dcv+1.5);
    return (gamma-0.1)/0.1*cv2+(0.2-gamma)/0.1*cv1;
  } else if (gamma<178.0) {
    return nionkb*(0.75*bcv*gamma14+1.25*ccv/gamma14+dcv+1.5);
  } else if (gamma<=210.0 && delta>=1.0e19) {
    double cv1=nionkb*(1.5+3.0*hcv/gamma/gamma+1.5);
    cv0[0]=1.5+3.0*hcv/gamma/gamma+1.5;
    int i1=((int)(delta*2.0e-20));
    if (i1>14 || i1<0) {
      std::cerr << "cvion failure." << std::endl;
      exit(-1);
    }
    double cv2=nionkb*(cv0[i1]+(delta*2.0e-20-i1)*(cv0[i1+1]-cv0[i1]));
    return (gamma-178.0)/32.0*cv2+(210.0-gamma)/32.0;
  } else if (delta<=1.0e19) {
    return nionkb*(1.5+3.0*hcv/gamma/gamma+1.5);
  } else if (delta>1.0e9 && delta<7.0e20) {
    cv0[0]=1.5+3.0*hcv/gamma/gamma+1.5;
    int i1=((int)(delta*2.0e-20));
    if (i1>14 || i1<0) {
      std::cerr << "cvion failure." << std::endl;
      exit(-1);
    }
    return nionkb*(cv0[i1]+(delta*2.0e-20-i1)*(cv0[i1+1]-cv0[i1]));
  }
  double delta1=delta*1.0e-20;
  return nionkb*cte/delta1/delta1/delta1;
}

double nscool_wrap::pressure(double T, double Rho, double A, double Z) {
  double hb=1.054588e-27;
  double kb=1.380662e-16;
  double c=2.997924e10;
  double NA=6.022045e23;
  double me=9.109e-28;
  double pi=3.141592653;

  double ne=Rho*NA*Z/A;
  double nion=Rho*NA/A;
  double Zeff=ne/nion;
  double gamma=2.273e5*Zeff*Zeff*cbrt(Rho/A)/T;
  double Uion;
  double gamma14=pow(gamma,0.25);
  if (gamma<210.0) {
    Uion=1.5-0.895929*gamma+3225.0*gamma14;
  } else {
    Uion=-0.897744*gamma+0.95043*gamma14+
      0.18956/gamma14-0.81487;
  }
  double Pion=nion*kb*T*(1.0+Uion/3.0);
    
  electron.n=ne;
  electron.m=me;
  fe.calc_density(electron,T);
  double Pel=electron.pr;
    
  return Pel+Pion;
}
  
void nscool_wrap::density(double T, double P, double A, double Z,
			  double &Rho) {

  double eps=1.0e-3;
  double dRho;
  do {
    double Rho0=Rho;
    double Pre0=pressure(T,Rho0,A,Z);
    double Rho1=(1.0+eps)*Rho0;
    double Pre1=pressure(T,Rho1,A,Z);
    double f=Pre0-P;
    double f1=(Pre1-Pre0)/(Rho1-Rho0);
    dRho=-f/f1;
    Rho=Rho0+dRho;
  } while (fabs(dRho/Rho)>=1.0e-5);
    
  return;
}
  
void nscool_wrap::default_star(std::string dir) {
    
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

void nscool_wrap::hhj_eos(double mass) {

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
  
void nscool_wrap::tptr_init
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
  
double nscool_wrap::Teff
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
	
void nscool_wrap::print_temp(int istep, int itprint,
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

void nscool_wrap::print_cv(int itprint, int imax, double *cv,
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
  
void nscool_wrap::num_param(double &time0, double &timemax, int &istepmax,
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

void nscool_wrap::bound_param(int &ifteff, double &eta_arg,
			      double &mag_coeff, double &tb_acc0) {
  ifteff=3;
  eta_arg=eta;
  mag_coeff=3.0;
  tb_acc0=0.0;
  return;
}
  
void nscool_wrap::cool_param(int &pscreen, double &debug, int &istep_debug,
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
  
void nscool_wrap::core_comp
(double *rho_t, double *nbar_t, double *yelect_t, double *ymuon_t,
 double *yneutr_t, double *yprot_t, double *ylambda_t,
 double *ysminus_t, double *yszero_t, double *ysplus_t,
 double *mstp_t, double *mstn_t, double *mstla_t, double *mstsm_t,
 double *msts0_t, double *mstsp_t, int *ix) {

  (*ix)=nscool_core.get_nlines();
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
      (*ix)=0;
      return;
    }
    if (fabs(yprot_t[i]+yneutr_t[i]+ylambda_t[i]+ysminus_t[i]+
	     yszero_t[i]+ysplus_t[i]-1.0)>1.0e-5) {
      std::cerr << "Baryon problem in core_comp." << std::endl;
      std::cout << i << " " << yneutr_t[i] << " "
		<< yprot_t[i] << " " << ylambda_t[i] << " "
		<< ysminus_t[i] << " " << yszero_t[i] << " "
		<< ysplus_t[i] << std::endl;
      (*ix)=0;
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
    
  return;
}

void nscool_wrap::urca_settings(double &durca, double &a_durca,
				double &b_durca) {
  durca=fix_durca;
  a_durca=alpha_durca;
  b_durca=beta_durca;
  return;
}
  
void nscool_wrap::crust_comp
(double *Z_ion_t, double *A_ion_t, double *A_cell_t,
 double *bar_t, double *pres_t, double *rho_t,
 int *jmax) {
    
  *jmax=((int)(nscool_crust.get_nlines()));
  if (nscool_crust.get_nlines()>=500) {
    O2SCL_ERR("Crust table too large in nscool_wrap::crust_comp().",
	      o2scl::exc_einval);
  }
  for(size_t i=0;i<nscool_crust.get_nlines();i++) {
    size_t i2=(*jmax)-1-i;
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
    
  return;
}

void nscool_wrap::crust_eos(double *rho2, double *pres2, int *idata) {

  *idata=((int)(nscool_crust.get_nlines()));
  for(size_t i=0;i<nscool_crust.get_nlines();i++) {
    rho2[i]=nscool_crust.get("rho",(*idata)-1-i);
    pres2[i]=nscool_crust.get("P",(*idata)-1-i);
  }

  return;
}

void nscool_wrap::sf_suppress_data(double *lgtau1, double *lgtau2,
				   double *lgr) {
  
#include "sf_suppression.h"
    
  return;
}
  
void nscool_wrap::pair_brem_data(double *logt, double *nalpha) {

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
  
void nscool_wrap::star_struct(int icore, double rhocore,
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
	       "nscool_wrap::star_struct().",o2scl::exc_einval);
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
  
void nscool_wrap::main_out(double &time, double &tptr,
			   double &lphot, double &lneut, double &lheat,
			   int &stop) {

  if (((int)v_time.size())%main_out_it==(main_out_it-1)) {
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

  stop=0;
  if (v_time.size()>10000) stop=1;
    
  return;
}

void nscool_wrap::write_cool_curve(std::string fname) {
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
  
void nscool_wrap::write_tl_prof(std::string fname) {

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

void nscool_wrap::gaps(int &sfn1s0_arg, double &n1_tc_arg,
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

int nscool_wrap::run(int irank) {
  if (nscool_wrap_ptrs.size()==0) {
    std::cerr << "Object 'nscool_wrap_ptrs' is empty." << std::endl;
    exit(-1);
  }
  if (v_time.size()>0) {
    v_time.clear();
    v_tptr.clear();
    v_lphot.clear();
    v_lneut.clear();
    v_lheat.clear();
  }
  int iret=0;

  if (false) {
    std::cout << "sfn1s0: " << sfn1s0 << std::endl;
    std::cout << "sfn3p2: " << sfn3p2 << std::endl;
    std::cout << "sfp1s0: " << sfp1s0 << std::endl;
    std::cout << "n3_tc: " << n3_tc << std::endl;
    std::cout << "n3_kf: " << n3_kf << std::endl;
    std::cout << "n3_dk: " << n3_dk << std::endl;
    std::cout << "p1_tc: " << p1_tc << std::endl;
    std::cout << "p1_kf: " << p1_kf << std::endl;
    std::cout << "p1_dk: " << p1_dk << std::endl;
    std::cout << "n1_tc: " << n1_tc << std::endl;
    std::cout << "n1_kf: " << n1_kf << std::endl;
    std::cout << "n1_dk: " << n1_dk << std::endl;
    std::cout << "T_fact_drip: " << T_fact_drip << std::endl;
    std::cout << "T_fact_surf: " << T_fact_surf << std::endl;
    std::cout << "ptemp: " << ptemp << std::endl;
    std::cout << "sxrt_mode: " << sxrt_mode << std::endl;
    std::cout << "sxrt_Tinit: " << sxrt_Tinit << std::endl;
    std::cout << "mdot_index: " << mdot_index << std::endl;
    o2scl::vector_out(std::cout,time_print,true);
  }
  
  // nscool_ is a subroutine and thus has no return value
  nscool_(&irank,&iret,pb_logt,pb_nalpha,pb_n2,
	  sf_lgtau1,sf_lgtau2,sf_lgr,sf_lgr2);
  return iret;
};

extern "C" void nscool_gaps_
(int *irank, double *sfn1s0, double *n1tc, double *n1kf, double *n1dk,
 double *sfn3p2, double *n3tc, double *n3kf, double *n3dk, double *sfp1s0,
 double *p1tc, double *p1kf, double *p1dk) {
  
  // The superfluid switches are actually double-precision
  // numbers in the cooling code, so we convert from int
  // to double here.
  int isfn3p2, isfp1s0, isfn1s0;
  nscool_wrap_ptrs[*irank]->gaps(isfn1s0,*n1tc,*n1kf,*n1dk,
				 isfn3p2,*n3tc,*n3kf,*n3dk,
				 isfp1s0,*p1tc,*p1kf,*p1dk);
				 
  *sfn3p2=((double)isfn3p2);
  *sfp1s0=((double)isfp1s0);
  *sfn1s0=((double)isfn1s0);
  return;
};

/// Main output function
extern "C" void nscool_main_out_(int *irank, double *time, double *tptr,
				 double *lphot, double *lneut, double *lheat,
				 int *stop) {
  nscool_wrap_ptrs[*irank]->main_out(*time,*tptr,*lphot,*lneut,*lheat,*stop);
  //std::cout << "main_out" << std::endl;
  return;
};

/// Obtain stellar structure
extern "C" void nscool_star_struct_
(int *irank, int *icore,
 double *rhocore, double *rad_t, double *bar_t, double *rho_t,
 double *pres_t, double *emas_t, double *phi_t, double *rad,
 int *jmax, int *jcore, double *w1, double *w2) {
  nscool_wrap_ptrs[*irank]->star_struct(*icore,*rhocore,rad_t,bar_t,rho_t,
					pres_t,emas_t,phi_t,rad,
					jmax,jcore,w1,w2);
  return;
};

/// Obtain core composition
extern "C" void nscool_core_comp_
(int *irank, double *rho_t, double *nbar_t, double *yelect_t, 
 double *ymuon_t, double *yneutr_t, double *yprot_t, double *ylambda_t,
 double *ysminus_t, double *yszero_t, double *ysplus_t,
 double *mstp_t, double *mstn_t, double *mstla_t, double *mstsm_t,
 double *msts0_t, double *mstsp_t, int *ix) {
  nscool_wrap_ptrs[*irank]->core_comp(rho_t,nbar_t,yelect_t,ymuon_t,
				      yneutr_t,yprot_t,ylambda_t,ysminus_t,
				      yszero_t,ysplus_t,mstp_t,mstn_t,mstla_t,
				      mstsm_t,msts0_t,mstsp_t,ix);
  return;
};

/// Obtain crust composition
extern "C" void nscool_crust_comp_
(int *irank, double *Z_ion_t, double *A_ion_t, double *A_cell_t,
 double *bar_t, double *pres_t, double *rho_t, int *jmax) {
  nscool_wrap_ptrs[*irank]->crust_comp(Z_ion_t,A_ion_t,A_cell_t,bar_t,
				       pres_t,rho_t,jmax);
  return;
};

/// Obtain crust EOS
extern "C" void nscool_crust_eos_(int *irank, double *rho2, double *pres2,
				  int *idata) {
  nscool_wrap_ptrs[*irank]->crust_eos(rho2,pres2,idata);
  return;
};

/// Set numerical parameters
extern "C" void nscool_num_param_
(int *irank, double *time0, double *timemax, int *istepmax,
 int *itrial_max, int *itrial_opt, double *tcut,
 double *dtime, double *dtlimit, double *scale_dt0,
 double *scale_dt1, double *repeat, int *istart,
 double *mratt, double *mratl, double *mrats,
 double *tvar, double *svar, double *tcon) {
  
  nscool_wrap_ptrs[*irank]->num_param
    (*time0,*timemax,*istepmax,*itrial_max,*itrial_opt,*tcut,
     *dtime,*dtlimit,*scale_dt0,*scale_dt1,*repeat,*istart,
     *mratt,*mratl,*mrats,*tvar,*svar,*tcon);

  return;
};

/// Set cooling parameters
extern "C" void nscool_cool_param_
(int *irank, int *pscreen, double *debug, int *istep_debug,
 double *pteff, double *ptemp, double *pstar,
 int *idump1, int *idump2, int *idump3,
 double *tempmin, double *tempini,
 int *icvel_nodeg, double *emnco, double *emncr,
 double *emp, double *p0, int *itpmax, double *tprint) {
  
  nscool_wrap_ptrs[*irank]->cool_param
    (*pscreen,*debug,*istep_debug,*pteff,*ptemp,*pstar,
     *idump1,*idump2,*idump3,*tempmin,*tempini,
     *icvel_nodeg,*emnco,*emncr,*emp,*p0,*itpmax,tprint);
  
  return;
};

/// Set initial temperature profile
extern "C" void nscool_tptr_init_(int *irank, int *ifteff, double *tempini,
				  double *ephi_surf, double *ephi_drip,
				  double *ephi_core, double *tsurface,
				  double *tdrip, double *tcore,
				  double *tb_acc0) {
  nscool_wrap_ptrs[*irank]->tptr_init(*ifteff,*tempini,*ephi_surf,*ephi_drip,
				      *ephi_core,*tsurface,*tdrip,
				      *tcore,*tb_acc0);
  //std::cout << "tempini: " << *tempini << std::endl;
  return;
};

/// Set atmosphere parameters
extern "C" double nscool_teff_(int *irank, double *Tb, int *ifteff,
			       double *eta, double *bfield, int *istep,
			       double *time, double *Ts1, double *Ts2,
			       double *Z, double *A, double *Rho,
			       int *debug, double *gs14,
			       double *compactness) {
  return nscool_wrap_ptrs[*irank]->Teff(*Tb,*ifteff,*eta,*bfield,*istep,
					*time,*Ts1,*Ts2,*Z,*A,*Rho,*debug,
					*gs14,*compactness);
};

extern "C" void nscool_bound_param_(int *irank, int *ifteff, double *eta,
				    double *mag_coeff, double *tb_acc0) {
  nscool_wrap_ptrs[*irank]->bound_param(*ifteff,*eta,*mag_coeff,*tb_acc0);
  return;
};

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
 double *qpbf_n3p2, double *qpbf_q) {
  
  nscool_wrap_ptrs[*irank]->print_temp
    (*istep,*itprint,*time,*t_effective,*imax,*w1,*w2,otemp,temp,
     olum,lum,rad,rrho,ephi,dvol,e2phi,tcn,tcp,qnu,qeebrem,qnpb,qplasma,
     qsynch,qbubble,qpair,qphoto,qbrem_nn,qmurca_nucl,qbrem_nucl,
     qmurca_hyp,qbrem_hyp,qdurca_np,qdurca_lap,qdurca_smn,qdurca_smla,
     qdurca_sms0,qfast,qdurca_q,qmurca_q,qpbf_n1s0,qpbf_p1s0,qpbf_n3p2,
     qpbf_q);
  return;
};

/// Print specific heats
extern "C" void nscool_print_cv_
(int *irank, int *itprint, int *imax, double *cv, double *cv_n, double *cv_p, 
 double *cv_e, double *cv_m, double *cv_la, double *cv_sm, double *cv_s0, 
 double *cv_sp, double *cv_q) {
  nscool_wrap_ptrs[*irank]->print_cv
    (*itprint,*imax,cv,cv_n,cv_p,cv_e,
     cv_m,cv_la,cv_sm,cv_s0,cv_sp,cv_q);
  return;
};

/// Handle Urca settings
extern "C" void nscool_urca_settings_(int *irank, double *fix_durca,
				      double *alpha_durca,
				      double *beta_durca) {
  nscool_wrap_ptrs[*irank]->urca_settings(*fix_durca,*alpha_durca,
					  *beta_durca);
  return;
};

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
			       int *isf) {

  /*
    std::cout << "H2: " << *icore << " " << *sfn1s0 << " "
    << *fn1s0 << std::endl;
    std::cout << "H2: " << *icore << " " << *sfp1s0 << " "
    << *fp1s0 << std::endl;
  */
  
  nscool_wrap_ptrs[*irank]->atc.get_tc(*imax,*idrip,*icore,
				       *sfn1s0,*n1tc,*n1kf,*n1dk,
				       *sfp1s0,*p1tc,*p1kf,*p1dk,
				       *sfn3p2,*n3tc,*n3kf,*n3dk,
				       *fn1s0,*fp1s0,*fn3p2,
				       kf_n,kf_p,kf_la,kf_sm,kf_s0,kf_sp,
				       kf_u,kf_d,kf_s,tc_n,tc_p,tc_la,
				       tc_uu,tc_dd,tc_ss,tc_ud,tc_us,
				       tc_ds,tc_u,tc_d,tc_s,isf);

  return;
};

extern "C" void nscool_density_(int *irank, double *T, double *P,
				double *A, double *Z, double *Rho) {
  nscool_wrap_ptrs[*irank]->density(*T,*P,*A,*Z,*Rho);
  return;
};
