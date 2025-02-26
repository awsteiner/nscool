#include "data_eval.h"

using namespace std;
using namespace o2scl;

//reverted back to r882 to make runs
int data_eval::ss_acc_solve(double Mdot, double Q_heat, double mass,
			    double rad, double &logT) {
  
  mroot_hybrids<> mh;
  mm_funct func=std::bind(std::mem_fn<int(size_t,const ubvector &,
					  ubvector &,double,double,
					  double,double)>
			  (&data_eval::match_lum),this,
			  std::placeholders::_1,std::placeholders::_2,
			  std::placeholders::_3,Q_heat,mass,rad,Mdot);
  ubvector x(1), y(1);
  x[0]=logT;
  // Ignore solver convergence errors
  mh.err_nonconv=false;
  mh.def_jac.err_nonconv=false;
  // First, ensure initial guess is valid
  //cout << "Im2 " << endl;
  int ret=match_lum(1,x,y,Q_heat,mass,rad,Mdot);
  //cout << "Im1 " << endl;
  // If so, then call solver
  if (ret==0) {
    //cout << "I0 " << endl;
    ret=mh.msolve(1,x,func);
    logT=x[0];
  }
  // If it failed, try a different initial guess
  if (ret!=0) {
    if (verbose>0) {
      cout << "Solver recovery:" << endl;
      cout << "logT: " << logT << endl;
    }
    for(double init=6.0;ret!=0 && init<8.501;init+=0.5) {
      //cout << "I1 " << init << endl;
      x[0]=init;
      // Try to get an initial guess from this value of log T
      ret=match_lum(1,x,y,Q_heat,mass,rad,Mdot);
      //cout << "I2 " << init << endl;
      if (verbose>0) {
        cout << "ret: " << init << " " << ret << endl;
      }
      // If it worked, call the solver
      if (ret==0) {
	//mh.verbose=1;
        //cout << "I3 " << init << endl;
	ret=mh.msolve(1,x,func);
        logT=x[0];
      }
      if (verbose>0) {
        cout << "ret1: " << init << " " << ret << endl;
        //cout << "hehe: " << x[0] << " " << logT << endl;
      }
      //cout << "I4 " << init << endl;
    }
  }
  //cout << "I5 " << endl;
  return ret;
}

int data_eval::match_lum(size_t nv, const ubvector &x, ubvector &y,
			 double Q_heat, double mass, double rad,
			 double Mdot) {
			 
  double logT=x[0];
  double lphot=0.0, lneut=0.0, lheat=0.0;
  //cout << "J1" << endl;
  int ret=acc_compute(logT,Q_heat,mass,rad,Mdot,lphot,lneut,lheat);
  //cout << "J2" << endl;
  // Heat balance equation (eq 1. in Han 2017)
  y[0]=(lphot+lneut-lheat)/lheat;
  return ret;
}

int data_eval::acc_compute(double logT, double Q_heat, double mass,
			   double rad, double Mdot, double &lphot,
			   double &lneut, double &lheat) {
        
  double acc_Tinit=pow(10.0,logT);
    
  // AWS: This function needs to be modified for the OpenMP threaded
  // option. Right now it only runs with one thread.

  // Setting Mdot as the ygrid in our tl_prof tables for SXRTs

  nw.sxrt_mode=true;
  nw.sxrt_Tinit=acc_Tinit;
  //cout << "K1" << endl;
  if (nw.sxrt_mode) {
    nw.time_print[0]=1.1e-12;
    nw.temp_ygrid.clear();
    for(size_t i=0;i<log_mdot_grid.size();i++) {
      nw.temp_ygrid.push_back(log_mdot_grid[i]);
    }
  }
  //cout << "K2" << endl;
#ifndef NO_OPENMP
  size_t i_thread=omp_get_thread_num();
#else
  size_t i_thread=0;
#endif

  //cout << "K3" << endl;
  int ret=nw.run(i_thread);
  //cout << "K4" << endl;
  if (ret!=0) {
    return ret;
  }
  //cout << "K5" << endl;
  lphot=nw.v_lphot[nw.v_lphot.size()-1];
  lneut=nw.v_lneut[nw.v_lneut.size()-1];
  //cout << "K6" << endl;
  //lheat=nw.v_lheat[nw.v_lheat.size()-1];

  // AWS, FIXME: Document 6.03e33 and 1.0e-10. Mdot is in solar
  // masses per year, Q heat is in MeV, lheat should be in erg/s.
  // See eq. 5 in Han's paper, there's a nucleon mass in here
  // somewhere as well. 
  lheat=(Q_heat*Mdot/1.0e-10)*6.03e33*sqrt(1.0-schwarz_km*mass/rad);
  return ret;
}

int data_eval::compute_ss(size_t nv, const ubvector &pars,
			  double &log_weight,
                          std::array<double,ndat> &dat) {

  // Q
  double Q_heat=pars[pvi["Q"]];

  if (verbose>1) {
    cout << "eta          M            Mdot         "
         << "log_T        L_phot" << endl;
  }

#ifdef PLOT_SCRIPTS  
  ofstream myfile;
  myfile.open("o2g_sxrt.scr");
  myfile << "o2graph -set logx 1 -set logy 1 \\" << endl;
#endif
  
  for(size_t j=0;j<6;j++) {
    
    double M=mass_grid[j];
      
    // ---------------------------------------------------------
    // Compute the neutron star structure and store the EOS in a
    // table for the NScool object
    
    int f_ret=nst.fixed(M);
    if (f_ret!=0) {
      if (verbose>0) {
	std::cout << "Fail: Object nstar_cold failed to "
		  << "compute profile of a "
		  << M << " neutron star." << std::endl;
      }
      log_weight=-844.0;
      //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
      return 0;
    }
    
    // Note that this is different from class variable 'tmvsr',
    // which stores the full M-R curve. 
    std::shared_ptr<o2scl::table_units<> > tov_tab=nst.get_tov_results();
    
    if (tov_tab->get_nlines()<5) {
      if (verbose>0) {
	std::cout << "Fail: TOV profile too small." << std::endl;
      }
      log_weight=-831.0;
      //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
      return 0;
    }

    //cout << "H1." << endl;
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
    //cout << "H2." << endl;
    
    for(size_t i=0;i<3;i++) {
      //cout << "H9." << endl;
      nw.eta=pow(10.0,log_eta_grid[i]);
      
      double R=tmvsr.interp("gm",M,"r");
      double logT=8.0;
      for(size_t k=0;k<30;k++) {
        //cout << "H10 " << log_mdot_grid.size() << endl;
        nw.mdot_index=k;
        //cout << "H10b." << k << endl;
	double Mdot=pow(10.0,log_mdot_grid[k]);
        //cout << "H11." << endl;
	int ret1=ss_acc_solve(Mdot,Q_heat,M,R,logT); 
        //cout << "H12." << endl;
	if (ret1!=0) {
	  if (verbose>0) {
	    cout << "SXRT curve failed." << endl;
	  }
	  log_weight=-838.0;
	  //for(size_t ik=0;ik<ndat;ik++) dat[ik]=0.0;
	  return 0;
	}
	double lphot=0.0, lneut=0.0, lheat=0.0;
        //cout << "H3." << endl;
	int iret2=acc_compute(logT,Q_heat,M,R,Mdot,
			      lphot,lneut,lheat);
        //cout << "H4." << endl;

	if (verbose>1) {
	  cout << nw.eta << " " << M << " " << Mdot << " "
	       << logT << " " << lphot << endl;

#ifdef PLOT_SCRIPTS  
	  myfile << "-error-point " << Mdot << " " << lphot
		 << " 0 0 \\" << endl;
#endif
	}
        //cout << "H5." << endl;
	sxrt_lum.set(i,j,k,lphot);
        //cout << "H6." << endl;
	if (iret2!=0) {
	  if (verbose>0) {
	    cout << "SXRT curve failed (2)." << endl;
	  }
	  log_weight=-839.0;
	  //for(size_t ik=0;ik<ndat;ik++) dat[ik]=0.0;
	  return 0;
	}
        //cout << "H8." << endl;
      }
      // Writing tl_prof data in a file
      if(w_tl_prof){
        //nw.write_tl_prof(((std::string)"SXRT_")+o2scl::szttos(i)+"_"+
        //                 o2scl::szttos(j));
      }
      //cout << "H7." << endl;
    } 
  }
#ifdef PLOT_SCRIPTS  
  myfile << "-modax labelsize=30 \\" << endl;
  myfile << "-ylimits 29 35 -xlimits -15 -8 \\" << endl;
  myfile << "-set font 30 -xtitle '$lg \\dot{M}$' -ytitle "
	 << "'$lg L_{\\gamma}^{\\infty} [ergs/s]$' \\" << endl;
  myfile << "-show" << endl;
  
  myfile.close();
#endif
  // Clearing temp_ygrid and getting ready for next run
  nw.temp_ygrid.clear();
  nw.time_print[0]=1.0e-10;
  for (size_t i=0;i<nw.time_print.size();i++) {
    nw.temp_ygrid.push_back(nw.time_print[i]);
  }
  return 0;
}

void data_eval::copy_ss(std::array<double,ndat> &dat) {
  for(size_t ieta=0;ieta<3;ieta++) {
    for(size_t imass=0;imass<6;imass++) {
      for(size_t i=0;i<30;i++) {
        dat[dvi[((string)"sL_")+o2scl::szttos(ieta)+"_"+
                o2scl::szttos(imass)+"_"+o2scl::szttos(i)]]=
          sxrt_lum.get(ieta,imass,i);
      }
    }
  }
  return;
}

void data_eval::no_sxrt_fun(std::array<double,ndat> &dat) {
  for (size_t i=0;i<sd.list.size();i++) {
    sxrt &c=sd.list[i];
    dat[dvi[((string)"log_wgt_")+c.name]]=0.0;
  }
  for(size_t ieta=0;ieta<3;ieta++) {
    for(size_t imass=0;imass<6;imass++) {
      for(size_t i=0;i<30;i++) {
        dat[dvi[((std::string)"sL_")+o2scl::szttos(ieta)+"_"+
                o2scl::szttos(imass)+"_"+o2scl::szttos(i)]]=0.0;
      }
    }
  }
  return;
}

int data_eval::sxrt_like(const ubvector &pars,
			 double &log_weight, std::array<double,ndat> &dat) {

  // New bool parameter to toggle using the new likelihood function. If
  // set to true, it will use the new likelihood function; if false will
  // resort back to the old method in Han, Steiner (2017).
  bool new_like=true;
  
  double mdot1, L1;
  double wgt_sxrt=1.0;

  /*
  // Extended log_mdot grid we use to interpolate later. 

  std::vector<double> log_mdot_ext_grid;
  o2scl::uniform_grid<double>
  log_mdot_ext=o2scl::uniform_grid_end<double>(-12,-9,299);
  
  for (size_t m=0;m<300;m++) {
  log_mdot_ext_grid.push_back(log_mdot_ext[m]);
  }

  // Bool to interpolate and find out mass and eta combination for 
  // lowest luminosity from heating curves
  bool guess_sxrt=false;

  if (guess_sxrt) {
  std::vector<double> m_ext_grid,eta_ext_grid;
  o2scl::uniform_grid<double>
  m_ext=o2scl::uniform_grid_end<double>(1,dat[dvi["Mns_max"]],99);

  o2scl::uniform_grid<double>
  eta_ext=o2scl::uniform_grid_end<double>(-17,-7,99);

  for (size_t m=0;m<100;m++) {
  m_ext_grid.push_back(m_ext[m]);
  eta_ext_grid.push_back(eta_ext[m]);
  }
    
  double guess_min_lum=1e99, guess_mass=1e99, guess_eta=1e99;

  for (size_t i=0; i<100;i++) {
  for (size_t j=0;j<100;j++ ) {
  double int_lum=sxrt_lum.interp_linear
  (eta_ext_grid[i],m_ext_grid[j],
  -12);

  if (int_lum<guess_min_lum) {
  guess_min_lum=int_lum;
  guess_mass=m_ext_grid[j];
  guess_eta=eta_ext_grid[i];
  }
  }
  }
  std::cout << "Mass,eta,lum: " << guess_mass << "  " 
  << guess_eta << "  " << guess_min_lum << endl;
  }*/

  // Vector to store interpolated luminosity from the cooling curves.
  // units in normalized log scale
  std::vector<double> L_hat_curve(log_mdot_grid.size());

  bool debug=false;
  if (verbose>=2) debug=true;
  
  for (size_t i=0;i<sd.list.size();i++) {

    sxrt &c=sd.list[i];

    if (pars[pvi[((string)"mf_")+c.name]]>1.0) {
      if (verbose>0) {
	cout << "Fail: Mass of " << c.name
             << "greater than Mns_max = " << dat[dvi["Mns_max"]] << endl;
      }
      log_weight=-811.0;
      //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
      return 0;
    }
    
    if (c.mdot_ul) {
      mdot1=pow(10.0,pars[pvi[((string)"log10_mdot_")+c.name]]);
      if (mdot1>c.mdot) {
        O2SCL_ERR("Mdot greater than upper limit.",o2scl::exc_einval);
      }
    } else {
      mdot1 = c.mdot;
    }
      
    L1 = c.L;

    // Normalized log luminosity from the data points
    double L_hat=0.25*(log10(L1)-30.0);
    double curve_sum=0.0;
    double exp_lum, exp_mdot, sqrt_fac;

    if (new_like) {
      
      // Mass of the star in solar masses
      double masst=pars[pvi[(std::string("mf_"))+c.name]]*
        (dat[dvi["Mns_max"]]-1.0)+1.0;
      
      dat[dvi[((string)"Rns_")+c.name]]=
        tmvsr.interp("gm",masst,"r");

      if (c.L_ul) {
	dat[dvi[((string)"L_")+c.name]]=sxrt_lum.interp_linear
	  (pars[pvi[((string)"eta_")+c.name]],masst,log10(mdot1));
      }
      
      if (debug) {
        cout << "Name: " << c.name << " M,R: " << masst << " "
             << dat[dvi[((string)"Rns_")+c.name]]
             << " km, and value of 'eta': " 
             << pars[pvi[((string)"eta_")+c.name]] << endl;
        if (c.mdot_ul) {
          cout << "M_dot is an upper limit." << endl;
          cout << "Upper limit: " << c.mdot << " M_sun/yr" << endl;
          cout << "Current value: "
	       << pars[pvi[((string)"log10_mdot_")+c.name]]
               << " M_sun/yr" << endl;
        }
        if (c.L_ul) {
          cout << "Luminosity is an upper limit." << endl;
          cout << "Upper limit: " << c.L << " ergs/s" << endl;
        }
      }
      
      if(debug) {
        cout << "  i Mdot         exp_mdot     "
             << "lum(ergs/s)  exp_lum      sqrt_fac     "
             << "curve_sum" << endl;
      }
      
      for (size_t j=0;j<L_hat_curve.size(); j++) {

        // Gaussian in Luminosity calculation

        // Luminosity in erg/s
        double lum=sxrt_lum.interp_linear
          (pars[pvi[((string)"eta_")+c.name]],masst,log_mdot_grid[j]);
        
        // Check if luminosity data is upper limit and only take 
        // contributions when interpolated luminosity is lower than 
        // the upper limit

        if (c.L_ul && lum>c.L) {
          exp_lum=0.0;
        } else if (c.L_ul && lum<c.L) {
          exp_lum=1.0;
        } else {
          L_hat_curve[j]=0.25*(log10(lum)-30.0);
          double exp_temp=(log10(lum)-log10(L1))/0.5;
          exp_lum=exp(-pow(exp_temp,2.0)/2.0);
        }

        // Gausian in Mdot calculation

        // Check if mdot data is upper limit and only take contributions
        // for mdots lower than the upper limit

        if (c.mdot_ul && pow(10,log_mdot_grid[j])>c.mdot) {
          exp_mdot=0.0;
        } else if (c.mdot_ul && pow(10,log_mdot_grid[j])<c.mdot) {
          exp_mdot=1.0;
        } else {
          double exp_temp2=(log10(mdot1)-log_mdot_grid[j])/0.5;
          exp_mdot=exp(-pow(exp_temp2,2.0)/2.0);
        }

        // Sqrt factor calculation
        if (j==0) {
          sqrt_fac=1.0;
        } else{
          double deriv_star=(L_hat_curve[j]-L_hat_curve[j-1])/
            (log_mdot_grid[j]-log_mdot_grid[j-1]);
          sqrt_fac=sqrt(1.0+deriv_star*deriv_star);
        }

        curve_sum+=sqrt_fac*exp_mdot*exp_lum;

        if (debug) {
          std::cout.width(3);
          std::cout << j << " " << pow(10,log_mdot_grid[j]) << " ";
          std::cout << exp_mdot << " " << lum << " "
                    << exp_lum << " " << sqrt_fac << " "
                    << curve_sum << endl;
        }

      }

      // Check log_weight. If its -inf for any star, then reject the
      // point.

      if (!std::isfinite(log(curve_sum))) {
        if (verbose>0) {
          std::cout << "Model cooling curves too far from data point "
                    << "for SXRT " << c.name << "." << std::endl;
        }
        log_weight=-845.0;
        //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
        return 0;
      }

      dat[dvi[((string)"log_wgt_")+c.name]]=log(curve_sum);
    
      log_weight+=log(curve_sum);
      if (verbose>1) {
        cout << "log weight for " << c.name << ": "
             << dat[dvi[((string)"log_wgt_")+c.name]] << endl;
        cout << "-------------------------------------" << endl;
      }

    } else {

      // The old likelihood
      
      double masst=pars[pvi[(std::string("mf_"))+c.name]]*
        (dat[dvi["Mns_max"]]-1.0)+1.0;
      double lum=sxrt_lum.interp_linear
        (pars[pvi[((string)"eta_")+c.name]],masst,log10(mdot1));

      // We presume half a decade (a factor of sqrt(10)) uncertainty
      // in all of the luminosities
      double exp_temp=(log10(lum)-log10(L1))/0.5;
    
      dat[dvi[((string)"log_wgt_")+c.name]]=-pow(exp_temp,2.0)/2.0;
      wgt_sxrt*=exp(-pow(exp_temp,2.0)/2.0);

      if (debug || verbose>1) {
        cout << "name,mdot,L,M,eta: ";
        cout.width(5);
        cout << c.name << " " << mdot1 << " "
             << L1 << " " << masst << " "
             << pars[pvi[((string)"eta_")+c.name]] << endl;
        cout << "  lum,log_wgt,wgt_sxrt: " << lum << " "
             << -pow(exp_temp,2.0)/2.0 << " " << wgt_sxrt << endl;
      }
    }
  }
  
  if (!new_like) {
    log_weight+=log(wgt_sxrt);

    if (verbose>0) {
      std::cout << "log_wgt_sxrt,log(wgt): " << log(wgt_sxrt) << " "
                << log_weight << std::endl;
    }
  }

  if (false && debug) exit(-1);
  
  return 0;
}

