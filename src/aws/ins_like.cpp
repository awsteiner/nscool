#include "data_eval.h"

using namespace std;
using namespace o2scl;

void data_eval::no_ins_fun(std::array<double,ndat> &dat) {
  for (size_t i=0;i<id.list.size();i++) {
    cooling_ns &c=id.list[i];
    dat[dvi[((string)"Rns_")+c.name]]=0.0;
    dat[dvi[((string)"log_wgt_")+c.name]]=0.0;
  }
  for(size_t ieta=0;ieta<3;ieta++) {
    for(size_t imass=0;imass<6;imass++) {
      for(size_t i=0;i<ngrid;i++) {
        dat[dvi[((std::string)"cT_")+o2scl::szttos(ieta)+"_"+
                o2scl::szttos(imass)+"_"+o2scl::szttos(i)]]=0.0;
      }
    }
  }
  
  for(size_t ieta=0;ieta<3;ieta++) {
    for(size_t imass=0;imass<6;imass++) {
      for(size_t i=0;i<ngrid;i++) {
        dat[dvi[((std::string)"cL_")+o2scl::szttos(ieta)+"_"+
                o2scl::szttos(imass)+"_"+o2scl::szttos(i)]]=0.0;
      }
    }
  }
  return;
}

int data_eval::ins_like(size_t nv, const ubvector &pars,
			double &log_weight, std::array<double,ndat> &dat) {

  for (size_t i=0;i<id.list.size();i++) {
    cooling_ns &c=id.list[i];
    if (pars[pvi[((string)"mf_")+c.name]]>1.0) {
      if (verbose>0) {
	cout << "Fail: Mass of " << c.name << " "
             << " is greater than Mns_max = "
             << dat[dvi["Mns_max"]] << " ." << endl;
      }
      log_weight=-851.0;
      return 0;
    }
  }
  
  // Dimensionless time grid
  std::vector<double> dimless_times(100);
  for(int i=0;i<100;i++) {
    dimless_times[i]=0.2*log10(0.01*t_grid[i]);
  }
      
  // --------------------------------------------------------------
  // Compute the cooling curves
  
    nw.sxrt_mode=false;
    nw.time_print[0]=1.0e-10;
    //nw.ptemp=2.0;
    
    for(size_t ieta=0;ieta<3;ieta++) {
      for(size_t imass=0;imass<6;imass++) { 
	
	double ns_mass=mass_grid[imass];
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
	
	// ---------------------------------------------------------
	// Compute the neutron star structure and store the EOS in a
	// table for the NScool object

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

	// Note that this is different from class variable 'tmvsr',
	// which stores the full M-R curve. 
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
	
	// --------------------------------------------------------------
	// Compute the cooling curve 

	// Control the number of timesteps between output. A
	// large number effectively specifies no output.
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
	
        // Write tl_prof in a file
        if (w_tl_prof) {
	  nw.write_tl_prof(((std::string)"INS_")+o2scl::szttos(ieta)+"_"+
                           o2scl::szttos(imass));
        }
	
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

	// Interpolate the temperature and luminosity from the cooling
	// calculation into the tensor_grid objects
	for(int k=0;k<100;k++) {
          itp.set(nw.v_time.size(),nw.v_time,nw.v_tptr);
	  double interp_tptr=itp.eval(t_grid[k]);
	  nscool_tptr.set(ieta,imass,k,interp_tptr);
          itp.set(nw.v_time.size(),nw.v_time,nw.v_lphot);
	  double interp_lum=itp.eval(t_grid[k]);
	  nscool_lum.set(ieta,imass,k,interp_lum);
	}	
      }
    }

  // --------------------------------------------------------------
  // Copy cooling curves to dat[] array
  
  for(size_t ieta=0;ieta<3;ieta++) {
    for(size_t imass=0;imass<6;imass++) {
      for(size_t i=0;i<ngrid;i++) {
        dat[dvi[((std::string)"cT_")+o2scl::szttos(ieta)+"_"+
                o2scl::szttos(imass)+"_"+o2scl::szttos(i)]]=
          nscool_tptr.get(ieta,imass,i);
      }
    }
  }
  
  for(size_t ieta=0;ieta<3;ieta++) {
    for(size_t imass=0;imass<6;imass++) {
      for(size_t i=0;i<ngrid;i++) {
        dat[dvi[((std::string)"cL_")+o2scl::szttos(ieta)+"_"+
                o2scl::szttos(imass)+"_"+o2scl::szttos(i)]]=
          nscool_lum.get(ieta,imass,i);
      }
    }
  }
      
  // --------------------------------------------------------------
  // Compute the contribution to the log weight for each star

  // Lower and upper limit of ix_time, thats different for stars with
  // exact age
  int ix_ll, ix_ul;

  // Vector to store the dimensionless luminosity from the
  // cooling curve. Not all values in this array are always
  // computed for all stars.
  std::vector<double> L_hat_curve(ngrid);

  // Vector to store star sum of each star
  std::vector<double> log_eta_values(id.list.size());
  std::vector<double> mass_values(id.list.size());	
  
  for (size_t i=0;i<id.list.size();i++) {

    cooling_ns &c=id.list[i];
    
    mass_values[i]=pars[pvi[(std::string("mf_"))+c.name]]*
      (dat[dvi["Mns_max"]]-1.0)+1.0;
    dat[dvi[((string)"Rns_")+c.name]]=
      tmvsr.interp("gm",mass_values[i],"r");

    if (debug_ins) {
      cout << "Name: " << c.name << " M,R: " << mass_values[i] << " "
           << dat[dvi[((string)"Rns_")+c.name]]
           << " km, and value of 'eta': " << c.eta << endl;
    }

    if (c.eta) {
      log_eta_values[i]=pars[pvi[(std::string("eta_"))+c.name]];
      if (debug_ins) {
        cout << "Eta is: " << log_eta_values[i] << endl;
      }
    } else {
      log_eta_values[i]=-17.0;
    }

    bool age_upper_limit=false;
    bool lum_upper_limit=false;
    bool new_like=true;
    
    if (c.tk==0.0 && c.tk_lo==0.0) {
      age_upper_limit=true;
      // In this case, only the spin-down age is known and it
      // is treated as an upper limit. 
      c.t_hat=0.2*(log10(pars[pvi[((string)"t_")+c.name]])-2.0);
      if (debug_ins) {
        cout << "Age is only an upper limit." << endl;
        cout << "Upper limit: " << c.tc << " yr" << endl;
        cout << "Current value: " << pars[pvi[((string)"t_")+c.name]]
             << " yr" << endl;
      }
      if (pars[pvi[((string)"t_")+c.name]]>c.tc) {
        O2SCL_ERR("Age greater than upper limit.",o2scl::exc_einval);
        exit(-1);
      }
    }

    if (c.L == 0.0 && c.L_lo==0.0) {
      lum_upper_limit=true;
      //c.L_hat=0.25*(log10(pars[pvi[((string)"L_")+c.name]])+3.0);
      if (debug_ins) {
        cout << "Luminosity is an upper limit." << endl;
        cout << "Upper limit: " << c.L_hi*1.0e33 << " erg/s" << endl;
        //cout << "Current value: " << pars[pvi[((string)"L_")+c.name]]*1.0e33
        //     << " erg/s" << endl;
        //cout << "Current value: " << pvi[((string)"L_")+c.name] << " "
        //     << pars.size() << endl;
      }
      //if (pars[pvi[((string)"L_")+c.name]]>c.L_hi) {
      //  O2SCL_ERR("Luminosity greater than upper limit.",o2scl::exc_einval);
      //  exit(-1);
      //}
    }

    if (new_like) {

      double dt_grid=dimless_times[1]-dimless_times[0];
      bool accurate_age=false;
    
      if (c.dt_hat>0.0 && dt_grid>c.dt_hat) {
        accurate_age=true;
        if (debug_ins) {
          cout << "Age is more accurate than grid." << endl;
        }
      }

      if (accurate_age || age_upper_limit) {
        // Find the time interval which encloses the exact age
        int j = 0;
        while (c.t_hat > dimless_times[j]) {
          j++;
        }
        ix_ll=j-1;
        ix_ul=j;
      } else {
        ix_ll=0;
        ix_ul=ngrid;
      }

      if (debug_ins) {
        cout << "name,ll,ul: " << c.name << " " << ix_ll << " " << ix_ul
             << endl;
      }

      if (debug_ins) {
        if (!lum_upper_limit) {
          cout << "t (yr), L (erg/s): " << pow(10.0,5.0*c.t_hat+2.0) << " "
               << pow(10.0,4.0*c.L_hat+30.0) << endl;
          cout << "t_hat, dt_hat, L_hat, dL_hat: "
               << c.t_hat << " " << c.dt_hat << " "
               << c.L_hat << " " << c.dL_hat << endl;
          cout << "ix time         exp_time     lum_curve    exp_lum      " 
               << "sqrt_fac     curve_sum" << endl;
        }
      }

      double curve_sum=0.0;

      if (lum_upper_limit) {
        
        double lum_interp1=nscool_lum.interp_linear
          (log_eta_values[i],mass_values[i],pow(10.0,c.t_hat*5.0+2.0));
        dat[dvi[((string)"L_")+c.name]]=lum_interp1;
	if (verbose>0) {
	  cout << "Current L for " << c.name << " is: "
	       << dat[dvi[((string)"L_")+c.name]]
	       << " erg/s" << endl;
	}
        if (lum_interp1>c.L_hi*1e33) {
          if (verbose>0) {
            std::cout << "Luminosity higher than the upper limit "
                      << "for NS " << c.name << "." << std::endl;
            cout << "log_eta,mass: "
                 << log_eta_values[i] << " " << mass_values[i] << endl;
            cout << "interp lum, upper limit: "
                 << lum_interp1 << " " << c.L_hi*1.0e33 << endl;
          }
          log_weight=-847.0;
          return 0;
        }
        curve_sum=1.0;
        
      } else {
        
        if (ix_ll<0 || ix_ul>((int)t_grid.size())) {
          cout << c.name << " " << accurate_age << " "
               << age_upper_limit << endl;
          cout << ix_ll << " " << ix_ul << " " << t_grid.size()
               << " " << ngrid << endl;
          cout << c.t_hat << " " << dimless_times[0] << " "
               << dimless_times[dimless_times.size()-1] << " "
               << dimless_times.size() << endl;
          
          O2SCL_ERR("Indexing problem in ins_like.cpp",o2scl::exc_esanity);
        }
        
        for (int ix_time = ix_ll; ix_time < ix_ul; ix_time++) {
          
          // These values shouldn't matter, so I'm temporarily
          // setting it to 1.0e99 to make sure.
          double exp_lum=1.0e99, exp_time=1.0e99, sqrt_fac=1.0e99;

          // If false, then the extrapolated luminosity of the
          // cooling curve is negative, so we skip those intervals
          bool lum_positive=true;

          // The luminosity from the cooling curve (in erg/s)
          double lum_curve=nscool_lum.interp_linear
            (log_eta_values[i],mass_values[i],t_grid[ix_time]);

          if (lum_curve<0.0) {
            lum_positive=false;
            // This value shouldn't matter, so I'm temporarily
            // setting it to 1.0e99 to make sure.
            
            //L_hat_curve[ix_time]=-1.0e99;
            L_hat_curve[ix_time]=L_hat_curve[ix_time-1];
          } else {
            L_hat_curve[ix_time]=0.25*(log10(lum_curve)-30.0);
          }

          // If first point in the cooling curve is below the luminosity
          // stored in c.L_hat for an upper limit star, then the entire
          // cooling curve will be below the c.L_hat and thus the
          // probability is zero.
          /*
            if (lum_upper_limit && ix_time==0 &&
            c.L_hat>L_hat_curve[0]) {
            if (verbose>0) {
            std::cout << "Model cooling curve is always below "
            << "star's luminosity for NS "
            << c.name << "." << std::endl;
            }
            log_weight=-810.0;
            for(size_t k=0;k<ndat;k++) dat[k]=0.0;
            return 0;
            }*/

          if (lum_positive) {
            double lum_interp=1.0e99;
            double lum_next=nscool_lum.interp_linear
              (log_eta_values[i],mass_values[i],t_grid[ix_time+1]);
          
            L_hat_curve[ix_time+1]=0.25*(log10(lum_next)-30.0);

            //

            // Square root factor calculation

            if (accurate_age || age_upper_limit) {
              double deriv_star=(L_hat_curve[ix_time+1]-L_hat_curve[ix_time])/
                (dimless_times[ix_time+1]-dimless_times[ix_time]);
              sqrt_fac=sqrt(1.0+deriv_star*deriv_star);
            } else {
              if (ix_time==0) {
                sqrt_fac=1.0;
              } else {  
                double deriv_star=(L_hat_curve[ix_time]-
				   L_hat_curve[ix_time-1])/
                  (dimless_times[ix_time]-dimless_times[ix_time-1]);
                sqrt_fac=sqrt(1.0+deriv_star*deriv_star);
              }
            }

            // Gaussian in luminosity

            if (!accurate_age && !age_upper_limit) {
              exp_lum=exp(-pow(L_hat_curve[ix_time]-c.L_hat,2.0)/
                          (2.0*c.dL_hat*c.dL_hat));
            } else {
              lum_interp=nscool_lum.interp_linear
                (log_eta_values[i],mass_values[i],pow(10.0,c.t_hat*5.0+2.0));
              double L_hat_interp=0.25*(log10(lum_interp)-30.0);
            
              // Use the age to interpolate the luminosity from the cooling
              // curve
              exp_lum=exp(-pow(L_hat_interp-c.L_hat,2.0)/
                          (2.0*c.dL_hat*c.dL_hat));
            }

            // Gaussian in time

            if(!accurate_age) {
              if (age_upper_limit && t_grid[ix_time]>c.tc) {
                exp_time=0.0;
              } else if (age_upper_limit && t_grid[ix_time]<c.tc) {
                exp_time=1.0;
              } else {
                exp_time=exp(-pow(dimless_times[ix_time]-c.t_hat,2.0)/
                             (2.0*c.dt_hat*c.dt_hat));
              }
            } else {
              exp_time=1.0;
            }

            curve_sum+=sqrt_fac*exp_time*exp_lum;

            if (debug_ins) {
              std::cout.width(2);
              if (accurate_age || age_upper_limit) {
                std::cout << ix_time << " "
			  << pow(10.0,c.t_hat*5.0+2.0) << " ";
                std::cout << exp_time << " " << lum_interp << " ";
              } else {
                std::cout << ix_time << " " << t_grid[ix_time] << " ";
                std::cout << exp_time << " " << lum_curve << " ";
              }
              std::cout << exp_lum << " " << sqrt_fac << " "
                        << curve_sum << endl;
            }
          }
        }
      }

      if (!std::isfinite(log(curve_sum))) {
        if (verbose>0) {
          std::cout << "Model cooling curves too far from data point "
                    << "for NS " << c.name << "." << std::endl;
        }
        log_weight=-848.0;
        //for(size_t k=0;k<ndat;k++) dat[k]=0.0;
        return 0;
      }
    
      dat[dvi[((string)"log_wgt_")+c.name]]=log(curve_sum);
    
      log_weight+=log(curve_sum);
      if (verbose>1) {
        cout << "log weight for " << c.name << ": "
             << dat[dvi[((string)"log_wgt_")+c.name]] << endl;
        if (debug_ins) {
          cout << "--------------------------------------"
               << "--------------------------------------" << endl;
          //char ch;
          //cin >> ch;
        }
      }
    } else {
      // ------------------------------------------------------------------
      // Determine if the age is known with an accuracy smaller than the
      // time grid, i.e. if c.dt_hat is non-zero but smaller than
      // dt_grid

      double dt_grid=dimless_times[1]-dimless_times[0];

      /*
        AWS: 2/12/21 this code just checks the new and old method
        of determining if the age is accurately measured. 

        cout.width(8);
        if (c.tk_hi != 0.0 && (c.tk_hi - c.tk_lo) < 2*t_grid[1]) {
        cout << c.name << " ZZZ 1 ";
        } else {
        cout << c.name << " ZZZ 0 ";
        }
      
        // If tk_lo is zero, then the age accuracy is not specified
        if (c.dt_hat>0.0 && dt_grid>c.dt_hat) {
        cout << "1 " << c.tc << " " << c.tk_lo << " " << c.tk << " "
        << c.tk_hi << " " << c.dt_hat << " " << dt_grid << endl;
        } else {
        cout << "0 " << c.tc << " " << c.tk_lo << " " << c.tk << " "
        << c.tk_hi << " " << c.dt_hat << " " << dt_grid <<  endl;
        }

      */

      bool accurate_age=false;
    
      if (lum_upper_limit==false && c.dt_hat>0.0 && dt_grid>c.dt_hat) {
        accurate_age=true;
        if (debug_ins) {
          cout << "Age is more accurate than grid." << endl;
        }
      }

      ix_ll=0;
      ix_ul=ngrid;

      // If the age is very accurate, or only an upper limit, then
      // we only need to consider the dimensionless time interval
      // around c.t_hat, so adjust ix_ll and ix_ul accordingly.
      // We have one star, J0007, which is a luminosity upper limit,
      // but it's age is determined quite accurately, so for this
      // star we treat it with the luminosity upper limit code below.
    
      if (!lum_upper_limit && (accurate_age || age_upper_limit)) {
        // Find the time interval which encloses the exact age
        int j = 0;
        while (c.t_hat > dimless_times[j]) {
          j++;
        }
        ix_ll=j-1;
        ix_ul=j;
      }

      if (debug_ins) {
        cout << "name,ll,ul: " << c.name << " " << ix_ll << " " << ix_ul
             << endl;
      }

      if (debug_ins) {
        cout << "t (yr), L (erg/s): " << pow(10.0,5.0*c.t_hat+2.0) << " "
             << pow(10.0,4.0*c.L_hat+30.0) << endl;
        cout << "t_hat, dt_hat, L_hat, dL_hat: "
             << c.t_hat << " " << c.dt_hat << " "
             << c.L_hat << " " << c.dL_hat << endl;
        cout << "ix time         exp_time     lum_curve    exp_lum      " 
             << "sqrt_fac     curve_sum" << endl;
      }

      // If the age is an upper limit, then we get the age from the
      // parameters and look up the luminosity from the cooling curve.
      // We then compare that luminosity with the measured one
      // and include the sqrt factor.

      // If the luminosity is an upper limit, we get L from the
      // parameters and look up the age from the cooling curve.
      // We then compare that age with the measured one and include
      // the sqrt factor

      double curve_sum=0.0;

      for (int ix_time = ix_ll; ix_time < ix_ul; ix_time++) {
      
        // These values shouldn't matter, so I'm temporarily
        // setting it to 1.0e99 to make sure.
        double exp_lum=1.0e99, exp_time=1.0e99, sqrt_fac=1.0e99;

        // If false, then the extrapolated luminosity of the
        // cooling curve is negative, so we skip those intervals
        bool lum_positive=true;

        // The luminosity from the cooling curve (in erg/s)
        double lum_curve=nscool_lum.interp_linear
          (log_eta_values[i],mass_values[i],t_grid[ix_time]);
      
        if (lum_curve<0.0) {
          lum_positive=false;
          // This value shouldn't matter, so I'm temporarily
          // setting it to 1.0e99 to make sure.
          L_hat_curve[ix_time]=-1.0e99;
        } else {
          L_hat_curve[ix_time]=0.25*(log10(lum_curve)-30.0);
        }

        // If first point in the cooling curve is below the luminosity
        // stored in c.L_hat for an upper limit star, then the entire
        // cooling curve will be below the c.L_hat and thus the
        // probability is zero.
        if (lum_upper_limit && ix_time==ix_ll &&
            c.L_hat>L_hat_curve[ix_time]) {
          if (verbose>0) {
            std::cout << "Model cooling curve is always below "
                      << "star's luminosity for NS "
                      << c.name << "." << std::endl;
          }
          log_weight=-849.0;
          return 0;
        }
      
        if (lum_positive) {

          if (!lum_upper_limit && (accurate_age || age_upper_limit)) {
          
            // Stars which have a t upper limit or a very accurately
            // measured age
          
            double lum_next=nscool_lum.interp_linear
              (log_eta_values[i],mass_values[i],t_grid[ix_time+1]);
          
            L_hat_curve[ix_time+1]=0.25*(log10(lum_next)-30.0);
          
            double deriv_star=(L_hat_curve[ix_time+1]-L_hat_curve[ix_time])/
              (dimless_times[ix_time+1]-dimless_times[ix_time]);
            sqrt_fac=sqrt(1.0+deriv_star*deriv_star);

            exp_time=1.0;

            // Use the age to interpolate the luminosity from the cooling
            // curve
            double lum_interp=nscool_lum.interp_linear
              (log_eta_values[i],mass_values[i],pow(10.0,c.t_hat*5.0+2.0));

            if (debug_ins) {
              cout << "dimless_times: "
                   << dimless_times[ix_time] << " " << c.t_hat << " "
                   << dimless_times[ix_time+1] << endl;
              cout << "lum_curve,lum_interp,lum_next: " << lum_curve << " "
                   << lum_interp << " " << lum_next << endl;
            }          
            double L_hat_interp=0.25*(log10(lum_interp)-30.0);
          
            exp_lum=exp(-pow(L_hat_interp-c.L_hat,2.0)/
                        (2.0*c.dL_hat*c.dL_hat));

            curve_sum+=sqrt_fac*exp_lum;

            if (debug_ins) {
              std::cout.width(2);
              std::cout << ix_time << " " << t_grid[ix_time] << " ";
              std::cout << exp_time << " " << lum_curve << " "
                        << exp_lum << " " << sqrt_fac << " "
                        << curve_sum << endl;
            }
          
          } else {

            // For stars which have a luminosity upper limit, choose to
            // include only the interval for which the cooling curve
            // encloses the value of c.L_hat determined from the
            // parameters above

            if (lum_upper_limit && ix_time>ix_ll &&
                L_hat_curve[ix_time-1]>c.L_hat &&
                c.L_hat>L_hat_curve[ix_time]) {

              // Stars which have a Luminosity upper limit

              double deriv_star=(L_hat_curve[ix_time]-L_hat_curve[ix_time-1])/
                (dimless_times[ix_time]-dimless_times[ix_time-1]);
              sqrt_fac=sqrt(1.0+deriv_star*deriv_star);

              exp_lum=1.0;

              double t_hat_interp=dimless_times[ix_time-1]+
                (dimless_times[ix_time]-dimless_times[ix_time-1])*
                (c.L_hat-L_hat_curve[ix_time-1])/
                (L_hat_curve[ix_time]-L_hat_curve[ix_time-1]);

              if (debug_ins) {
                cout << "dimless_times: "
                     << dimless_times[ix_time-1] << " " << t_hat_interp << " "
                     << dimless_times[ix_time] << endl;
                cout << "L_hat values: " << L_hat_curve[ix_time-1] << " "
                     << c.L_hat << " " << L_hat_curve[ix_time] << endl;
                cout << "exp: " << t_hat_interp << " " << c.t_hat << " "
                     << c.dt_hat << endl;
              }

              exp_time=exp(-pow(t_hat_interp-c.t_hat,2.0)/
                           (2.0*c.dt_hat*c.dt_hat));

              curve_sum+=sqrt_fac*exp_time;

              if (debug_ins) {
                std::cout.width(2);
                std::cout << ix_time << " " << t_grid[ix_time] << " ";
                std::cout << exp_time << " " << lum_curve << " "
                          << exp_lum << " " << sqrt_fac << " "
                          << curve_sum << endl;
              }
            
            } else if (!lum_upper_limit) {
            
              // Stars for which t and L are measurements and not
              // upper limits
            
              if (ix_time==0) {
                sqrt_fac=1.0;
              } else {
                double deriv_star=(L_hat_curve[ix_time]-L_hat_curve[ix_time-1])/
                  (dimless_times[ix_time]-dimless_times[ix_time-1]);
                sqrt_fac=sqrt(1.0+deriv_star*deriv_star);
              }
            
              exp_lum=exp(-pow(L_hat_curve[ix_time]-c.L_hat,2.0)/
                          (2.0*c.dL_hat*c.dL_hat));
              exp_time=exp(-pow(dimless_times[ix_time]-c.t_hat,2.0)/
                           (2.0*c.dt_hat*c.dt_hat));

              curve_sum+=sqrt_fac*exp_time*exp_lum;

              if (debug_ins) {
                std::cout.width(2);
                std::cout << ix_time << " " << t_grid[ix_time] << " ";
                std::cout << exp_time << " " << lum_curve << " "
                          << exp_lum << " " << sqrt_fac << " "
                          << curve_sum << endl;
              }
        
            
            }
          }
        
        }
      }
    

      // Check log_weight. If its -inf for any star, then reject the
      // point.

      if (!std::isfinite(log(curve_sum))) {
        if (verbose>0) {
          std::cout << "Model cooling curves too far from data point "
                    << "for NS " << c.name << "." << std::endl;
        }
        log_weight=-850.0;
        return 0;
      }
    
      dat[dvi[((string)"log_wgt_")+c.name]]=log(curve_sum);
    
      log_weight+=log(curve_sum);
      if (verbose>1) {
        cout << "log weight for " << c.name << ": "
             << dat[dvi[((string)"log_wgt_")+c.name]] << endl;
        if (debug_ins) {
          cout << "--------------------------------------"
               << "--------------------------------------" << endl;
          //char ch;
          //cin >> ch;
        }
      }
    } // end of old like

    // End of loop: for (size_t i = 0; i < id.list.size(); i++)
  }

  // AWS 12/30/21, I'm commenting this out because this
  // value is no longer available in this function
  //if (verbose>0) {
  //cout << "Value of log weight after including INSs: "
  //<< log_weight << endl;
  //}

  if (verbose>1) {
    cout << "INS likelihood done." << endl;
  }

  //if (debug_ins) exit(-1);

  return 0;
}
