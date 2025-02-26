#include "data_eval.h"

using namespace std;
using namespace o2scl;

int data_eval::ins_like(size_t nv, const ubvector &pars,
			double &log_weight, std::array<double,ndat> &dat) {
  
  // Dimensionless time grid
  std::vector<double> dimless_times(100);
  for(int i=0;i<100;i++) {
    dimless_times[i]=0.2*log10(0.01*t_grid[i]);
  }
      
  // --------------------------------------------------------------
  // Compute the cooling curves
  
  if (!no_cooling) {

    nw.sxrt_mode=false;
    nw.time_print[0]=1.0e-10;
    
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
	  log_weight=-805.0;
	  for(size_t k=0;k<ndat;k++) dat[k]=0.0;
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
	  for(size_t k=0;k<ndat;k++) dat[k]=0.0;
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
	
	//nw.write_tl_prof();
	//exit(-1);
	
	if (nw_ret!=0) {
	  if (verbose>0) {
	    std::cout << "Fail: Neutron star cooling calculation failed "
		      << "nw_ret: " << nw_ret << std::endl;
	  }
	  /*
	    for(size_t k=0;k<ndat;k++) {
	    dat[k]=0.0;
	    }
	  */
	  log_weight=-806.0;
	  return 0;
	}
	
	if (nw.v_time.size()<5) {
	  if (verbose>0) {
	    std::cout << "Fail: Cooling data not present." << std::endl;
	  }
	  log_weight=-832.0;
	  for(size_t k=0;k<ndat;k++) dat[k]=0.0;
	  return 0;
	}

	// Interpolate the temperature and luminosity from the cooling
	// calculation into the tensor_grid objects
	for(int k=0;k<100;k++) {
	  double interp_tptr=itp.eval(t_grid[k],nw.v_time.size(),
				      nw.v_time,nw.v_tptr);
	  nscool_tptr.set(ieta,imass,k,interp_tptr);
	  double interp_lum=itp.eval(t_grid[k],nw.v_time.size(),
				     nw.v_time,nw.v_lphot);
	  nscool_lum.set(ieta,imass,k,interp_lum);
	}	  
      }
    }
  }

  // --------------------------------------------------------------
  // Copy cooling curves to dat[] array
    
  if (!no_cooling) {
      
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
      
  } else {

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

  }

  //---------------------------------------------------------------------
  // Generates more cooling curve data to check for initial guesses later
  
  bool guess_ins=false;

  if (guess_ins) {
    std::vector<size_t> szzz={101,121,100};
    nscool_lum_all.resize(3,szzz);
    int teta=0; int tmass;

    for (double test_eta=-17.0;test_eta<-7.0;test_eta+=0.1) {
      tmass=0;
      for (double test_mass=1.0;test_mass<2.2;test_mass+=0.01) {
        for (double test_time=0;test_time<100;test_time++) {
          double int_lum = 1.0e-30*nscool_lum.interp_linear
            (test_eta,test_mass,
             t_grid[test_time]);
          double int_lum2 = 0.25*log10(int_lum);
          nscool_lum_all.set(teta,tmass,test_time,int_lum2);
        }
        tmass+=1;
      }
      teta+=1;
    }
  }

  // --------------------------------------------------------------
  // Compute the contribution to the log weight for each
  // star
  
  // Lower and upper limit of ix_time, thats different for stars with
  // exact age
  int ix_ll, ix_ul;

  // Vector to store each star's y_star[ix_time] value
  std::vector<double> y_star(100);

  // Vector to store star sum of each star
  std::vector<double> star_sum(id.list.size()); 
  std::vector<double> log_eta_values(id.list.size());
  std::vector<double> mass_values(id.list.size());	

  cout << "Value of log weight before including INSs: " << log_weight << endl;

  for (size_t i = 0; i < id.list.size(); i++) {
    
    cooling_ns &c=id.list[i];
    
    mass_values[i]=pars[pvi[(std::string("Mns_"))+c.name]];
    dat[dvi[((string)"Rns_")+c.name]]=
      tmvsr.interp("gm",pars[pvi[(string("Mns_"))+c.name]],"r");
    
    //cout << c.name << " M,R: " << mass_values[i] << " "
    //<< dat[dvi[((string)"Rns_")+c.name]] << " km" << endl;

    if (c.eta) {
      log_eta_values[i]=pars[pvi[(std::string("eta_"))+c.name]];
    } else {
      log_eta_values[i]=-17.0;
    }

    if (c.tk==0.0 && c.tk_lo==0.0) {
      c.t_hat=0.2*(log10(pars[pvi[((string)"t_")+c.name]])-2.0);
    }

    if (c.L == 0.0 && c.L_lo==0.0) {
      c.L_hat=0.25*(log10(pars[pvi[((string)"L_")+c.name]])+3.0);
    }

    //----------------------------------------------------------------------
    // Computes initial guesses for mass, eta and age from cooling
    // curves and data
    
    int tt_ll, tt_ul;

    if (c.tk_hi != 0.0 && (c.tk_hi - c.tk_lo) < 2*t_grid[1]) {
      int k = 0;
      while (c.tk_hi > t_grid[k]) {
        k++;
      }
      tt_ll=k-1;
      tt_ul=k+1;
    } else if (c.dt_hat==0.0) {
      int k =0;
      while (c.tc > t_grid[k]) {
        k++;
      }
      tt_ll=0;
      tt_ul=k+1;
    } else {
      int k=0;int ku=0;
      while (c.tk_lo>t_grid[k]) {
        k++;
      }
      tt_ll=k-1;
      while (c.tk_hi>t_grid[ku]) {
        ku++;
      }
      tt_ul=ku+1;
    }

    bool debug_ins=true;
    // cout << "Name " << " Age_Lowerlimit " << " Age_upperlimit" << endl;
    // cout << c.name << "  " << t_grid[tt_ll] << " "
    // << t_grid[tt_ul-1]<< endl;

    if (debug_ins) {
      cout << "Name: " << c.name << "  " << t_grid[tt_ll] << " "
           << t_grid[tt_ul-1] << endl;
    }
    
    if (guess_ins) {
      int teta=0; int tmass;
      double diff,diff_min,m_guess,eta_guess,t_guess;
      diff_min=1000;
      for (double test_eta=-17.0;test_eta<-7.0;test_eta+=0.1) {
        tmass=0;
        for (double test_mass=1.0;test_mass<2.2;test_mass+=0.01) {
          for (double test_time=tt_ll;test_time<tt_ul;test_time++) {
            double lum=nscool_lum_all.get(teta,tmass,test_time);
            diff = abs(lum-c.L_hat);
            if (diff < diff_min) {
              m_guess=test_mass;
              eta_guess=test_eta;
              diff_min=diff;
              if (c.dt_hat==0.0) {
                t_guess=t_grid[test_time];
              }
            }
          }
          tmass+=1;
        }
        teta+=1;
      }
      
      cout << "Guess_Mass: " << m_guess << endl;
      cout << "Guess_Eta: " << eta_guess << endl;
      cout << "diff_min: " << diff_min << endl;
      if (c.dt_hat==0.0) {
        cout << "Guess_age: " << t_guess << endl;
      }
    }
    
    //----------------------------------------------------------------------
    // check if tk is exact with no error

    if (c.tk_hi != 0.0 && (c.tk_hi - c.tk_lo) < 2*t_grid[1]) {
      double age=c.tk;
      double Lhat_from_data=c.L_hat;
      double L_from_curve=nscool_lum.interp_linear
        (log_eta_values[i],mass_values[i],
         age);
      double Lhat_from_curve=0.25*(log10(L_from_curve)+3.0);
      double exp_new=exp(-pow(Lhat_from_curve-Lhat_from_data,2.0)/
                         2.0/c.dL_hat/c.dL_hat);
      log_weight+=log(exp_new);
    } else (c.tk_lo==0.0 && c.tk==0.0) {
        double t_hat=c.t_hat;
      double L_from_curve=nscool_lum.interp_linear
        (log_eta_values[i],mass_values[i],t_hat);
      double Lhat_from_curve=0.25*(log10(L_from_curve)+3.0);
      double exp_new=exp(-pow(Lhat_from_curve-Lhat_from_data,2.0)/
                         2.0/c.dL_hat/c.dL_hat);
      log_weight+=log(exp_new);
    }
    
    
    if (c.tk_hi != 0.0 && (c.tk_hi - c.tk_lo) < 2*t_grid[1]) {    

      // Find the time interval which encloses the exact age
      int j = 0;
      while (c.tk_hi > t_grid[j]) {
        j++;
      }
      ix_ll=j-2;
      ix_ul=j;
    } else {
      ix_ll = 0;
      ix_ul = 100;
    }

    //cout << c.name << " tk_lo,tk_hi " << c.tk_lo << " " << c.tk_hi 
    //<< " tgrid " << t_grid[ix_ll] << " " << t_grid[ix_ul-1] << endl;

    if (debug_ins) {
      cout << "name,ll,ul: " << c.name << " " << ix_ll << " " << ix_ul << endl;
    }

    if (debug_ins) {
      cout << "t_hat, dt_hat, L_hat, dL_hat:\n  "
	   << c.t_hat << " " << c.dt_hat << " " << c.L_hat<< " "
	   << c.dL_hat << endl;
      cout << "ix_time t_grid[ixtime], exp_y_star, exp_time" <<
        ", sqrt_star, Lum" << endl;
    }
    
    for (int ix_time = ix_ll; ix_time < ix_ul; ix_time++) {
      
      double exp_y_star, exp_time, sqrt_star;
      bool arg_positive=true;
      
      // Luminosity from the cooling curve
      double arg_test=nscool_lum.interp_linear
	(log_eta_values[i],mass_values[i],
	 t_grid[ix_time]);
      double arg=1.0e-30*arg_test;
      if (arg<0.0) {
	arg_positive=false;
	y_star[ix_time]=-10.0;
      } else {
	y_star[ix_time]=0.25*log10(arg);
      }

      if (arg_positive) {
	if (c.dL_hat == 0.0) {
          //if (arg_test<=c.L_hi*1e33) {
          exp_y_star = 1.0;
          //} else exp_y_star = 0.0;
	}
        else {
          //if (arg_test>=c.L_lo*1e33 && arg_test<=c.L_hi*1e33) {
          exp_y_star = exp(-(y_star[ix_time] - c.L_hat) *
			   (y_star[ix_time] - c.L_hat) /
			   (2.0 * c.dL_hat * c.dL_hat));
          //} else {
          // exp_y_star=0.0;
          //}
	}
        
	// Gaussian for time
	if ((c.tk_hi - c.tk_lo) < 2*t_grid[1] && c.dt_hat != 0.0) {
	  exp_time = 1.0;
        } else if (c.dt_hat==0.0) {
          //if (t_grid[ix_time]<=c.tc) {
          exp_time = 1.0;
          //} else exp_time =0.0;
        } else {
	  exp_time = exp(-((dimless_times[ix_time] - c.t_hat) *
			   (dimless_times[ix_time] - c.t_hat)) /
			 (2.0*c.dt_hat*c.dt_hat));
	}

	// Square-root factor
	if (ix_time == ix_ll) {
	  sqrt_star = 1.0;
	} else {
	  double deriv_star =
	    (y_star[ix_time] - y_star[ix_time - 1]) /
	    (dimless_times[ix_time] - dimless_times[ix_time - 1]);
	  sqrt_star = abs(sqrt(1 + deriv_star * deriv_star));
	  if (!std::isfinite(sqrt_star)) {
	    std::cout << "Problem in sqrt_star." << std::endl;
	    std::cout << ix_time - 1 << " " << ix_time << std::endl;
	    std::cout << y_star[ix_time - 1] << " "
		      << y_star[ix_time] << std::endl;
	    std::cout << dimless_times[ix_time - 1] << " "
		      << dimless_times[ix_time] << std::endl;
	    std::cout << deriv_star << " "
		      << sqrt_star << std::endl;
	    exit(-1);
	  }
	}

	// Add contribution to sum

        if (ix_time==ix_ll) {
          star_sum[i] = 0.0;
        } else {
	  star_sum[i] += exp_y_star * exp_time * sqrt_star;
        }

	if (debug_ins) {
	  cout << ix_time << " " << t_grid[ix_time] //<< " " << y_star[ix_time]
               << " " << exp_y_star //<< "\n"
               << " " << exp_time << " " << sqrt_star << " " << arg_test 
               << " " << star_sum[i] << endl;
	}
      }
    }
	
    dat[dvi[((string)"wgt_")+c.name]]=log(star_sum[i]);
    /* likelihood check, if its not finite, go next
       if (!std::isfinite(log(star_sum[i]))) {
       if (verbose>0) {
       std::cout << "Data not matching." << std::endl;
       }
       log_weight=-810.0;
       for(size_t k=0;k<ndat;k++) dat[k]=0.0;
       return 0;
       }*/
    log_weight+=log(star_sum[i]);
    if (verbose>1) {
      cout << "log weight for " << c.name << ": "
           << dat[dvi[((string)"wgt_")+c.name]] << endl;
      cout << "---------------------------------" << endl;
    }

    char ch;
    cin >> ch;
  }

  cout << "Value of log weight after including INSs: " << log_weight << endl;

  if (verbose>1) {
    cout << "INS likelihood done." << endl;
  }
    
  return 0;
}
