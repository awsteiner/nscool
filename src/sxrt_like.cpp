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
  int ret=match_lum(1,x,y,Q_heat,mass,rad,Mdot);
  // If so, then call solver
  if (ret==0) {
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
      x[0]=init;
      // Try to get an initial guess from this value of log T
      ret=match_lum(1,x,y,Q_heat,mass,rad,Mdot);
      if (verbose>0) {
        cout << "ret: " << init << " " << ret << endl;
      }
      // If it worked, call the solver
      if (ret==0) {
	//mh.verbose=1;
	ret=mh.msolve(1,x,func);
        logT=x[0];
      }
      if (verbose>0) {
        cout << "ret1: " << init << " " << ret << endl;
        //cout << "hehe: " << x[0] << " " << logT << endl;
      }
    }
  }
  return ret;
}

int data_eval::match_lum(size_t nv, const ubvector &x, ubvector &y,
			 double Q_heat, double mass, double rad,
			 double Mdot) {
			 
  double logT=x[0];
  double lphot=0.0, lneut=0.0, lheat=0.0;
  int ret=acc_compute(logT,Q_heat,mass,rad,Mdot,lphot,lneut,lheat);
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
  if(nw.sxrt_mode){
    nw.time_print[0]=1.1e-12;
    nw.temp_ygrid.clear();
    for(size_t i=0;i<log_mdot_grid.size();i++) {
      nw.temp_ygrid.push_back(log_mdot_grid[i]);
    }
  }
#ifndef NO_OPENMP
  size_t i_thread=omp_get_thread_num();
#else
  size_t i_thread=0;
#endif

  int ret=nw.run(i_thread);
  if (ret!=0) {
    return ret;
  }
  lphot=nw.v_lphot[nw.v_lphot.size()-1];
  lneut=nw.v_lneut[nw.v_lneut.size()-1];
  //lheat=nw.v_lheat[nw.v_lheat.size()-1];

  // AWS, FIXME: Document 6.03e33 and 1.0e-10. Mdot is in solar
  // masses per year, Q heat is in MeV, lheat should be in erg/s.
  // See eq. 5 in Han's paper, there's a nucleon mass in here
  // somewhere as well. 
  lheat=(Q_heat*Mdot/1.0e-10)*6.03e33*sqrt(1.0-schwarz_km*mass/rad);
  return ret;
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

  // Vector to store interpolated luminosity from the cooling curves.
  // units in normalized log scale
  std::vector<double> L_hat_curve(log_mdot_grid.size());

  bool debug=false;
  if (verbose>=2) debug=true;
  
  for (size_t i=0;i<sd.list.size();i++) {

    sxrt &c=sd.list[i];

    if (pars[pvi[(std::string("mf_"))+c.name]]>1.0) {
      O2SCL_ERR("Fail: mass greater than maximum.",o2scl::exc_einval);
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

      // Store the radius in the output table
      dat[dvi[((string)"Rns_")+c.name]]=
        tmvsr.interp("gm",masst,"r");

      if (c.L_ul) {
	dat[dvi[((string)"L_")+c.name]]=sxrt_lum.interp_linear
	  (pars[pvi[((string)"eta_")+c.name]],masst,log10(mdot1));
      }
      
      if (debug) {
        cout << "Name: " << c.name << " Lum, Mdot: " 
             << c.L << " ergs/s, " << c.mdot << endl;
        cout << "M,R: " << masst << " M_sun, "
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

