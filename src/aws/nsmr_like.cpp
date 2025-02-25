#include "data_eval.h"

using namespace std;
using namespace o2scl;

int data_eval::nsmr_like(const ubvector &pars,
			 std::array<double,ndat> &dat,
			 std::vector<o2scl::table3d> &MR_data,
			 double &log_weight) {

  if (verbose>0) {
    cout << "NS  name M           R           eta         at  "
         << "log_wgt      log_wgt_sum" << endl;
  }
  
  for(size_t i=0;i<nd.list.size();i++) {
    
    nsmr &c=nd.list[i];
    string slice_name="like";

    if (i==nd.list.size()-1 || i==nd.list.size()-2) slice_name="prob";

    if (pars[pvi[((string)"mf_")+c.name]]>1.0) {
      if (verbose>0) {
	cout << "Fail: Mass of " << c.name
             << "greater than Mns_max = " << dat[dvi["Mns_max"]] << endl;
      }
      log_weight=-852.0;
      for(size_t k=0;k<ndat;k++) dat[k]=0.0;
      return 0;
    }

    // Mass of this neutron star
    double masst=pars[pvi[(std::string("mf_"))+c.name]]*
      (dat[dvi["Mns_max"]]-1.0)+1.0;
    
    dat[dvi[((string)"Rns_")+c.name]]=
        tmvsr.interp("gm",masst,"r");
    
    // Hydrogen results are in even-indexed data files, and helium
    // results are in odd-indexed data files. This section switches
    // to the He data file if the associated atmosphere parameter
    // is greater than 2/3.
    size_t ix=i*2;
    string atm="H";
    if (pars[pvi[((string)"eta_")+c.name]]>2.0/3.0) {
      ix++;
      atm="He";
    }
    
    if (verbose>0) {
      // index mass radius eta eta_index log_likelihood
      cout << "NS ";
      cout.precision(5);
      cout.width(5);
      cout << c.name << " " << masst << " "
	   << dat[dvi[((string)"Rns_")+c.name]] << " "
	   << pars[pvi[((string)"eta_")+c.name]] << " ";
      cout.width(2);
      cout << atm << " ";
      cout.setf(ios::showpos);
      cout << log(nd.MR_data[ix].interp(dat[dvi[((string)"Rns_")+c.name]],
                                        masst,slice_name)) << " ";
      cout.precision(6);
    }

    double weight=nd.MR_data[ix].interp(dat[dvi[((string)"Rns_")+c.name]],
                                        masst,slice_name);
    if (weight<=0.0) {
      if (verbose>0) {
	cout << "Fail: NSMR data negative or zero." << endl;
      }
      log_weight=-853.0;
      for(size_t k=0;k<ndat;k++) dat[k]=0.0;
      return 0;
    }

    dat[dvi[((string)"log_wgt_")+c.name]]=log(weight);
    
    log_weight+=log(weight);
    if (verbose>0) {
      cout.precision(5);
      cout << log_weight << endl;
      cout.unsetf(ios::showpos);
      cout.precision(6);
    }
  }

  return 0;
} 
