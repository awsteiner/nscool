#include <cmath>
#include <iostream>
#include <vector>
#include <functional>
#include <o2scl/table.h>

const double pi = 3.14159265;
const double kb = 1.38e-16;  // Boltzmann constant in erg/K
const double MeV = 1.602e-6; // MeV to erg conversion factor
const int isize = 10000;
const double strange_mass = 0.0;
double alpha_durca;
double dinput_n1tc,dinput_n1kf,
        dinput_n1dk,dinput_n3tc,dinput_n3kf,
        dinput_n3dk,dinput_p1tc,dinput_p1kf,dinput_p1dk;

double pf0_uu,dpf_uu,gap_uu,
        pf0_dd,dpf_dd,gap_dd,
        pf0_ss,dpf_ss,gap_ss,
        pf0_ud,dpf_ud,gap_ud,
        pf0_us,dpf_us,gap_us,
        pf0_ds,dpf_ds,gap_ds;
typedef std::function<double (double)> funct;
o2scl::table<std::vector<double>> nscool_core;
o2scl::table<std::vector<double>> nscool_crust;
o2scl::table<std::vector<double>> nscool_tov;

// Calculates the degenerate specific heat per cm^3 over T:    
// Cv/T                                   
// For spin 1/2 fermions !                                     
// pf and m must be in MeV, but cvt is returned in cgs units.  
// m must be the Landau effective mass, i.e., m* for baryons   
// and sqrt(m**2+pf**2) for leptons !                     
double cvt_deg(double pf, double m) {
  if (pf == 0.0) {
    return 0.0; } else {
    double N0 = 2.0*m*pf/(2.0*pi*pi*pi);
    double cvt = pi*pi/3.0*N0;
    cvt *= kb*kb/(MeV*197.0*197.0*197.0)*1.0e39; // Converts to cgs units
    return cvt; }
}

void get_degenerate_density(int irank, std::vector<double> rrho, 
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


void get_Tc(int irank, int imax, int icore, int idrip,
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
  // Just to be safe:
  for (int i=0; i<=imax; ++i) {
    tcn[i]=tcp[i]=tcla[i]=tcuu[i]=tcdd[i]=tcss[i]=
    tcud[i]=tcus[i]=tcds[i]=tcu[i]=tcd[i]=tcs[i]=1.0;
  }
  // 1s0 neutron superfluidity
  if (sfn1s0 == 1) {
    for (int i=0; i<=idrip; ++i) {
      tcn[i] = std::max(1.0, cpp_tcn1_sfb(kfn[i]))*fn1s0; }
  } else if (sfn1s0 == 2) {
    for (int i=0; i<=idrip; ++i) {
      tcn[i] = std::max(1.0, cpp_tcn1_ccdk(kfn[i]))*fn1s0; }
  } else if (sfn1s0 == 3) {
    for (int i=0; i<=idrip; ++i) {
      tcn[i] = std::max(1.0, cpp_tcn1_wap(kfn[i]))*fn1s0; }
  } else if (sfn1s0 == 4) {
    for (int i=0; i<=idrip; ++i) {
      tcn[i] = std::max(1.0, cpp_tcn1_gc(kfn[i]))*fn1s0; }
  } else if (sfn1s0 == 5) {
    for (int i=0; i<=idrip; ++i) {
      tcn[i] = std::max(1.0, cpp_tcn1_gipsf(kfn[i]))*fn1s0; }
  } else if (sfn1s0 == 201) {
    for (int i=0; i<=idrip; ++i) {
      tcn[i] = std::max(1.0, cpp_tc_Ioffe_1ns(kfn[i]))*fn1s0; }
  } else if (sfn1s0 == 202) {
    for (int i=0; i<=idrip; ++i) {
      tcn[i] = std::max(1.0, cpp_tc_Ioffe_2ns(kfn[i]))*fn1s0; }
  } else if (sfn1s0 == 203) {
    for (int i=0; i<=idrip; ++i) {
      tcn[i] = std::max(1.0, cpp_tc_Ioffe_3ns(kfn[i]))*fn1s0; }
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

  if (sfn3p2 == 1) fill_tcn(cpp_tcn3_hgrr,tcn,isf);
  else if (sfn3p2 == 2) fill_tcn(cpp_tcn3_ao,tcn,isf);
  else if (sfn3p2 == 3) fill_tcn(cpp_tcn3_ao_m1,tcn,isf);
  else if (sfn3p2 == 4) fill_tcn(cpp_tcn3_t72,tcn,isf);
  else if (sfn3p2 == 5) fill_tcn(cpp_tcn3_t72_m1,tcn,isf);
  else if (sfn3p2 == 6) fill_tcn(cpp_tcn3_bcll92,tcn,isf);
  else if (sfn3p2 == 7) fill_tcn(cpp_tcn3_eehjo96_nr,tcn,isf);
  else if (sfn3p2 == 8) fill_tcn(cpp_tcn3_eehjo96_r,tcn,isf);
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
        tcp[i] = std::max(1.0, cpp_tcp1_ccy_ms(kfp[i]))*fp1s0; }
  } else if (sfp1s0 == 2) {
    for (int i = 0; i <= icore; ++i) {
        tcp[i] = std::max(1.0, cpp_tcp1_ccy_ps(kfp[i]))*fp1s0; }
  } else if (sfp1s0 == 3) {
    for (int i = 0; i <= icore; ++i) {
        tcp[i] = std::max(1.0, cpp_tcp1_t73(kfp[i]))*fp1s0; }
  } else if (sfp1s0 == 4) {
    for (int i = 0; i <= icore; ++i) {
        tcp[i] = std::max(1.0, cpp_tcp1_ns(kfp[i]))*fp1s0; }
  } else if (sfp1s0 == 5) {
    for (int i = 0; i <= icore; ++i) {
        tcp[i] = std::max(1.0, cpp_tcp1_ao(kfp[i]))*fp1s0; }
  } else if (sfp1s0 == 6) {
    for (int i = 0; i <= icore; ++i) {
        tcp[i] = std::max(1.0, cpp_tcp1_bcll92(kfp[i]))*fp1s0; }
  } else if (sfp1s0 == 7) {
    for (int i = 0; i <= icore; ++i) {
        tcp[i] = std::max(1.0, cpp_tcp1_ccdk(kfp[i]))*fp1s0; }
  } else if (sfp1s0 == 21) {
    for (int i = 0; i <= icore; ++i) {
        tcp[i] = std::max(1.0, cpp_tcn1_t72(kfp[i]))*fp1s0; }
  } else if (sfp1s0 == 22) {
    for (int i = 0; i <= icore; ++i) {
        tcp[i] = std::max(1.0, cpp_tcn1_awp_2(kfp[i]))*fp1s0; }
  } else if (sfp1s0 == 23) {
    for (int i = 0; i <= icore; ++i) {
        tcp[i] = std::max(1.0, cpp_tcn1_awp_3(kfp[i]))*fp1s0; }
  } else if (sfp1s0 == 201) {
    for (int i = 0; i <= icore; ++i) {
        tcp[i] = std::max(1.0, cpp_tc_Ioffe_1p(kfp[i]))*fp1s0; }
  } else if (sfp1s0 == 202) {
    for (int i = 0; i <= icore; ++i) {
        tcp[i] = std::max(1.0, cpp_tc_Ioffe_2p(kfp[i]))*fp1s0; }
  } else if (sfp1s0 == 203) {
    for (int i = 0; i <= icore; ++i) {
        tcp[i] = std::max(1.0, cpp_tc_Ioffe_3p(kfp[i]))*fp1s0; }
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

void get_spec_heat_degenerate(
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

void get_effective_masses(const std::string version,
                          double emnco, double emncr, double emp,
                          const std::vector<double>& kfn, const std::vector<double>& kfp,
                          std::vector<double>& mstn, std::vector<double>& mstp, std::vector<double>& mstla,
                          std::vector<double>& mstsm, std::vector<double>& msts0, std::vector<double>& mstsp,
                          int idrip, int icore) {
  for (int i = icore + 1; i <= idrip; ++i) {
    mstn[i] = std::min(1.0, 1.09 - 0.11 * kfn[i]);
  }
}

void get_fermi_momenta(int irank, int imax, int icore,
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
  const double coeff = 3.0*pi*pi*pi;
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

    kfn[j] = pow(3.0*pi*pi*nn, 1.0/3.0);
    kfp[j] = pow(3.0*pi*pi*np, 1.0/3.0);
    kfla[j] = pow(3.0*pi*pi*nla, 1.0/3.0);
    kfsm[j] = pow(3.0*pi*pi*nsm, 1.0/3.0);
    kfs0[j] = pow(3.0*pi*pi*ns0, 1.0/3.0);
    kfsp[j] = pow(3.0*pi*pi*nsp, 1.0/3.0);
    kfqu[j] = pow(3.0*pi*pi*nqu, 1.0/3.0);
    kfqd[j] = pow(3.0*pi*pi*nqd, 1.0/3.0);
    kfqs[j] = pow(3.0*pi*pi*nqs, 1.0/3.0);

    double ne = std::abs(yelect[j]*bar[j]);
    double nm = std::abs(ymuon[j]*bar[j]);
    kfe[j] = pow(3.0*pi*pi*ne, 1.0/3.0);
    if (yelect[j]<=0.0) kfe[j] = -kfe[j];
    kfm[j] = pow(3.0*pi*pi*nm, 1.0/3.0);
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
        
    kfe[j] = pow(3.0*pi*pi*ne, 1.0/3.0);
    kfm[j] = 0.0;
    kfn[j] = pow(3.0*pi*pi*nn, 1.0/3.0);
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

void get_crust_chemistry(int irank, int debug, const std::string version, int imax, int icore,
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


void get_core_chemistry(int irank, const std::string version, int imax, int icore, 
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


void grid(int irank, int idec, double rhocore, double rhodrip, double rhoenv, double rhosurf,
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