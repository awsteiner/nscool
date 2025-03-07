/*
  -------------------------------------------------------------------
  
  Copyright (C) 2015-2020, Andrew W. Steiner and Satyajit Roy
  
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
#ifndef NSCOOL_PRECOOL_H
#define NSCOOL_PRECOOL_H

#include <boost/numeric/ublas/vector.hpp>

#include <o2scl/constants.h>
#include <o2scl/interp.h>
#include <o2scl/table.h>
#include <o2scl/hdf_file.h>
#include <o2scl/hdf_io.h>

typedef boost::numeric::ublas::vector<double> ubvector;
typedef boost::numeric::ublas::matrix<double> ubmatrix;

/** \brief Compute critical temperatures
 */
class precool {
  private:
    const double pi, kb, MeV, isize, strange_mass;
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

  public:

    precool() {
      pi = 3.14159265;
      kb = 1.38e-16;  // Boltzmann constant in erg/K
      MeV = 1.602e-6; // MeV to erg conversion factor
      isize = 10000;
      strange_mass = 0.0;
    }

    /** \brief Calculates the degenerate specific heat per cm^3 over T:    
      Cv/T                                   
      For spin 1/2 fermions !                                     
      pf and m must be in MeV, but cvt is returned in cgs units.  
      m must be the Landau effective mass, i.e., m* for baryons   
      and sqrt(m**2+pf**2) for leptons ! 
    */                    
    double cvt_deg(double pf, double m) {
      if (pf == 0.0) {
        return 0.0; } else {
        double N0 = 2.0*m*pf/(2.0*pi*pi*pi);
        double cvt = pi*pi/3.0*N0;
        cvt *= kb*kb/(MeV*197.0*197.0*197.0)*1.0e39; // Converts to cgs units
        return cvt; }
    }

    /** \brief Desc
     */
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
            std::vector<double>& tcn, std::vector<double>& tcp, std::vector<double>& tcla, 
            std::vector<double>& tcsm, std::vector<double>& tcs0, std::vector<double>& tcsp,
            std::vector<double>& tcuu, std::vector<double>& tcdd, std::vector<double>& tcss, 
            std::vector<double>& tcud, std::vector<double>& tcus, std::vector<double>& tcds,
            std::vector<double>& tcu, std::vector<double>& tcd, std::vector<double>& tcs,
            int& sfn1s0, int& sfn3p2, int& sfp1s0, int& sfl1s0,
            double fn1s0, double fn3p2, double fp1s0, double fl1s0,
            double &kfmax_n3p2, double &delkf_n3p2, double &tcmax_n3p2, int &isf,
            std::vector<double> kfn, std::vector<double> kfp, std::vector<double> kfla, 
            std::vector<double> kfqu, std::vector<double> kfqd, std::vector<double> kfqs,
            std::vector<double> bar, std::vector<double> fhad, std::vector<double> yquarku, 
            std::vector<double> yquarkd, std::vector<double> yquarks) {
      // AWS: determine gap parameters
      gaps(irank,sfn1s0,dinput_n1tc,dinput_n1kf,
          dinput_n1dk,sfn3p2,dinput_n3tc,dinput_n3kf,
          dinput_n3dk,sfp1s0,dinput_p1tc,dinput_p1kf,dinput_p1dk);
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

}
#endif