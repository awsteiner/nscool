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
#ifndef NSCOOL_CONDUCT_H
#define NSCOOL_CONDUCT_H

#include <boost/numeric/ublas/vector.hpp>

#include <o2scl/constants.h>
#include <o2scl/interp2_direct.h>
#include <o2scl/fermion.h>

typedef boost::numeric::ublas::vector<double> ubvector;
typedef boost::numeric::ublas::matrix<double> ubmatrix;

/** \brief Conductivity Calculations

    This work in progress will eventually replace Dany's Fortran code.
*/

class conduct {

  private:
    static constexpr double pi = 3.14159265;
    static constexpr double c = 2.997924e10;
    static constexpr double a_f = 1/137;
    static constexpr double mu=1.66e-24;
    static constexpr double Na = 6.022e23;           // Avogadro's number
    static constexpr double kb = 1.380e-16;          // Boltzmann's constant
    static constexpr double Mu = 1.0 / Na;            // Derived constant

    static constexpr double me = 9.109e-28;           // Electron mass in grams
    static constexpr double e = 4.803206e-10;         // Elementary charge in esu

    static constexpr double hbar = 1.054572e-27;      // Reduced Planck's constant
    static constexpr double cl = 2.99792458e10;       // Speed of light in cm/s

    static constexpr double hbc = 197.327;            // Conversion factor (hbar*c) in MeV*fm
    static constexpr double MeV = 1.602177e-6;        // MeV in grams*cm^2/s^2

    static constexpr double u_1 = 2.80;       
    static constexpr double u_2 = 13.00;      
    static constexpr double a_0 = 0.01740;     
    static constexpr double a_2 = 0.01180;     
  
  public:

    double c_con_str, p_con_str;

    conduct() {
      c_con_str = 0.0;
      p_con_str = 0.0;
    }

    /** \brief Core conductivity
     */
    void con_core(double icon_core, double debug, double Temp, 
                  double kf_e, double kf_mu,
                  double kf_p, double mst_p, double Tc_p,              // proton
                  double kf_n, double mst_n, double Tc_n, double isfn, // neutron
                  double kf_la, double mst_la, double Tc_la,           // lambda
                  double kf_sm, double mst_sm, double Tc_sm,           // sigma-
                  double kf_s0, double mst_s0, double Tc_s0,           // sigma0
                  double kf_sp, double mst_sp, double Tc_sp,           // sigma+
                  double f_had,
                  double& sigma, double& lambda,
                  double& nu_e_s, double& nu_e_l){
      
      double sigma_lep, lambda_e, lambda_mu, nu_e_s1, nu_e_l1;
      double sigma_bar, lambda_bar, nu_e_s2, nu_e_l2;
      // Use simple Flowers & Itoh (1981) formula:
      if (icon_core == 1) {
        lambda = 1e23*(kf_n*kf_n*kf_n/1.6)/(Temp/1e8);
        sigma = 0.0;
      } else if (icon_core == 2) {
        con_core_lep(Temp, kf_e, kf_mu,
                    kf_p, mst_p, Tc_p,
                    kf_sm, mst_sm, Tc_sm,
                    kf_sp, mst_sp, Tc_sp,
                    sigma_lep, lambda_e, lambda_mu, debug,
                    nu_e_s1, nu_e_l1);
        int icontrol = 1;
        con_core_bar(Temp, kf_e, kf_mu,
                    kf_p, mst_p, Tc_p,
                    kf_n, mst_n, Tc_n, isfn,
                    kf_la, mst_la, Tc_la,
                    kf_sm, mst_sm, Tc_sm,
                    kf_s0, mst_s0, Tc_s0,
                    kf_sp, mst_sp, Tc_sp,
                    sigma_bar, lambda_bar, debug,
                    nu_e_s2, nu_e_l2, icontrol);
        // Quark conductivity NOT defined !
        double lambda_qrk = 0.0;
        double sigma_qrk = 0.0;

        lambda = lambda_e+lambda_mu;
        sigma = 0.0;
      } else if (icon_core>20) {
        lambda = icon_core/(Temp/1e8);
        sigma = 0.0;
      } else if (icon_core<-20) {
        lambda = abs(icon_core);
      }
      return;
    }

    /** \brief Core conductivity Leptons
     */
    void con_core_lep(double Temp, double kf_e, double kf_m, 
                  double kf_p, double mst_p0, double Tc_p,
                  double kf_sm, double mst_sm0, double Tc_sm, 
                  double kf_sp, double mst_sp0, double Tc_sp,
                  double &sigma_lep, double &lambda_e, double &lambda_m, 
                  int debug, double &nu_e_s, double &nu_e_l) {

      // Define the Fermi momentum ratios for pairing correction
      auto u_1s0 = [] (double t) {
        return std::sqrt(1.0-t)*(1.456-0.157/std::sqrt(t)+1.764/t);
      };

      // Pairing correction factors for 1S0
      double t = Temp / Tc_p;
      double y = u_1s0(t);
      double nu_mm_par = 0.0;
      double nu_me_par = 0.0;
      double nu_mp_par = 0.0;
      double nu_em_pri = 0.0;
      double nu_mm_pri = 0.0;
      double nu_me_pri = 0.0;
      double R_tot_per = 1.0;
      double R_p_par = 1.0;
      double R_l_pri = 1.0;
      double lambda_lep;

      if (debug == 1)
        std::cout << "Entering con_core_lep: T, kfeo = " << Temp
                  << ", " << kf_e << std::endl;

      // In case there are no leptons
      if (kf_e == 0.0){
        lambda_e = 0.0;
        lambda_m = 0.0;
        lambda_lep = 0.0;
        sigma_lep = 0.0;
        if (debug == 1)
          std::cout << "Exiting con_core_lep: sigma_lep, lambda_lep = "
                    << sigma_lep << ", " << lambda_e << std::endl;
        return;
      }

      // In case there are no muons
      bool muons = (kf_m != 0.0);

      // THERMAL CONDUCTIVITY: ***********************************************
      // Trick to automatically eliminate absent baryons:
      // That's because they show up only through the phase space integrals
      // and this comes to zero if mst=0 !
      double mst_p = mst_p0;
      if (kf_p == 0.0) mst_p = 0.0;
      double mst_sm = mst_sm0;
      if (kf_sm == 0.0) mst_sm = 0.0;
      double mst_sp=mst_sp0;
      if (kf_sp == 0.0) mst_sp = 0.0;
      // Define Fermi momenta ratios (massively used):
      double rkf_0 = 1.68/kf_e;
      double rkf_m = kf_m/kf_e;
      double rkf_p = kf_p/kf_e;
      // Screening momenta ratios:
      double rkf_e_ql3 = 1.0/(0.00929*std::pow(1.0+rkf_m+2.83*mst_p0*rkf_0*rkf_p,-1.5));
      double rkf_e_qt2 = 1.0/(0.00929*(1.0+rkf_m*rkf_m+rkf_p*rkf_p));
      // Longitudinal collisional frequencies:
      double nu_ee_par = 1.43e11*rkf_0*rkf_e_ql3*std::pow(Temp/1.0e8,2);
      double nu_em_par = nu_ee_par*muons;
      double nu_ep_par = 1.15e12*mst_p0*mst_p0*rkf_0*rkf_0*rkf_e_ql3*std::pow(Temp/1.0e8,2);
      if (muons) {
        double nu_mm_par = nu_ee_par/rkf_m;
        double nu_me_par = nu_mm_par;
        double nu_mp_par = nu_ep_par/rkf_m;
      }
      // Transverse ("perpendicular") collisional frequencies:
      double nu_ee_per = 6.49e14*rkf_e_qt2*std::pow(Temp/1.0e8,1);
      double nu_em_per = nu_ee_per*rkf_m*rkf_m;
      double nu_ep_per = nu_ee_per*rkf_p*rkf_p;
      double nu_mm_per = nu_ee_per*rkf_m*rkf_m*rkf_m;
      double nu_me_per = nu_ee_per*rkf_m;
      double nu_mp_per = nu_ee_per*rkf_m*rkf_p*rkf_p;
      // Cross ("prime") collisional frequencies
      double nu_ee_pri = 4.38e12*std::pow(rkf_0,2.0/3.0)*std::pow(rkf_e_qt2,1.0/3.0)
                        *std::pow(rkf_e_ql3,2.0/3.0)*std::pow(Temp/1.0e8,5.0/3.0);
      if (muons) {
        double nu_em_pri = nu_ee_pri*rkf_m*rkf_m;
        double nu_mm_pri = nu_ee_pri*rkf_m*rkf_m*rkf_m;
        double nu_me_pri = nu_ee_pri/rkf_m;
      }
      // Effect of pairing for proton 1S0 pairing
      if ((Temp <= Tc_p) && (kf_p > 0.0)) {
        double r = (pow(kf_e,2)+pow(kf_m,2))/pow(kf_p,2);
        R_l_pri = pow(r+1.0,1.0/3.0) /
                        (pow(r+1.0,2.0)-0.757*y+pow(0.50651*y,2.0))/pow(6.0,1.0/6.0);
        double p1 = 0.48-0.17*r;
        double p3 = pow((1.0-p1)*54.0/(4.0*pi*pi*r),2.0);
        R_tot_per = p1*exp(-0.14*y*y)+(1.0-p1)/sqrt(1.0+p3*y*y);
        R_p_par = (0.998+(2.04+0.68*sqrt(y)+5.7*pow(y,2)+1.71*pow(y,4))*exp(-1.04*y))
                  *exp(-sqrt(1.23+pow(y,2)));
      }
      // Adjust collisional frequencies for pairing and add them
      double nu_e_par = nu_ee_par+nu_em_par+nu_ep_par*R_p_par;
      double nu_e_per = (nu_ee_per+nu_em_per+nu_ep_per)*R_tot_per;
      double nu_e_pri = nu_ee_pri*R_l_pri;
      double nu_m_par = nu_mm_par+nu_me_par+nu_mp_par*R_p_par;
      double nu_m_per = (nu_mm_per+nu_me_per+nu_mp_per)*R_tot_per;
      double nu_m_pri = nu_mm_pri*R_l_pri;
      nu_em_pri = nu_em_pri*R_l_pri;
      nu_me_pri = nu_me_pri*R_l_pri;
      // Total collisional frecuencies:
      double nu_e=nu_e_par+nu_e_per+nu_e_pri;
      double nu_m=nu_m_par+nu_m_per+nu_m_pri;
      // Relaxation times
      double tau_e, tau_m;
      if (muons) {
        tau_e = (nu_m-nu_em_pri)/(nu_e*nu_m-nu_em_pri*nu_me_pri);
        tau_m = (nu_e-nu_me_pri)/(nu_e*nu_m-nu_em_pri*nu_me_pri);
      } else {
        tau_e = 1.0/nu_e;
        tau_m = 0.0;
      }
      // Thermal conductivities
      double T8 = Temp/1.0e8;
      lambda_e = 1.70e24*T8*(1.0e15*tau_e)*std::pow(kf_e/1.68,2);
      lambda_m = 1.70e24*T8*(1.0e15*tau_m)*std::pow(kf_m/1.68,2);
      lambda_lep=lambda_e+lambda_m;
      // Electrical conductivity (not calculated in this function)
      sigma_lep = 0.0;
      nu_e_l = nu_e;
      nu_e_s = 0.0;

      if (debug == 1)
        std::cout << "Exiting con_core_lep: sigma_lep, lambda_lep = "
                  << sigma_lep << ", " << lambda_e + lambda_m << std::endl;
    }

    /** \brief Calculates the neutron thermal conductivity in the core from        
      Baiko, Haensel & Yakovlev, 2001, A&A 374, p 151                     
                                                                          
      Chemical potentials mu's in MeV                                     
      Fermi momenta kf's in fm^-1
    */
    void con_core_bar(double Temp, double kf_e, double kf_mu,
                      double kf_p, double mst_p, double Tc_p,
                      double kf_n, double mst_n, double Tc_n, int isfn,
                      double kf_la, double mst_la, double Tc_la,
                      double kf_sm, double mst_sm, double Tc_sm,
                      double kf_s0, double mst_s0, double Tc_s0,
                      double kf_sp, double mst_sp, double Tc_sp,
                      double &sigma_bar, double &lambda_bar, int debug,
                      double &nu_e_s, double &nu_e_l, int icontrol){

      // Define the Fermi momentum ratios for pairing correction
      auto u_1s0 = [] (double t) {
        return std::sqrt(1.0-t)*(1.456-0.157/std::sqrt(t)+1.764/t);
      };
      auto u_3p2 = [] (double t) {
        return std::sqrt(1.0-t)*(0.7893+1.188/t);
      };

      if (debug == 1)
        std::cout << "Entering con_core_bar: T, kfeo = " << Temp
                  << ", " << kf_e << std::endl;

      // Neutron contribution
      double Sn1 = 14.57/std::pow(kf_n,1.5)*(1.0-0.0788*kf_n+0.0883*kf_n*kf_n)
                    /(1.0-0.1114*kf_n);
      double Sn2 = 7.880/(kf_n*kf_n)*(1.0-0.2241*kf_n+0.2006*kf_n*kf_n)
                    /(1.0-0.1742*kf_n);
      double Sp1 = 0.8007*kf_p/(kf_n*kf_n)*(1.0+31.28*kf_p-0.0004285*kf_p*kf_p+
                    26.85*kf_n+0.08012*kf_n*kf_n)/
                  (1.0-0.5898*kf_n+0.2368*kf_n*kf_n+0.5838*kf_p*kf_p+0.884*kf_n*kf_p);
      double Sp2 = 0.3830*std::pow(kf_p,4)/std::pow(kf_n,5.5)*(1.0+102.0*kf_p+53.91*kf_n)/
                  (1.0-0.7087*kf_n+0.2537*kf_n*kf_n+9.404*kf_p*kf_p-1.589*kf_n*kf_p);

      double u;
      u = kf_n-1.665;
      double Kn1 = (0.4583+0.892*u*u-0.5497*u*u*u-0.06205*kf_p+
                    0.04022*kf_p*kf_p+0.2122*u*kf_p)/(mst_n*mst_n);
      u = kf_n-1.556;
      double Kn2 = (0.4891+1.111*u*u-0.2283*u*u*u+0.01589*kf_p-
                    0.02099*kf_p*kf_p+0.2773*u*kf_p)/(mst_n*mst_n);
      u = kf_n-2.126;
      double Kp1 = (0.04377+1.100*u*u+0.1180*u*u*u+0.1626*kf_p+
                    0.3871*u*kf_p-0.2990*u*u*u*u)/(mst_p*mst_p);
      u = kf_n-2.116;
      double Kp2 = (0.0001313+1.248*u*u+0.2403*u*u*u+0.3257*kf_p+
                    0.5536*u*kf_p-0.3237*u*u*u*u+0.09786*u*u*kf_p)/(mst_p*mst_p);
      if (icontrol >= 2) {
        Kn1 = 1.0;
        Kn2 = 1.0;
        Kp1 = 1.0;
        Kp2 = 1.0;
      }
      // Pairing correction factors for 1S0 and 3P2
      double tau, yn, yp;
      if (Temp <= Tc_p) {
        tau = Temp/Tc_p;
        yp = u_1s0(tau);
      } else {
        yp = 0.0;
      }
      if (Temp <= Tc_n) {
        tau = Temp/Tc_n;
        if (isfn == 1) {
          yn = u_1s0(tau);
        } else if (isfn == 3) {
          yn = u_3p2(tau);
        } else {
          std::cout << "Subroutine con_core_bar: isfn badly defined !" << std::endl;
          return;
        }
      } else {
        yn = 0.0;
      }

      double Rn1, Rn2, Rp1, Rp2, RC;
      con_core_bar_pairing_supr(yn, yp, Rn1, Rn2, Rp1, Rp2, RC);

      double Snn = Sn2*Kn2*Rn2+3.0*Sn1*Kn1*(Rn1-Rn2);
      double Snp = Sp2*Kp2*Rp2+0.5*Sp1*Kp1*(3.0*Rp1-Rp2);

      // When T << Tc_p and/or Tc_n, Snn and Snp may vanish, so better use:
      Snn = std::max(Snn, 1.0e-200);
      Snp = std::max(Snp, 1.0e-200);
      if (icontrol == 3)
        Snp = 0.0;

      double nu_nn = 3.48e15*mst_n*mst_n*mst_n*std::pow(Temp/1.0e8,2)*Snn;
      double nu_np = 3.48e15*mst_n*mst_p*mst_p*std::pow(Temp/1.0e8,2)*Snp;
      double tau_n = RC/(nu_nn+nu_np);

      double lambda_n = 7.2e23*(Temp/1.0e8)*RC*RC/
                      (mst_n*1.0e15*(nu_nn+nu_np))*std::pow(kf_n/1.68,3);

      // Lambda contribution
      double lambda_la = 0.0;

      // Sigma0 contribution
      double lambda_s0 = 0.0;

      lambda_bar = lambda_n+lambda_la+lambda_s0;

      // Electrical conductivity
      sigma_bar = 0.0;

      if (debug == 1) {
        std::cout << "Exiting con_core_bar: sigma_bar, lambda_bar = " << sigma_bar << ", " << lambda_bar << std::endl;
      }

      nu_e_l = 0.0;
      nu_e_s = 0.0;
      return;
    }

    /** \brief 
     */
    void con_core_bar_pairing_supr(double yn, double yp, double &Rn1, double &Rn2,
                               double &Rp1, double &Rp2, double &RC) {
      if (yn == 0.0) {
        Rn1 = 1.0;
        Rn2 = 1.0;
        RC = 1.0;
      } else {
        Rn1 = (2.0/3.0)*(std::pow(0.9468+std::sqrt(0.0532*0.0532+0.5346*yn*yn),3.0)*
                                std::exp(0.377-std::sqrt(0.377*0.377+4.0*yn*yn))+
                            (1.0/3.0)*(std::pow(1.0+1.351*yn*yn,2.0)*
                        std::exp(0.169-std::sqrt(0.169*0.169+9.0*yn*yn))));

        Rn2 = 0.5*(std::pow(0.6242+std::sqrt(0.3758*0.3758+0.07198*yn*yn),3.0)*
                        std::exp(3.6724-std::sqrt(3.6724*3.6724+4.0*yn*yn))+
                    0.5*(std::pow(1.0+0.01211*yn*yn,9.0)*
                  std::exp(7.5351-std::sqrt(7.5351*7.5351+9.0*yn*yn))));

        RC = (std::pow(0.647+std::sqrt(0.353*0.353+0.109*yn*yn),1.5)*
              std::exp(1.39-std::sqrt(1.39*1.39+yn*yn)));
      }

      if (yn == 0.0 && yp == 0.0) {
        Rp1 = 1.0;
        Rp2 = 1.0;
      } else if (yn > 0.0 && yp == 0.0) {
        Rp1 = (std::pow(0.4459+std::sqrt(0.5541*0.5541+0.03016*yn*yn),2.0)*
              std::exp(2.1178-std::sqrt(2.1178*2.1178+yn*yn)));

        Rp2 = (std::pow(0.801+std::sqrt(0.199*0.199+0.04645*yn*yn),2.0)*
              std::exp(2.3569-std::sqrt(2.3569*2.3569+yn*yn)));
      } else if (yn == 0.0 && yp > 0.0) {
        Rp1 = 0.5*(0.3695+std::sqrt(0.6305*0.6305+0.01064*yp*yp))*
                  std::exp(2.4451-std::sqrt(2.4451*2.4451+yp*yp))+
              0.5*(std::pow(1.0+0.1917*yp*yp,1.4)*
                    std::exp(4.6627-std::sqrt(4.6627*4.6627+4.0*yp*yp)));

        Rp2 = 0.0436*(std::sqrt(3.345*3.345+19.55*yp*yp)-3.345)*
                  std::exp(2.0247-std::sqrt(2.0247*2.0247+yp*yp))+
              0.0654*std::exp(8.992-std::sqrt(8.992*8.992+1.5*yp*yp))+
              0.8910*std::exp(9.627-std::sqrt(9.627*9.627+9.0*yp*yp));
      } else {
        double y_p = std::max(yn,yp);
        double y_m = std::min(yn,yp);
        double u_p = std::sqrt(y_p*y_p+(1.485*1.485))-1.485;
        double u_m = std::sqrt(y_m*y_m+(1.485*1.485))-1.485;
        double up = std::sqrt(yp*yp+(1.485*1.485))-1.485;
        double un = std::sqrt(yn*yn+(1.485*1.485))-1.485;

        Rp1 = std::exp(-u_p-u_m)*(0.7751+0.4823*un+0.1124*up+
                                      0.04991*un*un+0.08513*un*up+0.01284*un*un*up)+
              std::exp(-2.0*u_p)*(0.2249+0.3539*u_p-0.2189*u_m-0.6069*un*u_m+0.7362*up*up);

        u_p = std::sqrt(y_p*y_p+(1.761*1.761))-1.761;
        u_m = std::sqrt(y_m*y_m+(1.761*1.761))-1.761;
        up = std::sqrt(yp*yp+(1.761*1.761))-1.761;
        un = std::sqrt(yn*yn+(1.761*1.761))-1.761;

        Rp2 = std::exp(-u_p-u_m)*(1.1032+0.8645*un+0.2042*up+
                                      0.07937*un*un+0.1451*un*up+0.01333*un*un*up)+
              std::exp(-2.0*u_p)*(-0.1032-0.2340*u_p+0.06152*un*u_p+0.7533*un*u_m-1.007*up*up);
      }
      return;
    }

    /** \brief From Yakovlev & Urpin, Sov. Astron. 24 (1980): 303                
      Calculates the conductivity due to electron-impurity scattering   
      in the crust in the crystalline phase. 
    */
    void con_crust_e_imp_YU(double T, double rho, double A, double A1, double Z, double Q,
                            double &sigma, double &lambda, int debug,
                            double &nu_e_s, double &nu_e_l) {

      // calculates a Coulomb potential using the Yukawa potential form.
      auto Coulomb_imp_YU = [] (double x) {
        double beta = x/std::sqrt(1.0+x*x);
        double q = 0.048196/std::sqrt(beta);
        return std::log(1.0/q)-0.5*(1.0+beta*beta);
      };

      if (debug == 1) {
        std::cout << "Entering con_crust_e_imp_YU: T, rho, A, A1, Z, Q = " 
                  << T << ", " << rho << ", " << A << ", "
                  << A1 << ", " << Z << ", " << Q << std::endl;
      }                        
      double x = 1.00884*std::pow(Z/A*rho/1.0e6,1.0/3.0);
      double nu = 1.75e16*std::sqrt(1.0+x*x)*Q/Z*Coulomb_imp_YU(x);
      lambda = 4.04e17*std::pow(x, 3)/std::sqrt(1.0+x*x)*(T/1.0e8)*(1.0e16/nu);

      sigma = 1.0;
      nu_e_s = 1.0;
      nu_e_l = 1.0;

      if (debug == 1) {
        std::cout << "Exiting con_crust_e_imp_YU" << std::endl;
      }
      return;
    }

    /** \brief calculates the thermal conductivity due to electron-ion scattering 
      in the crust of a neutron star, specifically in the liquid phase.
      **fix
    */
    void con_crust_e_ion_YU(double T, double rho, double a, double z, double& lbdaei) {

      double mue = (1.0/z)*a;
      double rho6 = rho/1.0e6;

      double beta2 = 1.02*std::pow(rho6/mue,2.0/3.0)/(1.0+1.02*std::pow(rho6/mue,2.0/3.0));
      double beta = std::sqrt(beta2);
      double gamma = 2.273e5*z*z*std::pow(rho/a,1.0/3.0)/T;
      double lambda = 1.0/6.0*std::log(3.0*pi*pi/2.0*z*z)+
                      0.5*std::log(1.0+2.0/gamma)-beta2/2.0;

      // Uncomment the following lines if you want to include the 
      // correction from 2nd order cross coulomb section
      // double alphab = 1.0/137.0/beta*z;
      // lambda += pi/2.0*alphab*beta2*(1.0+1.30*alphab)/(1.0+alphab*alphab*(0.71-0.54*beta2));

      double vf = beta*c;
      double nukap = lambda*z*std::sqrt(1.0+std::pow(rho6/mue,2.0/3.0))/5.65e-17;

      lbdaei = 4.11e15*(rho6/mue)/std::sqrt(1.0+std::pow(rho6/mue,2.0/3.0))
                *(T/1.0e6)*(1.0e16/nukap);

      return;
    }

    /** \brief From Yakovlev & Urpin, Sov. Astron. 24 (1980): 303 
      Calculates the conductivity due to electron-phonon scattering     
      in the crust in the crystalline phase.  
    */

    void con_crust_e_phonon_YU(double T, double rho, double A, double A1, 
                                double Z, double& lambda) {

      double mue = A/Z;
      double rho6 = rho/1.0e6;
      double nion = Na*rho/A1;
      double mion = A/Na;

      double theta = 2.4e6*std::sqrt(rho6/mue*2.0/mue);
      double omegap = e*Z*std::sqrt(4.0*pi*nion/mion);
      double x = theta/0.45/T;

      double phi0 = 13.0/std::sqrt(1.0+(theta/3.46/T)*(theta/3.46/T))/(x*x);
      double phi2 = 13.0*(theta/5.1/T)*(theta/5.1/T)/
                  std::pow(1.0+(theta/4.17/T)*(theta/4.17/T),1.5)*pi*pi/(x*x);

      double beta2 = 1.02*std::pow(rho6/mue,2.0/3.0)/
                    (1.0+1.02*std::pow(rho6/mue,2.0/3.0));
      double L_Coulomb = 1.0/3.0*std::log(4.0*Z)-beta2/2.0;

      double beta = std::sqrt(beta2);
      double vf = beta*c;
      double nukap = (e*e/hbar/vf)*omegap*x*((2.0-beta2)*phi0+
                    (3.0*L_Coulomb-1.0+beta2/2.0)*phi2/(pi*pi));

      lambda = 4.11e15*(rho6/mue)/
                    std::sqrt(1.0+std::pow(rho6/mue,2.0/3.0))*
                    (T/1.0e6)*(1.0e16/nukap);

      return;
    }

    /** \brief From: N.Itoh, S.Mitake, H.Iyetomi & S.Ichimaru, ApJ 273 (1983): 774   
      - S.Mitake, S.Ichimaru & N.Itoh, ApJ 277 (1984): 375              
      - D.Yakovlev, Sov.Astron. 31 (1987): 346                          

      Calculate the conductivity due to electron-ion scattering         
      in the crust in the liquid phase.
    */
    void con_crust_e_ion_Itoh(double T, double rho, double A, double A1, double Z,
                                double& sigma, double& lambda, double debug,
                                double& nu_e_s, double& nu_e_l) {
      double aliq[4] = {1.4453, -0.1561, 0.0941, -0.0263};
      double bliq[3] = {-1.5213, 0.8369, -0.4364};
      double cliq[3] = {0.6087, -3.1264, 1.8772};
      double dliq[4] = {0.4764, -0.0024, -0.0003, -0.0014};
      double eliq[3] = {-0.6640, 0.0656, -0.0346};
      double fliq[3] = {-0.5154, -0.1940, 0.0982};

      double rho6 = rho/1.0e6;
      double T8 = T/1.0e8;
      double gamma = 0.2275*Z*Z/T8*std::pow(rho6/A,1.0/3.0);
      double x2 = 1.018*std::pow(rho6*Z/A,2.0/3.0);
      double y = 1.656e-2/A1/T8*std::pow(rho6*Z/A, 2.0/3.0);
      double R = x2/(1.0+x2);
      double rs = 1.388e-2*std::pow(A/Z/rho6, 1.0/3.0);
      double u = 0.45641*std::log(gamma)-1.31636;

      // Classical contribution to the scattering integral s
      double sum1 = 1.0*aliq[0];
      double sum2 = 1.0+bliq[0]*rs+cliq[0]*rs*rs;
      for (int j=1;j<=2;j++) {
        sum1 += aliq[j]*std::pow(u, j);
        sum2 += bliq[j]*std::pow(u, j)*rs+cliq[j]*std::pow(u, j)*rs*rs;
      }
      sum1 += aliq[3]*std::pow(u, 3);
      double sm1 = sum1*sum2;
      sm1 = (1.0/3.0)*std::log(Z/26.0)+sm1;

      sum1 = 1.0*dliq[0];
      sum2 = 1.0+eliq[0]*rs+fliq[0]*rs*rs;
      for (int j=1;j<=2;j++) {
        sum1 += dliq[j]*std::pow(u, j);
        sum2 += eliq[j]*std::pow(u, j)*rs+fliq[j]*std::pow(u, j)*rs*rs;
      }
      sum1 += dliq[3] * std::pow(u, 3);
      double sp1 = sum1 * sum2;
      sp1 = 0.5 - std::pow(Z / 26.0, 2.0 / 3.0) * (0.5 - sp1);

      double s_s = sm1-R*sp1;
      double s_l = sm1-R*sp1;

      // Semiclassical corrections to s
      if (y >= 0.01) {
        double sp3 = 0.2493-0.1081*rs;
        double rk2 = 9.291*std::sqrt(1.0+1.0/x2);
        double im1tf = 0.5*(std::log(1.0+4.0/rk2)-1.0/(1.0+0.25*rk2));
        double im1 = im1tf*(1.0028+0.0896*rs);
        double ip1 = 0.4893-0.4573*rs+0.3429*rs*rs;
        double ip3 = 0.2484-0.0953*rs;
        double ds_s = -2.0/3.0*y*(ip1-R*ip3+sp1-R*sp3);
        double ds_l = ds_s+2.0*y/(pi*pi)*(3.0*im1-(2.0+3.0*R)*ip1+2.0*R*ip3);
        s_s += ds_s;
        s_l += ds_l;
      }

      // Correction from 2nd order Coulomb cross-section
      double mue = A/Z;
      double beta2 = 1.018*std::pow(rho6/mue, 2.0/3.0)/(1.0+1.018*std::pow(rho6/mue, 2.0/3.0));
      double beta = std::sqrt(beta2);
      double alphab = 1.0/137.0/beta*Z;
      double ds = pi/2.0*alphab*beta2*(1.0+1.30*alphab)/(1.0+alphab*alphab);
      s_s += ds;
      s_l += ds;

      sigma = 8.693e21*rho6/A*(1.0-R)/s_s;
      lambda = 2.363e17*rho6*T8/A*(1.0-R)/s_l;
      nu_e_l = (4.11/2.363)*1.0e16*Z*std::sqrt(1.0+x2)*s_l;
      nu_e_s = (4.11/2.363)*1.0e16*Z*std::sqrt(1.0+x2)*s_s;

      return;
    }

    /** \brief 
     */
    void con_crust_e_phonon_Itoh(double T, double rho, double A, double A1, double Z,
                             double sigma, double lambda, int debug,
                             double nu_s, double nu_k) {

      std::cout << 'AWS: Removed because this required a data file.' << std::endl;

      return;
    }

    /** \brief his subroutine is from Oleg et al code !
      
      ----------------------------------------------------
      The following block has been copied from "conrt.pas" (D.G.Yakovlev)
      and converted into Fortran. It realizes the SMOOTH COMPOSITION model
      Version 24.02.99
      Input: BARD (baryon density in fm^{-3}), Index (of phase)
      Output: Z (total number of protons inside the nucleus)
      Anuc (num.of barions within the nucleus), A (within the cell)
      tp (smoothness param.of proton core)
      xnuc (effective proton-core radius divided by the WS cell radius)
      xnuct (a second proton-core parameter for use in a quantum crystal)
      Internal variables:
      Nin=A-Z - total number of neutrons inside the nucleus
      Nfree - total number of free neutrons (incl.ones penetrating nuclei)
    */
    void OYAFORM( double BARD, int Index, double &Z, double &Anuc, 
                  double &A, double &xnuc, double &xnuct) {
      double f, Rws, tn, Rn, dn_n, Nin, tp, Rp, np_in;
      double Nfree, nn_out, nn_in, Rp0eff, Rp2eff, Rp1eff, Rp3eff;
      double Rp0eff, Rp2eff, Rp1eff, Rp3eff;

      // The following factor corrects the volume for nucl.shape acc.to Oyamatsu:
      // nn_in=nn_out+dn_n*(1-(r/Rn)^t); x=min(1,Rws/Rn); same for protons.

      auto SOyam = [] (double t, double x) {
        return x*x*x-9.0*pow(x, 3.0-t)/(3.0-t)+
              9.0*pow(x,3.0+2.0*t)/(3.0+2.0*t)-
              pow(x,3.0+3.0*t)/(1.0+t);
      };

      if (Index == 30) { // {densities lower than the neutron drip}
        f = log(1.0+BARD/5.0e-9);
        Rp = 5.688+0.02628*f+0.009468*f*f;      // max.proton core radius
        Rn = 5.788+0.02077*f+0.01489*f*f;       // max.neutron core radius
        np_in = 0.0738+1.22e-4*f-1.641e-4*f*f;  // centr.num.dens.of protons
        nn_in = 0.0808+1.688e-4*f+9.439e-5*f*f; // same for neutrons
        nn_out = 0.0;
        tp = 6.0;
        tn = tp;
        Nin = pi/0.75*pow(Rn, 3)*nn_in;
        Z = pi/0.75*pow(Rp, 3)*np_in;
        Anuc = Z+Nin; // {nucleons within a nucleus}
        A = Anuc;
        Rws = pow(A*0.75/pi/BARD, 0.333333);
        if (Rws<Rn) {
          std::cout << "OYAFORM: too large Rn for outer envelope!" << std::endl;
          return;
        }
        double aa = pow(A/BARD, 0.333333); // {cube size}
        xnuc = Rp/Rws;
        xnuct = xnuc*tp/(0.6+tp);
      } else if (Index == 3) { // {spheres after drip}
        double g = BARD*100.0;
        f = log(g);
        Rws = 31.68-8.400*f-0.2380*f*f+0.1152*f*f*f;
        tn = 1.0/(0.2027+0.004506*g); // param.of shape
        Rn = 9.406+1.481*f+0.4625*f*f+0.05738*f*f*f;
        dn_n = (9.761-1.322*f-0.5544*f*f-0.07624*f*f*f)/100.0; // n-height
        Nin = pi/0.75*pow(Rn,3)*dn_n*SOyam(tn,std::min(1.0,Rws/Rn));
        tp = 1.0/(0.1558+2.225e-3*g+9.452e-4*g*g);
        Rp = 8.345+0.7767*f+0.1333*f*f+0.008707*f*f*f;
        np_in = (4.040-1.097*f-0.0723*f*f+0.0225*f*f*f)/100.0;
        Z = pi/0.75*pow(Rp, 3)*np_in*SOyam(tp,std::min(1.0,Rws/Rp));
        Nfree = BARD*pi/0.75*pow(Rws,3)-Z-Nin; // free neutrons outside+under nuc.
        nn_out = Nfree/(pi/0.75*pow(Rws,3));       // free neutron density
        nn_in = nn_out+dn_n;                            // max.n-density
        A = Z + Nfree+Nin;                              // total num.of barions in the cell (A')
        Anuc = Z+Nin+Nfree*pow(Rn/Rws,3);        // number of barions within Rn
        if (Rn>Rws)
          Anuc = A;
        double aa = pow(A/BARD,0.333333); // {cube size}
        xnuc = Rp/Rws;
        xnuct = xnuc*tp/(0.6+tp);
      } else {
        std::cout << "OYAFORM: invalid Index" << std::endl;
        // Probably have to set all the outputs to zero
        return;
      }

      Rp0eff = pow(Z/pi*0.75/np_in,0.333333);
      Rp2eff = Rp*sqrt((1.0-15.0/(5.0+tp)+15.0/(5.0+2.0*tp)-5.0/(5.0+3.0*tp))/SOyam(tp,1.0));
      Rp1eff = Rp*(1.0-12.0/(4.0+tp)+12.0/(4.0+2.0*tp)-4.0/(4.0+3.0*tp))/SOyam(tp,1.0);
      Rp3eff = Rp*pow((1.0-18.0/(6.0+tp)+18.0/(6.0+2.0*tp)-6.0/(6.0+3.0*tp))/SOyam(tp,1.0),0.333333);
      xnuc = Rp2eff/Rws;
      xnuct = xnuc*tp/(0.6+tp);
      return;
    }

    /** \brief CHECKED on Dec. 3, 2003 against table in Abramowitz & Stegun:
      accurate to 10^-5 for all x
    */
    double exp_int(double x) {
      if (x<=0.0) {
        std::cout << "exp_int: x must be > 0 !" << std::endl;
        exit(EXIT_FAILURE);
      }

      double result;
      if (x>=1.0) {
        double num = x*x*x*x + 8.5733287401*x*x*x+18.0590169730*x*x+
                    8.6347608925*x+0.2677737343;
        double den = x*x*x*x+9.5733223454*x*x*x+25.6329561486*x*x+
                    21.0996530827*x+3.9584969228;
        result = num/den/(x*exp(x));
      } else {
        result = -0.57721566+0.99999193*x-0.24991055*x*x+0.05519968*x*x*x-
                0.00976004*x*x*x*x+0.00107857*x*x*x*x*x-log(x);
      }

      return result;
    }

    /** \brief Desc
     */
    void get_lam(double s, double w, double &Lam1, double &Lam2) {
      const double eps = 0.05;

      if (s<=eps && s*w<=eps) {
        Lam1 = 0.5*(exp_int(w)+log(w)+0.5772156);
        Lam2 = (exp(-w)-1.0+w)/(2.0*w);
      } else if (w<=eps) {
        Lam1 = w*((2.0*s+1.0)/(2.0*s+2.0)-s*log((s+1.0)/s));
        Lam2 = w*((1.0-3.0*s-6.0*s*s)/(4.0*s+4.0)+1.5*log((s+1.0)/s));
      } else if (w>1.0/eps) {
        Lam1 = 0.5*(log((s+1.0)/s)-1.0/(s+1.0));
        Lam2 = (2.0*s+1.0)/(2.0*s+2.0)-s* log((s+1.0)/s);
      } else {
        double exp_sw = exp(s*w);
        Lam1 = 0.5*(log((s+1.0)/s)+s/(s+1.0)*(1.0-exp(-w))-
                      (1.0+s*w)*exp_sw*(exp_int(s*w)-exp_int(s*w+w)));

        Lam2 = (exp(-w)-1.0+w)/w-s*s/(s+1.0)*(1.0-exp(-w))-
              2.0*s*log((s+1.0)/s)+s*(2.0+s*w)*exp_sw*(exp_int(s*w)-exp_int(s*w+w));
        Lam2 = 0.5*Lam2;
      }
      return;
    }

    /** \brief Calculates the conductivity due to electron-phonon scattering by   
      umklapp processes in the crust in the crystalline phase AND        
      the electron-ion contribution in the liquid phase !                
                                                                         
      Units are cgs-K                                                    
                                                                         
      From Appendix of Gnedin et al, MNRAS 324 (2001): 725               
      modified from Potekhin et al, A&A 346 (1999): 345                  
      Checked on Jan. 9, 2010: (almost) exactly reproduces Fig 4 and A.1 
      of the paper 
    */
    void con_e_phon_ion_GYP(double T, double rho, double A_in, double A1_in, double Z_in, int ifs,
                            double &sigma, double &lambda, double debug, double &nu_e_s, double &nu_e_l, double rhodrip) {

      double BARD = rho/mu*1.0e-39;
      int Index = (rho>rhodrip) ? 3 : 30;

      double Z,A1,A,xnuc,xnuct;
      OYAFORM(BARD,Index,Z,A1,A,xnuc,xnuct);
      if (ifs==0){
        xnuc=0.0;
        xnuct=0.0;
      } else if (ifs!=1) {
        std::cout << "Sub. ''con_e_phon_ion_GYP'': ifs badly defined" << std::endl;
        exit(-1);
      }

      double n_i = rho/(A*mu);
      double n_e = Z*n_i;
      double kf = pow(3.0*pi*pi*n_e,1.0/3.0);
      double pf = hbar*kf;
      double m_st = sqrt(me*me+(pf/c)*(pf/c));
      double Ef = m_st*c*c;
      double vf = pf/m_st;
      double x = pf/me/c;

      double Omega_p = sqrt(4.0*pi*e*e*Z*Z*n_i/A1/mu);
      double T_p = hbar*Omega_p/kb;
      double tp = T/T_p;
      double betaZ = pi*a_f*Z*vf/c;
      double ai = pow(3.0/(4.0*pi*n_i),1.0/3.0);
      double Gamma = Z*Z*e*e/(kb*T*ai);

      double r_D = ai/sqrt(3.0*Gamma);
      double s_D = 1.0/((2.0*kf*r_D)*(2.0*kf*r_D));
      double s_i = s_D*(1.0+0.06*Gamma)*exp(-sqrt(Gamma));
      double s_e = a_f/pi*c/vf;
      double s = (s_i+s_e)*exp(-betaZ);
      double w = (u_2/s_D)*(1.0+betaZ/3.0);
      double w1 = 14.73*xnuc*xnuc*(1.0+Z*sqrt(xnuc)/13.0)*(1.0+betaZ/3.0);

      double G_s = 1.0/sqrt(1.0+0.0361/pow(Z, 1.0/3.0)/(tp*tp))*(1.0+0.122*betaZ*betaZ);
      double G_l = G_s+0.0105*tp/pow(tp*tp+0.0081,1.5)*(1.0+(vf/c)*
                  (vf/c)*vf*betaZ)*(1.0-1.0/Z)*(1.0+xnuct*xnuct*sqrt(2.0*Z));

      double D = exp(-0.42*sqrt(x/A/Z)*u_1*exp(-9.1*tp));

      // Get Lambda values from get_lam function
      double Lam1a, Lam2a, Lam1b, Lam2b;
      w = w+w1;
      get_lam(s, w, Lam1a, Lam2a);
      double Lama = Lam1a-(vf/c)*(vf/c)*Lam2a;
      w = w1; // Replace with your desired value
      get_lam(s, w, Lam1b, Lam2b);
      double Lamb = Lam1b-(vf/c)*(vf/c)*Lam2b;
        
      double Lam = Lama - Lamb;
      // Coulomb logarithms from umklapp processes, i.e., at high T:  
      double Lam_s_hT = Lam*G_s*D;
      double Lam_l_hT = Lam*G_l*D;
      // Coulomb logarithms with umklapp processes frozen, i.e., at low T:  
      double T_u = T_p*pow(Z, 1.0/3.0)*a_f/(3.0*vf)*c;
      double Lam_0_lT = 50.0*sqrt(x)/sqrt(A1)/Z;
      double Lam_s_lT = Lam_0_lT*(4.0/3.0)*a_f*c/vf*pow(tp,5);
      double Lam_l_lT = Lam_0_lT*pow(tp,3);
      // Inerpolation, for all T:    
      double ww = exp(-T_u/T);
      double Lam_s = Lam_s_hT*ww+Lam_s_lT*(1.0-ww);
      double Lam_l = Lam_l_hT*ww+Lam_l_lT*(1.0-ww);
      //  Collisional frequencies:  
      double nu0 = 4.0*Z*Ef*a_f*a_f/(3.0*M_PI*hbar);
      double nu_s = nu0*Lam_s;
      double nu_l = nu0*Lam_l;
      
      sigma = n_e*e*e/(m_st*nu_s);
      lambda = pi*pi*kb*kb*T*n_e/(3.0*m_st*nu_l);

      nu_e_s = nu_s;
      nu_e_l = nu_l;

      if (debug == 1.0) {
        std::cout << "Exiting con_crust_e_phon_ion: sigma, lambda=" << sigma << ", " << lambda << std::endl;
      }
      return;
    }

    /** \brief a = nucleons per cell   a1= nucleons per nucleus
 
      Calculates the conductivity due to electron-phonon scattering by  
      umklapp processes in the crust in the crystalline phase AND       
      the electron-ion contribution in the liquid phase !               
                                                                         
      Really valid only at Rho < 10^10 g/cm^3                           
                                                                         
      Units are cgs-K                                                   
                                                                        
      From Potekhin et al, A&A 346 (1999): 345                          
      Checked on Dec. 4, 2003                                           
    */
    void con_env_e_phon_ion_PBHY(double T, double rho, double A, double A1, 
                                  double Z, double &sigma, double &lambda, 
                                  double debug, double &nu_e_s, double &nu_e_l) {

      double n_i = rho/A*6.022e23;
      double n_e = Z*n_i;

      double kf = pow(3.0*pi*pi*n_e,1.0/3.0);
      double pf = hbar*kf;
      double m_st = sqrt(me*me+(pf/c)*(pf/c));
      double Ef = m_st*c*c;
      double vf = pf/m_st;

      double omega_p = sqrt(4.0*pi*e*e*Z*Z*n_i/A1/ mu);
      double T_p = hbar/kb*omega_p;
      double eta = T/T_p;
      double beta = pi*a_f*Z*vf/c;
      double ai = pow(3.0/4.0/pi/n_i,1.0/3.0);
      double Gamma = e*e*Z*Z/kb/T/ai;
      double q_D2 = 3.0*Gamma/ai/ai;
      double k_TF2 = 4.0*a_f/pi*c/vf*kf*kf;

      double q_i2 = q_D2*(1.0+0.06*Gamma)*exp(-sqrt(Gamma));
      double q_s2 = (q_i2+k_TF2)*exp(-beta);
      double s = q_s2/(2.0*kf)*(2.0*kf);
      double w = u_2*(2.0*kf)*(2.0*kf)/q_D2*(1.0+beta/3.0);

      double eta_02 = pow(0.19, 2)/pow(Z, 1.0/3.0);
      double G_s = eta/sqrt(eta*eta+eta_02)*(1.0+0.122*beta*beta);
      double G_l = G_s+
                    0.0105*(1.0-1.0/Z)*(1.0+pow(vf/c, 3)*beta)*
                        eta/pow(eta*eta+0.0081, 1.5);

      double Lam1, Lam2, Lam;
      if ((s<=0.01) && (s*w<=0.01)) {
        Lam1 = 0.5*(exp(w)-log(w)+0.5772156);
        Lam2 = (exp(-w)-1.0+w)/(2.0*w);
      } else if (w<=0.01) {
        Lam1 = w*((2.0*s+1.0)/(2.0*s+2.0)-s*log((s+1.0)/s));
        Lam2 = w*((1.0-3.0*s-6.0*s*s)/(4.0*s+4.0)+
                        1.5*log((s+1.0)/s));
      } else if (w>100.0) {
        Lam1 = 0.5*(log((s+1.0)/s)-1.0/(s+1.0));
        Lam2 = ((2.0*s+1.0)/(2.0*s+2.0)-s*log((s+1.0)/s));
      } else {
        Lam1 = log((s+1.0)/s)+s/(s+1.0)*(1.0-exp(-w))-
                  (1.0+s*w)*exp(s*w)*(exp(w)-exp(w+s*w));
        Lam1 = 0.5*Lam1;
        Lam2 = (exp(-w)-1.0+w)/w-s*s/(s+1.0)*(1.0-exp(-w)) -
                  2.0*s*log((s+1.0)/s)+
                  s*(2.0+s*w)*exp(s*w)*(exp(w)-exp(w+s*w));
        Lam2 = 0.5*Lam2;
      }
      Lam = Lam1-(vf/c)*(vf/c)*Lam2;

      double Lam_s = Lam*G_s;
      double Lam_l = Lam*G_l;

      double nu0 = 4.0*pi*Z*Z*e*e*n_i/(pf*pf)/vf;
      double nu_s = nu0*Lam_s;
      double nu_l = nu0*Lam_l;

      sigma = n_e*e*e/(m_st*nu_s);
      lambda = pi*pi*kb*kb*T*n_e/(3.0*m_st*nu_l);

      nu_e_s = nu_s;
      nu_e_l = nu_l;

      if (debug == 1.2) {
        std::cout << "Exiting con_crust_e_phon_ion: sigma, lambda=" << sigma << ", " << lambda << std::endl;
      }
    }

    /** \brief Calculates the conductivity due to electron-phonon scattering by  
      umklapp processes in the crust in the crystalline phase.          
      Units are cgs-K                                                   
                                                                         
      From Baiko & Yakovlev, Astron. Lett. 22 (1996): 708               
      Checked on JULY 19, 2000                                          
    */
    void con_crust_e_phonon_BY(double Temp, double rho, double A, double A1, double Z,
                              double &sigma, double &lambda, int debug,
                              double &nu_e_s, double &nu_e_l, double rhodrip) {

      auto ei = [] (double q) {
        return exp(-q*q*q/(q*q*q+0.1397))* 
              (log(1+1/q)-0.5772/(1+2.2757*q*q));
      };
      double rho6 = rho/1e6;
      double T8 = Temp/1e8;
      double n_i = rho/A*6.022e23;
      double a_WS = pow(3.0/(4.0*pi*n_i),1.0/3.0);
      double r_nucl;
        
      if (rho < rhodrip) {
        r_nucl = 1.15e-13*pow(A, 1.0/3.0);
      } else {
        r_nucl = 1.83e-13*pow(Z, 1.0/3.0);
      }
      double g2 = pow(r_nucl/a_WS, 2);
        
      double x = 1.0088*pow(rho6*Z/A,1.0/3.0);
      double beta = x/sqrt(1.0+x*x);
      double gam = 7.832e-2*Z/T8*pow(rho6/A/A1, 0.5);
      
      double G_0 = u_2/sqrt(1.0+a_0*gam*gam);
      double G_2 = gam*gam/(pi*pi*sqrt(1.0+a_2*gam*gam)*sqrt(1.0+a_2*gam*gam));
        
      double ue = 1.0/(137.0*pi*beta);
      double u1 = 1.0/pow(4.0*Z, 2.0/3.0)+ue;
      double alpha0 = 1.683*sqrt(x/A1/Z);
      double alpha = alpha0*(0.5*u_1*exp(-9.1/gam)+u_2/gam);
        
      double w = alpha*u1;
      double S__1 = ei(w)-ei(alpha);
      double S_0 = (exp(-w)-exp(-alpha))/alpha;
      double S_1 = (exp(-w)*(w+1)-exp(-alpha)*(alpha+1))/(alpha*alpha);
      double S_2 = (exp(-w)*(w*w+2*w+2)-exp(-alpha)*(alpha*alpha+2*alpha+2))/
                      (alpha*alpha*alpha);
        
      double Phi_0 = S__1-beta*beta*S_0;
      double Phi_1 = S_0-beta*beta*S_1;
      double Phi_2 = S_1-beta*beta*S_2;
        
      double P0 = 4.787-0.0346*Z;
      double P2 = 2.729-0.0204*Z;
        
      double K_0 = 2.0*Phi_1/(1.0+pow(18.0*pi*Z,2.0/3.0)*g2*Phi_2/(5.0*Phi_1*P0));
      double K_2 = 0.5 * Phi_0/(1.0+pow(18.0*pi*Z,2.0/3.0)*g2*Phi_1/(5.0*Phi_0*P2));
        
      double F_s = G_0*K_0;
      double F_k = G_0*K_0+G_2*(3.0*K_2-0.5*K_0);
        
      double nu_s = 0.9554*T8/beta*F_s;
      double nu_k = 0.9554*T8/beta*F_k;
        
      sigma = 1.49e22*x*x*beta/(1.0e16*nu_s);
      lambda = 4.04e15*x*x*beta*T8/(1.0e18*nu_k);
        
      nu_e_s = nu_s;
      nu_e_l = nu_k;

      if (debug == 1) {
        std::cout << "Exiting con_crust_e_phonon: sigma, lambda=" << sigma << ", " << lambda << std::endl;
      }
      return;
    }

    /** \brief Calculates the conductivity due to electron-electron scattering   
      in the crust.                                                     
      Units are cgs-K, except kfe in fm^-1                              
                                                                        
      From Shternin & Yakovlev, Phys. Rev. D74, 043004 (2006)           
                                                                         
      Checked on August 28, 2009, against Fig 2 of the paper.           
    */
    void con_crust_ee(double T, double ne, double kfe, double &lambda, int debug) {
      double I_l, I_t, I_lt;
        
      double xe = 197.3*kfe/0.511;
      double gammae = sqrt(1.0+xe*xe);
      double u = xe/gammae;
      double mste = gammae*me;
      double om_pe = sqrt(4.0*pi*e*e*ne/mste);
      double T_pe = hbar*om_pe/kb;
      double th = sqrt(3.0)*T_pe/T;
        
      I_l = (0.1587-0.02538/(1.0+0.0435*th))* 
            log(1.0+128.56/(37.1*th+10.83*th*th+th*th*th))/u;

      double A = 20.0+450.0*u*u*u;
      double C1 = 0.05067+0.03216*u*u;
      double C2 = 0.0254+0.04127*u*u*u*u;
      double C = A*exp(C1/C2);
      I_t = u*u*u*(2.404/C+(C2-2.404/C)/(1.0+0.1*th*u))* 
            log(1.0+C/(A*th*u+th*th*u*u));

      A = 12.2+25.2*u*u*u;
      double B = 1.0-0.75*u;
      double C1_lt = 0.123636+0.016234*u*u;
      double C2_lt = 0.0762+0.05714*u*u*u*u;
      double C_lt = A*exp(C1_lt/C2_lt);
      I_lt = u * (18.52*u*u/C_lt+(C2_lt-18.52*u*u/C_lt)/ 
                  (1.0+0.1558*pow(th,B)))* 
            log(1.0+C_lt/(A*th+10.83*th*th*u*u+pow(th*u,8.0/3.0)));
        
      double alpha = 1.0/137.036;
      lambda = pow(pi,3)*kb*kb*T*T / 
              (108*alpha*alpha*hbar*hbar*c*(I_l+I_t+I_lt));

      if (debug == 1) {
        std::cout << "Exiting con_crust_ee" << std::endl;
      }
    }

    /** \brief Desc
     */
    void con_crust(int icon_crust, double debug, double T, double rho, 
                double kfe, double A, double A1, double Z,
               double Q_imp, double &sigma, double &lambda, 
               double &nu_e_s, double &nu_e_l, double rhodrip) {
      const double gammacryst = 210.0;
      const double gammaliq = 180.0;
      double sigma_ph, lambda_ph, nu_e_s_ph, nu_e_l_ph;
      double sigma_imp, lambda_imp, nu_e_s_imp, nu_e_l_imp;
      double sigma1, lambda1, nu_e_s1, nu_e_l1;
      double sigma2, lambda2, nu_e_s2, nu_e_l2;
      double sigma2_ph, lambda2_ph, nu_e_s2_ph, nu_e_l2_ph;
      double sigma2_imp, lambda2_imp, nu_e_s2_imp, nu_e_l2_imp;
      double sigma_ee, lambda_ee;

      auto smooth = [] (double x) {
        return 6.0*x*x*x*x*x-15.0*x*x*x*x+10.0*x*x*x;
      };

      if (debug == 1.2) {
        std::cout << "Entering con_crust: T, rho, A, A1, Z, Q_imp =" 
                  << T << ", " << rho << ", " << A << ", " << A1
                  << ", " << Z << ", " << Q_imp << std::endl;
      }

      double gamma = 2.273e5*Z*Z*pow(rho/A, 1.0/3.0)/T;

      // Check the regime and call appropriate subroutines accordingly
      if (rho >= 6.0e7) {
        // CRUST REGIME
        if (gamma > gammacryst) {
          // SOLID
          if (icon_crust == 1) {
            // icon_crust = 1
            con_crust_e_phonon_Itoh(T, rho, A, A1, Z, sigma_ph, lambda_ph, debug, nu_e_s_ph, nu_e_l_ph);
            con_crust_e_imp_YU(T, rho, A, A1, Z, Q_imp, sigma_imp, lambda_imp, debug, nu_e_s_imp, nu_e_l_imp);
            nu_e_s = nu_e_s_ph+nu_e_s_imp;
            nu_e_l = nu_e_l_ph+nu_e_l_imp;
            sigma = 1.0/(1.0/sigma_ph+1.0/sigma_imp);
            lambda = 1.0/(1.0/lambda_ph+1.0/lambda_imp);
          } else if (icon_crust == 2) {
            // icon_crust = 2
            con_crust_e_phonon_BY(T, rho, A, A1, Z, sigma_ph, lambda_ph, debug, nu_e_s_ph, nu_e_l_ph, rhodrip);
            con_crust_e_imp_YU(T, rho, A, A1, Z, Q_imp, sigma_imp, lambda_imp, debug, nu_e_s_imp, nu_e_l_imp);
            nu_e_s = nu_e_s_ph+nu_e_s_imp;
            nu_e_l = nu_e_l_ph+nu_e_l_imp;
            sigma = 1.0/(1.0/sigma_ph+1.0/sigma_imp);
            lambda = 1.0/(1.0/lambda_ph+1.0/lambda_imp);
          } else if (icon_crust == 3) {
            // icon_crust = 3
            con_e_phon_ion_GYP(T, rho, A, A1, Z, 1, sigma_ph, lambda_ph, debug, nu_e_s_ph, nu_e_l_ph, rhodrip);
            con_crust_e_imp_YU(T, rho, A, A1, Z, Q_imp, sigma_imp, lambda_imp, debug, nu_e_s_imp, nu_e_l_imp);
            nu_e_s = nu_e_s_ph+nu_e_s_imp;
            nu_e_l = nu_e_l_ph+nu_e_l_imp;
            sigma = 1.0/(1.0/sigma_ph+1.0/sigma_imp);
            lambda = 1.0/(1.0/lambda_ph+1.0/lambda_imp);
          } else {
            std::cout << "sub. con_crust: icon_crust incorrectly def !" << std::endl;
            exit(1);
          }
        } else if (gamma < gammaliq) {
          // LIQUID
          if (icon_crust == 1 || icon_crust == 2) {
            // icon_crust = 1 or 2
            con_crust_e_ion_Itoh(T, rho, A, A1, Z, sigma, lambda, debug, nu_e_s, nu_e_l);
          } else if (icon_crust == 3) {
            // icon_crust = 3
            con_e_phon_ion_GYP(T, rho, A, A1, Z, 1, sigma, lambda, debug, nu_e_s, nu_e_l, rhodrip);
          } else {
            std::cout << "sub. con_crust: icon_crust incorrectly def !" << std::endl;
            exit(1);
          }
        } else {
          // SOLID-LIQUID
          if (icon_crust == 1) {
            // icon_crust = 1
            con_crust_e_ion_Itoh(T, rho, A, A1, Z, sigma1, lambda1, debug, nu_e_s1, nu_e_l1);
            con_crust_e_phonon_Itoh(T, rho, A, A1, Z, sigma2_ph, lambda2_ph, debug, nu_e_s2_ph, nu_e_l2_ph);
            con_crust_e_imp_YU(T, rho, A, A1, Z, Q_imp, sigma2_imp, lambda2_imp, debug, nu_e_s2_imp, nu_e_l2_imp);
            nu_e_s2 = nu_e_s2_ph+nu_e_s2_imp;
            nu_e_l2 = nu_e_l2_ph+nu_e_l2_imp;
            sigma2 = 1.0/(1.0/sigma2_ph+1.0/sigma2_imp);
            lambda2 = 1.0/(1.0/lambda2_ph+1.0/lambda2_imp);
            double w = (gamma-gammaliq)/(gammacryst-gammaliq);
            double w2 = smooth(w);
            double w1 = 1.0-w2;
            nu_e_s = w1*nu_e_s1+w2*nu_e_s2;
            nu_e_l = w1*nu_e_l1+w2*nu_e_l2;
            lambda = w1*lambda1+w2*lambda2;
            sigma = w1*sigma1+w2*sigma2;
          } else if (icon_crust == 2) {
            // icon_crust = 2
            con_crust_e_ion_Itoh(T, rho, A, A1, Z, sigma1, lambda1, debug, nu_e_s1, nu_e_l1);
            con_crust_e_phonon_BY(T, rho, A, A1, Z, sigma2_ph, lambda2_ph, debug, nu_e_s2_ph, nu_e_l2_ph, rhodrip);
            con_crust_e_imp_YU(T, rho, A, A1, Z, Q_imp, sigma2_imp, lambda2_imp, debug, nu_e_s2_imp, nu_e_l2_imp);
            nu_e_s2 = nu_e_s2_ph+nu_e_s2_imp;
            nu_e_l2 = nu_e_l2_ph+nu_e_l2_imp;
            sigma2 = 1.0/(1.0/sigma2_ph+1.0/sigma2_imp);
            lambda2 = 1.0/(1.0/lambda2_ph+1.0/lambda2_imp);
            double w = (gamma-gammaliq)/(gammacryst-gammaliq);
            double w2 = smooth(w);
            double w1 = 1.0-w2;
            nu_e_s = w1*nu_e_s1+w2*nu_e_s2;
            nu_e_l = w1*nu_e_l1+w2*nu_e_l2;
            lambda = w1*lambda1+w2*lambda2;
            sigma = w1*sigma1+w2*sigma2;
          } else if (icon_crust == 3) {
            // icon_crust = 3
            con_e_phon_ion_GYP(T, rho, A, A1, Z, 1, sigma, lambda, debug, nu_e_s, nu_e_l, rhodrip);
          } else {
            std::cout << "sub. con_crust: icon_crust incorrectly def !" << std::endl;
            exit(1);
          }
        }
      } else if (rho < 6.0e7) {
        // ENVELOPE REGIME
        con_env_e_phon_ion_PBHY(T, rho, A, A1, Z, sigma1, lambda1, debug, nu_e_s1, nu_e_l1);
        con_crust_e_imp_YU(T, rho, A, A1, Z, Q_imp, sigma2, lambda2, debug, nu_e_s2, nu_e_l2);
        sigma = sigma1*sigma2/(sigma1+sigma2);
        lambda = lambda1*lambda2/(lambda1+lambda2);
        nu_e_s = nu_e_s1+nu_e_s2;
        nu_e_l = nu_e_l1+nu_e_l2;
      }

      // Add the e-e contribution
      double ne = kfe*kfe*kfe/(3.0*pi*pi)*1.0e39;
      con_crust_ee(T, ne, kfe, lambda_ee, debug);

      // Combine with e-e contribution
      lambda = lambda*lambda_ee/(lambda+lambda_ee);

      if (debug == 1.2) {
        std::cout << "Exiting con_crust: sigma, lambda = " << sigma << ", " << lambda << std::endl;
      }
      return;
    }

    /** \brief Desc
     */
    void conductt(int i, double T, double rho, double A, double A1, double Z, double Q,
             double magfield, double &sigma, double &lambda, int debug,
             double &nu_e_s, double &nu_e_l, int icon_crust, int icon_core, double rhodrip, double rhocore,
             std::vector<double> kfe, std::vector<double> kfm, std::vector<double> kfn,
             std::vector<double> kfp, std::vector<double> kfla, std::vector<double> kfsm,
             std::vector<double> kfs0, std::vector<double> kfsp, std::vector<double> mstn,
             std::vector<double> mstp, std::vector<double> mstla, std::vector<double> mstsm,
             std::vector<double> msts0, std::vector<double> mstsp, std::vector<double> tcn,
             std::vector<double> tcp, std::vector<double> tcla, std::vector<double> tcsm,
             std::vector<double> tcs0, std::vector<double> tcsp, int isf,
             std::vector<double> fhad, int istrange) {
      int isfn = 0;
      //int istrange=0;

      if (debug>=2)
        std::cout << "Entering conduct: T, rho, A, A1, Z, Qimp = " 
                  << T << ", " << rho << ", " << A << ", " << A1 
                  << ", " << Z << ", " << Q << std::endl;

      if (rho>=rhocore) {
        if (istrange==0) {
          if (i<=isf)
            isfn=3;
          else
            isfn=1;

          con_core(icon_core, debug, T, kfe[i], kfm[i], kfp[i], mstp[i], tcp[i], kfn[i], mstn[i],
                  tcn[i], isfn, kfla[i], mstla[i], tcla[i], kfsm[i], mstsm[i], tcsm[i],
                  kfs0[i], msts0[i], tcs0[i], kfsp[i], mstsp[i], tcsp[i], fhad[i], sigma, lambda, nu_e_s, nu_e_l);
        } else if (istrange == 1) {
          lambda = c_con_str/std::pow(T/1e9,p_con_str);
        } else {
          std::cout << "conduct: istrange not defined !" << std::endl;
          exit(-1);
        }
      } else {
        con_crust(icon_crust, debug, T, rho, kfe[i], A, A1, Z, Q, sigma, lambda, nu_e_s, nu_e_l, rhodrip);
      }

      if (debug>=2)
        std::cout << "Exiting conduct: sigma, lambda = " 
                  << sigma << ", " << lambda << std::endl;
      return;
    }
};

#endif