/*
  -------------------------------------------------------------------
  
  Copyright (C) 2015-2024, Andrew W. Steiner and Satyajit Roy
  
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
#ifndef NSCOOL_NEUTRINO_H
#define NSCOOL_NEUTRINO_H

#include <boost/numeric/ublas/vector.hpp>

#include <o2scl/constants.h>
#include <o2scl/interp2_direct.h>
#include <o2scl/fermion.h>

typedef boost::numeric::ublas::vector<double> ubvector;
typedef boost::numeric::ublas::matrix<double> ubmatrix;

/** \brief Neutrino emissivities

    This work in progress will eventually replace Dany's Fortran code.
*/

class neutrino {

  private:
    static constexpr double pi = 3.14159265;
    static constexpr double g_fermi = 1.436e-49;
    static constexpr double theta_c = 0.239;
    static constexpr double mev = 1.6e-6;
    static constexpr double h_bar = 1.054e-27;
    static constexpr double kb = 1.38e-16;
    static constexpr double c_light = 3.0e10;

  public:

    /** \brief Interpolation object
     */
    o2scl::interp2_direct<> id;

    /** \brief Desc
     */
    ubvector lgtau1;

    /** \brief Desc
     */
    ubvector lgtau2;

    /** \brief Desc
     */
    ubmatrix lgr;

    double alpha_c;

    double murca_increase;

    double strange_mass;

    double rhoexo,cexo;

    int pexo, pexosn, pexosp;

    int nonothing1,nonothing2,nonothing3,nonothing4;

    int inu_durca,inu_eion,inu_plasma,inu_synch,inu_n1s0_pbf,inu_n3p2_pbf,
        inu_p_pbf,inu_bubble,inu_photo,inu_pair,inu_nuts1,inu_nuts2,
        inu_nuts3,inu_nuts4,inu_nuts5;

    neutrino() {
    
      alpha_c = 0.0;
      strange_mass = 0.0;
      murca_increase=0.0;
      inu_durca=1;
      inu_eion=1;
      inu_plasma=1;
      inu_synch=0;
      inu_n1s0_pbf=1;
      inu_n3p2_pbf=1;
      inu_p_pbf=1;
      inu_bubble=0;
      inu_photo=0;
      inu_pair=0;
      inu_nuts1=0;
      inu_nuts2=0;
      inu_nuts3=0;
      inu_nuts4=0;
      inu_nuts5=0;
      rhoexo=1.2e25;
      cexo=1.0e25;
      pexo=0;
      pexosn=0;
      pexosp=0;
      nonothing1=0;
      nonothing2=0;
      nonothing3=0;
      nonothing4=0;

      lgtau1.resize(35);
      lgtau2.resize(35);
      lgr.resize(35,35);
      #include "../sf_suppression2.h"
      id.set_data(35,35,lgtau1,lgtau2,lgr);

      return;
    }

    /** \brief Includes the Levenfish & Yakovlev 
     suppression factors for MURCA
    */
    void numurca_q(int i, double t, double rho, double& qmurca_q,
                   const std::vector<double>& kfqu,
                   const std::vector<double>& tcu,
                   const std::vector<double>& tcd) {
      double num_coeff = 1.0; // Remains to be calculated

      // Calculate the initial value of qmurca_q
      qmurca_q = num_coeff * 
                    std::pow((alpha_c * g_fermi * cos(theta_c) / 
                    std::pow(h_bar, 5) / std::pow(c_light, 4)), 2) *
                    (1.0e13 * kfqu[i] * h_bar) * std::pow(kb * t, 8);

      // Up quark pairing suppression
      double r_u = (t < tcu[i]) ? exp(-1.76 * tcu[i] / t) : 1.0;

      // Down quark pairing suppression
      double r_d = (t < tcd[i]) ? exp(-1.76 * tcd[i] / t) : 1.0;

      // If we need Strange quark suppression, uncomment the following:
      // double r_s = (t < tcs[i]) ? exp(-1.76 * tcs[i] / t) : 1.0;

      // Combine suppression factors
      double r_ud = r_u * r_d;
      // double r_us = r_u * r_s; // Uncomment if r_s is needed
      
      // Update qmurca_q with the suppression factor
      qmurca_q *= r_ud;
      
    }

    /** \brief Desc
    */
    double func_r_1s0_1s0(double v1, double v2) {
      double gamma = 5040.0/(457.0*pow(pi,6));
      double u = v1*v1+v2*v2;
      double w = v1*v1-v2*v2;
      double u1 = 1.8091+sqrt(v1*v1+2.2476*2.2476);
      double u2 = 1.8091+sqrt(v2*v2+2.2476*2.2476);
      double p = (u+12.421+sqrt(w*w+16.350*u+45.171))/2.0;
      double q = (u+12.421-sqrt(w*w+16.350*u+45.171))/2.0;
      double ps = (u+sqrt(w*w+5524.8*u+6.7737))/2.0;
      double pe = (u+0.43847+sqrt(w*w+8.3680*u+491.32))/2.0;
      double D = (pow(u1*u2,1.5)/(2.0*pow(4.0567,5)))*(u1*u1+u2*u2)*
                exp(-u1-u2+8.1134);
      double K0 = (sqrt(p-q)/120.0)*(6.0*p*p+83.0*p*q+16.0*q*q)-
                  (sqrt(p)*q/8.0)*(4.0*p+3.0*q)*
                      log((sqrt(p)+sqrt(p-q))/sqrt(q));
      double K1 = (pi*pi*sqrt(p-q)/6.0)*(p+2.0*q)-
                  (pi*pi/2.0)*q*sqrt(p)*log((sqrt(p)+sqrt(p-q))/sqrt(q));
      double K2 = (7.0*pi*pi*pi*pi/60.0)*sqrt(p-q);
      double S = gamma*(K0+K1+0.42232*K2)*sqrt(pi/2.0)*pow(ps,0.25)*
                exp(-sqrt(pe));

      double func_r_1s0_1s0 = u/(u+0.9163)*S+D;
      return func_r_1s0_1s0;
    }

    /** \brief Desc
    */
    double func_r_1s0_3p2B(int irank, double t1, double t2, 
                      const std::vector<double>& sf_lgtau1,
                      const std::vector<double>& sf_lgtau2, 
                      const std::vector<std::vector<double>>& sf_lgr,
                      const std::vector<std::vector<double>>& sf_lgr2) {
        double lt1 = log10(t1);
        double lt2 = log10(t2);
        double lr;
        splint2(sf_lgtau1, sf_lgtau2, sf_lgr, sf_lgr2, 35, 35, lt1, lt2, lr);
        // Calculate func_r_1s0_3p2B
        double result = pow(10.0, lr);
        // Optional: Apply exponential decay for large lt values
        double lt = sqrt(lt1*lt1+lt2*lt2);
        double lt_limit = 3.0;
        if (lt>lt_limit) {
            lt1 /= lt;
            lt2 /= lt;
            splint2(sf_lgtau1, sf_lgtau2, sf_lgr, sf_lgr2, 35, 35, lt1, lt2, lr);
            result *= exp(-lt/lt_limit);
        }
        return result;
    }

    /** \brief Includes the Levenfish & Yakovlev 
      suppression factors for nucleon DURCA
    */
    void nudurca_q(int i, double t, double rho, double &qdurca_q, 
                std::vector<double> tcu1, std::vector<double> tcu2, 
                std::vector<double> tcu3, std::vector<double> tcd1, 
                std::vector<double> tcd2, std::vector<double> tcd3, 
                std::vector<double> tcs1, std::vector<double> tcs2, 
                std::vector<double> tcs3, std::vector<double> kfe, 
                std::vector<double> kfm, std::vector<double> kfqu, 
                std::vector<double> kfqd, std::vector<double> kfqs, 
                std::vector<int> idurca_quqd, std::vector<int> idurca_quqs) {
      auto fexp = [] (double x) { return std::exp(std::max(x, -7.e2)); };
      auto u_1s0 = [] (double t) { 
        return std::sqrt(1.0-t)*(1.456-0.157/std::sqrt(t)+1.764/t); };
      auto r_1s0 = [fexp] (double u) { 
        return std::pow(0.2312+std::pow(0.7688*0.7688+0.1438*u,2.0),5.5)*
                fexp(3.427-std::sqrt(3.427*3.427+u*u)); };
      auto r_1s0_1s0 = [this] (double u1, double u2) { return func_r_1s0_1s0(u1,u2);};

      double alpha_c = 0.0; // AWS Needs to be fixed to restore quark functionality
      double coeff_ud = (1.0/3.0)*(914.0/315.0)*
        pow((g_fermi*cos(theta_c)/(h_bar*h_bar*h_bar*c_light*c_light*c_light)),2)*
        alpha_c;
      
      double qdurca_quqd, r_ud1, r_ud2, r_ud3, qdurca_quqs, r_us1, r_us2, r_us3;

      if (idurca_quqd[i] == 1) {
        qdurca_quqd = coeff_ud*pow(10.0, 39)*
            kfqd[i]*kfqu[i]*kfe[i]*pow(h_bar, 3)*pow(kb*t, 6);
      } else if (idurca_quqd[i] == 2) {
        qdurca_quqd = coeff_ud*pow(10.0, 39)*
            (kfqd[i]*kfqu[i]*kfe[i]*pow(h_bar, 3)*pow(kb*t, 6) +
            kfqd[i]*kfqu[i]*kfm[i]*pow(h_bar, 3)*pow(kb*t, 6));
      } else {
        qdurca_quqd = 0.0;
      }
      // Pairing suppression: COLOR 1
      if ((t>tcu1[i]) && (t>tcd1[i])) {
        r_ud1 = 1.0;
      } else if ((t>tcu1[i]) && (t<=tcd1[i])) {
        double tt = t/tcd1[i];
        double u = u_1s0(tt);
        r_ud1 = r_1s0(u);
      } else if ((t<=tcu1[i]) && (t>tcd1[i])) {
        double tt = t/tcu1[i];
        double u = u_1s0(tt);
        r_ud1 = r_1s0(u);
      } else {
        double tt1 = t/tcu1[i];
        double u1 = u_1s0(tt1);
        double tt2 = t/tcd1[i];
        double u2 = u_1s0(tt2);
        r_ud1 = r_1s0_1s0(u1, u2);
      }
      // Pairing suppression: COLOR 2
      if ((t>tcu2[i]) && (t>tcd2[i])) {
        r_ud2 = 1.0;
      } else if ((t>tcu2[i]) && (t<=tcd2[i])) {
        double tt = t/tcd2[i];
        double u = u_1s0(tt);
        r_ud2 = r_1s0(u);
      } else if ((t<=tcu2[i]) && (t>tcd2[i])) {
        double tt = t/tcu2[i];
        double u = u_1s0(tt);
        r_ud2 = r_1s0(u);
      } else {
        double tt1 = t/tcu2[i];
        double u1 = u_1s0(tt1);
        double tt2 = t/tcd2[i];
        double u2 = u_1s0(tt2);
        r_ud2 = r_1s0_1s0(u1, u2);
      }
      // Pairing suppression: COLOR 3
      if ((t>tcu3[i]) && (t>tcd3[i])) {
        r_ud3 = 1.0;
      } else if ((t>tcu3[i]) && (t<=tcd3[i])) {
        double tt = t/tcd3[i];
        double u = u_1s0(tt);
        r_ud3 = r_1s0(u);
      } else if ((t<=tcu3[i]) && (t>tcd3[i])) {
        double tt = t/tcu3[i];
        double u = u_1s0(tt);
        r_ud3 = r_1s0(u);
      } else {
        double tt1 = t/tcu3[i];
        double u1 = u_1s0(tt1);
        double tt2 = t/tcd3[i];
        double u2 = u_1s0(tt2);
        r_ud3 = r_1s0_1s0(u1, u2);
      }
      // Putting everything together (3 colors + pairing suppression) :
      qdurca_quqd = (r_ud1+r_ud2+r_ud3)*qdurca_quqd;
      // u-s: for each color
      double theta_34 = pi/4.0; // A rough approximation
      double coeff_us = (1.0/3.0)*(457.0*pi/840.0)*
        pow((g_fermi*sin(theta_c)/(h_bar*h_bar*h_bar*c_light*c_light*c_light)),2)*
        (1.0-cos(theta_34));
      if (idurca_quqs[i] == 1) {
        double mus = sqrt(kfqs[i]*kfqs[i]*pow(10.0, 26)*c_light*c_light*h_bar*h_bar+
            pow(mev*strange_mass,2));
        qdurca_quqs = coeff_us*pow(10.0,26)*
            mus/c_light*kfqu[i]*kfe[i]*pow(h_bar,2)*pow(kb*t,6);
      } else if (idurca_quqs[i] == 2) {
        double mus = sqrt(kfqs[i]*kfqs[i]*pow(10.0, 13)*c_light*c_light*h_bar*h_bar+
            pow(mev*strange_mass,2));
        qdurca_quqs = coeff_us*pow(10.0,26)*
            (mus/c_light*kfqu[i]*kfe[i]*pow(h_bar,2)*pow(kb*t,6) +
            mus/c_light*kfqu[i]*kfm[i]*pow(h_bar,2)*pow(kb*t,6));
      } else {
        qdurca_quqs = 0.0;
      }
      // Pairing suppression: COLOR 1
      if ((t>tcu1[i]) && (t>tcs1[i])) {
        r_us1 = 1.0;
      } else if ((t>tcu1[i]) && (t<=tcs1[i])) {
        double tt = t/tcs1[i];
        double u = u_1s0(tt);
        r_us1 = r_1s0(u);
      } else if ((t<=tcu1[i]) && (t>tcs1[i])) {
        double tt = t/tcu1[i];
        double u = u_1s0(tt);
        r_us1 = r_1s0(u);
      } else {
        double tt1 = t/tcu1[i];
        double u1 = u_1s0(tt1);
        double tt2 = t/tcs1[i];
        double u2 = u_1s0(tt2);
        r_us1 = r_1s0_1s0(u1, u2);
      }
      // Pairing suppression: COLOR 1
      if ((t>tcu2[i]) && (t>tcs2[i])) {
        r_us2 = 1.0;
      } else if ((t>tcu2[i]) && (t<=tcs2[i])) {
        double tt = t/tcs2[i];
        double u = u_1s0(tt);
        r_us2 = r_1s0(u);
      } else if ((t<=tcu2[i]) && (t>tcs2[i])) {
        double tt = t/tcu2[i];
        double u = u_1s0(tt);
        r_us2 = r_1s0(u);
      } else {
        double tt1 = t/tcu2[i];
        double u1 = u_1s0(tt1);
        double tt2 = t/tcs2[i];
        double u2 = u_1s0(tt2);
        r_us2 = r_1s0_1s0(u1, u2);
      }
      // Pairing suppression: COLOR 1
      if ((t>tcu3[i]) && (t>tcs3[i])) {
        r_us3 = 1.0;
      } else if ((t>tcu3[i]) && (t<=tcs3[i])) {
        double tt = t/tcs3[i];
        double u = u_1s0(tt);
        r_us3 = r_1s0(u);
      } else if ((t<=tcu3[i]) && (t>tcs3[i])) {
        double tt = t/tcu3[i];
        double u = u_1s0(tt);
        r_us3 = r_1s0(u);
      } else {
        double tt1 = t/tcu3[i];
        double u1 = u_1s0(tt1);
        double tt2 = t/tcs3[i];
        double u2 = u_1s0(tt2);
        r_us3 = r_1s0_1s0(u1, u2);
      }
      qdurca_quqs= (r_us1+r_us2+r_us3)*qdurca_quqs;
      qdurca_q = qdurca_quqd + qdurca_quqs;
      return;
    }

    /// \name PBF emissivities
    //@{
    /** \brief Desc
    */
    double control_pbf_1S0(double v) {
      double x = 0.602*pow(v,2)+0.5942*pow(v,4)+0.288*pow(v,6);
      double y = sqrt(0.5547+sqrt(pow(0.4453,2)+0.01130*pow(v,2)));
      double z = exp(-sqrt(4*pow(v,2)+pow(2.245,2))+2.245);
      return x*y*z;
    }

    /** \brief Desc
    */
    double control_pbf_3P2_B(double v) {
      double x = (1.204*pow(v,2)+3.733*pow(v,4)+0.3191*pow(v,6))/(1+0.3511*pow(v,2));
      double y = pow(0.7591+sqrt(pow(0.2409,2)+0.3145*pow(v,2)),2);
      double z = exp(-sqrt(4*pow(v,2)+pow(0.4616,2))+0.4616);
      return x*y*z;
    }

    /** \brief Desc
    */
    double control_pbf_3P2_C(double v) {
      double x = 0.4013*pow(v,2)-0.043*pow(v,4)+0.002172*pow(v,6);
      double y = 1/(1-2.018e-1*pow(v,2)+2.601e-2*pow(v,4)-1.477e-3*pow(v,6)+4.34e-5*pow(v,8));
      return x*y;
    }

    /** \brief This subroutine uses only the Axial part from
      Neutrino emission due to Cooper pairing of nucleons in cooling neutron stars
      Yakovlev, D. G.; Kaminker, A. D.; Levenfish, K. P.
      1999A&A...343..650Y
      Vector part is put to zero according to:
      Neutrino emission due to Cooper pairing in neutron stars
      Leinson, L.B., & Perez, A.
      2006astro.ph..6653L
    */
    void nu_n3p2_B_pbf(double T, double Tc, double mst, double kf, double& q_n3p2_pbf) {
      if (T<=Tc) {
        double pf = kf*197.0;
        double vf = pf/mst/940.0;
        double g_A = 1.26;
        double a_v = 0.0;
        double a_a = 0.5*pow(g_A,2);
        double a = a_v+a_a;
        double tau = T/Tc;
        double u = sqrt(1.0-tau)*(0.7893+1.764/tau);
        q_n3p2_pbf = 1.170e21*pow(mst, 2)*vf*pow((T/1.0e9), 7)*3.0*a*control_pbf_3P2_B(u);
      } else {
        q_n3p2_pbf = 0.0;
      }
    }

    /** \brief This subroutine uses only the Axial part: good for n & p !
      From : Neutrino emission due to proton pairing in neutron stars
      Kaminker, A. D.; Haensel, P.; Yakovlev, D. G.
      1999A&A...345L..14K
      Vector part is put to zero according to:
      "Vector current conservation and neutrino emission from 
      singlet-paired baryons in neutron stars"
      Leinson, L.B., & Perez, A.
      2006PhLB..638..114L
    */
    void nu_1s0_pbf(double T, double Tc, double mst, double kf, double& q_1s0_pbf) {
      if (T<=Tc) {
        double pf = kf*197.0;
        double vf = pf/mst/940.0;
        double a_v = 0.0;
        double a_a = 1.60*pow(vf, 2)*(pow(mst,2)+11.0/42.0);
        double a = a_v+a_a;
        double tau = T/Tc;
        double u = sqrt(1.0-tau)*(1.456-0.157/sqrt(tau)+1.764/tau);
        q_1s0_pbf = 1.170e21*pow(mst,2)*vf*pow((T/1.0e9),7)*3.0*a*control_pbf_1S0(u);
      } else {
        q_1s0_pbf = 0.0;
      }
    }

    /** \brief Includes the Levenfish & Yakovlev suppression factors for DURCA,
     */
    void nufast(int i, double t, double rho, double& qfast, 
                std::vector<double> tcn, std::vector<double> tcp, int isf,
                std::vector<double> bar, std::vector<double> theta_k,
                std::vector<double> theta_p, std::vector<double> yelect, 
                double rhoexo, double cexo, double pexo,
                std::vector<double> mstn, std::vector<double> mstp, std::vector<double> kfe) {
      
      auto fexp = [] (double x) { return exp(std::max(x, -7.0e2)); };
      auto  u_1s0 = [] (double t) {
        return sqrt(1.0-t)*(1.456-0.157/sqrt(t)+1.764/t); };
      auto r_1s0 = [fexp] (double u) {
        return pow((0.4186+sqrt(1.007*1.007+(0.5010*u)*(0.5010*u))),2.5)*
              fexp(1.456-sqrt(1.456*1.456+u*u)); };
      auto u_3p2B = [] (double t) {
        return sqrt(1.0-t)*(5.596+8.424/t); };
      auto r_3p2B = [] (double u) {
        double temp = pow(0.7454*0.7454+(0.01811*u)*(0.01811*u),0.5);
        return pow(0.2546+temp, 5)*exp(2.701-sqrt(2.701*2.701+(u*u)/(16.0*pi))); };
      
      double rmn = mstn[i];
      double rmp = mstp[i];

      double u = bar[i]/0.16;
      double ratio = 0.319/pow(abs(yelect[i])*u, 1.0/3.0);
      double zero = 0.0;
      double f = std::max(zero,1.0-pow(ratio,2));
      f = sqrt(f);
      // Kaon urca: 
      double qkaon = 0.0;
      if (theta_k[i] != 0.0) {
        double g_a = 1.0;
        double mu_el = kfe[i]*197.0;
        qkaon = 5.0/4.0*pow(sin(theta_k[i]),2)*pow(sin(0.223),2)*
                    2.21e26*rmn*rmp*(mu_el/100.0)*
                    (1.0+3.0*pow(g_a,2))*pow(t/1.0e9,6);
      }
      double qexo = 0.0;
      if (rho >= rhoexo) {
        qexo = cexo*pow(rho/2.8e14,2.0/3.0)*pow(t/1.0e9,pexo);
      }
      double r = 0.0;
      if ((t<tcp[i]) && (t<tcn[i])) {
        if (i >= isf) {
          double un = u_1s0(t/tcn[i]);
          double rn = r_1s0(un);
          double up = u_1s0(t/tcp[i]);
          double rp = r_1s0(up);
          r = std::min(rn, rp);
        } else {
          double un = u_3p2B(t/tcn[i]);
          r = r_3p2B(un);
        }
      } else if (t<tcn[i]) {
        if (i>=isf) {
          double u = u_1s0(t/tcn[i]);
          r = r_1s0(u);
        } else {
          double u = u_3p2B(t/tcn[i]);
          r = r_3p2B(u);
        }
      } else if (t<tcp[i]) {
        double u = u_1s0(t/tcp[i]);
        r = r_1s0(u);
      } else {
        r = 1.0;
      }
      qkaon = qkaon*r;
      double r_exo = r;
      if ((pexosn == 0.0) && (pexosp == 0.0)) {
        r_exo = 1.0;
      }
      qexo = qexo * r_exo;
      qfast = qkaon + qexo;
      return;
    }

    /** \brief From Levenfish & Yakovlev, Astron. Lett. 20 (1994), p. 43        *
     */
    void nudurca_h(int irank, int i, double t, double rho,
               double& qdurca_np, double& qdurca_lap,
               double& qdurca_smn, double& qdurca_smla, double& qdurca_sms0,
               std::vector<double>tcn, std::vector<double>tcp, 
               std::vector<double>tcla, int isf,
               std::vector<double>bar, std::vector<double>yelect, 
               std::vector<double>ymuon,
               std::vector<double>mstp, std::vector<double>mstn, 
               std::vector<double>mstla, std::vector<double>mstsm, 
               std::vector<double>msts0, std::vector<double>mstsp,
               std::vector<double> durca_ctrl_e, std::vector<double> durca_ctrl_m,
               std::vector<int> idurca_lap, std::vector<int> idurca_smla,
               std::vector<int> idurca_smn, std::vector<int> idurca_sms0, 
               std::vector<int> idurca_np,
               const std::vector<double>& sf_lgtau1, const std::vector<double>& sf_lgtau2,
               const std::vector<std::vector<double>>& sf_lgr, 
               const std::vector<std::vector<double>>& sf_lgr2) {

      auto fexp = [] (double x) { return std::exp(std::max(x, -7.e2)); };
      auto u_1s0 = [] (double t) { 
        return std::sqrt(1.0-t)*(1.456-0.157/std::sqrt(t)+1.764/t); };
      auto u_3p2B = [] (double t) { return std::sqrt(1.0-t)*(0.7893+1.188/t); };
      auto r_1s0 = [fexp] (double u) { 
        return std::pow(0.2312+std::pow(0.7688*0.7688+0.1438*u,2.0),5.5)*
                fexp(3.427-std::sqrt(3.427*3.427+u*u)); };
      auto r_3p2B = [fexp] (double u) { return std::pow(0.2546+std::pow(0.7454*0.7454+
                  0.1284*u,2.0),5.0)*fexp(2.701-std::sqrt(2.701*2.701+u*u)); };
      auto r_1s0_3p2B = [irank,sf_lgtau1,sf_lgtau2,sf_lgr,sf_lgr2,this] (double t1, double t2) { 
        return func_r_1s0_3p2B(irank,t1,t2,sf_lgtau1,sf_lgtau2,sf_lgr,sf_lgr2); };

      double rmn = mstn[i];
      double rmp = mstp[i];
      double rmla = mstla[i];
      double rmsm = mstsm[i];
      double rms0 = msts0[i];
      double rmsp = mstsp[i];

      double rate_np = 1.0000;
      double rate_lap = 0.0394;
      double rate_smn = 0.0125;
      double rate_smla = 0.2055;
      double rate_sms0 = 0.6052;

      if (false) {
        // Dany's method for neutron-proton direct Urca
        if (idurca_np[i] == 1) {
          qdurca_np = rate_np*4.24e27*rmn*rmp*std::pow(t/1.0e9,6)*
                      std::pow(std::abs(yelect[i])*bar[i]/0.16,1.0/3.0);
        } else if (idurca_np[i] == 2) {
          qdurca_np = rate_np*4.24e27*rmn*rmp*std::pow(t/1.0e9,6)*
                      (std::pow(std::abs(yelect[i])*bar[i]/0.16,1.0/3.0)+
                      std::pow(std::abs(ymuon[i])*bar[i]/0.16,1.0/3.0));
        } else {
            qdurca_np = 0.0;
        }
      } else {
        // New method for neutron-proton direct Urca
        qdurca_np = durca_ctrl_e[i]*rate_np*4.24e27*rmn*rmp*std::pow(t/1.0e9,6)*
                    std::pow(std::abs(yelect[i])*bar[i]/0.16,1.0/3.0)+
                    durca_ctrl_m[i]*rate_np*4.24e27*rmn*rmp*std::pow(t/1.0e9,6)*
                    std::pow(std::abs(ymuon[i])*bar[i]/0.16,1.0/3.0);
      }
      double r_np;
      // Pairing suppression:
      if (t>tcn[i] && t>tcp[i]) {
        r_np = 1.0;
      } else if (t>tcn[i] && t<=tcp[i]) {
        double tt = t/tcp[i];
        double u = u_1s0(tt);
        r_np = r_1s0(u);
      } else if (t<=tcn[i] && t>tcp[i]) {
        if (i>=isf) {
            double tt = t/tcn[i];
            double u = u_1s0(tt);
            r_np = r_1s0(u);
        } else {
            double tt = t/tcn[i];
            double u = u_3p2B(tt);
            r_np = r_3p2B(u);
        }
      } else {
        double tt1 = t/tcp[i];
        double u1 = u_1s0(tt1);
        if (i>=isf) {
            double tt2 = t/tcn[i];
            double u2 = u_1s0(tt2);
            r_np = func_r_1s0_1s0(u1, u2);
        } else {
            double tt2 = t/tcn[i];
            // u2 is calculated using u_3p2B function, but not needed for r_1s0_3p2B
            r_np = r_1s0_3p2B(tt1, tt2);
        }
      }
      // **** la-p:
      if (idurca_lap[i] == 1) {
        qdurca_lap = rate_lap * 4.24e27 * rmla * rmp * std::pow(t / 1.0e9, 6) *
                    std::pow(std::abs(yelect[i]) * bar[i] / 0.16, 1.0 / 3.0);
      } else if (idurca_lap[i] == 2) {
        qdurca_lap = rate_lap * 4.24e27 * rmla * rmp * std::pow(t / 1.0e9, 6) *
                    (std::pow(std::abs(yelect[i]) * bar[i] / 0.16, 1.0 / 3.0) +
                      std::pow(std::abs(ymuon[i]) * bar[i] / 0.16, 1.0 / 3.0));
      } else {
        qdurca_lap = 0.0;
      }
      // **** sm-n:
      if (idurca_smn[i] == 1) {
        qdurca_smn = rate_smn * 4.24e27 * rmn * rmsm * std::pow(t / 1.0e9, 6) *
                    std::pow(std::abs(yelect[i]) * bar[i] / 0.16, 1.0 / 3.0);
      } else if (idurca_smn[i] == 2) {
        qdurca_smn = rate_smn * 4.24e27 * rmn * rmsm * std::pow(t / 1.0e9, 6) *
                    (std::pow(std::abs(yelect[i]) * bar[i] / 0.16, 1.0 / 3.0) +
                      std::pow(std::abs(ymuon[i]) * bar[i] / 0.16, 1.0 / 3.0));
      } else {
        qdurca_smn = 0.0;
      }
      // **** sm-la:
      if (idurca_smla[i] == 1) {
        qdurca_smla = rate_smla * 4.24e27 * rmsm * rmla * std::pow(t / 1.0e9, 6) *
                      std::pow(std::abs(yelect[i]) * bar[i] / 0.16, 1.0 / 3.0);
      } else if (idurca_smla[i] == 2) {
        qdurca_smla = rate_smla * 4.24e27 * rmsm * rmla * std::pow(t / 1.0e9, 6) *
                      (std::pow(std::abs(yelect[i]) * bar[i] / 0.16, 1.0 / 3.0) +
                      std::pow(std::abs(ymuon[i]) * bar[i] / 0.16, 1.0 / 3.0));
      } else {
        qdurca_smla = 0.0;
      }
      // **** sm-s0:
      if (idurca_sms0[i] == 1) {
        qdurca_sms0 = rate_sms0 * 4.24e27 * rmsm * rms0 * std::pow(t / 1.0e9, 6) *
                      std::pow(std::abs(yelect[i]) * bar[i] / 0.16, 1.0 / 3.0);
      } else if (idurca_sms0[i] == 2) {
        qdurca_sms0 = rate_sms0 * 4.24e27 * rmsm * rms0 * std::pow(t / 1.0e9, 6) *
                      (std::pow(std::abs(yelect[i]) * bar[i] / 0.16, 1.0 / 3.0) +
                      std::pow(std::abs(ymuon[i]) * bar[i] / 0.16, 1.0 / 3.0));
      } else {
        qdurca_sms0 = 0.0;
      }

      double rn, rp, rla, rsm, rs0, rsp; // Declare variables for pairing suppression
      // Neutron pairing suppression:
      if (t < tcn[i]) {
        if (i >= isf) {
          double tt = t / tcn[i];
          double u = u_1s0(tt);
          rn = r_1s0(u);
        } else {
          double tt = t / tcn[i];
          double u = u_3p2B(tt);
          rn = r_3p2B(u);
        }
      } else {
        rn = 1.0;
      }
      // Proton pairing suppression:
      if (t < tcp[i]) {
        double tt = t / tcp[i];
        double u = u_1s0(tt);
        rp = r_1s0(u);
      } else {
        rp = 1.0;
      }
      // Lambda pairing suppression:
      if (t < tcla[i]) {
        double tt = t / tcla[i];
        double u = u_1s0(tt);
        rla = r_1s0(u);
      } else {
        rla = 1.0;
      }
      // Sigma- pairing suppression:
      rsm = 1.0;
      // Sigma0 pairing suppression:
      rs0 = 1.0;
      // Sigma+ pairing suppression:
      rsp = 1.0;
      // Calculate minimum values
      double r_np = std::min(rn, rp);
      double r_lap = std::min(rla, rp);
      double r_smn = std::min(rsm, rn);
      double r_smla = std::min(rsm, rla);
      double r_sms0 = std::min(rsm, rs0);

      // Apply pairing suppression to qdurca
      qdurca_np *= r_np;
      qdurca_lap *= r_lap;
      qdurca_smn *= r_smn;
      qdurca_smla *= r_smla;
      qdurca_sms0 *= r_sms0;
    }

    /** \brief Hyperons Urca process subroutine
     */
    void numurca_hyp(int i, double t, double& qmurca_hyp) {
        qmurca_hyp = 0.0;
    }

    /** \brief Hyperons Bremsstrahlung process subroutine
     */
    void nubrem_hyp(int i, double t, double& qbrem_hyp) {
        qbrem_hyp = 0.0;
    }

    /** \brief Desc
    */
    void nubrem_nucl(int i, double t, double& qbrem_nucl, std::vector<double> tcn, 
                std::vector<double> tcp, int isf,
                std::vector<double> kfn, std::vector<double> kfp, std::vector<double> mstn, std::vector<double> mstp) {
      auto fexp = [] (double x) { return exp(std::max(x, -7.0e2)); };
      // SINGLET PAIRING:
      auto u_1s0 = [] (double t) {
        return sqrt(1.0-t)*(1.456-0.157/sqrt(t)+1.764/t); };
      // Brem_nn:  n+n -> n+n+2nu
      auto rbrem_nn_p1s0 = [](double u) { return 1.0;}; // ! Not affected by neutron pairing !
      auto rbrem_nn_n1s0 = [fexp] (double u) {
        return pow(0.1747+sqrt(0.8253*0.8253+0.07933*u*0.07933*u),2)*
              fexp(4.228-sqrt(4.228*4.228+4*u*4*u))/2.0+
              pow(0.7333+sqrt(0.2667*0.2667+0.1678*u*0.1678*u),7.5)*
              fexp(7.762-sqrt(7.762*7.762+9*u*9*u))/2.0;
      };
      // Brem_np:  n+p -> n+p+2nu
      auto rbrem_np_p1s0 = [fexp] (double u) {
        return pow(0.9982+sqrt(0.0018*0.0018+0.3815*u*0.3815*u),1)*
              fexp(1.306-sqrt(1.306*1.306+u*u))/2.732+
              pow(0.3949+sqrt(0.6051*0.6051+0.2666*u*0.2666*u),7)*
              fexp(3.303-sqrt(3.303*3.303+4*u*4*u))/1.577;
      };
      auto rbrem_np_n1s0 = [rbrem_np_p1s0] (double u) { return rbrem_np_p1s0(u);};
      auto rbrem_pp_p1s0 = [rbrem_nn_n1s0] (double u) { return rbrem_nn_n1s0(u);};
      auto rbrem_pp_n1s0 = [](double u) { return 1.0;}; // ! Not affected by neutron pairing !
      // TRIPLET PAIRING:
      auto u_3p2B = [] (double t) { return sqrt(1.0 - t) * (0.7893 + 1.188 / t);};
      auto rbrem_nn_n3p2B = [rbrem_nn_n1s0] (double u) { return rbrem_nn_n1s0(u);};
      auto rbrem_np_n3p2B = [rbrem_np_n1s0] (double u) { return rbrem_np_n1s0(u);};
      auto rbrem_pp_n3p2B = [](double u) { return 1.0;};
      
      double qbrem_nn, qbrem_np, qbrem_pp;
      double alpha_nn, beta_nn, alpha_np, beta_np, alpha_pp, beta_pp;
      double n_nu = 3.0;

      alpha_nn = 0.59;
      beta_nn = 0.56;
      qbrem_nn = n_nu*7.4e19*pow(mstn[i],4)*(kfn[i]/1.68)*alpha_nn*beta_nn*pow(t/1e9,8);

      alpha_np = 1.06;
      beta_np = 0.66;
      qbrem_np = n_nu*1.5e20*mstn[i]*mstp[i]*(kfp[i]/1.68)*alpha_np*beta_np*pow(t/1e9,8);

      alpha_pp = 0.11;
      beta_pp = 0.7;
      qbrem_pp = n_nu*7.4e19*pow(mstp[i],4)*(kfp[i]/1.68)*alpha_pp*beta_pp*pow(t/1e9,8);

      double rbrem_nn_n, rbrem_nn_p, rbrem_np_n, rbrem_np_p, rbrem_pp_n, rbrem_pp_p;
        
      if (t < tcn[i]) {
        if (i >= isf) {
          double tt = t/tcn[i];
          double u = u_1s0(tt);
          rbrem_nn_n = rbrem_nn_n1s0(u);
          rbrem_np_n = rbrem_np_n1s0(u);
          rbrem_pp_n = rbrem_pp_n1s0(u);
        } else {
          double tt = t/tcn[i];
          double u = u_3p2B(tt);
          rbrem_nn_n = rbrem_nn_n3p2B(u);
          rbrem_np_n = rbrem_np_n3p2B(u);
          rbrem_pp_n = rbrem_pp_n3p2B(u);
        }
      } else {
        rbrem_nn_n = 1.0;
        rbrem_np_n = 1.0;
        rbrem_pp_n = 1.0;
      }

      if (t < tcp[i]) {
        double tt = t/tcp[i];
        double u = u_1s0(tt);
        rbrem_nn_p = rbrem_nn_p1s0(u);
        rbrem_np_p = rbrem_np_p1s0(u);
        rbrem_pp_p = rbrem_pp_p1s0(u);
      } else {
        rbrem_nn_p = 1.0;
        rbrem_np_p = 1.0;
        rbrem_pp_p = 1.0;
      }

      double rbrem_nn = std::min(rbrem_nn_p, rbrem_nn_n);
      double rbrem_np = std::min(rbrem_np_p, rbrem_np_n);
      double rbrem_pp = std::min(rbrem_pp_p, rbrem_pp_n);

      qbrem_nn = rbrem_nn*qbrem_nn;
      qbrem_np = rbrem_np*qbrem_np;
      qbrem_pp = rbrem_pp*qbrem_pp;

      qbrem_nucl = qbrem_nn+qbrem_np+qbrem_pp;
    }

    /** \brief Includes the Levenfish & Yakovlev suppression factors for DURCA,
      and only Boltzmann factors for MURCA.
    */
    void nubrem_crust_nn(int i, double t, double vion, double& qbrem_nn, 
                        std::vector<double> tcn, int isf, std::vector<double> kfn, std::vector<double> mstn) {
      auto fexp = [] (double x) { return exp(std::max(x, -7.0e2)); };
      auto u_1s0 = [] (double t) { return sqrt(1.0-t)*(1.456-0.157/sqrt(t)+1.764/t); };
      auto u_3p2B = [] (double t) { return sqrt(1.0-t)*(0.7893+1.188/t); };
      auto rbrem_nn_n1s0 = [fexp] (double u) {
        return pow(0.1747+sqrt(0.8253*0.8253+0.07933*u*0.07933*u),2)*
              fexp(4.228-sqrt(4.228*4.228+4*u*4*u))/2.0+
              pow(0.7333+sqrt(0.2667*0.2667+0.1678*u*0.1678*u),7.5)*
              fexp(7.762-sqrt(7.762*7.762+9*u*9*u))/2.0;
      };
      auto rbrem_nn_n3p2B = [rbrem_nn_n1s0] (double u) {
        return rbrem_nn_n1s0(u);
      };
      
      double alpha_nn = 0.59; double beta_nn = 0.56; double n_nu = 3.0;
      qbrem_nn = n_nu*7.4e19*pow(mstn[i],4)*(kfn[i]/1.68)*alpha_nn*beta_nn*pow(t/1e9,8);
      double rbrem_nn;
      if (t < tcn[i]) {
        if (i >= isf) {
          double tt = t/tcn[i];
          double u = u_1s0(tt);
          rbrem_nn = rbrem_nn_n1s0(u);
        } else {
          double tt = t/tcn[i];
          double u = u_3p2B(tt);
          rbrem_nn = rbrem_nn_n3p2B(u);
        }
      } else {
        rbrem_nn = 1.0;
      }

      qbrem_nn = rbrem_nn*qbrem_nn*(1.0-vion);
    }

    /** \brief Rates and suppression factors from:                              
      Yakovlev & Levenfish, A&A 297 (1995): p. 717.   
    */    
    void numurca_nucl(int i, double t, double& qmurca_nucl, std::vector<double> tcn, 
                      std::vector<double> tcp, int isf, std::vector<double> mstn, std::vector<double> mstp, 
                      std::vector<double> kfe, std::vector<double> kfm, std::vector<double> kfn, std::vector<double> kfp) {
      auto fexp = [] (double x) { return exp(std::max(x, -7.0e2)); };
      auto u_1s0 = [] (double t) {
        return sqrt(1.0-t)*(1.456-0.157/sqrt(t)+1.764/t); };
      auto u_3p2B = [] (double t) { return sqrt(1.0-t)*(0.7893+1.188/t); };
      auto rmurca_n_p1s0 = [fexp] (double u) {
        return fexp(3.4370 - sqrt(3.4370 * 3.4370 + 1 * u * 1 * u)) * 0.5 *
              (pow(0.1477 + sqrt(0.8523 * 0.8523 + 0.1175 * u * 0.1175 * u), 7.5) +
                pow(0.1477 + sqrt(0.8523 * 0.8523 + 0.1297 * u * 0.1297 * u), 5.5)); };
      auto rmurca_p_n1s0 = [fexp] (double u) {
        return fexp(3.4370 - sqrt(3.4370 * 3.4370 + 1 * u * 1 * u)) * 0.5 *
              (pow(0.1477 + sqrt(0.8523 * 0.8523 + 0.1175 * u * 0.1175 * u), 7.5) +
                pow(0.1477 + sqrt(0.8523 * 0.8523 + 0.1297 * u * 0.1297 * u), 5.5)); };
      auto rmurca_n_n1s0 = [fexp] (double u) {
        return fexp(5.3390 - sqrt(5.3390 * 5.3390 + 2 * u * 2 * u)) *
              pow(0.2414 + sqrt(0.7586 * 0.7586 + 0.1318 * u * 0.1318 * u), 7.0); };
      auto rmurca_p_p1s0 = [fexp] (double u) {
        return fexp(5.3390 - sqrt(5.3390 * 5.3390 + 2 * u * 2 * u)) *
              pow(0.2414 + sqrt(0.7586 * 0.7586 + 0.1318 * u * 0.1318 * u), 7.0); };

      auto rmurca_p_n3p2B = [fexp] (double u) {
        return fexp(2.3980 - sqrt(2.3980 * 2.3980 + 1 * u * 1 * u)) * 0.5 *
              (pow(0.1612 + sqrt(0.8388 * 0.8388 + 0.1117 * u * 0.1117 * u), 7) +
                pow(0.1612 + sqrt(0.8388 * 0.8388 + 0.1274 * u * 0.1274 * u), 5)); };
      auto rmurca_n_n3p2B = [fexp, rmurca_p_n3p2B] (double u, double t) {
        return 39.1*t*fexp(-1.188/t)*rmurca_p_n3p2B(u); };

      double alpha_n, beta_n, alpha_p, beta_p, n_nu = 3.0;
      if (kfn[i]*kfp[i] == 0.0) {
        qmurca_nucl = 0.0;
        return;
      }

      double rmn = mstn[i];
      double rmp = mstp[i];

      alpha_n = 1.76-0.63*pow(1.68/kfn[i],2);
      beta_n = 0.68;

      double qmurca_n = 8.55e21*pow(rmn,3)*rmp*(kfe[i]/1.68+kfm[i]/1.68)*
                        alpha_n*beta_n*pow(t/1e9,8);

      alpha_p = alpha_n;
      beta_p = beta_n;

      double qmurca_p = 8.55e21*rmn*pow(rmp,3)*kfe[i]/1.68*pow((kfe[i]+3*kfp[i]-kfn[i])/
                        (8*kfe[i]*kfp[i]),2)*alpha_p*beta_p*pow(t/1e9,8);

      double rmurca_n_n, rmurca_p_n, rmurca_n_p, rmurca_p_p;

        if (t < tcn[i]) {
            if (i >= isf) {
                double tt = t/tcn[i];
                double u = u_1s0(tt);
                rmurca_n_n = rmurca_n_n1s0(u);
                rmurca_p_n = rmurca_p_n1s0(u);
            } else {
                double tt = t/tcn[i];
                double u = u_3p2B(tt);
                rmurca_n_n = rmurca_n_n3p2B(u, tt);
                rmurca_p_n = rmurca_p_n3p2B(u);
            }
        } else {
            rmurca_n_n = 1.0;
            rmurca_p_n = 1.0;
        }

        if (t < tcp[i]) {
            double tt = t / tcp[i];
            double u = u_1s0(tt);
            rmurca_n_p = rmurca_n_p1s0(u);
            rmurca_p_p = rmurca_p_p1s0(u);
        } else {
            rmurca_n_p = 1.0;
            rmurca_p_p = 1.0;
        }

        double rmurca_n = std::min(rmurca_n_p, rmurca_n_n);
        double rmurca_p = std::min(rmurca_p_p, rmurca_p_n);

        qmurca_n = rmurca_n*qmurca_n;
        qmurca_p = rmurca_p*qmurca_p;
        qmurca_nucl = qmurca_n+qmurca_p;
    }

    /** \brief Desc

      Calculate the energy loss rate per cubic centimeter in the crust
      of a neutron star from synchrotron neutrinos.
      
      From Bezchastnov, Haensel, Kaminker & Yakovlev,                  
      A&A 328 (1997): p. 409
   */
    void nsynch(double t, double bfield, double kfe, double &qsynch) {

      static const double a1=2.036e-4;
      static const double b1=7.405e-8;
      static const double c1=3.675e-4;
      static const double a2=3.356e-3;
      static const double b2=1.536e-5;
      static const double c2=1.436e-2;
      static const double d2=1.024e-5;
      static const double e2=7.647e-8;

      double b13=bfield/1.0e13;
      double x=kfe/197/0.511;
      
      double tp=2.02e9*b13*x*x;
      double xi=tp/t;
      double y1=pow(pow(1.0+3172.0*pow(xi,2.0/3.0),2.0/3.0)-1.0,1.5);
      double y2=pow(pow(1.0+172.2*pow(xi,2.0/3.0),2.0/3.0)-1.0,1.5);
      double fp=44.01*pow(1.0+c1*y1,2.0)/pow(1.0+a1*y1+b1*y1*y1,4.0);
      double fm=36.97*(1.0+c2*y2+d2*y2*y2+e2*y2*y2*y2)/
        pow(1.0+a2*y2+b2*y2*y2,5.0);
      double s_ab=27.0*pow(xi,4.0)/o2scl_const::pi2/512.0/1.037*
        (fp-0.175/1.675*fm);
      
      double tb=1.34e9*b13/sqrt(1.0+x*x);
      double z=tb/t;
      double d_1=1.0+0.4228*z+0.1014*z*z+0.006240*z*z*z;
      double d_2=1.0+0.4535*pow(z,2.0/3.0)+0.03008*z-0.05043*z*z+
        0.004314*z*z*z;
      double s_bc=exp(-z/2.0)*d_1/d_2;
      
      qsynch=9.04e14*b13*b13*pow(t/1.0e9,5.0)*s_ab*s_bc;
      
      return;
    }

    /** \brief calculate the energy loss rate per cubic centimeter in
      bubble phase of the crust. From L. Leinson, ApJ 415, p. 759, 1993  
    */
    void nbub(int i, double t, double rho, double a, double z, 
              double& qbubble, double rhocore, std::vector<double> tcn, int isf) {
      auto fexp = [] (double x) { return exp(std::max(x, -7.0e2)); };
      auto  u_1s0 = [] (double t) {
        return sqrt(1.0-t)*(1.456-0.157/sqrt(t)+1.764/t); };
      auto r_1s0 = [fexp] (double u) {
        return pow((0.4186+sqrt(1.007*1.007+(0.5010*u)*(0.5010*u))),2.5)*
              fexp(1.456-sqrt(1.456*1.456+u*u)); };
      auto u_3p2B = [] (double t) {
        return sqrt(1.0-t)*(5.596+8.424/t); };
      auto r_3p2B = [] (double u) {
        double temp = pow(0.7454*0.7454+(0.01811*u)*(0.01811*u),0.5);
        return pow(0.2546+temp, 5)*exp(2.701-sqrt(2.701*2.701+(u*u)/(16.0*pi))); };

      double rhomin = 1.0e14;
      if ((rho < rhocore) && (rho >= rhomin)) {
        qbubble = 1.1e22 * pow(t / 1.0e9, 6);
      } else {
        qbubble = 0.0;
      }

      double r;
      if (t < tcn[i]) {
        if (i >= isf) {
          double u = u_1s0(t / tcn[i]);
          r = r_1s0(u);
        } else {
          double u = u_3p2B(t / tcn[i]);
          r = r_3p2B(u);
        }
      } else {
        r = 1.0;
      }

      qbubble *= r;
    }

    /** \brief calculate the energy loss rate per cubic centimeter in the crust of a
      neutron star from plasma neutrinos.                                     
      from h.munakata, y.kohyama & n.itoh, ap.j.296(1985),p.197 
    */              
    void nplasma_old(double t, double rho, double a, double z, double& qplasma) {
      auto fexp = [] (double x) { return exp(std::max(x, -7.0e2)); };
      double apl[] = {2.320e-7, 8.449e-8, 1.787e-8};
      double bpl[] = {2.581e-2, 1.734e-2, 6.990e-4};
      double cpl = 0.56457;

      if (z == 0.0) {
        qplasma = 0.0;
        return;
      }

      double l = t/5.930e9;
      double xi = pow(rho*z/a*1.0e-9,1.0/3.0)/l;

      double n = 2.0;
      double fplasma = (apl[0]+apl[1]*xi+apl[2]*(xi*xi))*fexp(-cpl*xi) /
                      ((xi*xi*xi)+bpl[1]/l+bpl[2]/(l*l)+bpl[3]/(l*l*l));
      qplasma = (0.872+n*0.004)*pow(rho*z/a,3)*fplasma;
    }

    /** \brief calculate the energy loss rate per cubic centimeter in the crust  
      of a neutron star from plasma neutrinos.                          
      M. Haft, G. Raffelt & A. Weiss                                    
      Ap. J. 425 (1996): p. 222                                         
      which is identical to:                                            
      N. Itoh, H. Hayashi, A. Nishikawa & Y. Kohyama                    
      Ap. J. Suppl. XXX (1996): p. yyy
    */
    void nplasma(double t, double rho, double a, double z, double& qplasma) {
      auto fexp = [] (double x) { return exp(std::max(x, -7.0e2)); };
      if (z == 0.0) {
        qplasma = 0.0;
        return;
      }

      double l = t/5.9302e9;
      double den = 1.0 + pow(1.019e-6 * rho * z / a, 2.0 / 3.0);
      double gamma2 = 1.1095e11 * (rho * z / a) / (t * t) / sqrt(den);
      double gamma = sqrt(gamma2);
      double f_t = 2.4 + 0.6 * sqrt(gamma) + 0.51 * gamma + 1.25 * pow(gamma, 1.5);
      double f_l = (8.6 * gamma2 + 1.35 * pow(gamma, 3.5)) / (225.0 - 17.0 * gamma + gamma2);
      double x = 1.0 / 6.0 * (17.5 + log10(2.0 * rho * z / a) - 3.0 * log10(t));
      double y = 1.0 / 6.0 * (-24.5 + log10(2.0 * rho * z / a) + 3.0 * log10(t));
      double f_xy;
        
      if (std::abs(x) > 0.7 || y < 0.0) {
        f_xy = 1.0;
      } else {
        double first = 0.39 - 1.25 * x - 0.35 * sin(4.5 * x) - 0.3 * fexp(-pow(4.5 * x + 0.9, 2.0));
        double sec = y - 1.6 + 1.25 * x;
        sec = std::min(0.0, sec);
        double second = sec / (0.57 - 0.25 * x);
        f_xy = 1.05 + first * fexp(-second * second);
      }

      qplasma = 3.00e21 * pow(l, 9) * pow(gamma, 6) * fexp(-gamma) * (f_l + f_t) * f_xy;
    }

    /** \brief calculate the energy loss rate per cubic centimeter in the crust  
      of a neutron star from photo neutrinos.                           
      N. Itoh, T. Adachi, M. Nakagawa, Y. Kohyama & H. Munakata         
      Ap. J. 339 (1989): p. 354                                         
      which is identical to:                                            
      N. Itoh, H. Hayashi, A. Nishikawa & Y. Kohyama                    
      Ap. J. Suppl. XXX (1996): p. yyy 
    */
    void nphoto(double t, double rho, double a, double z, double& qphoto) {
      double bph[] = { 6.290e-3, 7.483e-3, 3.061e-4 };
      double cph[3][7][3] = {
              {+1.008E+11 , +0.000E+0 , +0.000E+0 , +0.000E+0 ,
              +0.000E+0 , +0.000E+0 , +0.000E+0 ,
              +8.156E+10 , +9.728E+8 , -3.806E+9 , -4.384E+9 ,
              -5.774E+9 , -5.249E+9 , -5.153E+9 ,
              +1.067E+11 , -9.782E+9 , -7.193E+9 , -6.936E+9 ,
              -6.893E+9 , -7.041E+9 , -7.193E+9 ,},
              {+9.889E+10 , -4.524E+8 , -6.088E+6 , +4.269E+7 ,
              +5.172E+7 , +4.910E+7 , +4.388E+7 ,
              +1.813E+11 , -7.556E+9 , -3.304E+9 , -1.031E+9 ,
              -1.764E+9 , -1.851E+9 , -1.928E+9 ,
              +9.750E+10 , +3.484E+10, +5.199E+9 , -1.695E+9 ,
              -2.865E+9 , -3.395E+9 , -3.418E+9 ,},
              {+9.581E+10 , +4.107E+8 , +2.305E+8 , +2.236E+8 ,
              +1.580E+8 , +2.165E+8 , +1.721E+8 ,
              +1.459E+12 , +1.314E+11, -1.169E+11, -1.765E+11, 
              -1.867E+11, -1.983E+11, -1.896E+11,
              +2.424E+11 , -3.669E+9 , -8.691E+9 , -7.967E+9 ,
              -7.932E+9 , -7.987E+9 , -8.333E+9}
      };

      // dph array
      double dph[3][5][3] = {
            { 0.000E+00, 0.000E+0, 0.000E+0, 0.000E+0, 0.000E+0,
            -1.879E+10, -9.667E+9, -5.602E+9, -3.370E+9, -1.825E+9 ,
            -2.919E+10, -1.185E+10, -7.270E+9, -4.222E+9, -1.560E+9 },
            { -1.135E+08, 1.256E+8, 5.149E+7, 3.436E+7, 1.005E+7 ,
            1.652E+09, -3.119E+9, -1.839E+9, -1.458E+9, -8.956E+8 ,
            -1.548E+10, -9.338E+9, -5.899E+9, -3.035E+9, -1.598E+9 },
            { 4.724E+08, 2.976E+8, 2.242E+8, 7.937E+7, 4.859E+7,
            -7.094E+11, -3.697E+11, -2.189E+11, -1.273E+11, -5.705E+10 ,
            -2.254E+10, -1.551E+10, -7.793E+9, -4.489E+9, -2.185E+9 }
      };
        
      auto fexp = [] (double x) { return exp(std::max(x, -7.0e2)); };
      if (t < 1.0e7) {
        qphoto = 0.0;
        return; }
      double l = t/5.9302e9;
      double xi = pow(rho*z/a*1.0e-9, 1.0/3.0)/l;
      double cphot, tau;
      int k;
      if (t < 1.0e8) {
        cphot = 0.5654+log10(t/1.0e7);
        k = 1;
        tau = log10(t/1.0e7); } else {
        cphot = 1.5654;
        if (t < 1.0e9) {
          k = 2;
          tau = log10(t/1.0e8); } else {
          k = 3;
          tau = log10(t/1.0e9); } }

      double aph[3];
      for (int i = 0; i <= 2; ++i) {
        aph[i] = 0.5*cph[i][0][k]+0.5*cph[i][6][k]*cos(10.0*pi*tau);
        for (int j = 1; j <= 5; ++j) {
          aph[i] += cph[i][j][k]*cos(5.0/3.0*pi*static_cast<double>(j)*tau)
                    +dph[i][j][k]*sin(5.0/3.0*pi*static_cast<double>(j)*tau);
        }
      }

      double n = 2.0;
      double fphoto = (aph[0]+aph[1]*xi+aph[2]*pow(xi,2.0))/
                      (pow(xi,3.0)+bph[1]/l+bph[2]/(l*l)+bph[3]/(l*l*l));
      double fphot = 1.875e8*l+1.653e8*pow(l,2.0)+8.499e8*pow(l,3.0)-1.604e8*pow(l,4.0);
      double qph = 0.666*pow((1.0+2.045*l),-2.066)/(1.0+rho*z/a/fphot);
      qphoto = 0.5*(1.122+n*0.254)*
                (1.0-(0.622-n*0.246)/(1.122+n*0.254)*qph)*
                (rho*z/a)*pow(l,5)*fphoto*fexp(-cphot*xi);
    }

    /** \brief Calculates the energy loss rate per cubic centimeter in the crust 
      of aneutron star from pair neutrinos.                            
      N. Itoh, T. Adachi, M. Nakagawa, Y. Kohyama & H. Munakata         
      Ap. J. 339 (1989): p. 354                                         
      which is identical to:                                            
      N. Itoh, H. Hayashi, A. Nishikawa & Y. Kohyama                   
      Ap. J. Suppl. XXX (1996): p. yyy
    */ 
    void npair(double t, double rho, double a, double z, double& qpair) {
      double apa[3] = {+6.002e19, +2.084e20, +1.872e21};
      double bpa_l[3] = {+9.383e-1, -4.141e-1, +5.829e-2};
      double bpa_h[3] = {+1.2383, -0.8141, +0.0};
      double cpa_l = +5.5924;
      double cpa_h = +4.9924;
      auto fexp = [] (double x) { return exp(std::max(x, -7.0e2)); };

      if (t<1.0e7) {
        qpair = 0.0; // this is qphoto in the original code. WHY???
        return; }  
      double xi = std::pow(rho*z/a*1.0e-9,1.0/3.0)/(t/5.9302e9);
      double n = 2.0;
      double fpair, qpa, g;
      if (t<1.0e10) {
        fpair = (apa[0]+apa[1]*xi+apa[2]*std::pow(xi,2))*
                std::exp(-cpa_l*xi) /
                (std::pow(xi,3)+bpa_l[0]/t+bpa_l[1]/(t*t)+bpa_l[2]/(t*t*t));
      } else {
        fpair = (apa[0]+apa[1]*xi+apa[2]*std::pow(xi,2))*
                std::exp(-cpa_h*xi) /
                (std::pow(xi,3)+bpa_h[0]/t+bpa_h[1]/(t*t)+bpa_h[2]/(t*t*t));
      }
        
      g = 1.0-13.04*std::pow(xi,2)+133.5*std::pow(xi,4)+
          1534.0*std::pow(xi,6)+918.6*std::pow(xi,8); 
      qpa = (10.7480*std::pow(xi,2)+0.3967*std::sqrt(xi)+1.0050)*
            (1.0+rho*z/a/(7.692*std::pow(xi,3)+9.715*std::sqrt(xi)))*
            (-0.3);  
      qpair = 0.5*(1.122+n*0.254)*(1.0+(0.622-n*0.246)/(1.122+n*0.254)*qpa)
              *g*std::exp(-2.0/xi)*fpair;
    }

    /** \brief 
     */
    void npbc(double t, double rho, double a, double z, double& qnpbc) {
      qnpbc =0.0;
      std::cout << "AWS: Removed because this required a data file." << std::endl;
    }

    /** \brief 
     */
    void npbl(double t, double rho, double a, double z, double& qnpbl) {
      qnpbl =0.0;
      std::cout << "AWS: Removed because this required a data file." << std::endl;
    }

    /** \brief calculate the energy loss rate per cubic centimeter in the crust 
      of a neutron star when electrons are partially degenerate        
      (T>0.3Tf) from neutrino pair bremsstrahlung.                     
      from H. Munakata, Y. Kohyama & N. Itoh,                          
      Ap. J. 316 (1987): p. 708.                                       
      which is almost identical to (typing errors corrected):          
      N. Itoh, H. Hayashi, A. Nishikawa & Y. Kohyama                   
      Ap. J. Suppl. XXX (1996): p. yyy  
    */
    void npbpde(double t, double rho, double a, double z, double& qnpbpde) {
        double n = 2.0;
        double gamma = 2.273e5*pow(z,2)*pow(rho/a,1.0/3.0)/t;
        double t8 = t/1.0e8;

        double eta = (rho*z/a)/(7.05e6*pow(t8,1.5)+5.12e4*pow(t8,3));
        double f1 = 23.5+6.83e4/pow(t8,2)+7.81e8/pow(t8,5);
        double f2 = 1.0+1.47/eta+0.0329/(eta*eta);
        double f = 1.0/f1+1.26*(1.0+1.0/eta)/f2;

        double b3 = 7.75e5*pow(t8,1.5)+247.0*pow(t8,3.85);
        double b4 = 4.07+0.0240*pow(t8,1.40);
        double b5 = 4.59e-5/pow(t8,0.11);
        double g1 = 230.0+6.7e5/pow(t8,2)+7.66e9/pow(t8,5);
        double g2 = b3/(rho*z/a)+b4+b5*pow(rho*z/a,0.656);
        double g = 1.0/(1.0+1.0e-9*rho*z/a)/g1+1.0/g2;

        qnpbpde = 0.5738*pow(z,2)/a*pow(t8,6)*rho*
                  (0.5*(1.122+0.254*n)*f-0.5*(0.622-0.246*n)*g);
    }

    /** \brief calculate the energy loss rate per cubic centimeter in the crust
      of a neutron star from neutrino pair bremsstrahlung.             
      checked on February 27, 1996 against figures of Itoh et al 1996
    */
    void npb(double t, double rho, double a, double z, double& qnpb) {
        const double gammacryst = 210.0;
        const double gammaliq = 180.0;

        double mm = pow(rho/1.0e6*z/a, 2.0/3.0);
        double tf = 5.930e9*(sqrt(1.0+1.018*mm)-1.0);
        double gamma = 2.273e5*pow(z,2)*pow(rho/a,1.0/3.0)/t;

        if (rho <= 1.0e4) {
            npbpde(t, rho, a, z, qnpb);
            // For rho < 10^4 only npb_pde is accurately calculated (?)
            // so it is used as a default
        } else if (t>0.35*tf) {
            npbpde(t, rho, a, z, qnpb);
        } else if (t >= 0.30*tf) {
            double qnpb1, qnpb2;
            npbpde(t, rho, a, z, qnpb1);
            npbl(t, rho, a, z, qnpb2);
            qnpb = (t-0.25*tf)/(0.1*tf)*qnpb1+(0.35*tf-t)/(0.1*tf)*qnpb2;
        } else if (gamma < gammaliq) {
            npbl(t, rho, a, z, qnpb);
        } else if (gamma > gammacryst) {
            npbc(t, rho, a, z, qnpb);
        } else {
            double qnpb1, qnpb2;
            npbl(t, rho, a, z, qnpb1);
            npbc(t, rho, a, z, qnpb2);
            qnpb = (gammacryst-gamma)/(gammacryst-gammaliq)*qnpb1+
                  (gamma-gammaliq)/(gammacryst-gammaliq)*qnpb2;
        }
        //*********************************
        // For extremely impure crust:
        // npbl(t, rho, a, z, qnpb);
        //*********************************
    }

    /** \brief calculate the energy loss rate per cubic centimeter in the crust
      of a neutron star from neutrino pair bremsstrahlung.              
      From Kaminker et al, A&A 343 (1999), p. 1009, Equ. (40) 
    */
    void npb_new(double temp, double rho, double& qnpb) {
        const double gammacryst = 210.0;
        const double gammaliq = 180.0;

        double tau = log10(temp / 1.0e8);
        double r = log10(rho / 1.0e12);
        double rho0 = 2.8e14;
        double lgq = 11.204 + 7.304 * tau + 0.2976 * r - 0.370 * pow(tau, 2)
                    + 0.188 * tau * r - 0.103 * pow(r, 2) + 0.0547 * pow(tau, 2) * r
                    - 6.77 * log10(1.0 + 0.228 * rho / rho0);
        qnpb = pow(10.0, lgq);
    }

    /** \brief 
     */
    void neebrem(int irank, double T, double mu, double& qeebrem, 
                std::vector<double> neebrem_logt,
                std::vector<double> neebrem_nalpha, std::vector<double> neebrem_n2) {
        double mu10=mu/10.0;
        double alpha=T/mu10;
        double logalpha=log10(alpha);
        
        double naa=0.0;  // Placeholder for the SPLINT function result
        splint(neebrem_logt,neebrem_nalpha,neebrem_n2,56,logalpha,naa);
        
        qeebrem=2.16e14*pow(T/1.0e9,7)*pow(mu/10.0,2)*naa;
    }

    /** \brief 
     */
    void neutrinot(int irank, int i, double t, double rho, double a, double z, double &qtot,
              double &qeebrem, double &qnpb, double &qplasma, double &qsynch, double &qbubble,
              double &qpair, double &qphoto, double &qbrem_nn, double &qmurca_nucl,
              double &qbrem_nucl, double &qmurca_hyp, double &qbrem_hyp,
              double &qdurca_np, double &qdurca_lap, double &qdurca_smn,
              double qdurca_smla, double &qdurca_sms0, double qfast,
              double &qdurca_q, double &qmurca_q,
              double &qpbf_n1s0, double &qpbf_n3p2, double &qpbf_p1s0,
              double &qpbf_q, bool debug, int naa, std::vector<double> nbfield2,
              double rhodrip, double rhocore,
              std::vector<double>mstp, std::vector<double>mstn, std::vector<double>mstla, 
              std::vector<double>mstsm, std::vector<double>msts0, std::vector<double>mstsp,
              std::vector<double>kfe, std::vector<double>kfm, std::vector<double>kfp, std::vector<double>kfn,
              std::vector<double>kfqu, std::vector<double>kfqd, std::vector<double>kfqs,
              std::vector<double> bar, std::vector<double> yelect, std::vector<double> ymuon, std::vector<double> fhad,
              std::vector<double> theta_k, std::vector<double> theta_p, std::vector<double> v_ion,
              double rhoexo, double cexo, double pexo,
              double c_nu_str, double p_nu_str,
              double murca_increase, int inu_durca, int inu_eion, int inu_plasma, int inu_synch,
              int inu_n1s0_pbf, int inu_n3p2_pbf, int inu_p_pbf,
              int inu_bubble, int inu_photo, int inu_pair,
              std::vector<int> idurca_np, std::vector<int> idurca_lap, std::vector<double> durca_ctrl_e,
              std::vector<double> durca_ctrl_m,
              std::vector<int> idurca_smn, std::vector<int> idurca_smla, std::vector<int> idurca_sms0,
              std::vector<int> idurca_quqd, std::vector<int> idurca_quqs,
              std::vector<double>tcn, std::vector<double>tcp, std::vector<double>tcla,
              std::vector<double>tcu, std::vector<double>tcd, std::vector<double>tcs,
              std::vector<double>tcu1, std::vector<double>tcu2, std::vector<double>tcu3,
              std::vector<double>tcd1, std::vector<double>tcd2, std::vector<double>tcd3,
              std::vector<double>tcs1, std::vector<double>tcs2, std::vector<double>tcs3,
              int isf, std::vector<double> neebrem_logt, 
              std::vector<double> neebrem_nalpha, std::vector<double> neebrem_n2,
              std::vector<double> sf_lgtau1, std::vector<double> sf_lgtau2,
              std::vector<std::vector<double>> sf_lgr, std::vector<std::vector<double>> sf_lgr2) {
      int istrange = 0;
      double print_it = 1.0; double qstrange;
      if (debug >= 2.0) {
        std::cout << "Entering subroutine `neutrino' ";
        std::cout << "T, rho, A, Z = " << t << ", " << rho << ", " << a << ", " << z << std::endl;
      }
      // ELECTRON-ELECTRON PAIR BREMSSTRAHLUNG
      if (rho < rhocore) {
        double mu_el = kfe[i] * 197.0;
        neebrem(irank, t, mu_el, qeebrem, neebrem_logt, neebrem_nalpha, neebrem_n2);
      } else {
        qeebrem = 0.0; }
      // ELECTRON-ION PAIR BREMSSTRAHLUNG
      if (inu_eion == 1) {
        if (rho < rhocore) {
          npb_new(t, rho, qnpb);
        } else { qnpb = 0.0; }
      } else if (inu_eion == 2) {
        if (rho < rhocore) {
          npb(t, rho, a, z, qnpb);
        } else { qnpb = 0.0; }
      } else {
        if (rho < rhocore) {
          qnpb = 0.0;
          if (print_it != 1.0) {
            std::cout << "No npb: Rho, Qnpb=" << rho << ", " << qnpb << std::endl;
            print_it = 1.0; }
        } else { qnpb = 0.0; } }
      // PLASMA NEUTRINO
      if (inu_plasma == 1) {
        if (rho < rhocore) {
          nplasma(t, rho, a, z, qplasma);
        } else { qplasma = 0.0; }
      } else if (inu_plasma == -1) {
        if (rho < rhocore) {
          nplasma_old(t, rho, a, z, qplasma);
        } else { qplasma = 0.0; }
      } else { qplasma = 0.0; }
      // SYNCHROTRON NEUTRINO
      if (inu_synch == 1) {
        if (rho < rhocore) {
          nsynch(t, nbfield2[i], kfe[i], qsynch);
        } else { qsynch = 0.0; }
      } else { qsynch = 0.0; }
      // BUBBLE NEUTRINO
      if (inu_bubble == 1) {
        if (rho < rhocore) {
          nbub(i, t, rho, a, z, qbubble, rhocore, tcn, isf);
        } else { qbubble = 0.0; }
      } else { qbubble = 0.0; }
      // NEUTRINO PAIR
      if (inu_pair == 1) {
        if (rho < rhocore) {
          npair(t, rho, a, z, qpair);
        } else { qpair = 0.0; }
      } else { qpair = 0.0; }
      // PHOTO-NEUTRINO
      if (inu_photo == 1) {
        if (rho < rhocore) {
          nphoto(t, rho, a, z, qphoto);
        } else { qphoto = 0.0; }
      } else { qphoto = 0.0; }
      // NN-BREMSTRAHLUNG in the inner crust
      if ((rho<rhocore) && (rho>=rhodrip)) {
        nubrem_crust_nn(i, t, v_ion[i], qbrem_nn, tcn, isf, kfn, mstn);
      } else { qbrem_nn = 0.0; }
      // URCA et al. PROCESSES:
      if (rho >= rhocore) {
        if (istrange == 0) {
          numurca_nucl(i, t, qmurca_nucl, tcn, tcp, isf, mstn, mstp, kfe, kfm, kfn, kfp);
          qmurca_nucl *= (1.0 + murca_increase);
          qmurca_nucl *= fhad[i];
          nubrem_nucl(i, t, qbrem_nucl, tcn, tcp, isf, kfn, kfp, mstn, mstp);
          qbrem_nucl *= (1.0 + murca_increase);
          qbrem_nucl *= fhad[i];
          numurca_hyp(i, t, qmurca_hyp);
          qmurca_hyp *= fhad[i];
          nubrem_hyp(i, t, qbrem_hyp);
          qbrem_hyp *= fhad[i];
          if (inu_durca == 1) {
            nudurca_h(irank, i, t, rho, qdurca_np, qdurca_lap, qdurca_smn,
                      qdurca_smla, qdurca_sms0, tcn, tcp, tcla, isf, bar,
                      yelect, ymuon, mstp, mstn, mstla, mstsm, msts0, mstsp,
                      durca_ctrl_e, durca_ctrl_m, idurca_lap, idurca_smla,
                      idurca_smn, idurca_sms0, idurca_np, sf_lgtau1, sf_lgtau2,
                      sf_lgr, sf_lgr2);
            qdurca_np *= fhad[i];
            qdurca_lap *= fhad[i];
            qdurca_smn *= fhad[i];
            qdurca_smla *= fhad[i];
            qdurca_sms0 *= fhad[i];
          } else {
            qdurca_np = 0.0;
            qdurca_lap = 0.0;
            qdurca_smn = 0.0;
            qdurca_smla = 0.0;
            qdurca_sms0 = 0.0;
          } // FAST neutrino emission:
          nufast(i, t, rho, qfast, tcn, tcp, isf, bar, theta_k, theta_p, yelect,
                rhoexo, cexo, pexo, mstn, mstp, kfe);
          qfast *= fhad[i];
          // QUARK processes:
          nudurca_q(i, t, rho, qdurca_q, tcu1, tcu2, tcu3, tcd1, tcd2, tcd3,
                    tcs1, tcs2, tcs3, kfe, kfm, kfqu, kfqd, kfqs, idurca_quqd,
                    idurca_quqs);
          numurca_q(i, t, rho, qmurca_q, kfqu, tcu, tcd);
          qdurca_q *= (1.0-fhad[i]);
          qmurca_q *= (1.0-fhad[i]);
          qstrange = 0.0;
        } else if (istrange == 1) { //STRANGE QUARK MATTER processes:
          qstrange = c_nu_str*std::pow((t/1.0e9),p_nu_str);
          qmurca_nucl = 0.0;
          qbrem_nucl = 0.0;
          qmurca_hyp = 0.0;
          qbrem_hyp = 0.0;
          qdurca_np = 0.0;
          qdurca_lap = 0.0;
          qdurca_smn = 0.0;
          qdurca_smla = 0.0;
          qdurca_sms0 = 0.0;
          qfast = 0.0;
          qdurca_q = 0.0;
          qmurca_q = 0.0;
        } else {
            std::cout << "neutrino: istrange not defined !" << std::endl;
            exit(1); }
      } else {
        qmurca_nucl = 0.0;
        qbrem_nucl = 0.0;
        qmurca_hyp = 0.0;
        qbrem_hyp = 0.0;
        qdurca_np = 0.0;
        qdurca_lap = 0.0;
        qdurca_smn = 0.0;
        qdurca_smla = 0.0;
        qdurca_sms0 = 0.0;
        qfast = 0.0;
        qdurca_q = 0.0;
        qmurca_q = 0.0;
        qstrange = 0.0; }
      // PBF PROCESSES:
      if (istrange == 0) {
        // Neutrons 1S0:
        if ((inu_n1s0_pbf==1) && (i>isf)) {
          nu_1s0_pbf(t, tcn[i], mstn[i], kfn[i], qpbf_n1s0);
          qpbf_n1s0 *= fhad[i];
        } else { qpbf_n1s0 = 0.0; }
        // Neutron 3P2:
        if ((inu_n3p2_pbf == 1) && (i <= isf)) {
          nu_n3p2_B_pbf(t, tcn[i], mstn[i], kfn[i], qpbf_n3p2);
          qpbf_n3p2 *= fhad[i];
        } else { qpbf_n3p2 = 0.0; }
        // Protons:
        if (inu_p_pbf == 1) {
          nu_1s0_pbf(t, tcp[i], mstp[i], kfp[i], qpbf_p1s0);
          qpbf_p1s0 *= fhad[i];
        } else { qpbf_p1s0 = 0.0; }
        // Quarks: TO BE INCLUDED !!!!
        qpbf_q = 0.0;
        qpbf_q = qpbf_q*(1.0-fhad[i]);
      } else {
        qpbf_n1s0 = 0.0;
        qpbf_n3p2 = 0.0;
        qpbf_p1s0 = 0.0;
        qpbf_q = 0.0;
      }

      // Adding everything:
      qtot = qeebrem+qnpb+qplasma+qsynch+qbubble+qpair+qphoto+qbrem_nn+
          qmurca_nucl+qbrem_nucl+qmurca_hyp+qbrem_hyp+
          qdurca_np+qdurca_lap+qdurca_smn+qdurca_smla+qdurca_sms0+
          qfast+qdurca_q+qmurca_q+qstrange+
          qpbf_n1s0+qpbf_n3p2+qpbf_p1s0+qpbf_q;

      if (debug >= 2) {
        std::cout << "Exiting subroutine `neutrino'" << std::endl; }
      return;
    }

};

#endif