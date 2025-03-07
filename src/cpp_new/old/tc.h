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
#ifndef NSCOOL_TC_H
#define NSCOOL_TC_H

#include <boost/numeric/ublas/vector.hpp>

#include <o2scl/constants.h>
#include <o2scl/interp.h>

typedef boost::numeric::ublas::vector<double> ubvector;
typedef boost::numeric::ublas::matrix<double> ubmatrix;

/** \brief Compute critical temperatures
 */
class tc {
  protected:

    bool done;

    /** \brief Basic function to solve for Tc
     */
    double solve_tc(int imax, double *k0, double *t0, double *t2, double k) {
      if (k<=k0[0] || k>=k0[imax-1]) {
        return 0.0;
      } else {
        int i1 = 1;
        int i2 = imax;
        while (i2-i1 > 1) {
          int i = (i1+i2)/2;
          if (k0[i] > k) {
            i2 = i; } else {
            i1 = i; } }
        double delk = k0[i2]-k0[i1];
        double a = (k0[i2]-k)/delk;
        double b = (k-k0[i1])/delk;
        double tc = a*t0[i1]+b*t0[i2]
          + ((std::pow(a,3)-a)*t2[i1]+(std::pow(b,3)-b)*t2[i2])*
          (std::pow(delk,2))/6.0;
        return tc;
      }
    }

    /** \brief Calculate the y" coefficients for a spline interpolation procedure
      to be used by splint. yp1 and ypn are the first derivative of y(x) 
      at y(1) and y(in). From numerical recipes, p.88  
    */
    void spline_here(double *x, double *y, int in, double yp1, double ypn,
                 double *y2) {
  
      const int jmax = 100;
      double u[jmax];

      if (yp1>1.0e30) {
        y2[0] = 0.0;
        u[0] = 0.0; 
      } else {
        y2[0] = -0.5;
        u[0] = (3.0/(x[1]-x[0]))*((y[1]-y[0])/(x[1]-x[0])-yp1);
      }

      for (int i=1; i<in-1;++i){
        double sig = (x[i]-x[i-1])/(x[i+1]-x[i-1]);
        double p = sig*y2[i-1]+2.0;
        y2[i] = (sig-1.0)/p;
        u[i] = (6.0*(((y[i+1]-y[i])/(x[i + 1]-x[i]))-((y[i]-y[i-1])/(x[i]-x[i-1])))/
            (x[i+1]-x[i-1])-sig*u[i-1])/p; 
      }

      double qn, un;
      if (ypn > 1.0e30) {
        qn = 0.0;
        un = 0.0;
      } else {
        qn = 0.5;
        un = (3.0/(x[in-1]-x[in-2]))*(ypn-((y[in-1]-y[in-2])/(x[in-1]-x[in-2])));
      }

      y2[in-1] = (un-qn*u[in-2])/(qn*y2[in-2]+1.0);

      for (int ik=in-2;ik>=0;--ik){
        y2[ik] = y2[ik]*y2[ik+1]+u[ik];
      }
      return;
    }

    /** \brief Yakovlev et al. functions                       
      Values from:                                                     
      Yakovlev, Kaminker, & Gnedin, A&A 379, L5   (2001)               
      Kaminker, Yakovlev, & Gnedin, A&A 383, 1076 (2002)               
      Yakovlev, Kaminker, Haensel, & Gnedin, A&A 389, L24 (2002)       
      Proton 1S0:
    */
    double cpp_tc_Ioffe(double kf, double T0, double k0, double k1,
                    double k2, double k3) {
      if (kf > k0 && kf < k2) {
        double tc = T0*std::pow(kf-k0,2)/(std::pow(kf-k0,2)+std::pow(k1,2))*
        std::pow(kf-k2,2)/(std::pow(kf-k2,2)+std::pow(k3,2));
        return 1.0e9*tc;
      } else {
        return 1.0;
      }
    }

  public:
  
    tc() {
      done=false;
    }

    /** \brief Calculate the Lambda 1s0 superfluidity critical temperature from    
      S. Balberg & N. Barnea, nucl-th/9709013                             
      T_c depends on k_F and nbar !                                       
      T_c at 5rho_0 is exactly as in the paper, at lower densities it is  
      an interpolation which is OK at 10%                                
      uses a cubic spline interpolation.                                  
      k is the fermi momentum in fm^-1 and tc is given in kelvins   
     */
    double cpp_tcla1_bb(double k, double nbar) {
      const int imax = 14;
      double k0[imax] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
        1.0, 1.1, 1.2, 1.3, 1.4};
      double t0[imax] = {0.0, 0.130, 0.385, 0.720, 1.040, 1.306, 1.468, 
        1.500, 1.393, 1.156, 0.838, 0.535, 0.175, 0.0};
      double t2[imax];
      spline_here(k0,t0,imax,0.0,0.0,t2);
      return solve_tc(imax,k0,t0,t2,k);
    }

    /** \brief Calculate the neutron 1S0 superfluidity critical temperature from   
      Elgaroy, Engvik, Hjorth-Jensen & Osnes, Nucl. Phys. A604 (1996), p. 466
      uses a cubic spline interpolation.    
    */ 
    double cpp_tcp1_eeho(double k) {
      const int imax = 13;
      double k0[imax] = {0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 
        0.70, 0.80, 0.90, 1.00, 1.10, 1.20};
      double d0[imax] = {0.00, 0.02, 0.10, 0.37, 0.62, 0.79, 0.87, 
        0.86, 0.70, 0.49, 0.30, 0.15, 0.00};
      double t0[imax]; double t2[imax];  
      if (done!=1.0) {
        for (int i=0; i<imax; ++i) {
          t0[i] = d0[i]/1.76*1.1604e10; }
        spline_here(k0, t0, imax, 0.0, 0.0, t2);
        done=1.0; }
      return solve_tc(imax,k0,t0,t2,k);
    }

    /** \brief Calculate the proton 1S0 superfluidity critical temperature from    
      Baldo, Cugnon, Lejeune & Lombardo, NP A536 (1992), p. 349           
      uses a cubic spline interpolation.                                  
      k is the fermi momentum in fm^-1 and Tc is given in K  
    */
    double cpp_tcp1_bcll92(double k) {
      const int imax = 10;
      double k0[imax] = {5.00E-02, 2.00E-01, 3.10E-01, 4.00E-01, 4.60E-01, 
        5.70E-01, 6.30E-01, 7.60E-01, 8.80E-01, 1.10E+00};
      double t0[imax] = {0.00E+00, 9.92E+08, 3.77E+09, 5.36E+09, 5.62E+09, 
        5.36E+09, 5.03E+09, 3.77E+09, 1.32E+09, 0.00E+00};
      double t2[imax] = {1.25E+10, 2.40E+11, -1.33E+11, -2.10E+11, -7.34E+10, 
        -3.03E+10, -4.51E+09, -1.65E+11, 1.54E+11, 4.80E+09};
      return solve_tc(imax,k0,t0,t2,k);
    }

    /** \brief Calculate the proton 1s0 superfluidity critical temperature from    
      ao, nucl.phys. a437(1985): p.487 from dotted curve fig. 15.         
      uses a cubic spline interpolation.                                  
      k is the fermi momentum in fm^-1 and tc is given in k  
    */
    double cpp_tcp1_ao(double k) {
      const int imax = 12;
      double k0[imax] = {0.00, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 
        0.80, 0.90, 1.00, 1.10, 1.20};
      double t0[imax] = {0.00e0, 7.91e8, 1.51e9, 2.14e9, 2.34e9, 
        2.04e9, 1.42e9, 8.57e8, 4.61e8, 1.65e8, 6.59e7, 0.00e0};
      double t2[imax] = {+5.13e+10, +1.61e+10, -4.86e+09, -5.00e+10, 
        -5.30e+10, -3.79e+10, +1.27e+10, +2.14e+10, 
        +1.82e+09, +3.13e+10, -8.90e+09, +2.42e+10};
      return solve_tc(imax,k0,t0,t2,k);
    }

    /** \brief Calculate the proton 1s0 superfluidity critical temperature from    
      ns as cited by ao, nucl.phys. a437(1985): p.487 from fig. 15.       
      uses a cubic spline interpolation.                                  
      k is the fermi momentum in fm^-1 and tc is given in k    
    */           
    double cpp_tcp1_ns(double k) {
      const int imax = 12;
      double k0[imax] = {0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 
        0.50, 0.60, 0.70, 0.80, 0.90};
      double t0[imax] = {0.00e0, 1.09e9, 2.54e9, 4.30e9, 5.52e9, 
        6.20e9, 5.40e9, 3.76e9, 1.45e9, 5.60e8, 1.32e8, 0.00e0};
      double t2[imax] = {+1.96e+11, +2.61e+11, +2.13e+11, -3.69e+11, 
        -3.43e+10, -7.90e+11, -3.57e+11, +2.01e+11, 
        +1.57e+11, +2.17e+10, +3.32e+10, +2.30e+10};
      return solve_tc(imax,k0,t0,t2,k);
    }

    /** \brief Calculate the proton 1s0 superfluidity critical temperature from    
      t73, prog.th.phys. 50(1973):p.1754                                  
      uses a cubic spline interpolation.                                  
      k is the fermi momentum in fm^-1 and tc is given in k   
    */            
    double cpp_tcp1_t73(double k) {
      const int imax = 12;
      double k0[imax] = {0.00, 0.30, 0.41, 0.48, 0.53, 0.58, 0.63, 
        0.67, 0.72, 0.76, 0.80, 0.84};
      double t0[imax] = {0.00e0, 7.58e8, 2.27e9, 3.03e9, 3.29e9, 
        3.00e9, 2.55e9, 2.11e9, 1.50e9, 8.45e8, 
        3.29e7, 0.00e0};
      double t2[imax] = {-2.49e+10, +1.00e+11, -6.79e+10, -5.58e+10, 
        -3.16e+11, -1.03e+09, -6.41e+10, -1.04e+10, 
        -5.55e+10, -3.64e+11, +9.21e+11, -3.99e+11};
      return solve_tc(imax,k0,t0,t2,k);
    }

    /** \brief Calculate the neutron 1S0 superfluidity critical temperature from 
      Chen, Clark, Dave & Khodel, Nucl. Phys. A555 (1993), p. 59
      uses a cubic spline interpolation.  
    */
    double cpp_tcp1_ccdk(double k) {
      const int imax = 14;
      double k0[imax] = {0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 
        0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 
        1.20, 1.30};
      double d0[imax] = {0.00, 0.05, 0.19, 0.41, 0.66, 0.84, 
        0.97, 1.00, 0.90, 0.72, 0.49, 0.22, 
        0.07, 0.00};
      double t0[imax], t2[imax];

      if (done!=1.0) {
        for (int i = 0; i < imax; ++i) {
          t0[i] = d0[i] / 1.76 * 1.1604e10; }
        spline_here(k0, t0, imax, 0.0, 0.0, t2);
        done=1.0; }

      return solve_tc(imax,k0,t0,t2,k);
    }

    /** \brief Calculate the proton 1s0 superfluidity critical temperature from    
      ccy, nucl.phys. a179(1972): p.320. from curve psi) fig.4.           
      uses a cubic spline interpolation.                                  
      k is the fermi momentum in fm^-1 and tc is given in k    
    */           
    double cpp_tcp1_ccy_ps(double k) {
      const int imax = 12;
      double k0[imax] = {0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 
        0.55, 0.60, 0.70, 0.80, 0.90, 0.95};
      double t0[imax] = {0.00e0, 2.97e8, 1.12e9, 2.24e9, 3.53e9, 
        4.18e9, 4.28e9, 4.20e9, 3.59e9, 2.24e9, 
        4.94e8, 0.00e0};
      double t2[imax] = {5.94e+10, 5.94e+10, 1.84e+10, 4.50e+10, 
        -9.63e+10, -4.37e+10, -8.52e+10, -4.76e+10, 
        -8.46e+10, -5.79e+10, 7.87e+10, 5.53e+11};

      return solve_tc(imax,k0,t0,t2,k);
    }

    /** \brief calculate the proton 1s0 superfluidity critical temperature from    
      ccy, nucl.phys. a179(1972): p.320. from curve msii) fig. 4.         
      uses a cubic spline interpolation.                                  
      k is the fermi momentum in fm^-1 and tc is given in k  
    */             
    double cpp_tcp1_ccy_ms(double k) {
      const int imax = 14;
      double k0[imax] = {0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 
        0.60, 0.65, 0.70, 0.80, 0.90, 1.00, 
        1.05, 1.10};
      double t0[imax] = {0.00e0, 2.97e8, 1.12e9, 2.24e9, 3.66e9, 
        4.78e9, 5.28e9, 5.28e9, 5.14e9, 4.48e9, 
        3.23e9, 1.32e9, 1.32e8, 0.00e0};
      double t2[imax] = {5.91e+10, 5.99e+10, 1.68e+10, 5.12e+10, 
        -4.14e+10, -6.54e+10, -6.89e+10, -5.60e+10, 
        -4.33e+10, -7.02e+10, -2.99e+10, -2.06e+11, 
        7.38e+11, -2.10e+11};
      return solve_tc(imax,k0,t0,t2,k);
    }

    /** \brief Calculates the neutron 3p2 superfluidity critical temperature from  
      Elgaroy, Engvik, Hjorth-Jensen & Osnes, preprint 1996               
      relativistic case                                               
      Uses a cubic spline interpolation.                                  
      k is the fermi momentum in fm^-1 and Tc is given in K  
    */             
    double cpp_tcn3_eehjo96_r(double k) {
      const int imax = 4;
      double k0[imax] = {1.06E+00, 1.33E+00, 1.53E+00, 1.68E+00};
      double t0[imax] = {0.00E+00, 8.30E+07, 7.00E+07, 0.00E+00};
      double t2[imax] = {4.62E+09, -2.66E+09, -5.17E+09, 1.15E+10};
      return solve_tc(imax,k0,t0,t2,k);
    }

    /** \brief Calculates the neutron 3p2 superfluidity critical temperature from  
      Elgaroy, Engvik, Hjorth-Jensen & Osnes, preprint 1996               
      non-relativistic case                                               
      Uses a cubic spline interpolation.                                  
      k is the fermi momentum in fm^-1 and Tc is given in K 
    */             
    double cpp_tcn3_eehjo96_nr(double k) {
      const int imax = 11;
      double k0[imax] = {1.06E+00, 1.33E+00, 1.53E+00, 1.68E+00,
        1.81E+00, 1.92E+00, 2.02E+00, 2.12E+00,
        2.20E+00, 2.28E+00, 2.35E+00};
      double t0[imax] = {0.00E+00, 3.30E+07, 1.43E+08, 2.59E+08,
        3.09E+08, 3.23E+08, 3.27E+08, 3.14E+08,
        2.51E+08, 1.19E+08, 0.00E+00};
      double t2[imax] = {5.69E+07, 2.50E+09, 1.77E+09, -3.92E+09,
        -2.10E+09, -3.77E+08, -1.03E+09, -7.04E+09,
        -1.22E+10, -1.32E+10, 7.25E+10};
      return solve_tc(imax,k0,t0,t2,k);
    }

    /** \brief Calculates the neutron 3p2 superfluidity critical temperature from  
      Baldo, Cugnon, Leujeune & Lombardo, N.Ph. A536 (1992), p. 349       
      Uses a cubic spline interpolation.                                  
      k is the fermi momentum in fm^-1 and Tc is given in K  
    */
    double cpp_tcn3_bcll92(double k) {
      const int imax = 12;
      double k0[imax] = {1.50E00, 1.80E00, 2.00E00, 2.25E00, 2.50E00, 2.60E00,
        2.70E00, 2.80E00, 2.90E00, 3.00E00, 3.50E00, 4.50E00};
      double t0[imax] = {0.00E00, 6.89E08, 1.72E09, 4.48E09, 7.64E09, 9.16E09,
        1.03E10, 1.07E10, 1.03E10, 8.95E09, 4.13E09, 0.00E00};
      double t2[imax] = {2.17E10, 2.56E09, 4.08E10, -8.32E09, 3.22E10, -5.57E10,
        -5.72E10, -8.74E10, -1.30E11, 3.06E10, 2.14E09, 1.13E10};

      return solve_tc(imax,k0,t0,t2,k);
    }

    /** \brief Calculate the neutron 3p2 superfluidity critical temperature from   
      a0, nucl.phys. a442(1985):p.163. m*=1 gap from fig.2                
      uses a cubic spline interpolation.                                  
      k is the fermi momentum in fm^-1 and tc is given in k   
    */            
    double cpp_tcn3_ao_m1(double k) {
      const int imax = 23;
      double k0[imax] = {1.35, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4,
        2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5};
      double t0[imax] = {1.00e0, 6.92e7, 2.77e8, 5.26e8, 8.45e8, 1.14e9,
        1.51e9, 1.87e9, 2.16e9, 2.34e9, 2.44e9, 2.47e9,
        2.40e9, 2.23e9, 1.95e9, 1.56e9, 1.16e9, 8.10e8,
        5.26e8, 3.05e8, 1.52e8, 5.54e7, 1.00e0};
      double t2[imax] = {8.36e+10, -1.20e+09, 3.42e+09, 1.22e+10, -1.03e+10,
        1.47e+10, -3.60e+09, -6.32e+09, -1.31e+10, -7.25e+09,
        -5.91e+09, -1.11e+10, -9.56e+09, -1.06e+10, -1.40e+10,
        6.01e+08, 5.59e+09, 7.02e+09, 5.93e+09, 7.07e+09,
        6.58e+09, 4.33e+08, 1.64e+10};
      return solve_tc(imax,k0,t0,t2,k);
    }

    /** \brief calculate the neutron 3p2 superfluidity critical temperature from   
      t72, rep.prog.theor.phys. 48(1972):p.1517. from fig.2   m*=1        
      uses a cubic spline interpolation.                                  
      k is the fermi momentum in fm^-1 and tc is given in k 
    */             
    double cpp_tcn3_t72_m1(double k) {
      const int imax = 25;
      double k0[imax] = {1.20, 1.39, 1.56, 1.70, 1.84, 1.97, 2.09, 2.20, 2.31, 2.41,
       2.51, 2.60, 2.69, 2.78, 2.87, 2.95, 3.03, 3.11, 3.19, 3.26,
        3.34, 3.41, 3.48, 3.55, 3.61};
      double t0[imax] = {0.00e0, 1.66e8, 4.85e8, 1.05e9, 1.73e9, 2.27e9, 
        2.70e9, 3.02e9, 3.20e9, 3.20e9,
        3.10e9, 2.96e9, 2.79e9, 2.60e9, 2.38e9, 2.15e9, 
        1.91e9, 1.65e9, 1.38e9, 1.14e9,
        8.70e8, 6.23e8, 3.74e8, 1.38e8, 0.00e0};
      double t2[imax] = {1.37e+10, 1.32e+08, 1.95e+10, 6.07e+09, -8.58e+09, 
        -3.38e+09, -5.16e+09, -1.15e+10, -1.82e+10, -9.07e+09,
        -5.51e+09, -3.69e+09, -1.94e+09, -3.36e+09, -6.83e+09, 
        5.15e+08, -4.60e+09, -8.46e+08, -1.39e+09, 2.31e+09,
        -3.45e+09, -1.04e+09, 5.14e+09, -3.61e+09, 1.17e+11};
      return solve_tc(imax,k0,t0,t2,k);
    }

    /** \brief Calculate the neutron 3p2 superfluidity critical temperature from   
      t72, rep.prog.theor.phys. 48(1972):p.1517.                          
      uses a cubic spline interpolation.                                  
      k is the fermi momentum in fm^-1 and tc is given in k  
    */ 
    double cpp_tcn3_t72(double k) {
      const int imax = 12;
      double k0[imax] = {1.38, 1.42, 1.46, 1.53, 1.64, 1.73, 1.82, 1.89, 
        1.96, 1.98, 2.02, 2.06};
      double t0[imax] = {0.00e0, 1.00e7, 1.00e8, 3.02e8, 6.02e8, 7.76e8, 
        6.02e8, 3.49e8, 1.52e8, 1.00e8, 1.00e7, 0.00e0};
      double t2[imax] = {-2.08e+10, +7.91e+10, +4.34e+09, -4.36e+09, +2.86e+09, -6.03e+10,
        -1.94e+10, +2.21e+10, -3.03e+08, -1.02e+10, +8.33e+10, -2.29e+10};

      double k1_8 = 1.68;
      double k2_8 = 2.38;
      double slope = (1.98 - 1.46) / (k2_8 - k1_8);
      double kk = 1.46 + slope * (k - k1_8);
      return solve_tc(imax,k0,t0,t2,kk);
    }

    /** \brief calculate the neutron 3p2 superfluidity critical temperature from   
      a0, nucl.phys. a442(1985):p.163.                                    
      uses a cubic spline interpolation.                                  
      k is the fermi momentum in fm^-1 and tc is given in k
    */               
    double cpp_tcn3_ao(double k) {
      const int imax = 26;
      double k0[imax] = {1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
        2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
        3.1, 3.2, 3.3, 3.4, 3.5, 3.6};
      double t0[imax] = {0.00e0, 3.50e7, 1.25e8, 2.42e8, 3.67e8,
        4.85e8, 5.95e8, 7.06e8, 8.01e8, 8.72e8,
        8.80e8, 8.31e8, 6.79e8, 5.12e8, 3.95e8,
        3.12e8, 2.51e8, 1.99e8, 1.65e8, 1.37e8,
        1.12e8, 8.78e7, 6.45e7, 4.15e7, 1.88e7};
      double t2[imax] = {+7.63e+09, +5.74e+09, +2.41e+09, +8.37e+08, -9.52e+08,
        -1.23e+09, +1.07e+09, -2.46e+09, -8.48e+08, -8.55e+09,
        -2.74e+09, -1.47e+10, -3.01e+08, +6.89e+09, +2.72e+09,
        +2.61e+09, +3.23e+07, +2.66e+09, +1.29e+08, +4.24e+08,
        -2.62e+07, +1.60e+08, -7.58e+07, +3.23e+08, -1.04e+09,
        +6.16e+09};
      return solve_tc(imax,k0,t0,t2,k);
    }

    /** \brief calculates the neutron 3p2 superfluidity critical temperature from  
      hgrr, phys.rev.lett. 24(1970):p.775.    m*=1                        
      uses a cubic spline interpolation.                                  
      k is the fermi momentum in fm^-1 and tc is given in k
    */               
    double cpp_tcn3_hgrr(double k) {
      const int imax = 21;
      double k0[imax] = {1.0, 1.2, 1.4, 1.6, 1.8, 1.9, 2.0,
        2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
        2.8, 2.9, 3.0, 3.2, 3.4, 3.6, 3.8};
      double t0[imax] = {0.00e0, 1.75e8, 8.12e8, 2.20e9, 3.83e9, 4.58e9, 5.22e9,
        5.74e9, 6.09e9, 6.18e9, 6.06e9, 5.80e9, 5.41e9, 4.97e9,
        4.41e9, 3.83e9, 3.19e9, 1.97e9, 8.12e8, 1.48e8, 0.00e0};
      double t2[imax] = {+8.65e09, +8.95e09, +2.48e10, +4.34e09, -5.91e09,
        -1.22e10, -1.11e10, -1.52e10, -2.99e10, -2.12e10,
        -1.14e10, -1.71e10, +1.67e09, -1.96e10, +4.83e09,
        -1.17e10, +5.98e09, -3.09e09, +1.57e10, +1.45e10, +3.87e09};
      return solve_tc(imax,k0,t0,t2,k);
    }

    /** \brief Calculate the neutron 3p2 superfluidity critical temperature from   
      Schulze, Cugnon, Lejeune, Baldo & Lombardo, preprint                
      with medium polarization effects                                    
      uses a cubic spline interpolation.                                  
      k is the fermi momentum in fm^-1 and Tc is given in K
    */ 
    double cpp_tcn1_sclbl96_pol(double k) {
      const int imax = 14;
      double k0[imax] = {0.30, 0.50, 0.60, 0.70, 0.90,
        1.10, 1.20, 1.25, 1.30, 1.40,
        1.50, 1.55, 1.60, 1.70};
      double t0[imax] = {0.00e0, 3.31e8, 5.95e8, 1.46e9, 3.57e9,
        5.82e9, 6.75e9, 6.81e9, 6.75e9, 5.82e9,
        3.77e9, 1.92e9, 7.94e8, 0.00e0};
      double t2[imax] = {3.31e10, -1.65e10, 9.25e10, 3.57e9, 2.56e9,
        6.03e9, -1.60e11, -2.24e9, -1.48e11, -3.07e10,
        -4.04e11, 4.99e11, 1.54e11, 1.61e11};
      return solve_tc(imax,k0,t0,t2,k);
    }

    /** \brief Calculate the neutron 3p2 superfluidity critical temperature from   
      Schulze, Cugnon, Lejeune, Baldo & Lombardo, preprint                
      uses a cubic spline interpolation.                                  
      k is the fermi momentum in fm^-1 and Tc is given in K 
    */
    double cpp_tcn1_sclbl96(double k) {
      const int imax = 12;
      double k0[imax] = {0.00E+00, 0.25E+00, 0.40E+00, 0.60E+00,
        0.70E+00, 0.82E+00, 0.90E+00, 1.00E+00,
        1.10E+00, 1.20E+00, 1.30E+00, 1.50E+00};
      double t0[imax] = {0.00E+00, 4.17E+09, 9.06E+09, 1.59E+10,
        1.81E+10, 1.90E+10, 1.86E+10, 1.66E+10,
        1.34E+10, 9.19E+09, 4.50E+09, 0.00E+00};
      double t2[imax] = {1.70E+11, 6.08E+10, 3.14E+10, -1.13E+11,
        -1.21E+11, -1.40E+11, -1.73E+11, -1.08E+11,
        -1.10E+11, -8.58E+10, 1.76E+11, 2.49E+11};
      return solve_tc(imax,k0,t0,t2,k);
    }

    /** \brief Calculate the neutron 1s0 superfluidity critical temperature from   
      Broglia et al, Phys.Rev. D***, 1994, p. ****                        
      uses a cubic spline interpolation. 
    */  
    double cpp_tcn1_bbllp(double k) {
      std::cout << "AWS: Removed because this required a data file." << std::endl;
      return 0.0;
    }

    /** \brief Calculate the neutron 1s0 superfluidity critical temperature from   
      awp  phys.lett. 222(1989): p.173. case iii, from fig. 3.            
      uses a cubic spline interpolation. 
    */   
    double cpp_tcn1_awp_3(double k) {
      const int imax = 15;
      double k0[imax] = {0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9, 1.0,
        1.1, 1.2, 1.3, 1.4, 1.5};
      double t0[imax] = {0.00E+00, 2.64E+08, 7.91E+08, 1.78E+09, 3.36E+09,
        5.27E+09, 6.59E+09, 7.25E+09, 7.05E+09, 5.74E+09,
        3.96E+09, 1.98E+09, 7.91E+08, 1.32E+08, 0.00E+00};
      double t2[imax] = {7.54E+10, 7.60E+09, 5.20E+10, 6.16E+10, 5.62E+10,
        -8.83E+10, -5.71E+10, -7.94E+10, -1.41E+11, -2.12E+10,
        -5.59E+10, 1.25E+11, 3.07E+10, 7.03E+10, 4.47E+09};
      return solve_tc(imax,k0,t0,t2,k);
    }

    /** \brief Calculate the neutron 1s0 superfluidity critical temperature from   
      awp  phys.lett. 222(1989): p.173. case ii, from fig. 3.             
      uses a cubic spline interpolation.                                  
      k is the fermi momentum in fm^-1 and tc is given in kelvins  
    */ 
    double cpp_tcn1_awp_2(double k) {
      const int imax = 17;
      double k0[imax] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
        0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
        1.3, 1.4, 1.5, 1.6, 1.7};
      double t0[imax] = {0.00E+00, 3.30E+08, 1.18E+09, 2.44E+09, 4.20E+09, 6.13E+09,
        7.91E+09, 9.10E+09, 9.56E+09, 9.03E+09, 7.71E+09, 5.93E+09,
        4.15E+09, 2.50E+09, 1.12E+09, 3.61E+08, 0.00E+00};
      double t2[imax] = {7.58E+10, 4.64E+10, 5.06E+10, -2.87E+09, 1.17E+11,
        -7.45E+10, -5.28E+10, -6.83E+10, -1.12E+11, -7.75E+10,
        -5.19E+10, 9.16E+09, 1.53E+10, 7.71E+09, 1.16E+11, -3.87E+10, 1.58E+11};
      return solve_tc(imax,k0,t0,t2,k);
    }

    /** \brief Calculates the neutron 1s0 superfluidity critical temperature from  
      ccks nucl.phys. a451(1986): p.509. reid pot. with cbf method        
      uses a cubic spline interpolation.                                  
      k is the fermi momentum in fm^-1 and tc is given in kelvins
    */ 
    double cpp_tcn1_ccks_cbf(double k) {
      const int imax = 9;
      double k0[imax] = {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1};
      double t0[imax] = {0.00E+00, 1.25E+09, 3.43E+09, 3.49E+09, 3.03E+09, 2.11E+09, 9.23E+08, 6.59E+07, 0.00E+00};
      double t2[imax] = {2.97E+11, 1.57E+11, -3.65E+11, 3.05E+10, -6.93E+10, -2.93E+10, 2.62E+10, 1.23E+11, -4.15E+10};
      return solve_tc(imax,k0,t0,t2,k);
    }

    /** \brief Calculate the neutron 1s0 superfluidity critical temperature from    
      ccks nucl.phys. a451(1986): p.509. reid pot. with variational method 
      uses a cubic spline interpolation.                                   
      k is the fermi momentum in fm^-1 and tc is given in kelvins
    */
    double cpp_tcn1_ccks_var(double k) {
      const int imax = 10;
      double k0[imax] = {0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4};
      double t0[imax] = {1.32E10, 1.36E10, 1.39E10, 1.40E10, 1.30E10, 1.02E10, 6.06E09, 1.98E09, 1.32E08, 0.00E00};
      double t2[imax] = {1.48E11, -5.52E10, 1.30E10, -1.17E11, -2.05E11, -1.43E11, -2.83E10, 2.92E11, 1.99E11, -6.01E10};
      return solve_tc(imax,k0,t0,t2,k);
    }

    /** \brief Calculate the neutron 1s0 superfluidity critical temperature from   
      ao, nucl.phys. a437(1985): p.487 from case v(rsc)eff of fig. 11.    
      uses a cubic spline interpolation.                                  
      k is the fermi momentum in fm^-1 and tc is given in kelvins
    */
    double cpp_tcn1_ao(double k) {
      const int imax = 21;
      double k0[imax] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
        1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 
        1.9, 2.0, 2.1};
      double t0[imax] = {0.00E00, 8.89E08, 2.64E09, 5.93E09, 1.12E10, 
        1.68E10, 2.31E10, 2.80E10, 3.10E10, 3.23E10, 
        3.23E10, 3.06E10, 2.77E10, 2.34E10, 1.85E10, 
        1.22E10, 7.58E09, 3.96E09, 1.65E09, 5.25E08, 0.00E00};
      double t2[imax] = {3.09E11, -2.43E10, 1.86E11, 2.66E11, -5.97E10, 
        1.71E11, -2.06E11, -1.87E11, -1.84E11, -9.61E10, 
        -2.12E11, -7.73E10, -1.99E11, 3.43E10, -2.98E11, 
        3.18E11, 3.58E10, 1.39E11, 1.94E11, -8.32E10, 2.59E11};
      return solve_tc(imax,k0,t0,t2,k);
    }

    /** \brief Calculate the neutron 1s0 superfluidity critical temperature from   
      t84, prog.theor.phys. 71(1984): p.1432                              
      uses a cubic spline interpolation.                                  
      k is the fermi momentum in fm^-1 and tc is given in kelvins
    */  
    double cpp_tcn1_t84(double k) {
      const int imax = 14;
      double k0[imax] = {0.00, 0.14, 0.20, 0.24, 0.33, 0.45, 0.52, 0.61, 
        0.81, 0.97, 1.09, 1.19, 1.30, 1.50};
      double t0[imax] = {0.00E00, 1.12E09, 2.44E09, 3.76E09, 6.72E09, 
        1.11E10, 1.32E10, 1.57E10, 1.75E10, 1.54E10, 
        1.18E10, 7.25E09, 2.97E09, 0.00E00};
      double t2[imax] = {1.09E11, 1.25E11, 3.10E11, -8.90E10, 1.12E11, 
        -1.44E11, 3.47E10, -1.59E11, -1.17E11, -1.03E11, 
        -2.06E11, 9.83E10, 1.71E11, 1.37E11};
      return solve_tc(imax,k0,t0,t2,k);
    }

    /** \brief Calculate the neutron 1s0 superfluidity critical temperature from   
      ns, preprint 1981 as cited in                                       
      ao nucl. phys. a437(1986): p. 487, fig 14.                          
      uses a cubic spline interpolation.                                  
      k is the fermi momentum in fm^-1 and tc is given in kelvins 
    */
    double cpp_tcn1_ns(double k) {
      const int imax = 17;
      double k0[imax] = {0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 
        0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 
        1.15, 1.20, 1.25};
      double t0[imax] = {0.00E00, 1.38E09, 3.16E09, 5.27E09, 6.59E09, 
        8.50E09, 1.07E10, 1.42E10, 1.61E10, 1.74E10, 
        1.81E10, 1.78E10, 1.52E10, 1.22E10, 8.31E09, 
        2.31E09, 0.00E00};
      double t2[imax] = {4.48E11, -6.74E10, 6.18E10, 1.83E10, 4.03E11, 
        -2.14E11, 1.15E12, -1.26E12, 3.50E10, -3.25E11, 
        -1.77E11, -1.37E12, 1.31E11, -1.17E11, -1.80E12, 
        2.25E12, 1.65E12};
      return solve_tc(imax,k0,t0,t2,k);
    }

    /** \brief calculate the neutron 1s0 superfluidity critical temperature from   
      t72, prog.theor.phys. 48(1972): p.1517 with values as reported in   
      table i of t84, prog.theor.phys 71(1984): p.1432.              
      uses a cubic spline interpolation.                                  
      k is the fermi momentum in fm^-1 and tc is given in kelvins 
    */
    double cpp_tcn1_t72(double k) {
      const int imax = 14;
      double k0[imax] = {0.00, 0.14, 0.20, 0.24, 0.33, 0.45, 0.52, 
        0.61, 0.81, 0.97, 1.09, 1.19, 1.30, 1.50};
      double t0[imax] = {0.00E00, 1.12E09, 2.37E09, 3.69E09, 6.53E09, 
        1.07E10, 1.29E10, 1.52E10, 1.71E10, 1.50E10, 
        1.15E10, 7.51E09, 3.30E09, 0.00E00};
      double t2[imax] = {1.24E11, 9.50E10, 3.61E11, -1.20E11, 9.13E10, 
        -6.95E10, -6.40E10, -1.10E11, -1.34E11, -1.08E11, 
        -1.20E11, 1.15E10, 1.54E11, 1.71E11};
      return solve_tc(imax,k0,t0,t2,k);
    }

    /** \brief Calculates the neutron 1S0 superfluidity critical temperature from   
      Gondolfi, Illarionov, Pederiva, Schmidt & Fantoni,                  
      PRL 111 (2008), 132501                                              
      uses a cubic spline interpolation.                                  
      k is the fermi momentum in fm^-1 and Tc is given in K 
    */  
    double cpp_tcn1_gipsf(double k) {
      const int imax = 9;
      double k0[imax] = {0.000, 0.200, 0.300, 0.400, 0.600, 0.700, 0.800, 1.000, 1.200};
      double d0[imax] = {0.000, 0.300, 0.900, 1.500, 2.100, 1.900, 1.500, 0.500, 0.000};
      double d2[imax];
      if (done!=1.1111) {
        spline_here(k0, d0, imax, 0.0, 0.0, d2);
        done=1.1111; }
      return solve_tc(imax,k0,d0,d2,k)/1.76*1.1604e10;
    }

    /** \brief This attempts to mock-up results of Gezerlis & Carlson, by using the
      Takatsuka (1972) values scaled down !
      calculate the neutron 1s0 superfluidity critical temperature from   
      t72, prog.theor.phys. 48(1972): p.1517 with values as reported in   
      table i of t84, prog.theor.phys 71(1984): p.1432.              
      uses a cubic spline interpolation.                                  
      k is the fermi momentum in fm^-1 and tc is given in kelvins 
    */
    double cpp_tcn1_gc(double k) {
      const int imax = 14;
      double k0[imax] = {0.00, 0.14, 0.20, 0.24, 0.33, 0.45, 0.52, 
        0.61, 0.81, 0.97, 1.09, 1.19, 1.30, 1.50};
      double t0[imax] = {0.00e00, 1.12e09, 2.37e09, 3.69e09, 6.53e09, 
        1.07e10, 1.29e10, 1.52e10, 1.71e10, 1.50e10, 
        1.15e10, 7.51e09, 3.30e09, 0.00e00};
      double t2[imax] = {1.24e11, 9.50e10, 3.61e11, -1.20e11, 9.13e10, 
        -6.95e10, -6.40e10, -1.10e11, -1.34e11, -1.08e11, 
        -1.20e11, 1.15e10, 1.54e11, 1.71e11};
      return solve_tc(imax,k0,t0,t2,k)*0.7;
    }

    /** \brief Calculate the neutron 1S0 superfluidity critical temperature from   
      Wambach, Ainsworth & Pines Nucl. Phys. A555 (1993), p. 128
      uses a cubic spline interpolation.
    */  
    double cpp_tcn1_wap(double k) {
      const int imax = 14;
      double k0[imax] = {0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 
        0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40};
      double d0[imax] = {0.00, 0.03, 0.13, 0.30, 0.54, 0.74, 0.86, 
        0.90, 0.79, 0.59, 0.35, 0.14, 0.03, 0.00};
      double t0[imax]; double t2[imax];
      if (done!=1.0) {
        for (int i=0; i<imax; ++i) {
          t0[i] = d0[i]/1.76*1.1604e10; }
        spline_here(k0, t0, imax, 0.0, 0.0, t2);
        done=1.0; }
      return solve_tc(imax,k0,t0,t2,k);
    }

    /** \brief Calculate the neutron 1s0 superfluidity critical temperature from   
      SFB  nucl-th/0207004                                                
      uses a cubic spline interpolation.
    */  
    double cpp_tcn1_sfb(double k) {
      const int imax = 18;
      double k0[imax] = {0.000,
        0.100,0.200,0.300,0.400,0.500,
        0.600,0.700,0.800,0.900,1.000,
        1.100,1.175,1.250,1.300,1.350,
        1.400,1.450};
      double d0[imax] = {0.000,
        0.000,0.090,0.210,0.350,0.490,
        0.610,0.720,0.790,0.780,0.700,
        0.570,0.440,0.280,0.190,0.100,
        0.030,0.000};
      double d2[imax];
      if (done!=1.1111) {
        spline_here(k0, d0, imax, 0.0, 0.0, d2);
        done=1.1111; }
      return solve_tc(imax,k0,d0,d2,k)/1.76*1.1604e10;
    }

    /** \brief Calculate the neutron 1S0 superfluidity critical temperature from   
      Chen, Clark, Dave & Khodel, Nucl. Phys. A555 (1993), p. 59
      uses a cubic spline interpolation. 
    */ 
    double cpp_tcn1_ccdk(double k){
      const int imax = 11;
      double k0[imax] = {0.10,0.20,0.30,0.40,0.50,
        0.60,0.70,0.80,0.90,1.00,
        1.10};
      double d0[imax] = {0.00,0.02,0.14,0.36,0.60,
        0.83,0.86,0.67,0.35,0.07,
        0.00};
      double t0[imax]; double t2[imax];
      if (done!=1.0) {
        for (int i=0; i<imax; ++i) {
          t0[i] = d0[i]/1.76*1.1604e10; }
        spline_here(k0, t0, imax, 0.0, 0.0, t2);
        done=1.0; }
      return solve_tc(imax,k0,t0,t2,k);
    }

    double cpp_tc_Ioffe_1p(double kf) {
      return cpp_tc_Ioffe(kf, 20.29, 0.0, 1.117, 1.241, 0.1473); 
    }

    double cpp_tc_Ioffe_2p(double kf) {
      return cpp_tc_Ioffe(kf, 17.0, 0.0, 1.117, 1.329, 0.1179); 
    }

    double cpp_tc_Ioffe_3p(double kf) {
      return cpp_tc_Ioffe(kf, 14.5, 0.0, 1.117, 1.518, 0.1179); 
    }

    double cpp_tc_Ioffe_1ns(double kf) {
      return cpp_tc_Ioffe(kf, 10.2, 0.0, 0.6, 1.45, 0.1); 
    }

    double cpp_tc_Ioffe_2ns(double kf) {
      return cpp_tc_Ioffe(kf, 7.9, 0.0, 0.3, 1.45, 0.01); 
    }

    double cpp_tc_Ioffe_3ns(double kf) {
      return cpp_tc_Ioffe(kf, 1800.0, 0.0, 21.0, 1.45, 0.4125); 
    }

    double cpp_tc_Ioffe_1nt(double kf) {
      return cpp_tc_Ioffe(kf, 6.461, 1.0, 1.961, 2.755, 1.3); 
    }

    double cpp_tc_Ioffe_2nt(double kf) {
      return cpp_tc_Ioffe(kf, 2.0, 1.0, 1.961, 2.755, 1.3);  
    }

    double cpp_tc_Ioffe_3nt(double kf) {
      return cpp_tc_Ioffe(kf, 15.0, 1.0, 1.961, 2.755, 1.3); 
    }

};

#endif