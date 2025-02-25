/*
  -------------------------------------------------------------------
  
  Copyright (C) 2015-2019, Andrew W. Steiner
  
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

  /// \name Data for SFB singlet neutron gap
  //@{
  o2scl::interp_vec<std::vector<double> > itp_tcn1_sfb;
  bool flag_tcn1_sfb;
  std::vector<double> k0_tcn1_sfb;
  std::vector<double> d0_tcn1_sfb;
  //@}

  /// \name Data for T73 singlet proton gap
  //@{
  bool flag_tcp1_t73;
  std::vector<double> k0_tcp1_t73;
  std::vector<double> t0_tcp1_t73;
  std::vector<double> t2_tcp1_t73;
  //@}
  
 public:
  
  tc() {
    flag_tcn1_sfb=false;
    flag_tcp1_t73=false;
  }

  /** \brief Compute critical temperatures
   */
  void get_tc(int imax, int idrip, int icore,
	      double sfn1s0, double n1tc,
	      double n1kf, double n1dk,
	      double sfp1s0, double p1tc,
	      double p1kf, double p1dk,
	      double sfn3p2, double n3tc,
	      double n3kf, double n3dk,
	      double fn1s0, double fp1s0, double fn3p2,
	      double *kf_n, double *kf_p, double *kf_la,
	      double *kf_sm, double *kf_s0, double *kf_sp,
	      double *kf_u, double *kf_d, double *kf_s,
	      double *tc_n, double *tc_p, double *tc_la,
	      double *tc_uu, double *tc_dd, double *tc_ss,
	      double *tc_ud, double *tc_us, double *tc_ds,
	      double *tc_u, double *tc_d, double *tc_s, int *isf) {

    /*
      std::cout << "H0: " << kf_n[0] << " " << tc_n[0] << " " << sfn1s0 << " "
      << fn1s0 << std::endl;
      std::cout << "H0: " << kf_p[0] << " " << tc_p[0] << " " << sfp1s0 << " "
      << fp1s0 << std::endl;
      std::cout << "H0: " << kf_p[73] << " " << tc_p[73] << " " << sfp1s0 << " "
      << fp1s0 << std::endl;
    */
    
    for(int i=0;i<imax;i++) {
      tc_n[i]=1.0;
      tc_p[i]=1.0;
      tc_la[i]=1.0;
      tc_uu[i]=0.0;
      tc_dd[i]=0.0;
      tc_ss[i]=0.0;
      tc_ud[i]=0.0;
      tc_us[i]=0.0;
      tc_ds[i]=0.0;
      tc_u[i]=1.0;
      tc_d[i]=1.0;
      tc_s[i]=1.0;
    }
    // Neutron singlet
    if (fabs(sfn1s0-1.0)<1.0e-6) {
      for(int i=0;i<idrip;i++) {
	tc_n[i]=tcn1_sfb(kf_n[i])*fn1s0;
	if (tc_n[i]<1.0) tc_n[i]=1.0;
      }
    }
    if (fabs(sfn1s0-150.0)<1.0e-6) {
      for(int i=0;i<idrip;i++) {
	tc_n[i]=n1tc*exp(-pow(kf_n[i]-n1kf,2.0)/n1dk/n1dk)*fn1s0;
      }
    }
    // Proton singlet
    if (fabs(sfp1s0-3.0)<1.0e-6) {
      for(int i=0;i<icore;i++) {
	tc_p[i]=tcp1_t73(kf_p[i])*fp1s0;
	if (tc_p[i]<1.0) tc_p[i]=1.0;
      }
    }
    if (fabs(sfp1s0-150.0)<1.0e-6) {
      for(int i=0;i<idrip;i++) {
	tc_n[i]=p1tc*exp(-pow(kf_n[i]-p1kf,2.0)/p1dk/p1dk)*fp1s0;
      }
    }
    // Neutron triplet
    if (fabs(sfn3p2-101.0)<1.0e-6) {
      n3kf=1.8;
      n3dk=0.5;
      n3tc=1.0e9;
      for(int i=0;i<idrip;i++) {
	double temp=n3tc*exp(-pow(kf_n[i]-n3kf,2.0)/n3dk/n3dk)*fn3p2;
	if (temp>tc_n[i]) {
	  tc_n[i]=temp;
	  *isf=i;
	}
      }
    }
    if (fabs(sfn3p2-150.0)<1.0e-6) {
      for(int i=0;i<idrip;i++) {
	double temp=n3tc*exp(-pow(kf_n[i]-n3kf,2.0)/n3dk/n3dk)*fn3p2;
	if (temp>tc_n[i]) {
	  tc_n[i]=temp;
	  *isf=i;
	}
      }
    }
    for(int i=0;i<idrip;i++) {
      tc_n[i]=fabs(tc_n[i]);
      tc_p[i]=fabs(tc_p[i]);
      tc_la[i]=fabs(tc_la[i]);
      tc_uu[i]=fabs(tc_uu[i]);
      tc_dd[i]=fabs(tc_dd[i]);
      tc_ss[i]=fabs(tc_ss[i]);
      tc_ud[i]=fabs(tc_ud[i]);
      tc_us[i]=fabs(tc_us[i]);
      tc_ds[i]=fabs(tc_ds[i]);
      tc_u[i]=fabs(tc_u[i]);
      tc_d[i]=fabs(tc_d[i]);
      tc_s[i]=fabs(tc_s[i]);
    }

    /*
      std::cout << "H0: " << kf_n[0] << " " << tc_n[0] << " " << sfn1s0 << " "
      << fn1s0 << std::endl;
      std::cout << "H0: " << kf_p[0] << " " << tc_p[0] << " " << sfp1s0 << " "
      << fp1s0 << std::endl;
      std::cout << "H0: " << kf_p[73] << " " << tc_p[73] << " " << sfp1s0 << " "
      << fp1s0 << std::endl;
    */

    return;
  }

  double tcn1_sfb(double kf) {
    if (flag_tcn1_sfb==false) {
      k0_tcn1_sfb={0.000,0.100,0.200,0.300,0.400,0.500,
		   0.600,0.700,0.800,0.900,1.000,
		   1.100,1.175,1.250,1.300,1.350,
		   1.400,1.450};
      d0_tcn1_sfb={0.000,0.000,0.090,0.210,0.350,0.490,
		   0.610,0.720,0.790,0.780,0.700,
		   0.570,0.440,0.280,0.190,0.100,
		   0.030,0.000};
      itp_tcn1_sfb.set(k0_tcn1_sfb.size(),k0_tcn1_sfb,d0_tcn1_sfb);
      flag_tcn1_sfb=true;
    }
    if (kf<=0.0 || kf>=1.45) return 0.0;
    double Tc=itp_tcn1_sfb.eval(kf)/1.76e0*1.1604e10;
    return Tc;
  }

  double tcp1_t73(double kf) {
    if (flag_tcp1_t73==false) {
      k0_tcp1_t73={0.00,0.30,0.41,0.48,0.53,0.58,
		   0.63,0.67,0.72,0.76,0.80,0.84};
      t0_tcp1_t73={0.00e0,7.58e8,2.27e9,3.03e9,3.29e9,3.00e9,
		   2.55e9,2.11e9,1.50e9,8.45e8,3.29e7,0.00e0};
      t2_tcp1_t73={-2.49e+10,+1.00e+11,-6.79e+10,-5.58e+10,
		   -3.16e+11,-1.03e+09,-6.41e+10,-1.04e+10,
		   -5.55e+10,-3.64e+11,+9.21e+11,-3.99e+11};
      flag_tcp1_t73=true;
    }
    if (kf<=k0_tcp1_t73[0] || kf>=k0_tcp1_t73[11]) return 0.0;
    int i1=0;
    int i2=11;
    bool done=false;
    while (done==false) {
      if (i2-i1>1) {
	int i=(i1+i2)/2;
	if (k0_tcp1_t73[i]>kf) {
	  i2=i;
	} else {
	  i1=i;
	}
      } else {
	done=true;
      }
    }
    double delk=k0_tcp1_t73[i2]-k0_tcp1_t73[i1];
    double a=(k0_tcp1_t73[i2]-kf)/delk;
    double b=(kf-k0_tcp1_t73[i1])/delk;
    double t=a*t0_tcp1_t73[i1]+b*t0_tcp1_t73[i2]+
      ((a*a*a-a)*t2_tcp1_t73[i1]+(b*b*b-b)*t2_tcp1_t73[i2])*
      delk*delk/6.0;
    return t;
  }

  /*
      function tcn1_ccdk(k)

      data k0/0.10d0,0.20d0,0.30d0,0.40d0,0.50d0,
     1        0.60d0,0.70d0,0.80d0,0.90d0,1.00d0,
     2        1.10d0/

      data d0/0.00d0,0.02d0,0.14d0,0.36d0,0.60d0,
     1        0.83d0,0.86d0,0.67d0,0.35d0,0.07d0,
     2        0.00d0/

      function tcn1_wap(k)

      data k0/0.10d0,0.20d0,0.30d0,0.40d0,0.50d0,
     1        0.60d0,0.70d0,0.80d0,0.90d0,1.00d0,
     2        1.10d0,1.20d0,1.30d0,1.40d0/

      data d0/0.00d0,0.03d0,0.13d0,0.30d0,0.54d0,
     1        0.74d0,0.86d0,0.90d0,0.79d0,0.59d0,
     2        0.35d0,0.14d0,0.03d0,0.00d0/

      function tcn1_GC(k)

      data k0/0.00,0.14,0.20,0.24,0.33,0.45,0.52,
     1        0.61,0.81,0.97,1.09,1.19,1.30,1.50/
      data t0/0.00e00,1.12e09,2.37e09,3.69e09,6.53e09,
     1        1.07e10,1.29e10,1.52e10,1.71e10,1.50e10,
     2        1.15e10,7.51e09,3.30e09,0.00e00/

      function tcn1_gipsf(k)

      data k0/0.000d0,
     1        0.200d0,0.300d0,0.400d0,0.600d0,0.700d0,
     1        0.800d0,1.000d0,1.200d0/
      data d0/0.000d0,
     1        0.300d0,0.900d0,1.500d0,2.100d0,1.900d0,
     1        1.500d0,0.500d0,0.000d0/

      function tcn1_t72(k)

      data k0/0.00,0.14,0.20,0.24,0.33,0.45,0.52,
     1        0.61,0.81,0.97,1.09,1.19,1.30,1.50/
      data t0/0.00e00,1.12e09,2.37e09,3.69e09,6.53e09,
     1        1.07e10,1.29e10,1.52e10,1.71e10,1.50e10,
     2        1.15e10,7.51e09,3.30e09,0.00e00/
      data t2/+1.24e+11,+9.50e+10,+3.61e+11,-1.20e+11,+9.13e+10,
     1        -6.95e+10,-6.40e+10,-1.10e+11,-1.34e+11,-1.08e+11,
     2        -1.20e+11,+1.15e+10,+1.54e+11,+1.71e+11/

      function tcn1_ns(k)

      data k0/0.30,0.40,0.50,0.60,0.65,
     1        0.70,0.75,0.80,0.85,0.90,0.95,
     2        1.00,1.05,1.10,1.15,1.20,1.25/
      data t0/0.00e00,1.38e09,3.16e09,5.27e09,6.59e09,
     1        8.50e09,1.07e10,1.42e10,1.61e10,1.74e10,1.81e10,
     2        1.78e10,1.52e10,1.22e10,8.31e09,2.31e09,0.00e00/
      data t2/   +4.48e+11,-6.74e+10,+6.18e+10,+1.83e+10,+4.03e+11,
     1 -2.14e+11,+1.15e+12,-1.26e+12,+3.50e+10,-3.25e+11,-1.77e+11,
     2 -1.37e+12,+1.31e+11,-1.17e+11,-1.80e+12,+2.25e+12,+1.65e+12/

      function tcn1_t84(k)

      data k0/0.00,0.14,0.20,0.24,0.33,0.45,0.52,
     1        0.61,0.81,0.97,1.09,1.19,1.30,1.50/
      data t0/0.00e00,1.12e09,2.44e09,3.76e09,6.72e09,
     1        1.11e10,1.32e10,1.57e10,1.75e10,1.54e10,
     2        1.18e10,7.25e09,2.97e09,0.00e00/
      data t2/+1.09e+11,+1.25e+11,+3.10e+11,-8.90e+10,+1.12e+11,
     1        -1.44e+11,+3.47e+10,-1.59e+11,-1.17e+11,-1.03e+11,
     2        -2.06e+11,+9.83e+10,+1.71e+11,+1.37e+11/

      function tcn1_ao(k)

      data k0/0.1,0.2,0.3,0.4,0.5,0.6,0.7,
     1        0.8,0.9,1.0,1.1,1.2,1.3,1.4,
     2        1.5,1.6,1.7,1.8,1.9,2.0,2.1/
      data t0/0.00e00,8.89e08,2.64e09,5.93e09,1.12e10,
     1        1.68e10,2.31e10,2.80e10,3.10e10,3.23e10,
     2        3.23e10,3.06e10,2.77e10,2.34e10,1.85e10,
     3        1.22e10,7.58e09,3.96e09,1.65e09,5.25e08,0.00e00/
      data t2/+3.09e+11,-2.43e+10,+1.86e+11,+2.66e+11,-5.97e+10,
     1        +1.71e+11,-2.06e+11,-1.87e+11,-1.84e+11,-9.61e+10,
     2        -2.12e+11,-7.73e+10,-1.99e+11,+3.43e+10,-2.98e+11,
     3   +3.18e+11,+3.58e+10,+1.39e+11,+1.94e+11,-8.32e+10,+2.59e+11/

      function tcn1_ccks_var(k)

      data k0/0.5,0.6,0.7,0.8,0.9,
     1        1.0,1.1,1.2,1.3,1.4/
      data t0/1.32e10,1.36e10,1.39e10,1.40e10,1.30e10,
     1        1.02e10,6.06e09,1.98e09,1.32e08,0.00e00/
      data t2/+1.48e+11,-5.52e+10,+1.30e+10,-1.17e+11,-2.05e+11,
     1        -1.43e+11,-2.83e+10,+2.92e+11,+1.99e+11,-6.01e+10/

      function tcn1_ccks_cbf(k)

      data k0/0.3,0.4,0.5,0.6,0.7,
     1        0.8,0.9,1.0,1.1/
      data t0/0.00,1.25e9,3.43e9,3.49e9,3.03e9,
     1        2.11e9,9.23e8,6.59e7,0.00e0/
      data t2/+2.97e+11,+1.57e+11,-3.65e+11,+3.05e+10,-6.93e+10,
     1        -2.93e+10,+2.62e+10,+1.23e+11,-4.15e+10/

      function tcn1_awp_2(k)

      data k0/0.1,0.2,0.3,0.4,0.5,0.6,
     1        0.7,0.8,0.9,1.0,1.1,1.2,
     2        1.3,1.4,1.5,1.6,1.7/
      data t0/0.00,3.30e8,1.18e9,2.44e9,4.20e9,6.13e9,
     1        7.91e9,9.10e9,9.56e9,9.03e9,7.71e9,5.93e9,
     2        4.15e9,2.50e9,1.12e9,3.61e8,0.00e0/
      data t2/   +7.58e+10,+4.64e+10,+5.06e+10,-2.87e+09,+1.17e+11,
     1 -7.45e+10,-5.28e+10,-6.83e+10,-1.12e+11,-7.75e+10,-5.19e+10,
     2 +9.16e+09,+1.53e+10,+7.71e+09,+1.16e+11,-3.87e+10,+1.58e+11/

      function tcn1_awp_3(k)

      data k0/0.1,0.2,0.3,0.4,0.5,
     1        0.6,0.7,0.8,0.9,1.0,
     2        1.1,1.2,1.3,1.4,1.5/
      data t0/0.00,2.64e8,7.91e8,1.78e9,3.36e9,
     1        5.27e9,6.59e9,7.25e9,7.05e9,5.74e9,
     2        3.96e9,1.98e9,7.91e8,1.32e8,0.00e0/
      data t2/+7.54e+10,+7.60e+09,+5.20e+10,+6.16e+10,+5.62e+10,
     1        -8.83e+10,-5.71e+10,-7.94e+10,-1.41e+11,-2.12e+10,
     2        -5.59e+10,+1.25e+11,+3.07e+10,+7.03e+10,+4.47e+09/

      function tcn1_sclbl96(k)
      data k0/+0.00E+00,+2.50E-01,+4.00E-01,+6.00E-01,
     1        +7.00E-01,+8.20E-01,+9.00E-01,+1.00E+00,
     2        +1.10E+00,+1.20E+00,+1.30E+00,+1.50E+00/
     1        
      data t0/+0.00E+00,+4.17E+09,+9.06E+09,+1.59E+10,
     1        +1.81E+10,+1.90E+10,+1.86E+10,+1.66E+10,
     2        +1.34E+10,+9.19E+09,+4.50E+09,+0.00E+00/
      data t2/+1.70E+11,+6.08E+10,+3.14E+10,-1.13E+11,
     1        -1.21E+11,-1.40E+11,-1.73E+11,-1.08E+11,
     2        -1.10E+11,-8.58E+10,+1.76E+11,+2.49E+11/

      function tcn1_sclbl96_pol(k)
      data k0/+3.00E-01,+5.00E-01,+6.00E-01,+7.00E-01,+9.00E-01,
     1        +1.10E+00,+1.20E+00,+1.25E+00,+1.30E+00,+1.40E+00,
     2        +1.50E+00,+1.55E+00,+1.60E+00,+1.70E+00/
      data t0/+0.00E+00,+3.31E+08,+5.95E+08,+1.46E+09,+3.57E+09,
     1        +5.82E+09,+6.75E+09,+6.81E+09,+6.75E+09,+5.82E+09,
     2        +3.77E+09,+1.92E+09,+7.94E+08,+0.00E+00/
      data t2/+3.31E+10,-1.65E+10,+9.25E+10,+3.57E+09,+2.56E+09,
     1        +6.03E+09,-1.60E+11,-2.24E+09,-1.48E+11,-3.07E+10,
     2        -4.04E+11,+4.99E+11,+1.54E+11,+1.61E+11/

      function tcn3_hgrr(k)
      data k0/1.0,1.2,1.4,1.6,1.8,1.9,2.0,
     1        2.1,2.2,2.3,2.4,2.5,2.6,2.7,
     2        2.8,2.9,3.0,3.2,3.4,3.6,3.8/
      data t0/0.00,1.75e8,8.12e8,2.20e9,3.83e9,4.58e9,5.22e9,
     1        5.74e9,6.09e9,6.18e9,6.06e9,5.80e9,5.41e9,4.97e9,
     2        4.41e9,3.83e9,3.19e9,1.97e9,8.12e8,1.48e8,0.00e0/
      data t2/+8.65e09,+8.95e09,+2.48e10,+4.34e09,-5.91e09,
     1        -1.22e10,-1.11e10,-1.52e10,-2.99e10,-2.12e10,
     2        -1.14e10,-1.71e10,+1.67e09,-1.96e10,+4.83e09,
     3        -1.17e10,+5.98e09,-3.09e09,+1.57e10,+1.45e10,+3.87e09/

      function tcn3_ao(k)
      data k0/1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,
     1        2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,
     2        3.1,3.2,3.3,3.4,3.5,3.6/
      data t0/0.00,3.50e7,1.25e8,2.42e8,3.67e8,
     1        4.85e8,5.95e8,7.06e8,8.01e8,8.72e8,
     2        8.80e8,8.31e8,6.79e8,5.12e8,3.95e8,
     3        3.12e8,2.51e8,1.99e8,1.65e8,1.37e8,
     4        1.12e8,8.78e7,6.45e7,4.15e7,1.88e7,0.00e0/
      data t2/+7.63e+09,+5.74e+09,+2.41e+09,+8.37e+08,-9.52e+08,
     1        -1.23e+09,+1.07e+09,-2.46e+09,-8.48e+08,-8.55e+09,
     2        -2.74e+09,-1.47e+10,-3.01e+08,+6.89e+09,+2.72e+09,
     3        +2.61e+09,+3.23e+07,+2.66e+09,+1.29e+08,+4.24e+08,
     4        -2.62e+07,+1.60e+08,-7.58e+07,+3.23e+08,-1.04e+09,
     5        +6.16e+09/

      function tcn3_t72(k)
      data k0/1.38,1.42,1.46,1.53,1.64,1.73,
     1        1.82,1.89,1.96,1.98,2.02,2.06/
      data t0/0.00,1.00e7,1.00e8,3.02e8,6.02e8,7.76e8,
     1        6.02e8,3.49e8,1.52e8,1.00e8,1.00e7,0.00e0/
      data t2/-2.08e+10,+7.91e+10,+4.34e+09,-4.36e+09,
     1        +2.86e+09,-6.03e+10,-1.94e+10,+2.21e+10,
     2        -3.03e+08,-1.02e+10,+8.33e+10,-2.29e+10/

      function tcn3_t72_m1(k)
      data k0/1.20,1.39,1.56,1.70,1.84,
     1        1.97,2.09,2.20,2.31,2.41,
     2        2.51,2.60,2.69,2.78,2.87,
     3        2.95,3.03,3.11,3.19,3.26,
     4        3.34,3.41,3.48,3.55,3.61/
      data t0/0.00,1.66e8,4.85e8,1.05e9,1.73e9,
     1        2.27e9,2.70e9,3.02e9,3.20e9,3.20e9,
     2        3.10e9,2.96e9,2.79e9,2.60e9,2.38e9,
     3        2.15e9,1.91e9,1.65e9,1.38e9,1.14e9,
     4        8.70e8,6.23e8,3.74e8,1.38e8,0.00e0/
      data t2/+1.37e+10,+1.32e+08,+1.95e+10,+6.07e+09,-8.58e+09,
     1        -3.38e+09,-5.16e+09,-1.15e+10,-1.82e+10,-9.07e+09,
     2        -5.51e+09,-3.69e+09,-1.94e+09,-3.36e+09,-6.83e+09,
     3        +5.15e+08,-4.60e+09,-8.46e+08,-1.39e+09,+2.31e+09,
     4        -3.45e+09,-1.04e+09,+5.14e+09,-3.61e+09,+1.17e+11/

      function tcn3_ao_m1(k)
      data k0/1.35,1.4,1.5,1.6,1.7,1.8,
     1         1.9,2.0,2.1,2.2,2.3,2.4,
     2         2.5,2.6,2.7,2.8,2.9,3.0,
     3         3.1,3.2,3.3,3.4,3.5/
      data t0/1.00,6.92e7,2.77e8,5.26e8,8.45e8,1.14e9,
     1        1.51e9,1.87e9,2.16e9,2.34e9,2.44e9,2.47e9,
     2        2.40e9,2.23e9,1.95e9,1.56e9,1.16e9,8.10e8,
     3        5.26e8,3.05e8,1.52e8,5.54e7,1.00e0/

      data t2/+8.36e+10,-1.20e+09,+3.42e+09,+1.22e+10,-1.03e+10,
     1        +1.47e+10,-3.60e+09,-6.32e+09,-1.31e+10,-7.25e+09,
     2        -5.91e+09,-1.11e+10,-9.56e+09,-1.06e+10,-1.40e+10,
     3        +6.01e+08,+5.59e+09,+7.02e+09,+5.93e+09,+7.07e+09,
     4        +6.58e+09,+4.33e+08,+1.64e+10/

      function tcn3_bcll92(k)
      data k0/+1.50E00,+1.80E00,+2.00E00,+2.25E00,+2.50E00,+2.60E00,
     1        +2.70E00,+2.80E00,+2.90E00,+3.00E00,+3.50E00,+4.50E00/
      data t0/+0.00E00,+6.89E08,+1.72E09,+4.48E09,+7.64E09,+9.16E09,
     1        +1.03E10,+1.07E10,+1.03E10,+8.95E09,+4.13E09,+0.00E00/
      data t2/+2.17E10,+2.56E09,+4.08E10,-8.32E09,+3.22E10,-5.57E10,
     1        -5.72E10,-8.74E10,-1.30E11,+3.06E10,+2.14E09,+1.13E10/

   */
  
};

#endif
