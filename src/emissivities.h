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
#ifndef NSCOOL_EMISSIVITIES_H
#define NSCOOL_EMISSIVITIES_H

#include <boost/numeric/ublas/vector.hpp>

#include <o2scl/constants.h>
#include <o2scl/interp2_direct.h>
#include <o2scl/fermion.h>

typedef boost::numeric::ublas::vector<double> ubvector;
typedef boost::numeric::ublas::matrix<double> ubmatrix;

/** \brief Neutrino emissivities

    This work in progress will eventually replace Dany's Fortran code.
*/
class emissivities {
  
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
  
  emissivities() {

    lgtau1.resize(35);
    lgtau2.resize(35);
    lgr.resize(35,35);
#include "sf_suppression2.h"
    id.set_data(35,35,lgtau1,lgtau2,lgr);

    return;
  }

  /// \name Generic functions
  //@{
  /** \brief Desc
   */
  double fexp(double x) {
    if (x>-7.0e2) return exp(x);
    return exp(-700.0);
  }

  /** \brief Desc
   */
  double u_1s0(double t) {
    return sqrt(1.0-t)*(1.4560-0.1570/sqrt(t)+1.7640/t);
  }
      
  /** \brief Desc
   */
  double u_3p2B(double t) {
    return sqrt(1.0-t)*(0.78930+1.1880/t);
  }
  
  /** \brief Desc
   */
  double r_1s0(double u) {
    return pow(0.23120+hypot(0.7688,0.1438*u),5.50)*
      fexp(3.4270-hypot(3.4270,u));
  }
  
  /** \brief Desc
   */
  double r_3p2B(double u) {
    return pow(0.25460+hypot(0.7454,0.1284*u),5.0)*
      fexp(2.7010-hypot(2.7010,u));
  }
  //@}

  /// \name Modified Urca
  //@{
  /** \brief Desc
   */
  double rmurca_n_p1s0(double u) {
    return fexp(3.437-hypot(3.437,u))*0.5*
      (pow(0.1477+hypot(0.8523,0.1175*u),7.5)+
       pow(0.1447+hypot(0.8523,0.1297*u),5.5));
  }

  /** \brief Desc
   */
  double rmurca_n_n1s0(double u) {
    return fexp(5.339-hypot(5.339,2.0*u))*
      pow(0.2414+hypot(0.7586,0.1318),7.0);
  }

  /** \brief Desc
   */
  double rmurca_p_p1s0(double u) {
    return fexp(5.339-hypot(5.339,2.0*u))*
      pow(0.2414+hypot(0.7586,0.1318*u),7.0);
  }
  
  /** \brief Desc
   */
  double rmurca_p_n1s0(double u) {
    return fexp(3.437-hypot(3.437,u))*0.5*
      (pow(0.1477+hypot(0.8523,0.1175*u),7.5)+
       pow(0.1477+hypot(0.8523,0.1297*u),5.5));
  }

  /** \brief Desc
   */
  double rmurca_p_n3p2B(double u) {
    return fexp(2.398-hypot(2.398,u))*0.5*
      (pow(0.1612+hypot(0.8388,0.1117*u),7.0)+
       pow(0.1612+hypot(0.8388,0.1274*u),5.0));
  }

  /** \brief Desc
   */
  double rmurca_n_n3p2B(double u, double t) {
    return 39.10*t*fexp(-1.1880/t)*rmurca_p_n3p2B(u);
  }

  /** \brief \f$ n+x \rightarrow p+x+{\ell}+{\bar{\nu_{\ell}}} \f$
      modified Urca emissivity in \f$
      \mathrm{erg}/\mathrm{cm}^3/\mathrm{s} \f$
  */
  double emissivity_murca_nxpxl(double kfn, double kfp, double kfe, 
				double kfmu, double rmn, double rmp,
				double TK, double Tcn1s0, double Tcp1s0,
				double Tcn3p2) {

    double alpha_n=1.76-0.63*pow(1.68/kfn,2.0);
    double beta_n=0.68;
    double emis_n=8.55e21*pow(rmn,3.0)*rmp*(kfe/1.68+kfmu/1.68)*
      alpha_n*beta_n*pow(TK/1.0e9,8.0);

    double alpha_p=1.76-0.63*pow(1.68/kfn,2.0);
    double beta_p=0.68;
    double emis_p=8.55e21*rmn*pow(rmp,3.0)*(kfe/1.68)*
      pow(kfe+3.0*kfp-kfn,2.0)/(8.0*kfe*kfp)*
      alpha_p*beta_p*pow(TK/1.0e9,8.0);

    double corr_n_n=1.0;
    double corr_p_n=1.0;
    double corr_n_p=1.0;
    double corr_p_p=1.0;
    
    // Neutron pairing suppression
    if (TK<Tcn1s0 || TK<Tcn3p2) {
      double tn, un;
      if (Tcn1s0>Tcn3p2) {
	tn=TK/Tcn1s0;
	un=u_1s0(tn);
	corr_n_n=rmurca_n_n1s0(un);
	corr_p_n=rmurca_p_n1s0(un);
      } else {
	tn=TK/Tcn3p2;
	un=u_3p2B(tn);
	corr_n_n=rmurca_n_n3p2B(un,tn);
	corr_p_n=rmurca_p_n3p2B(un);
      }
    }

    // Proton pairing suppression
    if (TK<Tcp1s0) {
      double tp=TK/Tcp1s0;
      double up=u_1s0(tp);
      corr_n_p=rmurca_n_p1s0(up);
      corr_p_p=rmurca_p_p1s0(up);
    }

    emis_n*=std::min(corr_n_n,corr_n_p);
    emis_p*=std::min(corr_p_n,corr_p_p);
    
    return emis_n+emis_p;
  }
  //@}

  /// \name Bremsstrahlung emissivity functions
  //@{
  /** \brief Desc
   */
  double rbrem_nn_p1s0(double u) {
    return 1.0;
  }

  /** \brief Desc
   */
  double rbrem_nn_n1s0(double u) {
    return pow(0.1747+hypot(0.8253,0.07933*u),2.0)*
      fexp(4.228-hypot(4.228,4.0*u))/2.0+
      pow(0.7333+hypot(0.2667,0.1678*u),7.5)*
      fexp(7.762-hypot(7.762,9.0*u))/2.0;
  }
  
  /** \brief Desc
   */
  double rbrem_np_p1s0(double u) {
    return (0.9982+hypot(0.0018,0.3815*u))*
      fexp(1.306-hypot(1.3060,u))/2.732+
      pow(0.3949+hypot(0.6051,0.2666*u),7.0)*
      fexp(3.303-hypot(3.303,4.0*u))/1.577;
  }

  /** \brief Desc
   */
  double rbrem_np_n1s0(double u) {
    return rbrem_np_p1s0(u);
  }

  /** \brief Desc
   */
  double rbrem_pp_p1s0(double u) {
    return rbrem_nn_n1s0(u);
  }

  /** \brief Desc
   */
  double rbrem_pp_n1s0(double u) {
    return 1.0;
  }

  /** \brief Desc
   */
  double rbrem_nn_n3p2B(double u) {
    return rbrem_nn_n1s0(u);
  }

  /** \brief Desc
   */
  double rbrem_np_n3p2B(double u) {
    return rbrem_np_n1s0(u);
  }

  /** \brief Desc
   */
  double rbrem_pp_n3p2B(double u) {
    return 1.0;
  }

  /** \brief \f$ n+n \rightarrow n+n+{\nu}+{\bar{\nu}} \f$ 
      bremsstrahlung emissivity in
      \f$ \mathrm{erg}/\mathrm{cm}^3/\mathrm{s} \f$
   */
  double emissivity_brem_nn(double kfn, double kfp, double kfe, 
				double kfmu, double rmn, double rmp,
				double TK, double Tcn1s0, double Tcp1s0,
				double Tcn3p2) {
    double n_nu=3.0;
    double alpha_nn=0.59;
    double beta_nn=0.56;
    double alpha_np=1.06;
    double beta_np=0.66;
    double alpha_pp=0.11;
    double beta_pp=0.7;
    double emis_nn=n_nu*7.4e19*pow(rmn,4.0)*(kfn/1.68)*alpha_nn*
      beta_nn*pow(TK/1.0e9,8.0);
    double emis_np=n_nu*1.5e20*pow(rmn,2.0)*pow(rmp,2.0)*(kfp/1.68)*alpha_np*
      beta_np*pow(TK/1.0e9,8.0);
    double emis_pp=n_nu*7.4e19*pow(rmp,4.0)*(kfp/1.68)*alpha_pp*
      beta_pp*pow(TK/1.0e9,8.0);

    double corr_nn_n=1.0;
    double corr_np_n=1.0;
    double corr_pp_n=1.0;
    double corr_nn_p=1.0;
    double corr_np_p=1.0;
    double corr_pp_p=1.0;
    
    if (TK<Tcn1s0 || TK<Tcn3p2) {
      if (Tcn1s0>Tcn3p2) {
	double tn=TK/Tcn1s0;
	double un=u_1s0(tn);
	corr_nn_n=rbrem_nn_n1s0(un);
	corr_np_n=rbrem_np_n1s0(un);
	corr_pp_n=rbrem_pp_n1s0(un);
      } else {
	double tn=TK/Tcn3p2;
	double un=u_3p2B(tn);
	corr_nn_n=rbrem_nn_n3p2B(un);
	corr_np_n=rbrem_np_n3p2B(un);
	corr_pp_n=rbrem_pp_n3p2B(un);
      }
    }

    if (TK<Tcp1s0) {
      double tp=TK/Tcp1s0;
      double up=u_1s0(tp);
      corr_nn_p=rbrem_nn_p1s0(up);
      corr_np_p=rbrem_np_p1s0(up);
      corr_pp_p=rbrem_pp_p1s0(up);
    }

    emis_nn*=std::min(corr_nn_p,corr_nn_n);
    emis_np*=std::min(corr_np_p,corr_np_n);
    emis_pp*=std::min(corr_pp_p,corr_pp_n);
    
    return emis_nn+emis_np+emis_pp;
  }
  //@}

  /// \name Direct Urca functions
  //@{
  /** \brief Suppression factor 
   */
  double r_1s0_1s0(double v1, double v2) {
    double gamma=5040.0/457.0/pow(o2scl_const::pi,6.0);
    double u=v1*v1+v2*v2;
    double w=v1*v1-v2*v2;
    double u1=1.8091+hypot(v1,2.2476);
    double u2=1.8091+hypot(v2,2.2476);
    double p=(u+12.421+sqrt(w*w+16.35*u+45.171))/2.0;
    double q=(u+12.421-sqrt(w*w+16.35*u+45.171))/2.0;
    double ps=(u+sqrt(w*w+5524.80*u+6.77370))/2.0;
    double pe=(u+0.43847+sqrt(w*w+8.368*u+491.32))/2.0;
    double D=pow(u1*u2,1.5)/(2.0*pow(4.0567,5.0))*(u1*u1+u2*u2)*
      exp(-u1-u2+8.1134);
    double K0=sqrt(p-q)/120.0*(6.0*p*p+83.0*p*q+16.0*q*q)-
      sqrt(p)*q/8.0*(4.0*p+3.0*q)*log((sqrt(p)+sqrt(p-q))/sqrt(q));
    double K1=o2scl_const::pi2*sqrt(p-q)/6.0*(p+2.0*q)-
      o2scl_const::pi2/2.0*q*sqrt(p)*log((sqrt(p)+sqrt(p-q))/sqrt(q));
    double K2=7.0*o2scl_const::pi2*o2scl_const::pi2/60.0*sqrt(p-q);
    double S=gamma*(K0+K1+0.42232*K2)*sqrt(o2scl_const::pi/2.0)*
      pow(ps,0.25)*exp(-sqrt(pe));
    
    return u/(u+0.91630)*S+D;
  }

  /** \brief Suppression factor 
   */
  double r_1s0_3p2B(double t1, double t2) {
    double lt1=log10(t1);
    double lt2=log10(t2);
    double ret=10.0*id.eval(lt1,lt2);
    double lt=hypot(lt1,lt2);
    double lt_limit=3.0;
    if (lt>lt_limit) {
      ret*=exp(-lt/lt_limit);
    }
    return ret;
  }

  /** \brief \f$ n \rightarrow p+ \ell + {\bar{\nu_{\ell}}} \f$ Urca
      emissivity in \f$ \mathrm{erg}/\mathrm{cm}^3/\mathrm{s} \f$
   */
  double emissivity_durca_npl(double kfn, double kfp,
			     double kfe, double ne,
			     double kfmu, double nmu,
			     double rmn, double rmp, double TK, double Tcn1s0,
			     double Tcp1s0, double Tcn3p2) {

    double s_e=kfn+kfp+kfe;
    double heron_e=s_e*(s_e-kfn)*(s_e-kfp)*(s_e-kfe);
    double fact_e=1.0;
    if (heron_e<=0.0) fact_e=0.0;
    double s_mu=kfn+kfp+kfmu;
    double heron_mu=s_mu*(s_mu-kfn)*(s_mu-kfp)*(s_mu-kfmu);
    double fact_mu=1.0;
    if (heron_mu<=0.0) fact_mu=0.0;
    
    double emis=4.24e27*rmn*rmp*pow(TK,6.0)*
      (fact_e*cbrt(ne/0.16)+fact_mu*cbrt(nmu/0.16));

    // Pairing suppression
    if (TK<Tcn1s0 || TK<Tcn3p2) {
      if (TK<Tcp1s0) {
	double tp=TK/Tcp1s0;
	double up=u_1s0(tp);
	if (Tcn3p2>Tcn1s0) {
	  double tn=TK/Tcn3p2;
	  emis*=r_1s0_3p2B(tp,tn);
	} else {
	  double tn=TK/Tcn1s0;
	  double un=u_1s0(tn);
	  emis*=r_1s0_1s0(up,un);
	}
      } else {
	if (Tcn3p2>Tcn1s0) {
	  double tn=TK/Tcn3p2;
	  double u=u_3p2B(tn);
	  emis*=r_3p2B(u);
	} else {
	  double tn=TK/Tcn1s0;
	  double u=u_1s0(tn);
	  emis*=r_1s0(u);
	}
      }
    } else if (TK<Tcp1s0) {
      double tp=TK/Tcp1s0;
      double u=u_1s0(tp);
      emis*=r_1s0(u);
    }
    
    return emis;
  }
  //@}

  /// \name PBF emissivities
  //@{
  /** \brief Desc
   */
  double control_pbf_1s0(double v) {
    double x=0.602*v*v+0.5942*pow(v,4.0)+0.288*pow(v,6.0);
    double y=sqrt(0.5547+sqrt(0.4453*0.4453+0.0113*v*v));
    double z=exp(-sqrt(4.0*v*v+2.245*2.245)+2.245);
    return x*y*z;
  }

  /** \brief Desc
   */
  double control_pbf_3p2B(double v) {
    double x=(1.204*v*v+3.733*pow(v,4.0)+0.3191*pow(v,6.0))/
      (1.0+0.3511*v*v);
    double y=pow(0.7591+sqrt(0.2409*0.2409+0.3145*v*v),2.0);
    double z=exp(-sqrt(4.0*v*v+0.4616*0.4616)+0.4616);
    return x*y*z;
  }
  
  /** \brief Desc
   */
  double emissivity_1s0_pbf(double kf, double rm, double TK,
			    double Tc1s0) {
    double emis=0.0;
    if (TK<Tc1s0) {
      double pf=kf*197.0;
      double vf=pf/rm/940.0;
      double a_a=1.6*vf*vf*(rm*rm+11.0/42.0);
      double tau=TK/Tc1s0;
      double u=sqrt(1.0-tau)*(1.456-0.157/sqrt(tau)+1.765/tau);
      emis=1.17e21*rm*rm*vf*pow(TK/1.0e9,7.0)*3.0*a_a*
	control_pbf_1s0(u);
    }
    return emis;
  }

  /** \brief Desc
   */
  double emissivity_3p2B_pbf(double kf, double rm, double TK,
			     double Tc3p2) {
    double emis=0.0;
    if (TK<Tc3p2) {
      double pf=kf*197.0;
      double vf=pf/rm/940.0;
      double g_A=1.26;
      double a_a=2.0*g_A*g_A;
      double tau=TK/Tc3p2;
      double u=sqrt(1.0-tau)*(0.7893+1.764/tau);
      emis=1.17e21*rm*rm*vf*pow(TK/1.0e9,7.0)*3.0*a_a*
	control_pbf_3p2B(u);
    }
    return emis;
  }
  //@}

  /** \brief the energy loss rate per cubic centimeter in the crust
      of a neutron star from neutrino pair bremsstrahlung.          
      
      From Kaminker et al, A&A 343 (1999), p. 1009, Equ. (40) 
  */
  void npb_new(double temp, double rho, double &qnpb) {
    double tau=log10(temp/1.0e8);
    double r=log10(rho/1.0e12);
    double rho0=2.8e14;
    double lgq=11.204+7.304*tau+0.2976*r-0.370*tau*tau+0.188*tau*r-
      0.103*r*r+0.0547*tau*tau*r-6.77*log10(1.0+0.228*rho/rho0);
    qnpb=10.0*lgq;
    return;
  }

  /** \brief the energy loss rate per cubic centimeter in the crust
      of a neutron star from neutrino pair bremsstrahlung.         
      
      checked on February 27, 1996 against figures of Itoh et al 1996
  */
  void npb(double t, double rho, double a, double z, double qnpb) {
    return;
  }
  
  /** \brief The energy loss rate per cubic centimeter in the crust
      of a neutron star from plasma neutrinos.
      
      from h.munakata, y.kohyama & n.itoh, ap.j.296(1985),p.197      
   */
  void nplasma_old(double t, double rho, double a, double z,
		   double &qplasma) {
    static const double apl[3]={2.320e-7,8.449e-8,1.787e-8};
    static const double bpl[3]={2.581e-2,1.734e-2,6.990e-4};
    static const double cpl=0.56457;
    if (z==0.0) {
      qplasma=0.0;
      return;
    }
    double l=t/5.93e9;
    double xi=cbrt(rho*z/a*1.0e-9)/l;
    int n=2;
    double fplasma=(apl[0]+apl[1]*xi+apl[2]*xi*xi)*fexp(-cpl*xi)/
      (xi*xi*xi+bpl[0]/l+bpl[1]/l/l+bpl[2]/l/l/l);
    qplasma=(0.872+n*0.004)*pow(rho*z/a,3.0)*fplasma;
    return;
  }
  
  /** \brief The energy loss rate per cubic centimeter in     
       bubble phase of the crust.                          
       
       from L. Leinson, ApJ 415, p. 759, 1993
  */
  void nbub(int i, double t, double rho, double a, double z,
	      double rhocore, double *tcn, int isf, double &qbubble) {
    double rhomin=1.0e14;
    if (rho<rhocore && rho>=rhomin) {
      qbubble=1.1e22*pow(t/1.0e9,6.0);
    } else {
      qbubble=0.0;
    }
    double r;
    if (t<tcn[i]) {
      if (i>=isf) {
	double u=sqrt(1.0-(t/tcn[i]))*(1.456-0.157/sqrt(t/tcn[i])+
				       1.764/t*tcn[i]);
	r=(0.2312+sqrt(0.7688*0.7688+pow(0.1438*u*0.1438*u,5.5))*
	   fexp(3.427-sqrt(3.427*3.427*u*u)));
      } else {
	double u=sqrt(1.0-t/tcn[i])*(5.596+9.424/t*tcn[i]);
	r=(0.2546+pow(0.7454*0.7454+0.01811*u*0.01811*u,5.0/2.0))*
	  fexp(2.701-sqrt(2.701*2.701+u*u/16.0/o2scl_const::pi));
      } 
    } else {
      r=1.0;
    }
    qbubble*=r;
    return;
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
  
};

#endif
