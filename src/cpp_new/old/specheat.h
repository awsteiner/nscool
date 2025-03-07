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
#ifndef NSCOOL_SPECHEAT_H
#define NSCOOL_SPECHEAT_H

#include <boost/numeric/ublas/vector.hpp>

#include <o2scl/constants.h>
#include <o2scl/interp2_direct.h>
#include <o2scl/fermion.h>

typedef boost::numeric::ublas::vector<double> ubvector;
typedef boost::numeric::ublas::matrix<double> ubmatrix;

/** \brief Neutrino emissivities

    This work in progress will eventually replace Dany's Fortran code.
*/

class specheat {

  private:

    static constexpr double pi = 3.14159265;
    static constexpr double rhodrip = 4.3e11;
    static constexpr double na = 6.022045e23;
    static constexpr double hb = 1.054588e-27;
    static constexpr double c = 2.997924e10;
    static constexpr double me=9.109e-28;
    static constexpr double kb=1.380662e-16;

    double cd[4][4] = {{2.315472, 7.128660, 7.504998, 2.665350},
                  {7.837752, 23.507934, 23.311317, 7.987465},
                  {9.215560, 26.834068, 25.082745, 8.020509},
                  {3.693280, 10.333176, 9.168960, 2.668248}};

    double cp[4][4] = {{2.315472, 6.748104, 6.564912, 2.132280},
                      {7.837752, 21.439740, 19.080088, 5.478100},
                      {9.215560, 23.551504, 19.015888, 4.679944},
                      {3.693280, 8.859868, 6.500712, 1.334124}};

    double cu[4][4] = {{3.473208, 10.122156, 9.847368, 3.198420},
                      {16.121172, 43.477194, 37.852852, 10.496830},
                      {23.971040, 60.392810, 47.782844, 11.361074},
                      {11.079840, 26.579604, 19.502136, 4.002372}};

  public:

    double c_cv_str;

    specheat() {
      c_cv_str = 0.0;
    }

    /** \brief Specific Heat of ions
     */
    void cvion(double t, double rho, double a, double z, double& cv) {
      double gamma = 2.273e5*z*z*std::pow(rho/a,1./3.)/t;
      double a1 = (rho>=rhodrip) ? 3.*z : a;
      double nionkb = 1.38e-16*6.022e23*rho/a;
      double delta = 1./t*z*std::sqrt(rho/(a1*a))*6.022e23;

      double bcv = 0.95043;
      double ccv = 0.18956;
      double dcv = -0.81487;
      double hcv = 3225;
      double cte = 1.417e2;

      double cv0[15] = {0.0, 2.956, 2.829, 2.633, 2.389, 2.118, 1.840, 1.572,
                        1.323, 1.102, 0.909, 0.745, 0.609, 0.496, 0.404};

      if (gamma <= 0.1) {
        cv = 1.5*nionkb;
      } else if (gamma <= 0.2) {
        double cv1 = 1.5*nionkb;
        double cv2 = nionkb*(0.75*std::pow(gamma,0.25)*bcv+1.25*ccv/std::pow(gamma,0.25)+dcv+1.5);
        cv = (gamma-0.1)/0.1*cv2+(0.2-gamma)/0.1*cv1;
      } else if (gamma <= 178.) {
        cv = nionkb*(0.75*std::pow(gamma,0.25)*bcv+1.25*ccv/std::pow(gamma,0.25)+dcv+1.5);
      } else if (gamma <= 210. && delta >= 1.e19) {
        double cv1 = nionkb*(1.5+3.*hcv/std::pow(gamma,2)+1.5);
        cv0[0] = 1.5+3.*hcv/std::pow(gamma,2)+1.5;
        int i1 = static_cast<int>(delta*2.e-20);
        double cv2 = nionkb*(cv0[i1]+(delta*2.e-20-i1)*(cv0[i1+1]-cv0[i1]));
        cv = (gamma-178.)/32.*cv2+(210.-gamma)/32.*cv1;
      } else if (delta <= 1.e19) {
        cv = nionkb*(1.5+3.*hcv/std::pow(gamma,2)+1.5);
      } else if (delta > 1.e19 && delta < 7.e20) {
        cv0[0] = 1.5+3.* hcv/std::pow(gamma,2)+1.5;
        int i1 = static_cast<int>(delta * 2.e-20);
        cv = nionkb*(cv0[i1]+(delta*2.e-20-i1)*(cv0[i1+1]-cv0[i1]));
      } else {
        double delta1 = delta*1.e-20;
        cv = nionkb*cte/std::pow(delta1,3);
      }
      return;
    }

    /** \brief Sub-function of Specific Heat of electrons
     */
    double sub_cvelec(double f1, double t1, double nehat){
      double f = std::abs(f1);
      double g = t1*std::sqrt(1.0+f);

      double sum1=0.0; double sum2=0.0; double sum3=0.0;
      for (int j1=0;j1<=3;j1++) {
        for (int j2=0;j2<=3;j2++) {
          sum1+=cd[j1][j2]*pow(f, j1)*pow(g,j2); } }

      for (int j1=1;j1<=3;j1++) {
        for (int j2=0;j2<=3;j2++) {
          sum2+=j1*cd[j1][j2]*pow(f,j1-1)*pow(g,j2); } }

      for (int j1=0;j1<=3;j1++) {
        for (int j2=1;j2<=3;j2++) {
          sum3+=j2*cd[j1][j2]*pow(f,j1)*pow(g,j2); } }

      double pf_value = 1.0+f;
      double pg_value = 1.0+g;
      double nef = 0.0, nef1 = 0.0;

      if (f<1.0) {
        nef = f*pow(g,1.5)*pow(pf_value,-4)*pow(pg_value,-1.5)*sum1;
        nef1 = (pow(g,1.5)*pow(pf_value,-4)/pow(pg_value,1.5)-
                3.25*f*pow(g,1.5)*pow(pf_value,-5)*pow(pg_value,-1.5)-
                0.75*f*pow(g,2.5)*pow(pf_value,-5)*pow(pg_value,-2.5))*sum1+
                f*pow(g,1.5)*pow(pf_value,-4)*pow(pg_value,-1.5)*(sum2+0.5/pf_value*sum3);
      } else {
        nef = pow(f,0.25)/pf_value*pow(g/pg_value,1.5)*sum1;
        nef1 = pow(f,0.25)/pf_value*pow(g/pg_value,1.5)*
              ((1.0/f-3.25/pf_value-0.75*g/(pf_value*pg_value))*sum1+sum2+0.5/pf_value*sum3);}

      double f1_value = std::abs(f+(nehat-nef)/nef1);
      while (std::abs(nef-nehat)/std::abs(nehat)>1.0e-10 ||
            std::abs(f1_value-f)/std::abs(f1_value)>1.0e-12) {
        f1 = f1_value;
        sub_cvelec(f1, t1, nehat);}
      return f1_value;
    }

    /** \brief Specific Heat of electrons
     */
    void cvelec(double t, double rho, double a, double z, int ikeep, double& cv) {
      double ne = na*rho*z/a;
      double pf = hb*std::pow(3.0*pi*pi*ne,1.0/3.0);
      double ef = std::sqrt(std::pow(me*c*c,2)+std::pow(pf*c,2))-me*c*c;
      double tf = ef/kb;
      double xe = pf/(me*c);
      double ae = xe*xe/std::sqrt(1.0+xe*xe);
      double cvt = ne*kb*kb*pi*pi/(me*c*c)/ae;

      double t0 = tf/50.0;

      if (t<=0.5*t0) {
        cv = cvt*t;
        return;}
      double nehat = ne/1.7595e30;

      // calculate f at .9999*t

      double t1 = 0.9999*t / 5.93e9;
      double f1 = 1.0;
      if (fkeep[ikeep]!=0.0) {
        f1 = fkeep[ikeep];}
      fkeep[ikeep]=sub_cvelec(f1, t1, nehat);

      // calculate f at 1.0001*t

      double t2 = 1.0001*t/5.93e9;
      double f2 = f1;
      f2=sub_cvelec(f2,t2,nehat);

      // Calculate cv
      
      double sumu1 = 0.0; double sumu2 = 0.0;
      double g2 = t2*std::sqrt(1.0+f2);
      double g1 = t1*std::sqrt(1.0+f1);
      for (int j1=0;j1<=3;j1++) {
        for (int j2=0;j2<=3;j2++) {
          sumu2+=cd[j1][j2]*pow(f2, j1)*pow(g2,j2);} }

      for (int j1=0;j1<=3;j1++) {
        for (int j2=0;j2<=3;j2++) {
          sumu1+=cd[j1][j2]*pow(f1, j1)*pow(g1,j2);} }

      double u1, u2;
      if (f1<1) {
        u1 = 1.44e24*f1*pow(g1,2.5)/pow(1.0+f1,4)/pow(1.0+g1,1.5)*sumu1;
        u2 = 1.44e24*f2 *pow(g2,2.5)/pow(1.0+f2,4)/pow(1.0+g2,1.5)*sumu2;
      } else {
        u1 = 1.44e24*pow(f1,0.25)/pow(1.0+f1,4)*g1*pow(g1/(1.0+g1),1.5)*sumu1;
        u2 = 1.44e24*pow(f2,0.25)/pow(1.0+f2,4)*g2*pow(g2/(1.0+g2),1.5)*sumu2;}

      cv = (u2-u1)/(0.0002*t);
      if (t<1.5*t0) {
        double w1 = (t-0.5*t0)/t0;
        double w2 = 1.0-w1;
        cv = w1*cv+w2*cvt*t;}
      return;
    }

    /** \brief Total specific Heat
     */
    void specheatt(int i, double t, double rho, double aion, double zion, double &cv,
              double &cvneutron, double &cvproton, double &cvelectron, double &cvmuon,
              double &cvlambda, double &cvsminus, double &cvszero, double &cvsplus,
              double &cvquark, double &cvions, std::vector<double> cve, std::vector<double> cvm, std::vector<double> cvn,
              std::vector<double> cvp, std::vector<double> cvla, std::vector<double> cvsm, std::vector<double> cvs0, std::vector<double> cvsp,
              std::vector<double> cvqu, std::vector<double> dvqd, std::vector<double> cvqs, double rhodrip, double rhocore,
              std::vector<double> fhad, int istrange, std::vector<double> tcn, std::vector<double> tcp, std::vector<double> tcla,
              std::vector<double> tcsm, std::vector<double> tcs0, std::vector<double> tcsp, int isf) {
      double raise, t0, u, r, w1, w2;
      int icvel_nodeg=1;

      auto fexp = [] (double x) { return exp(std::max(x, -7.0e2)); };
      auto u_1s0 = [] (double t) { return sqrt(1.0-t)*(1.456-0.157/sqrt(t)+1.764/t); };
      auto r_1s0 = [fexp] (double u) {
        return pow((0.4186+sqrt(1.007*1.007+(0.5010*u)*(0.5010*u))),2.5)*
              fexp(1.456-sqrt(1.456*1.456+u*u)); };
      auto u_3p2 = [] (double t) { return sqrt(1.0-t)*(5.596+8.424/t); };
      auto r_3p2 = [fexp] (double u) {
        return pow((0.6893+sqrt(0.790*0.790+(0.03983*u)*(0.03983*u))),2.0)*
              fexp(1.934-sqrt(1.934*1.934+u*u/(16.0*pi))); };

      auto solve_cv_mem = [rho,rhocore,istrange,u_1s0,r_1s0,i,t] 
                          (std::vector<double>tcmem, std::vector<double>cvmem) {
        double cv = 0.0;
        if ((rho>=rhocore) && (istrange==0)) {
          double raise=1.1;
          double r=1.0;
          if (t<raise*tcmem[i]) {
            double t0 = std::min(0.999999999999,t/tcmem[i]);
            double u = u_1s0(t0);
            double r = r_1s0(u);
            if (t > tcmem[i]) {
              double w1 = (raise*tcmem[i]-t)/((raise-1.0)*tcmem[i]);
              double w2 = 1.0-w1;
              r = w1*r+w2*1.0; 
            }
          }
          double cv = cvmem[i]*t*r;
        }
        return cv;
      };
        
      // Get Cv-ions
      if (rho<rhocore) {
        cvion(t,rho,aion,zion,cvions);
      } else { cvions = 0.0;}
      // Get Cv-electrons
      if ((rho<rhodrip) && (icvel_nodeg == 1)) {
        cvelec(t,rho,aion,zion,i,cvelectron);
      } else { cvelectron = cve[i]*t; }
      // Get Cv-muons
      cvmuon = cvm[i]*t;
      // get Cv-neutrons :
      if ((rho>=rhodrip) && (istrange==0)) {
        raise = 1.0;
        if (t<raise*tcn[i]) {
          t0 = std::min(0.999999999999, t/tcn[i]);
          if (i<=isf) {
            u = u_3p2(t0);
            r = r_3p2(u);
          } else {
            u = u_1s0(t0);
            r = r_1s0(u); }
          if (t>tcn[i]) {
            w1 = (raise*tcn[i]-t)/((raise-1.0)*tcn[i]);
            w2 = 1.0-w1;
            r = w1*r+w2*1.0; }
        } else { r=1.0; }
        cvneutron = cvn[i]*t*r;
      } else { cvneutron = 0.0; }
      cvproton=solve_cv_mem(tcp,cvp);   // get Cv-protons
      cvlambda=solve_cv_mem(tcla,cvla); // get Cv-lambdas
      cvsminus=solve_cv_mem(tcsm,cvsm); // get Cv-Sigma-
      cvszero=solve_cv_mem(tcs0,cvs0);  // get Cv-Sigma0
      cvsplus=solve_cv_mem(tcsp,cvsp);  // get Cv-Sigma+ 
      double cvphot=0.0;                //get Cv-photons 
      // get Cv-quarks:
      if (rho>=rhocore) {
        cvquark = c_cv_str*(t/1.0e9);
      } else { cvquark = 0.0; }
      // Total Cv
      cvelectron = cvelectron*fhad[i];
      cvmuon = cvmuon*fhad[i];
      cvproton = cvproton*fhad[i];
      cvneutron = cvneutron*fhad[i];
      cvlambda = cvlambda*fhad[i];
      cvsminus = cvsminus*fhad[i];
      cvszero = cvszero*fhad[i];
      cvsplus = cvsplus*fhad[i];
      cvquark = cvquark*(1.0-fhad[i]);

      cv = cvions+cvelectron+cvmuon+cvproton+cvneutron+
            cvlambda+cvsminus+cvszero+cvsplus+cvphot+cvquark;

        // Return statement
    }

};

#endif