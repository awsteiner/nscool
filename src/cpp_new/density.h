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
#ifndef NSCOOL_DENSITY_H
#define NSCOOL_DENSITY_H

#include <boost/numeric/ublas/vector.hpp>

#include <o2scl/constants.h>

typedef boost::numeric::ublas::vector<double> ubvector;
typedef boost::numeric::ublas::matrix<double> ubmatrix;

/** \brief Density Calculations

    This work in progress will eventually replace Dany's Fortran code.
*/

class density {

  private:
    static constexpr double pi = 3.14159652; 
    static constexpr double NA=6.022045e23;
    static constexpr double epsilon=1.e-12;
    static constexpr double eps = 1.0e-3;
    static constexpr double hb=1.054588e-27;
    static constexpr double kb=1.380662e-16;
    static constexpr double c=2.997924e10;
    static constexpr double me=9.109e-28;

    double CD[4][4] = {{2.315472, 7.128660, 7.504998, 2.665350},
                   {7.837752, 23.507934, 23.311317, 7.987465},
                   {9.215560, 26.834068, 25.082745, 8.020509},
                   {3.693280, 10.333176, 9.168960, 2.668248}};

    double CP[4][4] = {{2.315472, 6.748104, 6.564912, 2.132280},
                      {7.837752, 21.439740, 19.080088, 5.478100},
                      {9.215560, 23.551504, 19.015888, 4.679944},
                      {3.693280, 8.859868, 6.500712, 1.334124}};

    double CU[4][4] = {{3.473208, 10.122156, 9.847368, 3.198420},
                      {16.121172, 43.477194, 37.852852, 10.496830},
                      {23.971040, 60.392810, 47.782844, 11.361074},
                      {11.079840, 26.579604, 19.502136, 4.002372}};

    double DS[4];

  public:

    density() {
      
    }

    /** \brief Desc
     */
    bool iter(double T, double NEHAT, double &F1) {
      double F = std::abs(F1);
      double G = T/5.93e9*std::sqrt(1.0+F);
      double PF = 1.0+F;
      double PG = 1.0+G;
      for (int J1=0;J1<=3;J1++) {
        DS[J1] = CD[J1][0]+(CD[J1][1]+CD[J1][2]*G+CD[J1][3]*G*G)*G;
        DS[J1] = DS[J1]/pow(PF,3);
      }
      double SUM1 = DS[0]+(DS[1]+DS[2]*F+DS[3]*F*F)*F;
      for (int J1=1;J1<=3;J1++) {
        DS[J1] = CD[J1][0]+(CD[J1][1]+CD[J1][2]*G+CD[J1][3]*G*G)*G;
        DS[J1] = DS[J1]/pow(PF,3);
      }
      double SUM2 = DS[1]+(2.0*DS[2]+3.0*DS[3]*F)*F;
      for (int J1=0;J1<=3;J1++) {
        DS[J1] = (CD[J1][1]+2.0*CD[J1][2]*G+3.0*CD[J1][3]*G*G)*G;
        DS[J1] = DS[J1]/pow(PF,3);
      }
      double SUM3 = DS[0]+(DS[1]+DS[2]*F+DS[3]*F*F)*F;
      double COEF;
      double NEF, NEF1;
      if (F<1.0) {
        COEF = F*std::pow(G,1.5)/PF/std::pow(PG,1.5);
        NEF = COEF*SUM1;
        NEF1 = (COEF/F-3.25*COEF/PF-0.75*COEF*G/(PF*PG))*SUM1+
          COEF*(SUM2+0.5/PF*SUM3);
      } else {
        COEF = (F/PF)*std::pow(G/PG,1.5);
        NEF = COEF*SUM1;
        NEF1 = COEF*((1.0/F-3.25/PF-0.75*G/(PF*PG))*SUM1+SUM2+0.5/PF*SUM3);
      }
      F1 = std::abs(F+(NEHAT-NEF)/NEF1);
      //std::cout << "iter: " << F1 << " " << NEF << " " << NEHAT << std::endl;
      if (std::abs((NEF-NEHAT)/NEHAT)>epsilon ||
          std::abs((F1-F)/F1)>epsilon) {
        return true;
        //std::cout << std::abs((NEF-NEHAT)/NEHAT) << " " << std::abs((F1-F)/F1) << std::endl;
        //iter(T,NEHAT,F1);
      }
      return false;
    }

    /** \brief Pressure Electron
     */
    void P_electron(double T, double ne, double& Pres) {
      double F1 = 1.0e-3*(ne/NA)*pow(1.0e7/T,3);
      double NEHAT = ne/1.7595e30;

      bool test;
      do {
        test=iter(T,NEHAT,F1);
        //std::cout << "P_electron: " << F1 << " " << NEHAT << std::endl;
      } while (test==false);
      
      double G1 = T/5.93e9*std::sqrt(1.0+F1);
      for (int J1=0;J1<=3;J1++) {
        DS[J1]=CP[J1][0]+(CP[J1][1]+CP[J1][2]*G1+CP[J1][3]*G1*G1)*G1;
      }
      double SUMP1=DS[0]+(DS[1]+DS[2]*F1+DS[3]*F1*F1)*F1;
      double P1;
      if (F1<1.0) {
        P1 = 1.44e24*F1*std::pow(G1,2.5)/std::pow(1.0+F1,4)/
          std::pow(1.0+G1,1.5)*SUMP1;
      } else {
        P1 = 1.44e24*std::pow(F1,0.25)/(1.0+F1)*std::pow(G1,1.5)*
          SUMP1*(G1/(1.0+G1));
      }
      Pres = P1;
      return;
    }

    /** \brief Main Pressure function
     */
    void pressure(double T, double Rho, double A, double Z, double& Pres) {
      // Calculate the electron density and ionization
      double ne = Rho*NA*Z/A;
      double nion = Rho*NA/A;
      double Zeff = ne/nion;
      // Calculate the ionic pressure
      double gamma = 2.273e5*std::pow(Zeff, 2)*std::pow(Rho/A,1.0/3)/T;
      double Uion;
      if (gamma>=210.0) {
        Uion = 1.5-0.895929*gamma+3225.0/std::pow(gamma,2);
      } else {
        Uion = -0.897744*gamma+0.95043*std::pow(gamma,0.25)+
          0.18956/std::pow(gamma,0.25)-0.81487;
      }
      double Pion = nion*kb*T*(1.0+Uion/3.0);
      double Pel;
      P_electron(T,ne,Pel);
      Pres = Pel+Pion;
      return;
    }

    /** \brief Main Density function
     */
    void densityt(double T, double P, double A, double Z, double& Rho) {
      double dRho, Pre0, Pre1, Rho1, Rho0;
      do {
        Rho0 = Rho;
        pressure(T, Rho0, A, Z, Pre0);
        Rho1 = (1.0+eps)*Rho0;
        pressure(T, Rho1, A, Z, Pre1);
        double f = Pre0-P; 
        double f1 = (Pre1-Pre0)/(Rho1-Rho0);
        dRho = -f/f1;
        Rho = Rho0+dRho;
      } while (std::abs(dRho/Rho)>1.0e-5);
      return;  
    }

};

#endif