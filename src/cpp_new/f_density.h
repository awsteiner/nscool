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
#ifndef F_DENSITY_H
#define F_DENSITY_H

/***************************************************************
 * f_density.cpp
 * 
 * Highly optimized C++ translation of your Fortran code:
 *   1) p_electron(T, ne)
 *   2) pressure(T, rho, A, Z)
 *   3) density(T, Ptarget, A, Z, rhoInitial)
 *
 * Aggressive optimizations are used without breaking logic:
 *  - inline functions
 *  - replace std::pow with faster calls or direct multiplications
 *  - unify loops carefully
 *  - persistent variables in p_electron for reuse
 * 
 * Compile with:
 *    g++ fdensity_optimized.cpp -o fdensity_optimized -O3
 *    (Optionally add -march=native -ffast-math for further speed.)
 *
 ***************************************************************/

#include <iostream>     // for demonstration (main)
#include <cmath>        // std::fabs, std::sqrt, std::cbrt
#include <limits>       // for numeric_limits

// -----------------------------------------------------------
// Physical constants (cgs units) — declare as constexpr
// -----------------------------------------------------------
static constexpr double KB  = 1.380662e-16; // Boltzmann constant
static constexpr double NA  = 6.022045e23;  // Avogadro's number
static constexpr double PI  = 3.141592653;  // Pi

// Forward declarations (optional but nice)
inline double p_electron(double T, double ne);
inline double pressure(double T, double rho, double A, double Z);
inline double density(double T, double Ptarget, double A, double Z, double rhoInitial);

// -----------------------------------------------------------------------------
// p_electron: Fermi-Dirac approximation from Eggleton et al. (A&A 23, 1973),
// with the data tables (CD, CP, CU) and iterative solution F1, NEF, etc.
// 
// This function is made `inline` for possible performance gain when called often.
// -----------------------------------------------------------------------------
inline double p_electron(double T, double ne)
{
    // Persistent variables to mimic Fortran "static" or COMMON block
    static double OLDNE  = -1.0;  
    static double OLDT   = -1.0;
    static double OLDF1  =  0.0;  

    // Fortran-like parameters
    static constexpr double EPSILON   = 1.0e-12;  // iteration tolerance
    static constexpr double BIGFACTOR = 5.0e-1;   // 50% threshold for reusing old F1
    static constexpr int    MAX_ITER  = 100;      // limit iteration in case of slow conv.

    // Data arrays from your Fortran code
    static constexpr double CD[4][4] = {
        {2.315472,  7.128660,  7.504998,  2.665350},
        {7.837752, 23.507934, 23.311317,  7.987465},
        {9.215560, 26.834068, 25.082745,  8.020509},
        {3.693280, 10.333176,  9.168960,  2.668248}
    };

    static constexpr double CP[4][4] = {
        {2.315472,  6.748104,  6.564912,  2.132280},
        {7.837752, 21.439740, 19.080088,  5.478100},
        {9.215560, 23.551504, 19.015888,  4.679944},
        {3.693280,  8.859868,  6.500712,  1.334124}
    };

    // The original Fortran code had CU, but it wasn't directly used in final P:
    // We'll keep it for completeness (unused).
    // static constexpr double CU[4][4] = { ... };

    // Convert the Fortran code:
    // NEHAT = ne / 1.7595d30
    // T1    = T  / 5.93d9
    // initial F1 either from formula or from OLD if T,ne haven't changed much
    const double NEHAT = ne * (1.0 / 1.7595e30);
    const double T1    = T  * (1.0 / 5.93e9);

    // Proposed new F1
    double F1 = 1.0e-3 * ( ne / NA ) * std::pow(1.0e7 / T, 3.0);

    // Reuse old F1 if NE, T not changed by more than 50%
    if (OLDNE > 0.0 && ne > 0.0 && T > 0.0) {
        double relNE = std::fabs(OLDNE - ne) / ne;
        double relT  = std::fabs(OLDT  - T ) / T;
        if (relNE <= BIGFACTOR && relT <= BIGFACTOR) {
            F1 = OLDF1;
        }
    }

    double F   = 0.0; 
    for (int iter = 0; iter < MAX_ITER; ++iter)
    {
        // F = |F1|
        F = (F1 >= 0.0 ? F1 : -F1);

        // G = T1 * sqrt(1 + F)
        const double onePlusF = 1.0 + F;
        const double sqrt1pF  = std::sqrt(onePlusF);
        const double G  = T1 * sqrt1pF;

        // PF = 1 + F
        // PG = 1 + G
        const double PF = onePlusF;
        const double PG = 1.0 + G;

        // DS(J1) loop: 
        // DS[J1] = [CD[J1][0] + (CD[J1][1] + ... )*G] / PF^3
        double DS[4];
        const double PF3 = PF * PF * PF;
        {
            const double G2 = G * G;
            for (int j1 = 0; j1 < 4; ++j1) {
                double tmp = CD[j1][0] 
                           + (CD[j1][1] + CD[j1][2]*G + CD[j1][3]*G2)*G;
                DS[j1] = tmp / PF3;
            }
        }

        // SUM1 = DS[0] + (DS[1] + DS[2]*F + DS[3]*F^2)*F
        const double F2   = F * F;
        const double sum1 = DS[0] + (DS[1] + DS[2]*F + DS[3]*F2)*F;

        // We need DS(1..3) again for SUM2; let's recompute them the same way 
        // but from CD (the code re-evaluates them in Fortran).
        // We'll do it carefully (like the original).
        double tmpDS1, tmpDS2, tmpDS3;
        {
            const double G2 = G * G;
            tmpDS1 = (CD[1][0] 
                    + (CD[1][1] + CD[1][2]*G + CD[1][3]*G2)*G) / PF3;
            tmpDS2 = (CD[2][0] 
                    + (CD[2][1] + CD[2][2]*G + CD[2][3]*G2)*G) / PF3;
            tmpDS3 = (CD[3][0] 
                    + (CD[3][1] + CD[3][2]*G + CD[3][3]*G2)*G) / PF3;
        }
        // SUM2 = tmpDS1 + (2*tmpDS2 + 3*tmpDS3*F)*F
        const double sum2 = tmpDS1 + (2.0*tmpDS2 + 3.0*tmpDS3*F)*F;

        // Next block for SUM3:
        // DS0_3[J1] = (CD[J1][1] + 2.*CD[J1][2]*G + 3.*CD[J1][3]*G^2)*G / PF^3
        // SUM3= DS0_3[0] + ( DS0_3[1]+ DS0_3[2]*F + DS0_3[3]*F^2 )*F
        double DS0_3[4];
        {
            const double G2 = G * G;
            for (int j1 = 0; j1 < 4; ++j1) {
                double tmp = (CD[j1][1] + 2.0*CD[j1][2]*G + 3.0*CD[j1][3]*G2)*G;
                DS0_3[j1] = tmp / PF3;
            }
        }
        const double sum3 = DS0_3[0] + (DS0_3[1] + DS0_3[2]*F + DS0_3[3]*F2)*F;

        // Now the if(F<1) or else branch for NEF, NEF1
        double NEF  = 0.0;
        double NEF1 = 0.0;
        const double G15 = std::pow(G, 1.5);
        if (F < 1.0) {
            // COEF = F * G^1.5 / PF / (PG^1.5)
            double COEF = F * G15 / PF / std::pow(PG, 1.5);
            NEF  = COEF * sum1;
            double tmpTerm = (COEF / F) - 3.25*(COEF / PF) - 0.75*COEF*G/(PF*PG);
            NEF1 = tmpTerm * sum1 + COEF*(sum2 + 0.5*(sum3 / PF));
        } 
        else {
            // COEF = (F/PF) * (G/PG)^1.5
            double COEF = (F / PF) * std::pow((G / PG), 1.5);
            NEF = COEF * sum1;
            double tmpTerm = (1.0/F - 3.25/PF - 0.75*G/(PF*PG));
            NEF1 = COEF * (tmpTerm*sum1 + sum2 + 0.5*(sum3 / PF));
        }

        // F1 = |F + (NEHAT - NEF)/NEF1|
        double diff = (NEHAT - NEF)/NEF1;
        double newF1 = std::fabs(F + diff);

        // Check convergence 
        double rel1 = std::fabs(NEF - NEHAT) / (std::fabs(NEHAT) + 1.0e-40);
        double rel2 = std::fabs(newF1 - F)   / (std::fabs(newF1) + 1.0e-40);
        F1 = newF1; // update for next iteration

        if (rel1 <= EPSILON && rel2 <= EPSILON) {
            break; 
        }
    }

    // After convergence, compute final P1 using CP array:
    const double onePlusF = 1.0 + F; 
    const double sqrt1pF  = std::sqrt(onePlusF);
    const double G1 = T1 * sqrt1pF;

    // DS_cp[j1] = CP[j1][0] + (CP[j1][1]+...)*G1
    double DS_cp[4];
    {
        double G1_2 = G1 * G1;
        for (int j1 = 0; j1 < 4; ++j1) {
            DS_cp[j1] = CP[j1][0] 
                      + (CP[j1][1] + CP[j1][2]*G1 + CP[j1][3]*G1_2)*G1;
        }
    }
    double sumP1 = DS_cp[0] 
                 + (DS_cp[1] + DS_cp[2]*F + DS_cp[3]*F*F)*F;

    double P1 = 0.0;
    if (F < 1.0) {
        // P1=1.44E24 * F * G1^2.5 / (1+F)^4 / (1+G1)^1.5 * sumP1
        double denom = std::pow(onePlusF, 4.0) * std::pow(1.0 + G1, 1.5);
        double G1_25 = std::pow(G1, 2.5);
        P1 = 1.44e24 * F * G1_25 / denom * sumP1;
    } 
    else {
        // P1=1.44E24*(F^0.25/(1+F))^4 * G1*(G1/(1+G1))^1.5 * sumP1
        double F025 = std::sqrt(std::sqrt(F)); // F^(0.25)
        double tmp1 = (F025 / onePlusF);
        double factor = std::pow(tmp1, 4.0) * G1 * std::pow(G1/(1.0+G1), 1.5);
        P1 = 1.44e24 * factor * sumP1;
    }

    // Update persistent variables
    OLDNE  = ne;
    OLDT   = T;
    OLDF1  = F1;

    // Return final e- pressure
    return P1;
}

// -----------------------------------------------------------------------------
// pressure: Summation of ionic pressure + e- pressure
// -----------------------------------------------------------------------------
inline double pressure(double T, double rho, double A, double Z)
{
    // Ion/Electron densities
    const double ne   = rho * NA * Z / A;
    const double nion = rho * NA     / A;

    // gamma = 2.273e5 * Zeff^2 * (rho/A)^(1/3) / T
    // Zeff = ne / nion
    double Zeff  = (nion == 0.0 ? 0.0 : (ne / nion));
    double ratio = (rho / A);
    // Avoid std::pow(ratio, 1.0/3.0) -> use std::cbrt
    double gamma = 2.273e5 * (Zeff*Zeff) * std::cbrt(ratio) / T;

    // Uion depends on gamma
    double Uion = 0.0;
    if (gamma >= 210.0) {
        // Uion = 1.5 - 0.895929*gamma + 3225.0/(gamma*gamma)
        Uion = 1.5 - 0.895929*gamma + 3225.0/(gamma*gamma);
    } else {
        // Uion= -0.897744*gamma + 0.95043*gamma^0.25 + 0.18956/gamma^0.25 -0.81487
        double g025 = std::sqrt(std::sqrt(gamma)); // gamma^(0.25)
        Uion = -0.897744*gamma 
               + 0.95043*g025
               + 0.18956/g025
               - 0.81487;
    }

    // Pion = nion*KB*T*(1.0 + Uion/3.0)
    double Pion = nion * KB * T * (1.0 + Uion/3.0);

    // Electron pressure
    double Pel = p_electron(T, ne);

    return (Pion + Pel);
}

// -----------------------------------------------------------------------------
// density: Solve P(rho)= Ptarget via Newton's method
// -----------------------------------------------------------------------------
inline double density(double T, double Ptarget, double A, double Z, double rhoInitial)
{
    static constexpr double EPS = 1.0e-3;  // For Rho1=(1+EPS)*Rho0
    static constexpr double TOL = 1.0e-5;  // Convergence on |dRho/Rho|
    static constexpr int    MAX_ITER = 200;

    double Rho = (rhoInitial > 1.0e-40) ? rhoInitial : 1.0e-10; 

    for (int iter = 0; iter < MAX_ITER; ++iter)
    {
        double Rho0 = Rho;
        double Pre0 = pressure(T, Rho0, A, Z);

        // Rho1 = (1+EPS)*Rho0
        double Rho1 = (1.0 + EPS)*Rho0;
        double Pre1 = pressure(T, Rho1, A, Z);

        double f   = Pre0 - Ptarget;
        double f1  = (Pre1 - Pre0) / (Rho1 - Rho0);
        // Guard against extremely small f1
        if (std::fabs(f1) < 1.0e-30) {
            break;
        }

        double dRho = -f / f1;
        Rho = Rho0 + dRho;
        if (Rho < 1.0e-40) {
            Rho = 1.0e-40;
        }

        // Convergence if |dRho/Rho| < TOL
        if (std::fabs(dRho / Rho) < TOL) {
            break;
        }
    }

    return Rho;
}

#endif