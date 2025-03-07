/***************************************************************
 * fdensity_optimized.cpp
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

#include "f_density.h"

// ------------------------------------------------------------------
// Declarations for the Fortran subroutines.
// Make sure the names match the BIND(C, name="...") in your Fortran.
//
// For example, in your Fortran you have:
//   subroutine density(T, P, A, Z, Rho) bind(C, name="density")
// ------------------------------------------------------------------
extern "C" {
    void density_(double* T, double* P, double* A, double* Z, double* Rho);
    void pressure_(double* T, double* Rho, double* A, double* Z, double* Pres);

    // If you want to call your Fortran p_electron as well:
    void P_electron_(double* T, double* ne, double* Pres);
}

// -----------------------------------------------------------------------------
// Minimal main() for demonstration
// -----------------------------------------------------------------------------
int main()
{
    // Example parameters
    double T       = 1.0e7;     // K
    double Ptarget = 1.0e14;    // dyn/cm^2
    double A       = 4.0;       // Helium
    double Z       = 2.0;       // Helium
    double rhoGuess = 1.0;      // initial guess for Newton

    // 1) C++ approach
    double rho_cpp = density(T, Ptarget, A, Z, rhoGuess);
    double P_cpp   = pressure(T, rho_cpp, A, Z);

    // 2) Fortran approach
    //   Fortran "density" modifies Rho in-place
    double rho_fort = rhoGuess;
    density_(&T, &Ptarget, &A, &Z, &rho_fort);

    double P_fort;
    pressure_(&T, &rho_fort, &A, &Z, &P_fort);

    // 3) Print out comparisons
    std::cout << "=========================================\n";
    std::cout << "     T         = " << T       << " K\n";
    std::cout << "     P_target  = " << Ptarget << " dyn/cm^2\n";
    std::cout << "     A         = " << A       << "\n";
    std::cout << "     Z         = " << Z       << "\n";
    std::cout << "-----------------------------------------\n";
    std::cout << "   C++ density = " << rho_cpp 
              << "   => P(rho_cpp) = " << P_cpp << "\n";
    std::cout << "Fortran density= " << rho_fort
              << " => P(rho_fort)=  " << P_fort << "\n";
    std::cout << "-----------------------------------------\n";

    // Show the difference
    double relDiffRho = std::fabs(rho_cpp - rho_fort)/ (std::fabs(rho_cpp)+1e-40);
    double relDiffP   = std::fabs(P_cpp   - P_fort)  / (std::fabs(P_cpp)  +1e-40);

    std::cout << "   Relative diff in density = " << relDiffRho << "\n";
    std::cout << "   Relative diff in P       = " << relDiffP   << "\n";
    std::cout << "=========================================\n";

    // 4) [Optional] If you want to compare p_electron as well:
    //    Let's do an example with some NE value:
    double ne_val = 1.0e24; // arbitrary example electron density
    double pE_cpp  = p_electron(T, ne_val);
    double pE_fort = 0.0;
    //P_electron_(&T, &ne_val, &pE_fort);

    std::cout << "[p_electron check]   C++= " << pE_cpp; 
    //          << "   Fortran= " << pE_fort 
    //          << "\n";

    return 0;
}
