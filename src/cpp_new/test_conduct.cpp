/**
 * test_conductivity.cpp
 *
 * Example driver file that includes the two header files
 * ("core_conductivity.h" and "crust_conductivity.h") and calls
 * some of the functions they define. 
 */

#include <iostream>
#include <iomanip>

// Adjust these includes to match the filenames you gave your header files
#include "conductivity.h"

extern "C" {

// Declaration of the Fortran subroutine con_core() 
// with the underscore appended:

void con_core_(
    int*    icon_core, 
    double* debug,
    double* Temp, double* kf_e, double* kf_mu,
    double* kf_p, double* mst_p, double* Tc_p,
    double* kf_n, double* mst_n, double* Tc_n, int* isfn,
    double* kf_la, double* mst_la, double* Tc_la,
    double* kf_sm, double* mst_sm, double* Tc_sm,
    double* kf_s0, double* mst_s0, double* Tc_s0,
    double* kf_sp, double* mst_sp, double* Tc_sp,
    double* f_had,
    double* sigma, double* lambda,
    double* nu_e_s, double* nu_e_l);

void con_crust_(int* icon_crust, double* debug,
              double* T, double* rho, double* kfe, double* A, 
              double* A1, double* Z, double* Q_imp,
              double* sigma_crust, double* lambda_crust,
              double* nu_e_s_crust, double* nu_e_l_crust,
              double* rhodrip);

void conduct_(
    int* i, double* T, double* rho, double* A,
    double* A1, double* Z, double* Q, double* magfield,      // currently unused
    double* sigma, double* lambda, double* debug,
    double* nu_e_s, double* nu_e_l, int* icon_crust,
    int* icon_core, double* rhodrip, double* rhocore,
    const std::vector<double>* kfe, const std::vector<double>* kfm,
    const std::vector<double>* kfn, const std::vector<double>* kfp,
    const std::vector<double>* kfla, const std::vector<double>* kfsm,
    const std::vector<double>* kfs0, const std::vector<double>* kfsp,
    const std::vector<double>* mstn, const std::vector<double>* mstp,
    const std::vector<double>* mstla, const std::vector<double>* mstsm,
    const std::vector<double>* msts0, const std::vector<double>* mstsp,
    const std::vector<double>* tcn, const std::vector<double>* tcp,
    const std::vector<double>* tcla, const std::vector<double>* tcsm,
    const std::vector<double>* tcs0, const std::vector<double>* tcsp,
    int* isf, const std::vector<double>* fhad, int* istrange
);

} // extern "C"

int main() {

  conductivity con;

  // Set precision for floating-point output
    std::cout << std::scientific << std::setprecision(5);

  // --------------------------------------------------
  // Example usage for the "core conductivity" routines
  // --------------------------------------------------
  {
    double sigma_core   = 0.0;
    double lambda_core  = 0.0;
    double nu_e_s_core  = 0.0;
    double nu_e_l_core  = 0.0;

    // Some dummy parameters for the core
    int icon_core = 2;        // e.g. Yakovlev et al. model
    double debug  = 0;      // debug level
    double Temp   = 1.0e8;    // temperature in K
    double kf_e   = 0.5;      // Fermi momentum for electrons
    double kf_mu  = 0.3;      // Fermi momentum for muons
    double kf_p   = 0.8, mst_p=1.0, Tc_p=1.0e9;  // proton
    double kf_n   = 1.0, mst_n=1.0, Tc_n=1.0e9;  // neutron
    int isfn      = 1;        // type of neutron pairing
    double kf_la  = 0.0, mst_la=0.0, Tc_la=0.0;  // lambda (not used)
    double kf_sm  = 0.0, mst_sm=0.0, Tc_sm=0.0;  // sigma- 
    double kf_s0  = 0.0, mst_s0=0.0, Tc_s0=0.0;  // sigma0
    double kf_sp  = 0.0, mst_sp=0.0, Tc_sp=0.0;  // sigma+
    double f_had  = 1.0;      // hadronic fraction
    double nu_e_s_tmp=0.0, nu_e_l_tmp=0.0; // unused placeholders

    // rhodrip, etc.
    double rhodrip = 4.0e11;  

    // --------------------------------------------------------
  // 1) Call the C++ version of con_core(...)
  // --------------------------------------------------------
  double sigma_core_cpp    = 0.0;
  double lambda_core_cpp   = 0.0;
  double nu_e_s_core_cpp   = 0.0;
  double nu_e_l_core_cpp   = 0.0;

  con.con_core(
    icon_core, debug,
    Temp, kf_e, kf_mu,
    kf_p, mst_p, Tc_p,
    kf_n, mst_n, Tc_n, isfn,
    kf_la, mst_la, Tc_la,
    kf_sm, mst_sm, Tc_sm,
    kf_s0, mst_s0, Tc_s0,
    kf_sp, mst_sp, Tc_sp,
    f_had,
    sigma_core_cpp, lambda_core_cpp,
    nu_e_s_core_cpp, nu_e_l_core_cpp
  );

  // --------------------------------------------------------
  // 2) Call the Fortran version of con_core(...)
  // --------------------------------------------------------
  double sigma_core_f   = 0.0;
  double lambda_core_f  = 0.0;
  double nu_e_s_core_f  = 0.0;
  double nu_e_l_core_f  = 0.0;

  // Fortran subroutines require pointers, so set up local copies:
  int    icon_core_f    = icon_core;
  double debug_f        = debug;
  double Temp_f         = Temp;
  double kf_e_f         = kf_e;
  double kf_mu_f        = kf_mu;
  double kf_p_f         = kf_p, mst_p_f= mst_p, Tc_p_f= Tc_p;
  double kf_n_f         = kf_n, mst_n_f= mst_n, Tc_n_f= Tc_n;
  int    isfn_f         = isfn;
  double kf_la_f=0.0, mst_la_f=0.0, Tc_la_f=0.0;
  double kf_sm_f=0.0, mst_sm_f=0.0, Tc_sm_f=0.0;
  double kf_s0_f=0.0, mst_s0_f=0.0, Tc_s0_f=0.0;
  double kf_sp_f=0.0, mst_sp_f=0.0, Tc_sp_f=0.0;
  double f_had_f        = f_had;

  con_core_(
    &icon_core_f, &debug_f,
    &Temp_f, &kf_e_f, &kf_mu_f,
    &kf_p_f, &mst_p_f, &Tc_p_f,
    &kf_n_f, &mst_n_f, &Tc_n_f, &isfn_f,
    &kf_la_f, &mst_la_f, &Tc_la_f,
    &kf_sm_f, &mst_sm_f, &Tc_sm_f,
    &kf_s0_f, &mst_s0_f, &Tc_s0_f,
    &kf_sp_f, &mst_sp_f, &Tc_sp_f,
    &f_had_f,
    &sigma_core_f, &lambda_core_f,
    &nu_e_s_core_f, &nu_e_l_core_f
  );

  // --------------------------------------------------------
  // 3) Print a side-by-side comparison
  // --------------------------------------------------------
  std::cout << "==== Comparison of C++ vs. Fortran con_core ====\n";
  std::cout << "sigma_core   : " << sigma_core_cpp   << "  vs.  " << sigma_core_f   << "\n";
  std::cout << "lambda_core  : " << lambda_core_cpp  << "  vs.  " << lambda_core_f  << "\n";
  std::cout << "nu_e_s_core  : " << nu_e_s_core_cpp  << "  vs.  " << nu_e_s_core_f  << "\n";
  std::cout << "nu_e_l_core  : " << nu_e_l_core_cpp  << "  vs.  " << nu_e_l_core_f  << "\n";
  }

  // ----------------------------------------------------
  // Example usage for the "crust conductivity" routines
  // ----------------------------------------------------
  {
    double sigma_crust   = 0.0;
    double lambda_crust  = 0.0;
    double nu_e_s_crust  = 0.0;
    double nu_e_l_crust  = 0.0;

    int icon_crust = 3;       // e.g. "con_crust" with phonons (Itoh)
    double debug   = 0;     // debug level
    double T       = 1.0e8;   // temperature
    double rho     = 1.0e8;   // density 
    double kfe     = 1.0;     // electron fermi momentum in fm^-1
    double A       = 56.0;    // total nucleons per cell
    double A1      = 56.0;    // nucleons per nucleus
    double Z       = 26.0;    // protons per nucleus
    double Q_imp   = 1.0;     // impurity parameter
    double rhodrip = 4.0e11;  // drip density

    // -------------------------
    // 1) Call the C++ version
    // -------------------------
    double sigma_crust_cpp   = 0.0;
    double lambda_crust_cpp  = 0.0;
    double nu_e_s_crust_cpp  = 0.0;
    double nu_e_l_crust_cpp  = 0.0;

    con.con_crust(
      icon_crust, debug,
      T, rho, kfe, A, A1, Z, Q_imp,
      sigma_crust_cpp, lambda_crust_cpp,
      nu_e_s_crust_cpp, nu_e_l_crust_cpp,
      rhodrip);

  // ---------------------------
  // 2) Call the Fortran version
  // ---------------------------
  double sigma_crust_f   = 0.0;
  double lambda_crust_f  = 0.0;
  double nu_e_s_crust_f  = 0.0;
  double nu_e_l_crust_f  = 0.0;

  // Make local copies of your arguments in pointer-friendly form
  int    icon_crust_f = icon_crust;
  double debug_f      = debug;
  double T_f          = T;
  double rho_f        = rho;
  double kfe_f        = kfe;
  double A_fort          = A;
  double A1_f         = A1;
  double Z_f          = Z;
  double Q_imp_f      = Q_imp;
  double rhodrip_f    = rhodrip;

  // Fortran subroutine expects pointers:
  con_crust_(
      &icon_crust_f, &debug_f,
      &T_f, &rho_f, &kfe_f,
      &A_fort, &A1_f, &Z_f, &Q_imp_f,
      &sigma_crust_f, &lambda_crust_f,
      &nu_e_s_crust_f, &nu_e_l_crust_f,
      &rhodrip_f);

  // ---------------------------
  // 3) Print side-by-side
  // ---------------------------
  std::cout << "==== Comparison (C++ vs. Fortran) ====\n";
  std::cout << "sigma_crust  : " 
            << sigma_crust_cpp << "  vs.  " << sigma_crust_f << "\n";
  std::cout << "lambda_crust : "
            << lambda_crust_cpp << "  vs.  " << lambda_crust_f << "\n";
  std::cout << "nu_e_s_crust : "
            << nu_e_s_crust_cpp << "  vs.  " << nu_e_s_crust_f << "\n";
  std::cout << "nu_e_l_crust : "
            << nu_e_l_crust_cpp << "  vs.  " << nu_e_l_crust_f << "\n";
  }

  // ---------------------------------------------------------
  // Example usage for the neutrino class (if you have one)
  // ---------------------------------------------------------
  // If your "neutrino.h" defines a class named `neutrino`, you could do:
  // 
  // #include "neutrino.h"
  // int i = 0;
  // double Tn = 1.0e9, rhoN = 1.0e14;
  // double qmurca_q = 0.0;
  // neutrino neu;
  // std::vector<double> kfqu(10, 0.1), tcu(10, 1.0e9), tcd(10, 1.0e9);
  // neu.numurca_q(i, Tn, rhoN, qmurca_q, kfqu, tcu, tcd);
  // std::cout << "qmurca_q = " << qmurca_q << std::endl;

  // Done
  std::cout << "All tests completed.\n";
  return 0;
}
