#ifndef F_PRECOOL_H
#define F_PRECOOL_H

#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cstdlib>

#include <boost/numeric/ublas/vector.hpp>

#include <o2scl/constants.h>
#include <o2scl/interp.h>
#include <o2scl/table.h>
#include <o2scl/hdf_file.h>
#include <o2scl/hdf_io.h>

#include "nscool_wrap.h"
#include "f_tc.h"

class precool : public nscool_wrap {

  public:

    precool() : nscool_wrap("./")  {
    }

    typedef std::function<double (double)> funct;

    //nscool_wrap nw;

#ifdef PLOT_SCRIPTS 
/**
 * @brief Calculates the chemical composition, A, A', Z, and the electron & neutron Y's in the crust.
 *
 * This function interpolates particle concentrations using the crust EOS table and fills
 * the provided arrays with the computed values.
 *
 * @param irank    Rank identifier (unused in this context).
 * @param debug    Debug level (1 or higher for verbose output).
 * @param version  Version string ("old", "new", "NEW", "QRK").
 * @param imax     (Output) Maximum index of the grid.
 * @param icore    (Output) Core index.
 * @param rrho     (Input/Output) Density array (size 10001).
 * @param pres     (Input/Output) Pressure array (size 10001).
 * @param debar    (Output) Baryon differential volume array (size 10001).
 * @param dvol     (Input/Output) Volume array (size 10001).
 * @param bar      (Output) Baryon density array (size 10001).
 * @param A_cell   (Output) A_cell array (size 10001).
 * @param A_ion    (Output) A_ion array (size 10001).
 * @param Z_ion    (Output) Z_ion array (size 10001).
 * @param v_ion    (Output) Ion volume fraction array (size 10001).
 * @param yelect   (Output) Electron concentration array (size 10001).
 * @param yneutr   (Output) Neutrino concentration array (size 10001).
 */
void get_crust_chemistry(int irank, int debug, const std::string& version,
                         int& imax, int& icore,
                         std::vector<double>& rrho, std::vector<double>& pres,
                         std::vector<double>& debar, std::vector<double>& dvol,
                         std::vector<double>& bar, std::vector<double>& A_cell,
                         std::vector<double>& A_ion, std::vector<double>& Z_ion,
                         std::vector<double>& v_ion,
                         std::vector<double>& yelect, std::vector<double>& yneutr) {
    constexpr int isize = 10000;
    const double pi = 3.14159265358979323846;

    // Validate input vector sizes
    auto validate_vector = [&](const std::vector<double>& vec, const std::string& name) {
      if (vec.size() < static_cast<size_t>(isize + 1)) {
        throw std::invalid_argument("Vector " + name + " must have size at least " + std::to_string(isize + 1));
      }
    };

    // Function to compute log safely
    auto safe_log10 = [](double x) -> double {
      if (x <= 0.0) {
        O2SCL_ERR2("Logarithm of non-positive in ",
        "nscool::core_comp",o2scl::exc_einval);
      }
      return std::log(x);
    };

    validate_vector(rrho, "rrho");
    validate_vector(pres, "pres");
    validate_vector(debar, "debar");
    validate_vector(dvol, "dvol");
    validate_vector(bar, "bar");
    validate_vector(A_cell, "a_cell");
    validate_vector(A_ion, "a_ion");
    validate_vector(Z_ion, "z_ion");
    validate_vector(v_ion, "v_ion");
    validate_vector(yelect, "yelect");
    validate_vector(yneutr, "yneutr");

    // Temporary arrays for EOS tables
    std::vector<double> rho_t(500, 0.0);
    std::vector<double> pres_t(500, 0.0);
    std::vector<double> bar_t(500, 0.0);
    std::vector<double> A_cell_t(500, 0.0);
    std::vector<double> A_ion_t(500, 0.0);
    std::vector<double> Z_ion_t(500, 0.0);

    int jmax = 0;

    // Debug message
    if (debug >= 1) {
      std::cout << "Entering subroutine get_crust_chemistry" << std::endl;
    }

    // Call external subroutine to fill temporary arrays
    crust_comp_cpp(Z_ion_t, A_ion_t, A_cell_t, bar_t, pres_t, rho_t, jmax);

    // Find jdrip: first j from jmax downto 1 where A_cell_t(j) != A_ion_t(j)
    int jdrip = 0;
    for (int j = jmax; j >= 1; --j) {
      if (A_cell_t[j - 1] != A_ion_t[j - 1]) { // Adjusting for 0-based indexing
        jdrip = j;
        break;
      }
    }

    // Adjust jmax to ensure rho_t(jmax) < rrho(icore)
    int jjmax = jmax;
    for (int j = jmax; j >= 1; --j) {
      if (rho_t[j - 1] >= rrho[icore]) { // Adjusting for 0-based indexing
          jjmax = j - 1;
      }
    }
    jmax = jjmax;

    // Interpolation from the last crust_EOS line up to the core
    // First version:
    // - Take rho & bar from core EOS:
    jmax += 1;
    if (jmax >= static_cast<int>(rho_t.size())) {
      throw std::out_of_range("jmax exceeds rho_t size after increment.");
    }
    rho_t[jmax - 1] = rrho[icore];
    bar_t[jmax - 1] = bar[icore];
    pres_t[jmax - 1] = pres[icore];
    Z_ion_t[jmax - 1] = Z_ion_t[jmax - 2];
    A_ion_t[jmax - 1] = A_ion_t[jmax - 2];
    A_cell_t[jmax - 1] = Z_ion_t[jmax - 1] / yelect[icore];

    // Debug message
    if (debug >= 1) {
      std::cout << "Interpolating particle concentrations..." << std::endl;
    }

    // Loop from icore+1 to imax to interpolate
    int j = jmax - 1; // Adjusting for 0-based indexing
    for (int i = icore + 1; i <= imax; ++i) {
      // Decrement j until rho_t[j] <= rrho[i]
      while (j > 0 && rho_t[j] > rrho[i]) {
        --j;
      }

      // Ensure j is within bounds
      if (j >= jmax - 1) {
        j = jmax - 2;
      }
      if (j < 0 || (j + 1) >= static_cast<int>(rho_t.size())) {
        throw std::out_of_range("j index out of bounds during interpolation.");
      }

      // Prevent division by zero in logarithms
      double denom_log = safe_log10(rho_t[j + 1]) - safe_log10(rho_t[j]);
      if (denom_log == 0.0) {
        throw std::runtime_error("Division by zero in interpolation denominator.");
      }

      // Compute interpolation weights
      double x1 = (safe_log10(rho_t[j + 1]) - safe_log10(rrho[i])) / denom_log;
      double x2 = (safe_log10(rrho[i]) - safe_log10(rho_t[j])) / denom_log;

      // Interpolate particle concentrations
      bar[i]     = x1 * bar_t[j]     + x2 * bar_t[j + 1];
      yelect[i]  = x1 * yelect[j]    + x2 * yelect[j + 1];
      yneutr[i]  = x1 * yneutr[j]    + x2 * yneutr[j + 1];

      // Handle 'fhad' based on version
      //if (version == "old" || version == "new" || version == "NEW") {
        //fhad[i] = 1.0;
      //}
      //else if (version == "QRK") {
        //fhad[i] = x1 * bar_t[j] + x2 * bar_t[j + 1]; // Assuming fhad_t is bar_t based on Fortran code
        // Clamp fhad[i] to [0.0, 1.0]
        //fhad[i] = std::clamp(fhad[i], 0.0, 1.0);
      //}

      // Interpolate A_cell, A_ion, Z_ion
      if (A_ion_t[j + 1] == A_cell_t[j + 1]) {
        A_ion[i]  = A_ion_t[j + 1];
        A_cell[i] = A_cell_t[j + 1];
        Z_ion[i]  = Z_ion_t[j + 1];
      }
      else {
        // Old version: A_cell and A_ion are interpolated
        // New version: A_cell is interpolated, A_ion and Z_ion are set from j+1
        A_cell[i] = x1 * A_cell_t[j] + x2 * A_cell_t[j + 1];
        A_ion[i]  = A_ion_t[j + 1];
        Z_ion[i]  = Z_ion_t[j + 1];
      }

      // Calculate the fraction of volume occupied by ions
      const double r1 = 1.1; // Scale parameter in fm
      double vion_fm3 = (4.0 / 3.0) * pi * std::pow(r1, 3) * A_ion[i]; // fm^3
      double vion_cm3 = 1.0e-39 * vion_fm3; // cm^3

      double nion = rrho[i] / (1.66e-24 * A_ion[i]); // ion density per cm^3
      v_ion[i] = nion * vion_cm3;
      v_ion[i] = std::min(1.0, v_ion[i]); // Ensure v_ion[i] <= 1.0

      //// Check for consistency:
      //double bnuc  = yneutr[i] + yprot[i];
      //double bhyp  = 0.0; // ylambda, ysminus, yszero, ysplus not defined in this subroutine
      //// Assuming ylambda, ysminus, yszero, ysplus are available from previous steps
      //// If not, they should be interpolated similarly to yelect and yneutr
      //// For demonstration, we'll set bhyp = 0
      //double bqua  = (1.0 / 3.0) * (0.0); // yquarku, yquarkd, yquarks not defined here
      //double btot  = bnuc + bhyp + bqua;
//
      //double qlep  = -yelect[i]; // ymuon not interpolated in this subroutine
      //double qnuc  = yprot[i];
      //double qhyp  = 0.0; // ysplus - ysminus not defined here
      //double qqua  = 0.0; // yquarku, yquarkd, yquarks not defined here
      //double qtot  = qlep + qnuc + qhyp + qqua;
//
      //// Consistency checks
      //if (std::abs(btot - 1.0) > 1e-2) {
      //  if (debug >= 1) {
      //    std::cout << std::fixed << std::setprecision(3);
      //    std::cout << "i, rho, Btot, Qtot = " << i << ", " << rrho[i]
      //              << ", " << btot << ", " << qtot << std::endl;
      //    std::cout << "Btot not equal to 1! " << i << ", " << yneutr[i]
      //              << ", " << yprot[i] << std::endl;
      //    std::cout << "bnuc: " << bnuc << ", bhyp: " << bhyp << ", bqua: " << bqua << std::endl;
      //  }
      //  icore = 0;
      //  return;
      //}
//
      //if (std::abs(qtot) > 1e-2) {
      //    if (debug >= 1) {
      //        std::cout << std::fixed << std::setprecision(3);
      //        std::cout << "i, rho, Btot, Qtot = " << i << ", " << rrho[i]
      //                  << ", " << btot << ", " << qtot << std::endl;
      //        std::cout << "Qtot not equal to 0! " << i << ", " << qlep
      //                  << ", " << qnuc << ", " << qhyp << ", " << qqua
      //                  << ", " << qtot << std::endl;
      //    }
      //    icore = 0;
      //    return;
      //}
    }

    // Calculate the Y's of the electrons and neutrinos
    //for (int i = icore + 1; i <= imax; ++i) {
    //  if (A_cell[i] == 0.0) {
    //    yelect[i] = 0.0;
    //    yneutr[i] = 0.0;
    //  }
    //  else {
    //    yelect[i] = Z_ion[i] / A_cell[i];
    //    yneutr[i] = (A_cell[i] - a_ion[i]) / A_cell[i];
    //  }
    //}

    // Clean up the core, just in case
    for (int i = 0; i <= icore; ++i) {
      Z_ion[i]  = 0.0;
      A_ion[i]  = 0.0;
      A_cell[i] = 0.0;
      v_ion[i]  = 0.0;
    }

    // Calculate debar
    debar[0] = 0.0;
    for (int i = 1; i <= imax; ++i) {
      debar[i] = (bar[i - 1] + bar[i]) / 2.0 * dvol[i];
    }

    // Debug message
    if (debug >= 1) {
        std::cout << "Exiting subroutine get_crust_chemistry" << std::endl;
    }
  

}


/**
 * @brief Calculates the Fermi momenta of all fermions in the core and crust.
 *
 * @param irank             Rank identifier (unused in this context).
 * @param imax              Maximum index of the grid.
 * @param icore             Core index.
 * @param rrho              Density array (size 10001).
 * @param bar               Baryon density array (size 10001).
 * @param yneutr            Neutrino concentration array (size 10001).
 * @param yprot             Proton concentration array (size 10001).
 * @param yelect            Electron concentration array (size 10001).
 * @param ymuon             Muon concentration array (size 10001).
 * @param ylambda           Lambda concentration array (size 10001).
 * @param ysminus           S- concentration array (size 10001).
 * @param yszero            S0 concentration array (size 10001).
 * @param ysplus            S+ concentration array (size 10001).
 * @param yquarku           Up quark concentration array (size 10001).
 * @param yquarkd           Down quark concentration array (size 10001).
 * @param yquarks           Strange quark concentration array (size 10001).
 * @param fhad              Hadron fraction array (size 10001).
 * @param theta_k           Theta_k array (size 10001).
 * @param theta_p           Theta_p array (size 10001).
 * @param kfn               Neutron Fermi momentum array (size 10001).
 * @param kfp               Proton Fermi momentum array (size 10001).
 * @param kfe               Electron Fermi momentum array (size 10001).
 * @param kfm               Muon Fermi momentum array (size 10001).
 * @param kfla              Lambda Fermi momentum array (size 10001).
 * @param kfsm              S- Fermi momentum array (size 10001).
 * @param kfs0              S0 Fermi momentum array (size 10001).
 * @param kfsp              S+ Fermi momentum array (size 10001).
 * @param kfqu              Up quark Fermi momentum array (size 10001).
 * @param kfqd              Down quark Fermi momentum array (size 10001).
 * @param kfqs              Strange quark Fermi momentum array (size 10001).
 * @param idurca_np         Neutron-Proton direct Urca process array (size 10001).
 * @param idurca_lap        Lambda-Proton direct Urca process array (size 10001).
 * @param durca_ctrl_e      Electron direct Urca control function array (size 10001).
 * @param durca_ctrl_m      Muon direct Urca control function array (size 10001).
 * @param idurca_smn        S- Neutron direct Urca process array (size 10001).
 * @param idurca_smla       S- Lambda direct Urca process array (size 10001).
 * @param idurca_sms0       S- S0 direct Urca process array (size 10001).
 * @param idurca_quqd       Quark down direct Urca process array (size 10001).
 * @param idurca_quqs       Quark strange direct Urca process array (size 10001).
 * @param durca_henon_e     Electron Henon direct Urca function array (size 10001).
 * @param durca_henon_m     Muon Henon direct Urca function array (size 10001).
 */
void get_fermi_momenta(int irank, int imax, int icore,
                       const std::vector<double>& rrho,
                       const std::vector<double>& bar,
                       const std::vector<double>& yneutr,
                       const std::vector<double>& yprot,
                       const std::vector<double>& yelect,
                       const std::vector<double>& ymuon,
                       const std::vector<double>& ylambda,
                       const std::vector<double>& ysminus,
                       const std::vector<double>& yszero,
                       const std::vector<double>& ysplus,
                       const std::vector<double>& yquarku,
                       const std::vector<double>& yquarkd,
                       const std::vector<double>& yquarks,
                       const std::vector<double>& fhad,
                       const std::vector<double>& theta_k,
                       const std::vector<double>& theta_p,
                       std::vector<double>& kfn,
                       std::vector<double>& kfp,
                       std::vector<double>& kfe,
                       std::vector<double>& kfm,
                       std::vector<double>& kfla,
                       std::vector<double>& kfsm,
                       std::vector<double>& kfs0,
                       std::vector<double>& kfsp,
                       std::vector<double>& kfqu,
                       std::vector<double>& kfqd,
                       std::vector<double>& kfqs,
                       std::vector<int>& idurca_np,
                       std::vector<int>& idurca_lap,
                       std::vector<int>& durca_ctrl_e,
                       std::vector<int>& durca_ctrl_m,
                       std::vector<int>& idurca_smn,
                       std::vector<int>& idurca_smla,
                       std::vector<int>& idurca_sms0,
                       std::vector<int>& idurca_quqd,
                       std::vector<int>& idurca_quqs,
                       std::vector<double>& durca_henon_e,
                       std::vector<double>& durca_henon_m) {
  try {
    constexpr int isize = 10000;
    const double pi = 3.14159265358979323846;

    // Validate input vector sizes
    auto validate_vector = [&](const std::vector<double>& vec, const std::string& name) {
      if (vec.size() < static_cast<size_t>(isize + 1)) {
          throw std::invalid_argument("Vector " + name + " must have size at least " + std::to_string(isize + 1));
      }
    };
    auto validate_int_vector = [&](const std::vector<int>& vec, const std::string& name) {
      if (vec.size() < static_cast<size_t>(isize + 1)) {
          throw std::invalid_argument("Integer vector " + name + " must have size at least " + std::to_string(isize + 1));
      }
    };

    // Validate all required vectors
    validate_vector(rrho, "rrho");
    validate_vector(bar, "bar");
    validate_vector(yneutr, "yneutr");
    validate_vector(yprot, "yprot");
    validate_vector(yelect, "yelect");
    validate_vector(ymuon, "ymuon");
    validate_vector(ylambda, "ylambda");
    validate_vector(ysminus, "ysminus");
    validate_vector(yszero, "yszero");
    validate_vector(ysplus, "ysplus");
    validate_vector(yquarku, "yquarku");
    validate_vector(yquarkd, "yquarkd");
    validate_vector(yquarks, "yquarks");
    validate_vector(fhad, "fhad");
    validate_vector(theta_k, "theta_k");
    validate_vector(theta_p, "theta_p");

    validate_vector(kfn, "kfn");
    validate_vector(kfp, "kfp");
    validate_vector(kfe, "kfe");
    validate_vector(kfm, "kfm");
    validate_vector(kfla, "kfla");
    validate_vector(kfsm, "kfsm");
    validate_vector(kfs0, "kfs0");
    validate_vector(kfsp, "kfsp");
    validate_vector(kfqu, "kfqu");
    validate_vector(kfqd, "kfqd");
    validate_vector(kfqs, "kfqs");

    validate_int_vector(idurca_np, "idurca_np");
    validate_int_vector(idurca_lap, "idurca_lap");
    validate_int_vector(durca_ctrl_e, "durca_ctrl_e");
    validate_int_vector(durca_ctrl_m, "durca_ctrl_m");
    validate_int_vector(idurca_smn, "idurca_smn");
    validate_int_vector(idurca_smla, "idurca_smla");
    validate_int_vector(idurca_sms0, "idurca_sms0");
    validate_int_vector(idurca_quqd, "idurca_quqd");
    validate_int_vector(idurca_quqs, "idurca_quqs");

    validate_vector(durca_henon_e, "durca_henon_e");
    validate_vector(durca_henon_m, "durca_henon_m");

    // Temporary variables for consistency checks
    double coeff = 3.0 * pi * pi;

    // Calculate the Fermi momenta in the core
    for (int j = 0; j <= icore; ++j) {
      double nn = 0.0, np = 0.0, nla = 0.0, nsm = 0.0, ns0 = 0.0, nsp = 0.0;
      double nqu = 0.0, nqd = 0.0, nqs = 0.0;

      if (fhad[j] != 0.0) {
        nn = std::max(0.0, yneutr[j] * bar[j]) / fhad[j];
        np = std::max(0.0, yprot[j] * bar[j]) / fhad[j];
        nla = std::max(0.0, ylambda[j] * bar[j]) / fhad[j];
        nsm = std::max(0.0, ysminus[j] * bar[j]) / fhad[j];
        ns0 = std::max(0.0, yszero[j] * bar[j]) / fhad[j];
        nsp = std::max(0.0, ysplus[j] * bar[j]) / fhad[j];
      }

      if (fhad[j] != 1.0) {
        nqu = std::max(0.0, yquarku[j] * bar[j]) / (1.0 - fhad[j]);
        nqd = std::max(0.0, yquarkd[j] * bar[j]) / (1.0 - fhad[j]);
        nqs = std::max(0.0, yquarks[j] * bar[j]) / (1.0 - fhad[j]);
      }

      kfn[j]  = std::pow(3.0 * pi * pi * nn, 1.0 / 3.0);
      kfp[j]  = std::pow(3.0 * pi * pi * np, 1.0 / 3.0);
      kfla[j] = std::pow(3.0 * pi * pi * nla, 1.0 / 3.0);
      kfsm[j] = std::pow(3.0 * pi * pi * nsm, 1.0 / 3.0);
      kfs0[j] = std::pow(3.0 * pi * pi * ns0, 1.0 / 3.0);
      kfsp[j] = std::pow(3.0 * pi * pi * nsp, 1.0 / 3.0);
      kfqu[j] = std::pow(3.0 * pi * pi * nqu, 1.0 / 3.0);
      kfqd[j] = std::pow(3.0 * pi * pi * nqd, 1.0 / 3.0);
      kfqs[j] = std::pow(3.0 * pi * pi * nqs, 1.0 / 3.0);

      double ne = std::abs(yelect[j] * bar[j]);
      double nm = std::abs(ymuon[j] * bar[j]);

      kfe[j] = std::pow(3.0 * pi * pi * ne, 1.0 / 3.0);
      if (yelect[j] <= 0.0) {
          kfe[j] = -kfe[j];
      }

      kfm[j] = std::pow(3.0 * pi * pi * nm, 1.0 / 3.0);
      if (ymuon[j] <= 0.0) {
          kfm[j] = -kfm[j];
      }

      // Consistency Checks
      double bnuc  = yneutr[j] * bar[j] + yprot[j] * bar[j];
      double bhyp  = ylambda[j] * bar[j] + ysminus[j] * bar[j] + yszero[j] * bar[j] + ysplus[j] * bar[j];
      double bqua  = (1.0 / 3.0) * (yquarku[j] * bar[j] + yquarkd[j] * bar[j] + yquarks[j] * bar[j]);
      double btot  = bnuc + bhyp + bqua;

      double qlep  = -yelect[j] * bar[j] - ymuon[j] * bar[j];
      double qnuc  = yprot[j] * bar[j];
      double qhyp  = ysplus[j] * bar[j] - ysminus[j] * bar[j];
      double qqua  = (1.0 / 3.0) * (2.0 * yquarku[j] * bar[j] - yquarkd[j] * bar[j] - yquarks[j] * bar[j]);
      double qtot  = qlep + qnuc + qhyp + qqua;

      if (std::abs(btot - 1.0) >= 1e-2) {
        std::cerr << std::fixed << std::setprecision(3);
        std::cerr << "Charge neutrality violated at index " << j << " (rho = " << rrho[j]
                  << "): charge/fm3 = " << qtot << "\n";
        std::cerr << "btot = " << btot << " != 1.0\n";
        icore = 0;
        return;
      }

      if (std::abs(qtot) >= 1e-2) {
        std::cerr << std::fixed << std::setprecision(3);
        std::cerr << "Baryons do not sum up to baryon density at index " << j << " (rho = " << rrho[j]
                  << "): sum(baryons)/baryon# = " << btot << "\n";
        icore = 0;
        return;
      }
    }

    // Calculate the Fermi momenta in the crust
    for (int j = icore + 1; j <= imax; ++j) {
        double ne = yelect[j] * bar[j];
        double nn = yneutr[j] * bar[j];
        kfe[j] = std::pow(3.0 * pi * pi * ne, 1.0 / 3.0);
        kfm[j] = 0.0;
        kfn[j] = std::pow(3.0 * pi * pi * nn, 1.0 / 3.0);
        kfp[j] = 0.0;
        kfla[j] = 0.0;
        kfsm[j] = 0.0;
        kfs0[j] = 0.0;
        kfsp[j] = 0.0;
        kfqu[j] = 0.0;
        kfqd[j] = 0.0;
        kfqs[j] = 0.0;
    }

    // Check for direct Urca process (placeholder for actual implementation)
    // This section is left as a placeholder since the Fortran code has conditional blocks that
    // are not fully implemented. You can implement similar logic based on your specific requirements.
    /*
    if (some_condition) {
        // Implement direct Urca process checks and computations
    }
    */

    return;
  }
}



void get_fermi_momenta(int irank, int& imax, int& icore,
        const std::vector<double>& rrho, const std::vector<double>& bar,
        const std::vector<double> yneutr, const std::vector<double> yprot,
        const std::vector<double> yelect, const std::vector<double> ymuon,
        const std::vector<double> ylambda, const std::vector<double> ysminus,
        const std::vector<double> yszero, const std::vector<double> ysplus,
        const std::vector<double> yquarku, const std::vector<double> yquarkd,
        const std::vector<double> yquarks, const std::vector<double> fhad,
        const std::vector<double>& theta_k, const std::vector<double>& theta_p,
        std::vector<double>& kfn, std::vector<double>& kfp,
        std::vector<double>& kfe, std::vector<double>& kfm,
        std::vector<double>& kfla, std::vector<double>& kfsm,
        std::vector<double>& kfs0, std::vector<double>& kfsp,
        std::vector<double>& kfqu, std::vector<double>& kfqd,
        std::vector<double>& kfqs, std::vector<int>& idurca_np,
        std::vector<int>& idurca_lap, std::vector<double>& durca_ctrl_e,
        std::vector<double>& durca_ctrl_m, std::vector<int>& idurca_smn,
        std::vector<int>& idurca_smla, std::vector<int>& idurca_sms0,
        std::vector<int>& idurca_quqd, std::vector<int>& idurca_quqs,
        std::vector<double>& durca_henon_e, std::vector<double>& durca_henon_m) {

  const double pi = 3.14159265358979323846;
  const double coeff = 3.0*pi*pi;

  // Temporary variables
  double nn, np, nla, nsm, ns0, nsp;
  double nqu, nqd, nqs;
  double ne, nm;
  double charge_l, charge_h, charge_q, charge, baryon_h, baryon_q, baryon, barrel;
  double s, alpha_durca_frac, beta_durca_frac;
  double nb_low, nb_high, nb_durca_e, nb_durca_m;

  // Loop over j=0 to icore to calculate Fermi momenta in the core
  for(int j = 0; j <= icore; ++j) {
    // Calculate fermion densities based on fhad[j]
    if(fhad[j] != 0.0) {
        nn =std::max(0.0, yneutr[j] *bar[j])/fhad[j];
        np =std::max(0.0, yprot[j]  *bar[j])/fhad[j];
        nla=std::max(0.0, ylambda[j]*bar[j])/fhad[j];
        nsm=std::max(0.0, ysminus[j]*bar[j])/fhad[j];
        ns0=std::max(0.0, yszero[j] *bar[j])/fhad[j];
        nsp=std::max(0.0, ysplus[j] *bar[j])/fhad[j];
    } else {
        nn = np = nla = nsm = ns0 = nsp = 0.0;
    }

    if(fhad[j] != 1.0) {
        double denominator = 1.0 - fhad[j];
        nqu=std::max(0.0, yquarku[j]*bar[j])/denominator;
        nqd=std::max(0.0, yquarkd[j]*bar[j])/denominator;
        nqs=std::max(0.0, yquarks[j]*bar[j])/denominator;
    } else {
        nqu = nqd = nqs = 0.0;
    }

    // Calculate Fermi momenta using cube roots for performance
    kfn[j] =std::cbrt(3.0*pi*pi*nn);
    kfp[j] =std::cbrt(3.0*pi*pi*np);
    kfla[j]=std::cbrt(3.0*pi*pi*nla);
    kfsm[j]=std::cbrt(3.0*pi*pi*nsm);
    kfs0[j]=std::cbrt(3.0*pi*pi*ns0);
    kfsp[j]=std::cbrt(3.0*pi*pi*nsp);
    kfqu[j]=std::cbrt(3.0*pi*pi*nqu);
    kfqd[j]=std::cbrt(3.0*pi*pi*nqd);
    kfqs[j]=std::cbrt(3.0*pi*pi*nqs);

    // Calculate electron and muon densities and their Fermi momenta
    ne = std::abs(yelect[j] * bar[j]);
    nm = std::abs(ymuon[j] * bar[j]);

    kfe[j] = std::cbrt(3.0 * pi * pi * ne);
    if(yelect[j] <= 0.0)
        kfe[j] = -kfe[j];

    kfm[j] = std::cbrt(3.0 * pi * pi * nm);
    if(ymuon[j] <= 0.0)
        kfm[j] = -kfm[j];

    // Verify charge neutrality and baryon number conservation
    double nn_new =std::pow(kfn[j], 3.0)/coeff*fhad[j];
    double np_new =std::pow(kfp[j], 3.0)/coeff*fhad[j];
    double nla_new=std::pow(kfla[j], 3.0)/coeff*fhad[j];
    double nsm_new=std::pow(kfsm[j], 3.0)/coeff*fhad[j];
    double ns0_new=std::pow(kfs0[j], 3.0)/coeff*fhad[j];
    double nsp_new=std::pow(kfsp[j], 3.0)/coeff*fhad[j];
    double nqu_new=std::pow(kfqu[j], 3.0)/coeff*(1.0-fhad[j]);
    double nqd_new=std::pow(kfqd[j], 3.0)/coeff*(1.0-fhad[j]);
    double nqs_new=std::pow(kfqs[j], 3.0)/coeff*(1.0-fhad[j]);
    double ne_new =std::pow(kfe[j], 3.0)/coeff*1.0;
    double nm_new =std::pow(kfm[j], 3.0)/coeff*1.0;

    charge_l=-ne_new - nm_new;
    charge_h=np_new + nsp_new - nsm_new;
    charge_q=(2.0 / 3.0 * nqu_new) - (1.0 / 3.0 * nqd_new) - (1.0 / 3.0 * nqs_new);
    charge  =charge_l + charge_h + charge_q;

    baryon_h=nn_new + np_new + nla_new + nsm_new + ns0_new + nsp_new;
    baryon_q=(nqu_new + nqd_new + nqs_new) / 3.0;
    baryon  =baryon_h + baryon_q;

    // Check for charge neutrality violation
    if(std::abs(charge) >= 1e-2) {
      std::cerr << "Charge neutrality violated at j=" << j
                << ", Rho= " << rrho[j]
                << ": charge/fm3= " << charge << std::endl;
      icore = 0;
      return;
    }

    // Check for baryon number conservation
    barrel = baryon / bar[j];
    if(std::abs(barrel - 1.0) >= 1e-2) {
      std::cerr << "Baryons do not sum up to baryon density at j=" << j
                << ", Rho= " << rrho[j]
                << ": sum(baryons)/baryon#= " << baryon << std::endl;
      icore = 0;
      return;
    }
  }

  // Calculate the Fermi momenta in the crust
  for(int j = icore + 1; j <= imax; ++j) {
    ne=yelect[j]*bar[j];
    nn=yneutr[j]*bar[j];

    kfe[j] =std::cbrt(3.0*pi*pi*ne);
    kfm[j] =0.0;
    kfn[j] =std::cbrt(3.0*pi*pi*nn);
    kfp[j] =0.0;
    kfla[j]=0.0;
    kfsm[j]=0.0;
    kfs0[j]=0.0;
    kfsp[j]=0.0;
    kfqu[j]=0.0;
    kfqd[j]=0.0;
    kfqs[j]=0.0;
  }

  // Handle Direct Urca Process
  // The original Fortran code has a condition `if (.FALSE.)` which implies the else block is always executed
  // Implementing the 'else' part here

  // New method for neutron-proton direct Urca
  fix_durca = 0.0;
  alpha_durca_frac = 1.0e-8;
  beta_durca_frac  = 1.0;

  // Get direct Urca settings
  urca_settings(fix_durca, alpha_durca_frac, beta_durca_frac);

  // Compute triangle squared areas
  for(int j = 0; j <= icore; ++j) {
    // Calculate semi-perimeter s for electrons
    double s_e = (kfp[j] + kfn[j] + kfe[j]) / 2.0;
    durca_henon_e[j] = s_e * (s_e - kfp[j]) * (s_e - kfe[j]) * (s_e - kfn[j]);

    // Calculate semi-perimeter s for muons
    double s_m = (kfp[j] + kfn[j] + kfm[j]) / 2.0;
    durca_henon_m[j] = s_m * (s_m - kfp[j]) * (s_m - kfm[j]) * (s_m - kfn[j]);
  }

  // Compute direct Urca density thresholds using linear interpolation
  if(fix_durca == 0.0) {
    nb_durca_e = 0.0;
    nb_durca_m = 0.0;

    for(int j = 0; j < icore; ++j) {
      // Find where durca_henon_e changes sign
      if(nb_durca_e == 0.0 && (durca_henon_e[j] * durca_henon_e[j+1] < 0.0)) {
        double slope_e = durca_henon_e[j+1] - durca_henon_e[j];
        if(slope_e != 0.0) {
          nb_low  = bar[j];
          nb_high = bar[j+1];
          nb_durca_e = nb_low - (nb_high - nb_low) * durca_henon_e[j] / slope_e;
        }
      }
      // Find where durca_henon_m changes sign
      if(nb_durca_m == 0.0 && (durca_henon_m[j] * durca_henon_m[j+1] < 0.0)) {
        double slope_m = durca_henon_m[j+1] - durca_henon_m[j];
        if(slope_m != 0.0) {
          nb_low  = bar[j];
          nb_high = bar[j+1];
          nb_durca_m = nb_low - (nb_high - nb_low) * durca_henon_m[j] / slope_m;
        }
      }
      // Exit early if both thresholds are found
      if(nb_durca_e > 0.0 && nb_durca_m > 0.0)
          break;
    }
  }
  else {
    nb_durca_e = fix_durca;
    nb_durca_m = fix_durca;
  }

  // Compute direct Urca control functions from threshold densities
  for(int j = 0; j <= icore; ++j) {
    idurca_np[j]      = 0;
    durca_ctrl_e[j]   = 0.0;
    durca_ctrl_m[j]   = 0.0;

    // Electron direct Urca control
    if(nb_durca_e > 0.0) {
      if(bar[j] >= (1.0 + alpha_durca_frac) * beta_durca_frac * nb_durca_e) {
        idurca_np[j]    = 1;
        durca_ctrl_e[j] = 1.0;
      }
      else if(bar[j] >= (1.0 - alpha_durca_frac) * beta_durca_frac * nb_durca_e) {
        idurca_np[j]    = 3;
        durca_ctrl_e[j] = 0.5 + (1.0 / (2.0 * alpha_durca_frac)) * (bar[j] - nb_durca_e) / nb_durca_e;
      }
    }

    // Muon direct Urca control
    if(nb_durca_m > 0.0) {
      if(bar[j] >= (1.0 + alpha_durca_frac) * beta_durca_frac * nb_durca_m) {
        idurca_np[j]    = 2;
        durca_ctrl_m[j] = 1.0;
      }
      else if(bar[j] >= (1.0 - alpha_durca_frac) * beta_durca_frac * nb_durca_m) {
        idurca_np[j]    = 4;
        durca_ctrl_m[j] = 0.5 + (1.0 / (2.0 * alpha_durca_frac)) * (bar[j] - nb_durca_m) / nb_durca_m;
      }
    }
  }

  // Dany's method for hyperon and quark direct Urca
  for(int j = 0; j <= icore; ++j) {
    // la-p
    if((kfp[j] < kfla[j] + kfe[j]) &&
      (kfla[j] < kfp[j] + kfe[j]) &&
      (kfe[j]  < kfp[j] + kfla[j])) {
      idurca_lap[j] = 1;
      if((kfp[j] < kfla[j] + kfm[j]) &&
        (kfla[j] < kfp[j] + kfm[j]) &&
        (kfm[j]  < kfla[j] + kfp[j])) {
        idurca_lap[j] = 2;
       }
    }
    else {
      idurca_lap[j] = 0;
    }

    // sm-n
    if((kfsm[j] < kfn[j] + kfe[j]) &&
      (kfn[j] < kfsm[j] + kfe[j]) &&
      (kfe[j]  < kfsm[j] + kfn[j])) {
      idurca_smn[j] = 1;
      if((kfsm[j] < kfn[j] + kfm[j]) &&
        (kfn[j] < kfsm[j] + kfm[j]) &&
        (kfm[j]  < kfsm[j] + kfn[j])) {
        idurca_smn[j] = 2;
      }
    }
    else {
      idurca_smn[j] = 0;
    }

    // sm-la
    if((kfsm[j] < kfla[j] + kfe[j]) &&
      (kfla[j] < kfsm[j] + kfe[j]) &&
      (kfe[j]  < kfsm[j] + kfla[j])) {
      idurca_smla[j] = 1;
      if((kfsm[j] < kfla[j] + kfm[j]) &&
        (kfla[j] < kfsm[j] + kfm[j]) &&
        (kfm[j]  < kfsm[j] + kfla[j])) {
        idurca_smla[j] = 2;
      }
    }
    else {
      idurca_smla[j] = 0;
    }

    // sm-s0
    if((kfsm[j] < kfs0[j] + kfe[j]) &&
      (kfs0[j] < kfsm[j] + kfe[j]) &&
      (kfe[j]  < kfsm[j] + kfs0[j])) {
      idurca_sms0[j] = 1;
      if((kfsm[j] < kfs0[j] + kfm[j]) &&
        (kfs0[j] < kfsm[j] + kfm[j]) &&
        (kfm[j]  < kfsm[j] + kfs0[j])) {
        idurca_sms0[j] = 2;
      }
    }
    else {
      idurca_sms0[j] = 0;
    }

    // qu-qd
    if((kfqu[j] < kfqd[j] + kfe[j]) &&
      (kfqd[j] < kfqu[j] + kfe[j]) &&
      (kfe[j]  < kfqu[j] + kfqd[j])) {
      idurca_quqd[j] = 1;
      if((kfqu[j] < kfqd[j] + kfm[j]) &&
        (kfqd[j] < kfqu[j] + kfm[j]) &&
        (kfm[j]  < kfqu[j] + kfqd[j])) {
        idurca_quqd[j] = 2;
      }
    }
    else {
      idurca_quqd[j] = 0;
    }

    // qu-qs
    if((kfqu[j] < kfqs[j] + kfe[j]) &&
      (kfqs[j] < kfqu[j] + kfe[j]) &&
      (kfe[j]  < kfqu[j] + kfqs[j])) {
      idurca_quqs[j] = 1;
      if((kfqu[j] < kfqs[j] + kfm[j]) &&
        (kfqs[j] < kfqu[j] + kfm[j]) &&
        (kfm[j]  < kfqu[j] + kfqs[j])) {
        idurca_quqs[j] = 2;
      }
    }
    else {
      idurca_quqs[j] = 0;
    }
  }

  // Function completes successfully
  return;
}
#endif


};
#endif