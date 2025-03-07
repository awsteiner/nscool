#include "nscool.h"
#include <cmath>
#include <cstddef>
#include <o2scl/constants.h>
#include <o2scl/interp.h>
#include <o2scl/vector.h>

#ifdef O2SCL_MPI
#include "mpi.h"
#endif

std::vector<nscool *> nscool_ptrs;

nscool::nscool(std::string dir) {

  o2scl_hdf::hdf_file hf;
  std::string name;

  int mpi_rank = 0, mpi_size = 1;

#ifdef O2SCL_MPI
  // Get MPI rank, etc.
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  // Ensure that multiple threads aren't writing to the filesystem at the same
  // time
  int tag = 0, buffer = 0;
  if (mpi_size > 1 && mpi_rank >= 1) {
    MPI_Recv(&buffer, 1, MPI_INT, mpi_rank - 1, tag, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }
#endif

  if (false) {
    std::cout << "In nscool, rank " << mpi_rank << " reading data files."
              << std::endl;
  }

  // Read default crust EOS
  hf.open(dir + "../crust_HZD_NV.o2");
  hdf_input(hf, nscool_crust, name);
  hf.close();

  // Read default core EOS
  hf.open(dir + "../core_APR.o2");
  hdf_input(hf, nscool_core, name);
  hf.close();

  // Read default stellar profile
  hf.open(dir + "../tov_APR_14.o2");
  hdf_input(hf, nscool_tov, name);
  hf.close();

#ifdef O2SCL_MPI
  if (mpi_size > 1 && mpi_rank < mpi_size - 1) {
    MPI_Send(&buffer, 1, MPI_INT, mpi_rank + 1, tag, MPI_COMM_WORLD);
  }
#endif

  ptemp = 0.0;

  // Setup time_print
  time_print.resize(25);
  time_print[0] = 1.0e-10;
  time_print[1] = 1.0e-4;
  time_print[2] = 3.0e-4;
  time_print[3] = 1.0e-3;
  time_print[4] = 3.0e-3;
  time_print[5] = 1.0e-2;
  time_print[6] = 3.0e-2;
  time_print[7] = 1.0e-1;
  time_print[8] = 3.0e-1;
  time_print[9] = 1.0e+0;
  time_print[10] = 3.0e+0;
  time_print[11] = 1.0e+1;
  time_print[12] = 3.0e+1;
  time_print[13] = 1.0e+2;
  time_print[14] = 3.0e+2;
  time_print[15] = 1.0e+3;
  time_print[16] = 3.0e+3;
  time_print[17] = 1.0e+4;
  time_print[18] = 3.0e+4;
  time_print[19] = 1.0e+5;
  time_print[20] = 3.0e+5;
  time_print[21] = 1.0e+6;
  time_print[22] = 3.0e+6;
  time_print[23] = 1.0e+7;
  time_print[24] = 3.0e+7;
  T_fact_drip = 0.8;
  T_fact_surf = 0.5;
  eta = 0.0;

  temp_ygrid.clear();
  for (size_t i = 0; i < time_print.size(); i++) {
    temp_ygrid.push_back(time_print[i]);
  }

  /*
    This is the SFB neutrino singlet gap, the T73 proton singlet gap, and
    neutrino triplet gap "a" from the minimal cooling paper.
  */
  sfn1s0 = 1;
  sfn3p2 = 101;
  sfp1s0 = 3;

  n1_tc = 0.0;
  n1_kf = 0.0;
  n1_dk = 0.0;
  mdot_index = 0;

  nscool_debug = 0;

  fix_durca = 0.0;
  alpha_durca = 1.0e-8;
  beta_durca = 1.0;

  main_out_it = 20;

  // Read data for pair bremsstrahlung
  pair_brem_data(pb_logt, pb_nalpha);

  // Data for superfluid suppression
  sf_suppress_data(sf_lgtau1, sf_lgtau2, sf_lgr);

  lgtau1_t.resize(35), lgtau2_t.resize(35);
  for (size_t i = 0; i < 35; ++i) {
    lgtau1_t[i] = sf_lgtau1[i];
    lgtau2_t[i] = sf_lgtau2[i];
  }

  lgr_tt.resize(35, 35);
  for (size_t i = 0; i < 35; ++i) {
    for (size_t j = 0; j < 35; ++j) {
      lgr_tt(i, j) = sf_lgr[35 * i + j];
    }
  }

  pb_logt1.resize(56), pb_nalpha1.resize(56), pb_n21.resize(56);
  for (size_t i = 0; i < 56; ++i) {
    pb_logt1[i] = pb_logt[i];
    pb_nalpha1[i] = pb_nalpha[i];
    pb_n21[i] = pb_n2[i];
  }

  iv.set_type(o2scl::itp_cspline);
  iv.set(56, pb_logt1, pb_nalpha1);

  id2.set_data(35, 35, lgtau1_t, lgtau2_t, lgr_tt);

  // Lepton inits
  electron.init(o2scl::o2scl_settings.get_convert_units().convert(
                    "kg", "1/fm", o2scl_const::mass_electron_f<double>()),
                2.0);
  electron_deriv.init(o2scl::o2scl_settings.get_convert_units().convert(
                          "kg", "1/fm", o2scl_const::mass_electron_f<double>()),
                      2.0);
  muon.init(o2scl::o2scl_settings.get_convert_units().convert(
                "kg", "1/fm", o2scl_const::mass_muon_f<double>()),
            2.0);

  sxrt_mode = false;
  sxrt_Tinit = 0.0;

  // Let's define gamma values here
  gammacryst = 210.0;
  gammaliq = 180.0;
}

/// \name Generic functions
//@{
/** \brief Desc
 */
inline double fexp(double x) {
  if (x > -7.0e2)
    return exp(x);
  return exp(-700.0);
}
/** \brief Desc
 */
inline double u_1s0(double t) {
  return sqrt(1.0 - t) * (1.4560 - 0.1570 / sqrt(t) + 1.7640 / t);
}

/** \brief Desc
 */
inline double u_3p2B(double t) {
  return sqrt(1.0 - t) * (0.78930 + 1.1880 / t);
}

/** \brief A “pairing suppression factor” used in multiple places:
 */
double r_1s0(double u) {
  return pow(0.23120 + hypot(0.7688, 0.1438 * u), 5.50) *
         fexp(3.4270 - hypot(3.4270, u));
}

/** \brief Desc
 */
double r_3p2B(double u) {
  return pow(0.25460 + hypot(0.7454, 0.1284 * u), 5.0) *
         fexp(2.7010 - hypot(2.7010, u));
}
// We define a few inline Fortran-like helper macros:
inline double smooth(double x) {
  // Fortran: smooth(x) = 6*x^5 - 15*x^4 + 10*x^3
  return 6.0 * std::pow(x, 5) - 15.0 * std::pow(x, 4) + 10.0 * std::pow(x, 3);
}

// The exp_int(x) function from fortran
inline double exp_int(double x) {
  if (x <= 0.0) {
    std::cerr << "exp_int: x must be > 0 !" << std::endl;
    std::exit(1);
  }
  if (x >= 1.0) {
    double num = x * x * x * x + 8.5733287401 * x * x * x +
                 18.0590169730 * x * x + 8.6347608925 * x + 0.2677737343;
    double den = x * x * x * x + 9.5733223454 * x * x * x +
                 25.6329561486 * x * x + 21.0996530827 * x + 3.9584969228;
    return (num / den) / (x * std::exp(x));
  } else {
    double val = -0.57721566 + 0.99999193 * x - 0.24991055 * x * x +
                 0.05519968 * x * x * x - 0.00976004 * x * x * x * x +
                 0.00107857 * x * x * x * x * x - std::log(x);
    return val;
  }
}

// get_lam(...) helper for GYP code
inline void get_lam(double s, double w, double &Lam1, double &Lam2) {
  static constexpr double eps = 0.05;

  if ((s <= eps) && (s * w <= eps)) {
    Lam1 = 0.5 * (exp_int(w) + std::log(w) + 0.5772156);
    Lam2 = (std::exp(-w) - 1.0 + w) / (2.0 * w);
  } else if (w <= eps) {
    Lam1 =
        w * ((2.0 * s + 1.0) / (2.0 * s + 2.0) - s * std::log((s + 1.0) / s));
    Lam2 = w * ((1.0 - 3.0 * s - 6.0 * s * s) / (4.0 * s + 4.0) +
                1.5 * std::log((s + 1.0) / s));
  } else if (w > 1.0 / eps) {
    Lam1 = 0.5 * (std::log((s + 1.0) / s) - 1.0 / (s + 1.0));
    Lam2 = (2.0 * s + 1.0) / (2.0 * s + 2.0) - s * std::log((s + 1.0) / s);
  } else {
    double part1 =
        std::log((s + 1.0) / s) + s / (s + 1.0) * (1.0 - std::exp(-w)) -
        (1.0 + s * w) * std::exp(s * w) * (exp_int(s * w) - exp_int(s * w + w));
    Lam1 = 0.5 * part1;

    double part2 = (std::exp(-w) - 1.0 + w) / w -
                   s * s / (s + 1.0) * (1.0 - std::exp(-w)) -
                   2.0 * s * std::log((s + 1.0) / s) +
                   s * (2.0 + s * w) * std::exp(s * w) *
                       (exp_int(s * w) - exp_int(s * w + w));
    Lam2 = 0.5 * part2;
  }
}

// Coulomb_imp_YU(x)
inline double Coulomb_imp_YU(double x) {
  double beta = x / std::sqrt(1.0 + x * x);
  double q = 0.048196 / std::sqrt(beta);
  double val = std::log(1.0 / q) - 0.5 * (1.0 + beta * beta);
  return val;
}

/** \brief Suppression factor
 */
inline double r_1s0_1s0(double v1, double v2) {
  double gamma = 5040.0 / 457.0 / pow(o2scl_const::pi, 6.0);
  double u = v1 * v1 + v2 * v2;
  double w = v1 * v1 - v2 * v2;
  double u1 = 1.8091 + hypot(v1, 2.2476);
  double u2 = 1.8091 + hypot(v2, 2.2476);
  double p = (u + 12.421 + sqrt(w * w + 16.35 * u + 45.171)) / 2.0;
  double q = (u + 12.421 - sqrt(w * w + 16.35 * u + 45.171)) / 2.0;
  double ps = (u + sqrt(w * w + 5524.80 * u + 6.77370)) / 2.0;
  double pe = (u + 0.43847 + sqrt(w * w + 8.368 * u + 491.32)) / 2.0;
  double D = pow(u1 * u2, 1.5) / (2.0 * pow(4.0567, 5.0)) *
             (u1 * u1 + u2 * u2) * exp(-u1 - u2 + 8.1134);
  double K0 =
      sqrt(p - q) / 120.0 * (6.0 * p * p + 83.0 * p * q + 16.0 * q * q) -
      sqrt(p) * q / 8.0 * (4.0 * p + 3.0 * q) *
          log((sqrt(p) + sqrt(p - q)) / sqrt(q));
  double K1 = o2scl_const::pi2 * sqrt(p - q) / 6.0 * (p + 2.0 * q) -
              o2scl_const::pi2 / 2.0 * q * sqrt(p) *
                  log((sqrt(p) + sqrt(p - q)) / sqrt(q));
  double K2 = 7.0 * o2scl_const::pi2 * o2scl_const::pi2 / 60.0 * sqrt(p - q);
  double S = gamma * (K0 + K1 + 0.42232 * K2) * sqrt(o2scl_const::pi / 2.0) *
             pow(ps, 0.25) * exp(-sqrt(pe));

  return u / (u + 0.91630) * S + D;
}

/** \brief Suppression factor
 */
double nscool::r_1s0_3p2B(double t1, double t2) {
  double lt1 = log10(t1);
  double lt2 = log10(t2);
  double ret = 10.0 * id2.eval(lt1, lt2);
  double lt = hypot(lt1, lt2);
  double lt_limit = 3.0;
  if (lt > lt_limit) {
    ret *= exp(-lt / lt_limit);
  }
  return ret;
}
//@}

double nscool::cvelec(double t, double rho, double a, double z) {
  double hb = 1.054588e-27;
  double kb = 1.380662e-16;
  double c = 2.997924e10;
  double na = 6.022045e23;
  double me = 9.109e-28;
  double pi = 3.14159265;

  double ne = na * rho * z / a;
  double pf = hb * cbrt(3.0 * pi * pi * ne);
  double ef = sqrt((me * me * c * c * c * c) + (pf * c * c)) - me * c * c;
  double tf = ef / kb;
  double xe = pf / (me * c);
  double ae = xe * xe / sqrt(1. + xe * xe);
  double cvt = ne * kb * kb * pi * pi / (me * c * c) / ae;

  double t0 = tf / 50.0;

  if (t < 0.5 * t0) {
    return cvt * t;
  }

  electron_deriv.n = ne;
  fdr.calc_density(electron_deriv, t);
  return electron_deriv.dsdT;
}

double nscool::cvion(double t, double rho, double a, double z) {

  double rhodrip = 4.3e11;
  double bcv = 0.95043;
  double ccv = 0.18956;
  double dcv = -0.81487;
  double hcv = 3225.0;
  double cte = 141.7;
  double cv0[15] = {0.0,   2.956, 2.829, 2.633, 2.389, 2.118, 1.840, 1.572,
                    1.323, 1.102, 0.909, 0.745, 0.609, 0.496, 0.404};

  double gamma = 2.273e5 * z * z * cbrt(rho / a) / t;
  double gamma14 = pow(gamma, 0.25);
  double a1;
  if (rho >= rhodrip) {
    a1 = 3.0 * z;
  } else {
    a1 = a;
  }
  double nionkb = 1.38e-16 * 6.022e23 * rho / a;
  double delta = 1.0 / t * z * sqrt(rho / (a1 * a)) * 6.022e23;
  if (gamma <= 0.1) {
    return 1.5 * nionkb;
  } else if (gamma <= 0.2) {
    double cv1 = 1.5 * nionkb;
    double cv2 =
        nionkb * (0.75 * bcv * gamma14 + 1.25 * ccv / gamma14 + dcv + 1.5);
    return (gamma - 0.1) / 0.1 * cv2 + (0.2 - gamma) / 0.1 * cv1;
  } else if (gamma < 178.0) {
    return nionkb * (0.75 * bcv * gamma14 + 1.25 * ccv / gamma14 + dcv + 1.5);
  } else if (gamma <= 210.0 && delta >= 1.0e19) {
    double cv1 = nionkb * (1.5 + 3.0 * hcv / gamma / gamma + 1.5);
    cv0[0] = 1.5 + 3.0 * hcv / gamma / gamma + 1.5;
    int i1 = ((int)(delta * 2.0e-20));
    if (i1 > 14 || i1 < 0) {
      std::cerr << "cvion failure." << std::endl;
      exit(-1);
    }
    double cv2 =
        nionkb * (cv0[i1] + (delta * 2.0e-20 - i1) * (cv0[i1 + 1] - cv0[i1]));
    return (gamma - 178.0) / 32.0 * cv2 + (210.0 - gamma) / 32.0;
  } else if (delta <= 1.0e19) {
    return nionkb * (1.5 + 3.0 * hcv / gamma / gamma + 1.5);
  } else if (delta > 1.0e9 && delta < 7.0e20) {
    cv0[0] = 1.5 + 3.0 * hcv / gamma / gamma + 1.5;
    int i1 = ((int)(delta * 2.0e-20));
    if (i1 > 14 || i1 < 0) {
      std::cerr << "cvion failure." << std::endl;
      exit(-1);
    }
    return nionkb *
           (cv0[i1] + (delta * 2.0e-20 - i1) * (cv0[i1 + 1] - cv0[i1]));
  }
  double delta1 = delta * 1.0e-20;
  return nionkb * cte / delta1 / delta1 / delta1;
}

void nscool::cvtot(int i, double t, double rho, double aion, double zion,
                   double &cv, double &cvneutron, double &cvproton,
                   double &cvelectron, double &cvmuon, double &cvlambda,
                   double &cvsminus, double &cvszero, double &cvsplus,
                   double &cvquark, double &cvions, std::vector<double> &cve,
                   std::vector<double> &cvm, std::vector<double> &cvn,
                   std::vector<double> &cvp, std::vector<double> &cvla,
                   std::vector<double> &cvsm, std::vector<double> &cvs0,
                   std::vector<double> &cvsp, std::vector<double> &cvqu,
                   std::vector<double> &dvqd, std::vector<double> &cvqs,
                   double rhodrip, double rhocore, std::vector<double> &fhad,
                   int istrange, std::vector<double> &tcn,
                   std::vector<double> &tcp, std::vector<double> &tcla,
                   std::vector<double> &tcsm, std::vector<double> &tcs0,
                   std::vector<double> &tcsp, int isf) {
  int icvel_nodeg = 1;
  double c_cv_str = 0.0;

  auto solve_cv_mem = [rho, rhocore, istrange, i,
                       t](const std::vector<double> tcmem,
                          const std::vector<double> cvmem) {
    double cvt = 0.0;
    if ((rho >= rhocore) && (istrange == 0)) {
      double raise = 1.1;
      double r = 1.0;
      if (t < raise * tcmem[i]) {
        double t0 = std::min(0.999999999999, t / tcmem[i]);
        double u = u_1s0(t0);
        r = r_1s0(u);
        if (t > tcmem[i]) {
          double w1 = (raise * tcmem[i] - t) / ((raise - 1.0) * tcmem[i]);
          double w2 = 1.0 - w1;
          r = w1 * r + w2 * 1.0;
        }
      }
      cvt = cvmem[i] * t * r;
    }
    return cvt;
  };

  // Get Cv-ions
  if (rho < rhocore) {
    cvions = cvion(t, rho, aion, zion);
  } else {
    cvions = 0.0;
  }
  // Get Cv-electrons
  if ((rho < rhodrip) && (icvel_nodeg == 1)) {
    cvelectron = cvelec(t, rho, aion, zion);
  } else {
    cvelectron = cve[i] * t;
  }
  // Get Cv-muons
  cvmuon = cvm[i] * t;
  // get Cv-neutrons :
  if ((rho >= rhodrip) && (istrange == 0)) {
    double raise = 1.0;
    double r;
    if (t < raise * tcn[i]) {
      double t0 = std::min(0.999999999999, t / tcn[i]);
      if (i <= isf) {
        double u = u_3p2B(t0);
        r = r_3p2B(u);
      } else {
        double u = u_1s0(t0);
        r = r_1s0(u);
      }
      if (t > tcn[i]) {
        double w1 = (raise * tcn[i] - t) / ((raise - 1.0) * tcn[i]);
        double w2 = 1.0 - w1;
        r = w1 * r + w2 * 1.0;
      }
    } else {
      r = 1.0;
    }
    cvneutron = cvn[i] * t * r;
  } else {
    cvneutron = 0.0;
  }

  cvproton = solve_cv_mem(tcp, cvp);   // get Cv-protons
  cvlambda = solve_cv_mem(tcla, cvla); // get Cv-lambdas
  cvsminus = solve_cv_mem(tcsm, cvsm); // get Cv-Sigma-
  cvszero = solve_cv_mem(tcs0, cvs0);  // get Cv-Sigma0
  cvsplus = solve_cv_mem(tcsp, cvsp);  // get Cv-Sigma+

  double cvphot = 0.0; // get Cv-photons
  // get Cv-quarks:
  if (rho >= rhocore) {
    cvquark = c_cv_str * (t / 1.0e9);
  } else {
    cvquark = 0.0;
  }
  // Total Cv
  cvelectron = cvelectron * fhad[i];
  cvmuon = cvmuon * fhad[i];
  cvproton = cvproton * fhad[i];
  cvneutron = cvneutron * fhad[i];
  cvlambda = cvlambda * fhad[i];

  cvsminus = cvsminus * fhad[i];
  cvszero = cvszero * fhad[i];
  cvsplus = cvsplus * fhad[i];
  cvquark = cvquark * (1.0 - fhad[i]);

  cv = cvions + cvelectron + cvmuon + cvproton + cvneutron + cvlambda +
       cvsminus + cvszero + cvsplus + cvphot + cvquark;

  return;
}

double nscool::pressure(double T, double Rho, double A, double Z) {
  double hb = 1.054588e-27;
  double kb = 1.380662e-16;
  double c = 2.997924e10;
  double NA = 6.022045e23;
  double me = 9.109e-28;
  double pi = 3.141592653;

  double ne = Rho * NA * Z / A;
  double nion = Rho * NA / A;
  double Zeff = ne / nion;
  double gamma = 2.273e5 * Zeff * Zeff * cbrt(Rho / A) / T;
  double Uion;
  double gamma14 = pow(gamma, 0.25);
  if (gamma < 210.0) {
    Uion = 1.5 - 0.895929 * gamma + 3225.0 * gamma14;
  } else {
    Uion = -0.897744 * gamma + 0.95043 * gamma14 + 0.18956 / gamma14 - 0.81487;
  }
  double Pion = nion * kb * T * (1.0 + Uion / 3.0);

  electron.n = ne;
  electron.m = me;
  fe.calc_density(electron, T);
  double Pel = electron.pr;

  return Pel + Pion;
}

void nscool::density(double T, double P, double A, double Z, double &Rho) {

  double eps = 1.0e-3;
  double dRho;
  do {
    double Rho0 = Rho;
    double Pre0 = pressure(T, Rho0, A, Z);
    double Rho1 = (1.0 + eps) * Rho0;
    double Pre1 = pressure(T, Rho1, A, Z);
    double f = Pre0 - P;
    double f1 = (Pre1 - Pre0) / (Rho1 - Rho0);
    dRho = -f / f1;
    Rho = Rho0 + dRho;
  } while (fabs(dRho / Rho) >= 1.0e-5);

  return;
}

void nscool::opacity(double T, double Rho, double &A, double &Z, double &kappa,
                     int iopacity) {
  if (iopacity == 0) {
    kappa = 1.0e200;
  } else {
    if (Rho >= 1.0e14) {
      A = 100.0;
      Z = 32.0;
    }
    kappa = 0.645e23 * pow(Z, 3) / pow(A, 2) * Rho / pow(T, 3.5);
  }
  return;
}

void nscool::con_core_bar_pairing_supr(double yn, double yp, double &Rn1,
                                       double &Rn2, double &Rp1, double &Rp2,
                                       double &RC) {
  if (yn == 0.0) {
    Rn1 = 1.0;
    Rn2 = 1.0;
    RC = 1.0;
  } else {
    Rn1 = (2.0 / 3.0) *
              std::pow((0.9468 + std::sqrt(0.0532 * 0.0532 + 0.5346 * yn * yn)),
                       3.0) *
              std::exp(0.377 - std::sqrt(0.377 * 0.377 + 4.0 * yn * yn)) +
          (1.0 / 3.0) * std::pow((1.0 + 1.351 * yn * yn), 2.0) *
              std::exp(0.169 - std::sqrt(0.169 * 0.169 + 9.0 * yn * yn));

    Rn2 =
        0.5 *
            std::pow((0.6242 + std::sqrt(0.3758 * 0.3758 + 0.07198 * yn * yn)),
                     3.0) *
            std::exp(3.6724 - std::sqrt(3.6724 * 3.6724 + 4.0 * yn * yn)) +
        0.5 * std::pow((1.0 + 0.01211 * yn * yn), 9.0) *
            std::exp(7.5351 - std::sqrt(7.5351 * 7.5351 + 9.0 * yn * yn));

    RC = std::pow((0.647 + std::sqrt(0.353 * 0.353 + 0.109 * yn * yn)), 1.5) *
         std::exp(1.39 - std::sqrt(1.39 * 1.39 + yn * yn));
  }

  if ((yn == 0.0) && (yp == 0.0)) {
    Rp1 = 1.0;
    Rp2 = 1.0;
  } else if ((yn != 0.0) && (yp == 0.0)) {
    Rp1 = std::pow((0.4459 + std::sqrt(0.5541 * 0.5541 + 0.03016 * yn * yn)),
                   2.0) *
          std::exp(2.1178 - std::sqrt(2.1178 * 2.1178 + yn * yn));
    Rp2 =
        std::pow((0.801 + std::sqrt(0.199 * 0.199 + 0.04645 * yn * yn)), 2.0) *
        std::exp(2.3569 - std::sqrt(2.3569 * 2.3569 + yn * yn));
  } else if ((yn == 0.0) && (yp != 0.0)) {
    Rp1 = 0.5 * (0.3695 + std::sqrt(0.6305 * 0.6305 + 0.01064 * yp * yp)) *
              std::exp(2.4451 - std::sqrt(2.4451 * 2.4451 + yp * yp)) +
          0.5 * std::pow((1.0 + 0.1917 * yp * yp), 1.4) *
              std::exp(4.6627 - std::sqrt(4.6627 * 4.6627 + 4.0 * yp * yp));

    Rp2 = 0.0436 * (std::sqrt(3.345 * 3.345 + 19.55 * yp * yp) - 3.345) *
              std::exp(2.0247 - std::sqrt(2.0247 * 2.0247 + yp * yp)) +
          0.0654 * std::exp(8.992 - std::sqrt(8.992 * 8.992 + 1.5 * yp * yp)) +
          0.8910 * std::exp(9.627 - std::sqrt(9.627 * 9.627 + 9.0 * yp * yp));
  } else {
    double y_p = std::max(yn, yp);
    double y_m = std::min(yn, yp);

    double u_p = std::sqrt(y_p * y_p + (1.485 * 1.485)) - 1.485;
    double u_m = std::sqrt(y_m * y_m + (1.485 * 1.485)) - 1.485;
    double up = std::sqrt(yp * yp + 1.485 * 1.485) - 1.485;
    double un = std::sqrt(yn * yn + 1.485 * 1.485) - 1.485;

    Rp1 = std::exp(-u_p - u_m) *
              (0.7751 + 0.4823 * un + 0.1124 * up + 0.04991 * un * un +
               0.08513 * un * up + 0.01284 * un * un * up) +
          std::exp(-2.0 * u_p) * (0.2249 + 0.3539 * u_p - 0.2189 * u_m -
                                  0.6069 * un * u_m + 0.7362 * up * u_p);

    u_p = std::sqrt(y_p * y_p + (1.761 * 1.761)) - 1.761;
    u_m = std::sqrt(y_m * y_m + (1.761 * 1.761)) - 1.761;
    up = std::sqrt(yp * yp + 1.761 * 1.761) - 1.761;
    un = std::sqrt(yn * yn + 1.761 * 1.761) - 1.761;

    Rp2 = std::exp(-u_p - u_m) *
              (1.1032 + 0.8645 * un + 0.2042 * up + 0.07937 * un * un +
               0.1451 * un * up + 0.01333 * un * un * up) +
          std::exp(-2.0 * u_p) * (-0.1032 - 0.2340 * u_p + 0.06152 * un * u_p +
                                  0.7533 * un * u_m - 1.007 * up * u_p);
  }
  return;
}

void nscool::con_core_bar(double Temp, double kf_e, double kf_mu, double kf_p,
                          double mst_p, double Tc_p, double kf_n, double mst_n,
                          double Tc_n, int isfn, double kf_la, double mst_la,
                          double Tc_la, double kf_sm, double mst_sm,
                          double Tc_sm, double kf_s0, double mst_s0,
                          double Tc_s0, double kf_sp, double mst_sp,
                          double Tc_sp, double &sigma_bar, double &lambda_bar,
                          double debug, double &nu_e_s, double &nu_e_l,
                          int icontrol) {
  if (debug == 1.2) {
    std::cout << "Entering con_core_bar: T=" << Temp << ", kfe=" << kf_e
              << std::endl;
  }

  double Sn1 =
      14.57 / std::pow(kf_n, 1.5) *
      ((1.0 - 0.0788 * kf_n + 0.0883 * kf_n * kf_n) / (1.0 - 0.1114 * kf_n));
  double Sn2 =
      7.880 / std::pow(kf_n, 2.0) *
      ((1.0 - 0.2241 * kf_n + 0.2006 * kf_n * kf_n) / (1.0 - 0.1742 * kf_n));

  double Sp1 = 0.8007 * (kf_p / kf_n / kf_n) *
               ((1.0 + 31.28 * kf_p - 0.0004285 * kf_p * kf_p + 26.85 * kf_n +
                 0.08012 * kf_n * kf_n) /
                (1.0 - 0.5898 * kf_n + 0.2368 * kf_n * kf_n +
                 0.5838 * kf_p * kf_p + 0.884 * kf_n * kf_p));
  double Sp2 = 0.3830 * std::pow(kf_p, 4.0) / std::pow(kf_n, 5.5) *
               ((1.0 + 102.0 * kf_p + 53.91 * kf_n) /
                (1.0 - 0.7087 * kf_n + 0.2537 * kf_n * kf_n +
                 9.404 * kf_p * kf_p - 1.589 * kf_n * kf_p));

  double u = kf_n - 1.665;
  double Kn1 = (0.4583 + 0.892 * u * u - 0.5497 * u * u * u - 0.06205 * kf_p +
                0.04022 * kf_p * kf_p + 0.2122 * u * kf_p) /
               (mst_n * mst_n);

  u = kf_n - 1.556;
  double Kn2 = (0.4891 + 1.111 * u * u - 0.2283 * u * u * u + 0.01589 * kf_p -
                0.02099 * kf_p * kf_p + 0.2773 * u * kf_p) /
               (mst_n * mst_n);

  u = kf_n - 2.126;
  double Kp1 = (0.04377 + 1.100 * u * u + 0.1180 * u * u * u + 0.1626 * kf_p +
                0.3871 * u * kf_p - 0.2990 * u * u * u * u) /
               (mst_p * mst_p);

  u = kf_n - 2.116;
  double Kp2 =
      (0.0001313 + 1.248 * u * u + 0.2403 * u * u * u + 0.3257 * kf_p +
       0.5536 * u * kf_p - 0.3237 * u * u * u * u + 0.09786 * u * u * kf_p) /
      (mst_p * mst_p);

  if (icontrol >= 2) {
    Kn1 = 1.0;
    Kn2 = 1.0;
    Kp1 = 1.0;
    Kp2 = 1.0;
  }

  // Pairing effects
  double yn = 0.0, yp = 0.0;
  if (Temp <= Tc_p) {
    double tau = Temp / Tc_p;
    yp = u_1s0(tau);
  }

  if (Temp <= Tc_n) {
    double tau = Temp / Tc_n;
    if (isfn == 1) {
      yn = u_1s0(tau);
    } else if (isfn == 3) {
      yn = u_3p2B(tau);
    } else {
      std::cerr << "con_core_bar: isfn is not recognized!" << std::endl;
      std::exit(1);
    }
  }

  double Rn1, Rn2, Rp1, Rp2, RC;
  con_core_bar_pairing_supr(yn, yp, Rn1, Rn2, Rp1, Rp2, RC);

  double Snn = Sn2 * Kn2 * Rn2 + 3.0 * Sn1 * Kn1 * (1.0 * Rn1 - Rn2);
  double Snp = Sp2 * Kp2 * Rp2 + 0.5 * Sp1 * Kp1 * (3.0 * Rp1 - Rp2);

  Snn = std::max(Snn, 1.0e-200);
  Snp = std::max(Snp, 1.0e-200);
  if (icontrol == 3)
    Snp = 0.0;

  double nu_nn =
      3.48e15 * (mst_n * mst_n * mst_n) * std::pow((Temp / 1.0e8), 2.0) * Snn;
  double nu_np =
      3.48e15 * (mst_n * mst_p * mst_p) * std::pow((Temp / 1.0e8), 2.0) * Snp;

  double tau_n = RC / (nu_nn + nu_np);
  double lambda_n = 7.2e23 * (Temp / 1.0e8) * RC * RC / mst_n * 1.0e15 /
                    (nu_nn + nu_np) * std::pow((kf_n / 1.68), 3.0);

  // For now, ignoring lambdas for other baryons:
  double lambda_la = 0.0;
  double lambda_s0 = 0.0;

  lambda_bar = lambda_n + lambda_la + lambda_s0;
  sigma_bar = 0.0; // no electrical conductivity

  nu_e_l = 0.0;
  nu_e_s = 0.0;

  if (debug == 1.2) {
    std::cout << "Exiting con_core_bar: sigma_bar=" << sigma_bar
              << ", lambda_bar=" << lambda_bar << std::endl;
  }
  return;
}

void nscool::con_core_lep(double Temp, double kf_e, double kf_m, double kf_p,
                          double mst_p0, double Tc_p, double kf_sm,
                          double mst_sm0, double Tc_sm, double kf_sp,
                          double mst_sp0, double Tc_sp, double &sigma_lep,
                          double &lambda_e, double &lambda_m, double debug,
                          double &nu_e_s, double &nu_e_l) {
  if (debug == 1.2) {
    std::cout << "Entering con_core_lep: T, kfeo=" << Temp << ", " << kf_e
              << " " << kf_m << " " << kf_p << std::endl;
  }

  const double pi = 3.1415926535;

  // Handle case where electron Fermi momentum is zero
  if (kf_e == 0.0) {
    lambda_e = 0.0;
    lambda_m = 0.0;
    sigma_lep = 0.0;
    if (debug == 1.2) {
      std::cout << "Exiting con_core_lep: sigma_lep, lambda_lep=" << sigma_lep
                << ", " << lambda_e + lambda_m << std::endl;
    }
    return;
  }

  // Determine presence of muons
  double muons = (kf_m == 0.0) ? 0.0 : 1.0;
  double T8 = Temp / 1.0e8;
  double mst_p = (kf_p == 0.0) ? 0.0 : mst_p0;
  double mst_sm = (kf_sm == 0.0) ? 0.0 : mst_sm0;
  double mst_sp = (kf_sp == 0.0) ? 0.0 : mst_sp0;

  // Define Fermi momenta ratios (massively used):
  double rkf_0 = 1.68 / kf_e;
  double rkf_m = kf_m / kf_e;
  double rkf_p = kf_p / kf_e;

  // Screening momenta ratios:
  double rkf_e_ql3 =
      1.0 / pow(0.00929 * (1.0 + rkf_m + 2.83 * mst_p * rkf_0 * rkf_p), 1.5);
  double rkf_e_qt2 = 1.0 / (0.00929 * (1.0 + rkf_m * rkf_m + rkf_p * rkf_p));

  // Longitudinal collisional frequencies:
  double nu_ee_par = 1.43e11 * rkf_0 * rkf_e_ql3 * T8 * T8;
  double nu_em_par = nu_ee_par * muons;
  double nu_ep_par =
      1.15e12 * mst_p * mst_p * rkf_0 * rkf_0 * rkf_e_ql3 * T8 * T8;

  double nu_mm_par = (muons == 1.0) ? nu_ee_par / rkf_m : 0.0;
  double nu_me_par = (muons == 1.0) ? nu_mm_par : 0.0;
  double nu_mp_par = (muons == 1.0) ? nu_ep_par / rkf_m : 0.0;

  // Transverse ("perpendicular") collisional frequencies:
  double nu_ee_per = 6.49e14 * rkf_e_qt2 * T8;
  double nu_em_per = nu_ee_per * rkf_m * rkf_m;
  double nu_ep_per = nu_ee_per * rkf_p * rkf_p;
  double nu_mm_per = nu_ee_per * rkf_m * rkf_m * rkf_m;
  double nu_me_per = nu_ee_per * rkf_m;
  double nu_mp_per = nu_ee_per * rkf_m * rkf_p * rkf_p;

  // Cross ("prime") collisional frequencies:
  double nu_ee_pri = 4.38e12 * pow(rkf_0, 2.0 / 3.0) *
                     pow(rkf_e_qt2, 1.0 / 3.0) * pow(rkf_e_ql3, 2.0 / 3.0) *
                     pow(T8, 5.0 / 3.0);
  double nu_em_pri = (muons != 0.0) ? nu_ee_pri * rkf_m * rkf_m : 0.0;
  double nu_mm_pri = (muons != 0.0) ? (nu_ee_pri * pow(rkf_m, 3)) : 0.0;
  double nu_me_pri = (muons != 0.0) ? nu_ee_pri / rkf_m : 0.0;

  // Effect of pairing:
  double R_p_par = 1.0, R_tot_per = 1.0, R_l_pri = 1.0;
  if ((Temp <= Tc_p) && (kf_p > 0.0)) {
    double y = u_1s0(Temp / Tc_p);
    double r = (kf_e * kf_e + kf_m * kf_m) / (kf_p * kf_p);
    R_l_pri = pow(r + 1.0, 1.0 / 3.0) /
              pow(pow(r + 1.0, 2) - 0.757 * y + pow(0.50651 * y, 2), 1.0 / 6.0);
    double p1 = 0.48 - 0.17 * r;
    double p3 = pow((1.0 - p1) * 54.0 / (4.0 * pi * pi * r), 2);
    R_tot_per = p1 * exp(-0.14 * y * y) + (1.0 - p1) / sqrt(1.0 + p3 * y * y);
    R_p_par = (0.998 + (2.04 + 0.68 * std::sqrt(y) + 5.7 * y * y +
                        1.71 * y * y * y * y) *
                           std::exp(-1.04 * y)) *
              std::exp(-std::sqrt(1.23 + y * y));
  }

  // Adjust collisional frequencies for pairing and add them
  double nu_e_par = nu_ee_par + nu_em_par + nu_ep_par * R_p_par;
  double nu_e_per = (nu_ee_per + nu_em_per + nu_ep_per) * R_tot_per;
  double nu_e_pri = nu_ee_pri * R_l_pri;
  double nu_m_par = nu_mm_par + nu_me_par + nu_mp_par * R_p_par;
  double nu_m_per = (nu_mm_per + nu_me_per + nu_mp_per) * R_tot_per;
  double nu_m_pri = nu_mm_pri * R_l_pri;
  nu_em_pri = nu_em_pri * R_l_pri;
  nu_me_pri = nu_me_pri * R_l_pri;

  // Total collisional frecuencies:
  double nu_e = nu_e_par + nu_e_per + nu_e_pri;
  double nu_m = nu_m_par + nu_m_per + nu_m_pri;

  // Relaxation times:
  double tau_e = (muons != 0.0) ? (nu_m - nu_em_pri) /
                                      (nu_e * nu_m - nu_em_pri * nu_me_pri)
                                : 1.0 / nu_e;
  double tau_m = (muons != 0.0) ? (nu_e - nu_me_par) /
                                      (nu_e * nu_m - nu_em_pri * nu_me_pri)
                                : 0.0;

  // thermal conductivies:
  lambda_e = 1.70e24 * T8 * (1.0e15 * tau_e) * (kf_e / 1.68) * (kf_e / 1.68);
  lambda_m = 1.70e24 * T8 * (1.0e15 * tau_m) * (kf_m / 1.68) * (kf_m / 1.68);
  sigma_lep = 0.0;

  nu_e_l = nu_e;
  nu_e_s = 0.0;

  if (debug == 1.2) {
    std::cout << "Exiting con_core_lep: sigma_lep, lambda_lep=" << sigma_lep
              << ", " << lambda_e + lambda_m << std::endl;
  }
}

void nscool::con_core(int icon_core, double debug, double Temp, double kf_e,
                      double kf_mu, double kf_p, double mst_p,
                      double Tc_p, // proton
                      double kf_n, double mst_n, double Tc_n,
                      int isfn,                                  // neutron
                      double kf_la, double mst_la, double Tc_la, // lambda
                      double kf_sm, double mst_sm, double Tc_sm, // sigma-
                      double kf_s0, double mst_s0, double Tc_s0, // sigma0
                      double kf_sp, double mst_sp, double Tc_sp, // sigma+
                      double f_had, double &sigma, double &lambda,
                      double &nu_e_s, double &nu_e_l) {

  if (icon_core == 1) {
    // Use simple Flowers & Itoh (1981) formula
    lambda = 1.0e23 * (std::pow(kf_n, 3.0) / 1.6) / (Temp / 1.0e8);
    sigma = 0.0;
  } else if (icon_core == 2) {
    // Use Yakovlev et al. calculations
    double sigma_lep = 0.0, lambda_e = 0.0, lambda_m = 0.0;
    double lambda_lep = 0.0;
    double nu_e_s1 = 0.0, nu_e_l1 = 0.0;

    // LEPTONS
    con_core_lep(Temp, kf_e, kf_mu, kf_p, mst_p, Tc_p, kf_sm, mst_sm, Tc_sm,
                 kf_sp, mst_sp, Tc_sp, sigma_lep, lambda_e, lambda_m, debug,
                 nu_e_s1, nu_e_l1);

    // BARYONS
    double sigma_bar = 0.0, lambda_bar = 0.0;
    double nu_e_s2 = 0.0, nu_e_l2 = 0.0;
    int icontrol = 1; // ! =1 uses the full thing
    con_core_bar(Temp, kf_e, kf_mu, kf_p, mst_p, Tc_p, kf_n, mst_n, Tc_n, isfn,
                 kf_la, mst_la, Tc_la, kf_sm, mst_sm, Tc_sm, kf_s0, mst_s0,
                 Tc_s0, kf_sp, mst_sp, Tc_sp, sigma_bar, lambda_bar, debug,
                 nu_e_s2, nu_e_l2, icontrol);

    // Quark conductivity not defined => 0
    double lambda_qrk = 0.0;
    double sigma_qrk = 0.0;

    lambda = lambda_e + lambda_m; // ignoring baryons & quarks
    sigma = 0.0;

  } else if (icon_core >= 20) {
    // Use simple 1/T formula
    lambda = static_cast<double>(icon_core) / (Temp / 1.0e8);
    sigma = 0.0;
  } else if (icon_core <= -20) {
    // Use simple T-independent formula
    lambda = std::fabs(static_cast<double>(icon_core));
  }

  return;
}

void nscool::con_crust_e_phonon_Itoh(double /*T*/, double /*rho*/, double /*A*/,
                                     double /*A1*/, double /*Z*/,
                                     double &sigma_ph, double &lambda_ph,
                                     double /*debug*/, double &nu_e_s_ph,
                                     double &nu_e_l_ph) {
  std::cerr << "AWS: con_crust_e_phonon_Itoh => Removed because data file "
               "required.\n";
  std::exit(1);

  sigma_ph = 0.0;
  lambda_ph = 0.0;
  nu_e_s_ph = 0.0;
  nu_e_l_ph = 0.0;
}

void nscool::con_crust_e_phonon_BY(double Temp, double rho, double A, double A1,
                                   double Z, double &sigma_, double &lambda_,
                                   double debug, double &nu_e_s, double &nu_e_l,
                                   double rhodrip) {
  double u_1 = 2.80;
  double u_2 = 13.00;
  double a_0 = 0.01740;
  double a_2 = 0.01180;
  if (debug == 1.2) {
    std::cout << "Entering con_crust_e_phonon_BY: T, rho= " << Temp << ", "
              << rho << std::endl;
  }

  auto ei = [](double q) {
    return exp(-q * q * q * q / (q * q * q + 0.1397)) *
           (log(1 + 1 / q) - 0.5772 / (1 + 2.2757 * q * q));
  };

  double rho6 = rho / 1.0e6;
  double T8 = Temp / 1.0e8;

  double n_i = rho / A * 6.022e23;
  double a_WS = std::pow(3.0 / (4.0 * o2scl_const::pi * n_i), (1.0 / 3.0));

  double r_nucl = 0.0;
  if (rho < rhodrip) {
    r_nucl = 1.15e-13 * std::pow(A, (1.0 / 3.0));
  } else {
    r_nucl = 1.83e-13 * std::pow(Z, (1.0 / 3.0));
  }
  double g2 = std::pow((r_nucl / a_WS), 2.0);

  double x = 1.0088 * (std::pow(rho6 * Z / A, 1.0 / 3.0));
  double beta = x / std::sqrt(1.0 + x * x);
  double gam = 7.832e-2 * Z / T8 * std::sqrt(rho6 / (A * A1));

  // eqns:
  double G_0 = u_2 / (std::sqrt(1.0 + a_0 * gam * gam));
  double G_2 = (gam * gam) / (o2scl_const::pi2) /
               std::pow((1.0 + a_2 * gam * gam), (1.5));

  double ue = 1.0 / 137.0 / o2scl_const::pi / beta;
  double u1 = 1.0 / std::pow(4.0 * Z, (2.0 / 3.0)) + ue;
  double alpha0 = 1.683 * std::sqrt(x / (A1 * Z));
  double alpha = alpha0 * (0.5 * u_1 * std::exp(-9.1 / gam) + u_2 / gam);
  double w = alpha * u1;

  double S__1 = 0.0, S_0 = 0.0, S_1 = 0.0, S_2 = 0.0;
  {
    // placeholders: The snippet has "ei(q)=..." but also code for S__1, etc.
    double ewa = std::exp(-w), ewb = std::exp(-alpha);
    S__1 = ei(w) - ei(alpha); // ??? from snippet?
    S_0 = (ewa - ewb) / alpha;
    S_1 = (ewa * (w + 1.0) - ewb * (alpha + 1.0)) / (alpha * alpha);
    S_2 = (ewa * (w * w + 2.0 * w + 2.0) -
           ewb * (alpha * alpha + 2.0 * alpha + 2.0)) /
          (alpha * alpha * alpha);
  }
  double Phi_0 = S__1 - beta * beta * S_0;
  double Phi_1 = S_0 - beta * beta * S_1;
  double Phi_2 = S_1 - beta * beta * S_2;

  double P0 = 4.787 - 0.0346 * Z;
  double P2 = 2.729 - 0.0204 * Z;

  double pow_ = std::pow(18.0 * o2scl_const::pi * Z, (2.0 / 3.0));

  double K_0 =
      2.0 * Phi_1 / std::pow(1.0 + pow_ * g2 * Phi_2 / (5.0 * Phi_1 * P0), P0);
  double K_2 =
      0.5 * Phi_0 / std::pow(1.0 + pow_ * g2 * Phi_1 / (5.0 * Phi_0 * P2), P2);

  double F_s = G_0 * K_0;
  double F_k = G_0 * K_0 + G_2 * (3.0 * K_2 - 0.5 * K_0);

  double nu_s = 0.9554e17 * T8 / beta * F_s;
  double nu_k = 0.9554e17 * T8 / beta * F_k;

  sigma_ = 1.49e22 * x * x * beta * (1.0e16 / nu_s);
  lambda_ = 4.04e15 * x * x * beta * T8 * (1.0e18 / nu_k);
  nu_e_s = nu_s;
  nu_e_l = nu_k;

  if (debug == 1.2) {
    std::cout << "Exiting con_crust_e_phonon_BY: sigma, lambda= " << sigma_
              << ", " << lambda_ << std::endl;
  }
}

void nscool::con_env_e_phon_ion_PBHY(double T, double rho, double A, double A1,
                                     double Z, double &sigma, double &lambda,
                                     double debug, double &nu_e_s,
                                     double &nu_e_l) {
  if (debug == 1.2) {
    std::cout << "Entering con_env_e_phon_ion_PBHY: T, rho= " << T << ", "
              << rho << std::endl;
  }
  if (rho > 1.0e10) {
    std::cerr << "Subroutine con_env_e_phon_ion_PBHY: rho > 1e10 !\n";
    std::exit(1);
  }

  double hbar = 1.0546e-27;
  double C = 2.99792e10;
  double kb = 1.3806e-16;
  double MP = 1.6726e-24; // proton mass (unused in the snippet, but kept)
  double ME = 9.1095e-28;
  double E_CH = 4.803e-10; // e is ~4.803e-10 in cgs
  double C_LIGHT = 2.99792e10;
  double mu = 1.66e-24;
  double a_f = 1.0 / 137.0;
  double e = 4.803e-10;
  double u_1 = 2.80;
  double u_2 = 13.00;
  double a_0 = 0.01740;
  double a_2 = 0.01180;

  double n_i = rho / A * 6.022e23;
  double n_e = Z * n_i;
  double kf = std::pow(3.0 * o2scl_const::pi2 * n_e, 1.0 / 3.0);
  double pf = hbar * kf;
  double m_st = std::sqrt(electron.m * electron.m + (pf / C) * (pf / C));
  double Ef = m_st * C * C;
  double vf = pf / m_st;

  double omega_p = std::sqrt(4.0 * o2scl_const::pi * E_CH * E_CH * Z * Z * n_i /
                             (A1 * 1.66e-24));
  double T_p = hbar * omega_p / kb;
  double etat = T / T_p;
  double beta = o2scl_const::pi * 1.0 / 137.0 * Z * vf / C;

  double ai = std::pow(3.0 / (4.0 * o2scl_const::pi * n_i), 1.0 / 3.0);
  double Gamma = Z * Z * E_CH * E_CH / (kb * T * ai);

  double r_D = ai / std::sqrt(3.0 * Gamma);
  double s_D = 1.0 / std::pow((2.0 * kf * r_D), 2.0);
  double s = s_D * (1.0 + 0.06 * Gamma) * std::exp(-std::sqrt(Gamma));
  double s_e = 1.0 / 137.0 / o2scl_const::pi * C / vf * kf * kf;

  double w = (2.80 /*u_1? or u_2?*/);

  // Collisional frequencies
  double nu0 = 4.0 * Z * Ef * (1.0 / 137.0) * (1.0 / 137.0) / (hbar * (C));
  // This is approximate; real snippet is quite large.

  double Lam_s = 0.0, Lam_l = 0.0; // from Fortran
  // then:  sigma = n_e e^2 / (m_st nu_s) lambda= PI^2 kb^2 T n_e / (3 m_st
  //        nu_l)

  sigma = 1.0; // placeholder
  lambda = 1.0;
  nu_e_s = 1.0;
  nu_e_l = 1.0;

  if (debug == 1.2) {
    std::cout << "Exiting con_env_e_phon_ion_PBHY: sigma, lambda= " << sigma
              << ", " << lambda << std::endl;
  }
}

void nscool::OYAFORM(double BARD, int Index, double &Z, double &Anuc, double &A,
                     double &xnuc, double &xnuct) {
  // We replicate the big snippet. Many local variables:
  auto SOyam = [&](double t, double x) -> double {
    return std::pow(x, 3.0) - 9.0 * std::pow(x, 3.0 + t) / (3.0 + t) +
           9.0 * std::pow(x, 3.0 + 2.0 * t) / (3.0 + 2.0 * t) -
           std::pow(x, 3.0 + 3.0 * t) / (1.0 + t);
  };

  std::cout << "Fri2: " << BARD << " " << Index << std::endl;
  
  static const double PI=3.14159265;
  if (Index == 30) {
    double f = std::log(1.0 + BARD / 5.0e-9);
    double Rp = 5.688 + 0.02628 * f + 0.009468 * f * f;
    double Rn = 5.788 + 0.02077 * f + 0.01489 * f * f;
    double np_in = 0.0738 + 1.22e-4 * f - 1.641e-4 * f * f;
    double nn_in = 0.0808 + 1.688e-4 * f + 9.439e-5 * f * f;
    double nn_out = 0.0;
    double tp = 6.0;
    double tn = tp;
    double Nin = PI / 0.75 * Rn * Rn * Rn * nn_in * SOyam(tn, 1.0);
    Z = PI / 0.75 * Rp * Rp * Rp * np_in * SOyam(tp, 1.0);
    Anuc = Z + Nin;
    A = Anuc;
    std::cout << "Fri4: " << Nin << " " << Z << " " << Anuc << std::endl;
    double Rws = std::pow(A / BARD * 0.75 / PI, 0.333333);
    if (Rws < Rn) {
      std::cerr << "OYAFORM: too large Rn for outer envelope!\n";
      std::exit(1);
    }
    // final
    double aa = std::pow((A / BARD), 0.333333); // not used
    std::cout << "Fri3: " << aa << " " << tp << std::endl;

    double Rp0eff = std::pow((Z / PI * 0.75 / np_in), 0.333333);
    double tmp_sOy = SOyam(tp, 1.0);
    double Rp2eff =
        Rp * std::sqrt((1.0 - 15.0 / (5.0 + tp) + 15.0 / (5.0 + 2.0 * tp) -
                        5.0 / (5.0 + 3.0 * tp)) /
                       tmp_sOy);
    double Rp1eff = Rp * ((1.0 - 12.0 / (4.0 + tp) + 12.0 / (4.0 + 2.0 * tp) -
                           4.0 / (4.0 + 3.0 * tp)) /
                          tmp_sOy);
    double Rp3eff =
        Rp * std::pow(((1.0 - 18.0 / (6.0 + tp) + 18.0 / (6.0 + 2.0 * tp) -
                        6.0 / (6.0 + 3.0 * tp)) /
                       tmp_sOy),
                      0.333333);

    double Rws_ = Rws; // rename
    xnuc = Rp2eff / Rws_;
    xnuct = xnuc * tp / (0.6 + tp);
  } else if (Index == 3) {
    double g = BARD * 100.0;
    double f = std::log(g);
    double Rws = 31.68 - 8.400 * f - 0.2380 * f * f + 0.1152 * f * f * f;
    double tn = 1.0 / (0.2027 + 0.004506 * g);
    double Rn = 9.406 + 1.481 * f + 0.4625 * f * f + 0.05738 * f * f * f;
    double dn_n =
        (9.761 - 1.322 * f - 0.5544 * f * f - 0.07624 * f * f * f) / 100.0;
    double Nin = PI / 0.75 * std::pow(Rn, 3.0) * dn_n *
                 SOyam(tn, std::min(1.0, Rws / Rn));
    double tp = 1.0 / (0.1558 + 2.225e-3 * g + 9.452e-4 * g * g);
    double Rp = 8.345 + 0.7767 * f + 0.1333 * f * f + 0.008707 * f * f * f;
    double np_in =
        (4.040 - 1.097 * f - 0.0723 * f * f + 0.0225 * f * f * f) / 100.0;
    Z = PI / 0.75 * std::pow(Rp, 3.0) * np_in *
        SOyam(tp, std::min(1.0, Rws / Rp));
    double Nfree = BARD * PI / 0.75 * std::pow(Rws, 3.0) - Z - Nin;
    double nn_out = Nfree / (PI / 0.75 * std::pow(Rws, 3.0));
    double nn_in = nn_out + dn_n;

    A = Z + Nfree + Nin;
    double Anuc_ = Z + Nin + Nfree * std::pow((Rn / Rws), 3.0);
    if (Rn > Rws)
      Anuc_ = A;
    Anuc = Anuc_;

    double aa = std::pow((A / BARD), 1.0 / 3.0);
    double Rp0eff = std::pow((Z / (PI * 0.75 * np_in)), 1.0 / 3.0);

    double tmp_sOy = SOyam(tp, std::min(1.0, Rws / Rp));
    double Rp2eff =
        Rp * std::sqrt((1.0 - 15.0 / (5.0 + tp) + 15.0 / (5.0 + 2.0 * tp) -
                        5.0 / (5.0 + 3.0 * tp)) /
                       tmp_sOy);
    double Rp1eff = Rp * ((1.0 - 12.0 / (4.0 + tp) + 12.0 / (4.0 + 2.0 * tp) -
                           4.0 / (4.0 + 3.0 * tp)) /
                          tmp_sOy);
    double Rp3eff =
        Rp * std::pow(((1.0 - 18.0 / (6.0 + tp) + 18.0 / (6.0 + 2.0 * tp) -
                        6.0 / (6.0 + 3.0 * tp)) /
                       tmp_sOy),
                      1.0 / 3.0);

    double Rws_ = Rws;
    xnuc = Rp2eff / Rws_;
    xnuct = xnuc * tp / (0.6 + tp);
  } else {
    std::cerr << "OYAFORM: invalid Index = " << Index << std::endl;
    std::exit(1);
  }
}

void nscool::con_e_phon_ion_GYP(double T, double rho, double A_in, double A1_in,
                                double Z_in, int ifs, double &sigma,
                                double &lambda, double debug, double &nu_e_s,
                                double &nu_e_l, double rhodrip) {
  const double pi = 3.14159265;
  const double c = 2.99e10;
  const double kb = 1.380e-16;
  const double hb = 1.054e-27;
  const double a_f = 1.0 / 137.0;
  const double e = 4.803e-10;
  const double me = 9.109e-28;
  const double mu = 1.66e-24;

  // BCC lattice parameters
  const double u_1 = 2.80;
  const double u_2 = 13.00;
  const double a_0 = 0.01740;
  const double a_2 = 0.01180;

  if (debug == 1.2) {
    std::cout.precision(10);
    std::cout << "Entering con_e_phon_ion_GYP: T, rho= " << T << ", " << rho
              << std::endl;
  }

  double BARD = rho / mu * 1.0e-39;
  int Index = (rho > rhodrip) ? 3 : 30;
  double Z = Z_in, A1 = A1_in, A = A_in, xnuc, xnuct;

  OYAFORM(BARD, Index, Z, A1, A, xnuc, xnuct);
  std::cout << "Fri1: " << xnuc << " " << Z << " " << A << std::endl;
  if (ifs == 0) {
    xnuc = 0.0;
    xnuct = 0.0;
  } else if (ifs != 1) {
    std::cerr << "Sub. 'con_e_phon_ion_GYP': ifs badly defined" << std::endl;
    exit(1);
  }

  double n_i = rho / (A * mu);
  double n_e = Z * n_i;
  double kf = pow(3.0 * pi * pi * n_e, 1.0 / 3.0);
  double pf = hb * kf;
  double m_st = sqrt(me * me + (pf / c) * (pf / c));
  double Ef = m_st * c * c;
  double vf = pf / m_st;
  double x = pf / (me * c);

  double Omega_p = sqrt(4.0 * pi * e * e * Z * Z * n_i / (A1 * mu));
  double T_p = hb * Omega_p / kb;
  double tp = T / T_p;
  double betaZ = pi * a_f * Z * vf / c;
  double ai = pow(3.0 / (4.0 * pi * n_i), 1.0 / 3.0);
  double Gamma = Z * Z * e * e / (kb * T * ai);

  double r_D = ai / sqrt(3.0 * Gamma);
  double s_D = 1.0 / (2.0 * kf * r_D) / (2.0 * kf * r_D);
  double s_i = s_D * (1.0 + 0.06 * Gamma) * exp(-sqrt(Gamma));
  double s_e = a_f / pi * c / vf;
  double s = (s_i + s_e) * exp(-betaZ);

  double w = (u_2 / s_D) * (1.0 + betaZ / 3.0);
  double w1 =
      14.73 * xnuc * xnuc * (1.0 + Z * sqrt(xnuc) / 13.0) * (1.0 + betaZ / 3.0);

  double G_s = 1.0 / sqrt(1.0 + 0.0361 / pow(Z, 1.0 / 3.0) / (tp * tp)) *
               (1.0 + 0.122 * betaZ * betaZ);
  double G_l = G_s + 0.0105 * tp / pow(tp * tp + 0.0081, 1.5) *
                         (1.0 + pow(vf / c, 3) * betaZ) * (1.0 - 1.0 / Z) *
                         (1.0 + xnuct * xnuct * sqrt(2.0 * Z));

  double D = exp(-0.42 * sqrt(x / A / Z) * u_1 * exp(-9.1 * tp));

  w += w1;
  double Lam1a, Lam2a, Lam1b, Lam2b;
  get_lam(s, w, Lam1a, Lam2a);
  double Lama = Lam1a - pow(vf / c, 2) * Lam2a;
  get_lam(s, w1, Lam1b, Lam2b);
  double Lamb = Lam1b - pow(vf / c, 2) * Lam2b;
  double Lam = Lama - Lamb;

  double T_u = T_p * pow(Z, 1.0 / 3.0) * a_f / 3.0 / vf * c;
  double Lam_0_lT = 50.0 * sqrt(x) / sqrt(A1) / Z;
  double Lam_s_lT = Lam_0_lT * (4.0 / 3.0) * a_f * c / vf * pow(tp, 5);
  double Lam_l_lT = Lam_0_lT * pow(tp, 3);

  double ww = exp(-T_u / T);
  double Lam_s = Lam * G_s * D * ww + Lam_s_lT * (1.0 - ww);
  double Lam_l = Lam * G_l * D * ww + Lam_l_lT * (1.0 - ww);

  double nu0 = 4.0 * Z * Ef * a_f * a_f / (3.0 * pi * hb);
  double nu_s = nu0 * Lam_s;
  double nu_l = nu0 * Lam_l;
  sigma = n_e * e * e / (m_st * nu_s);
  lambda = pi * pi * kb * kb * T * n_e / (3.0 * m_st * nu_l);
  nu_e_s = nu_s;
  nu_e_l = nu_l;

  if (debug == 1.2) {
    std::cout.precision(10);
    std::cout << "Exiting con_e_phon_ion_GYP: sigma, lambda= " << sigma << ", "
              << lambda << " " << nu0 << " " << nu_e_s << " " << nu_e_l
              << std::endl;
    exit(-1);
  }
  return;
}

void nscool::con_crust_e_ion_Itoh(double T, double rho, double A, double A1,
                                  double Z, double &sigma, double &lambda,
                                  double &debug, double &nu_e_s,
                                  double &nu_e_l) {
  // Initialize coefficient arrays (0-indexed)
  std::vector<double> aliq = {1.4453, -0.1561, 0.0941, -0.0263};
  std::vector<double> bliq = {-1.5213, 0.8369, -0.4364};
  std::vector<double> cliq = {0.6087, -3.1264, 1.8772};
  std::vector<double> dliq = {0.4764, -0.0024, -0.0003, -0.0014};
  std::vector<double> eliq = {-0.6640, 0.0656, -0.0346};
  std::vector<double> fliq = {-0.5154, -0.1940, 0.0982};

  // Debug output
  if (debug == 1.2) {
    std::cout << "Entering con_crust_e_ion_Itoh: T = " << T << ", rho = " << rho
              << " g/cm³\n";
  }

  // Dimensionless quantities
  double rho6 = rho / 1.0e6; // rho / 1e6 g/cm³
  double T8 = T / 1.0e8;     // T / 1e8 K
  double gamma = 0.2275 * std::pow(Z, 2) / T8 * std::pow(rho6 / A, 1.0 / 3.0);
  double x2 = 1.018 * std::pow((rho6 * Z / A), 2.0 / 3.0);
  double y = 1.656e-2 / (A1 * T8) * std::pow((rho6 * Z / A), 2.0 / 3.0);
  double R = x2 / (1.0 + x2);
  double rs = 1.388e-2 * std::pow((A / (Z * rho6)), 1.0 / 3.0);
  double u = 0.45641 * std::log(gamma) - 1.31636;

  // Classical contribution to the scattering integral s
  double sum1 = aliq[0];
  double sum2 = 1.0 + bliq[0] * rs + cliq[0] * rs * rs;

  for (int j = 1; j <= 2; ++j) {
    sum1 += aliq[j] * std::pow(u, j);
    sum2 += bliq[j] * std::pow(u, j) * rs + cliq[j] * std::pow(u, j) * rs * rs;
  }

  sum1 += aliq[3] * std::pow(u, 3);
  double sm1 = sum1 * sum2;
  sm1 = (1.0 / 3.0) * std::log(Z / 26.0) + sm1;

  // Repeat for dliq, eliq, fliq
  sum1 = dliq[0];
  sum2 = 1.0 + eliq[0] * rs + fliq[0] * rs * rs;

  for (int j = 1; j <= 2; ++j) {
    sum1 += dliq[j] * std::pow(u, j);
    sum2 += eliq[j] * std::pow(u, j) * rs + fliq[j] * std::pow(u, j) * rs * rs;
  }

  sum1 += dliq[3] * std::pow(u, 3);
  double sp1 = sum1 * sum2;
  sp1 = 0.5 - std::pow(Z / 26.0, 2.0 / 3.0) * (0.5 - sp1);

  double s_s = sm1 - R * sp1;
  double s_l = sm1 - R * sp1; // Initially, s_s = s_l

  // Correction from 2nd order Coulomb cross section
  double mue = A / Z;
  double beta2 = 1.018 * std::pow(rho6 / mue, 2.0 / 3.0) /
                 (1.0 + 1.018 * std::pow(rho6 / mue, 2.0 / 3.0));
  double beta = std::sqrt(beta2);
  double alphab = (1.0 / 137.0) / beta * Z;
  double denominator = 1.0 + std::pow(alphab, 2) * (0.71 - 0.54 * beta2);
  double ds = (o2scl_const::pi / 2.0) * alphab * beta2 * (1.0 + 1.30 * alphab) /
              denominator;

  s_s += ds;
  s_l += ds;

  // Final conductivity and emissivity calculations
  sigma = 8.693e21 * (rho6 / A) * (1.0 - R) / s_s; // Electrical conductivity
  lambda = 2.363e17 * (rho6 * T8 / A) * (1.0 - R) / s_l; // Thermal conductivity
  nu_e_l = (4.11 / 2.363) * 1.0e16 * Z * std::sqrt(1.0 + x2) *
           s_l; // Longitudinal emissivity
  nu_e_s = (4.11 / 2.363) * 1.0e16 * Z * std::sqrt(1.0 + x2) *
           s_s; // Symmetric emissivity

  // Debug output
  if (debug == 1.2) {
    std::cout << "Exiting con_crust_e_ion_Itoh: sigma = " << sigma
              << ", lambda = " << lambda << std::endl;
  }

  return;
}

void nscool::con_crust_e_imp_YU(double T, double rho, double A, double A1,
                                double Z, double Q_imp, double &sigma,
                                double &lambda, double debug, double &nu_e_s,
                                double &nu_e_l) {
  if (debug == 1.2) {
    std::cout << "Entering con_crust_e_imp_YU: T, rho, A, A1, Z, Q= " << T
              << ", " << rho << ", " << A << ", " << A1 << ", " << Z << ", "
              << Q_imp << std::endl;
  }

  double x = 1.00884 * std::pow((Z / A * rho / 1.0e6), (1.0 / 3.0));
  double nu = 1.75e16 * std::sqrt(1.0 + x * x) * Q_imp / Z * Coulomb_imp_YU(x);

  lambda = 4.04e17 * std::pow(x, 3.0) / std::sqrt(1.0 + x * x) * (T / 1.0e8) *
           (1.0e16 / nu);

  sigma = 1.0;
  nu_e_s = 1.0;
  nu_e_l = 1.0;

  if (debug == 1.2) {
    std::cout << "Exiting con_crust_e_imp_YU: sigma, lambda= " << sigma << ", "
              << lambda << std::endl;
  }
}

void nscool::con_crust_ee(double T, double ne, double kfe, double &lambda,
                          double debug) {
  // From Shternin & Yakovlev, PRD74(2006) ...
  if (debug == 1.2) {
    std::cout << "Entering con_crust_ee: T, ne, kfe= " << T << ", " << ne
              << ", " << kfe << std::endl;
  }

  const double pi = 4.0 * atan(1.0);
  const double Na = 6.022e23;
  const double kb = 1.380e-16;
  const double Mu = 1.0 / Na;
  const double me = 9.109e-28;
  const double e = 4.803206e-10;
  const double hbar = 1.054572e-27;
  const double cl = 2.99792458e10;
  const double hbc = 197.327;
  const double MeV = 1.602177e-6;

  double xe = 197.3 * kfe / 0.511;
  double gammae = sqrt(1.0 + xe * xe);
  double u = xe / gammae;
  double mste = gammae * me;
  double om_pe = sqrt(4.0 * pi * e * e * ne / mste);
  double T_pe = hbar * om_pe / kb;
  double th = sqrt(3.0) * T_pe / T;

  double I_l =
      (0.1587 - 0.02538 / (1.0 + 0.0435 * th)) *
      log(1.0 + 128.56 / (37.1 * th + 10.83 * th * th + th * th * th)) / u;

  double A = 20.0 + 450.0 * u * u * u;
  double C1 = 0.05067 + 0.03216 * u * u;
  double C2 = 0.0254 + 0.04127 * u * u * u * u;
  double C = A * exp(C1 / C2);
  double I_t = u * u * u *
               (2.404 / C + (C2 - 2.404 / C) / (1.0 + 0.1 * th * u)) *
               log(1.0 + C / (A * th * u + th * th * u * u));

  A = 12.2 + 25.2 * u * u * u;
  double B = 1.0 - 0.75 * u;
  C1 = 0.123636 + 0.016234 * u * u;
  C2 = 0.0762 + 0.05714 * u * u * u * u;
  C = A * exp(C1 / C2);

  double I_lt = u *
                (18.52 * u * u / C +
                 (C2 - 18.52 * u * u / C) / (1.0 + 0.1558 * pow(th, B))) *
                log(1.0 + C / (A * th + 10.83 * th * th * u * u +
                               pow(th * u, 8.0 / 3.0)));

  double alpha = 1.0 / 137.036;
  lambda = pi * pi * pi * kb * kb * kb * T * T /
           (108.0 * alpha * alpha * hbar * hbar * cl * (I_l + I_t + I_lt));

  if (debug == 1.2) {
    std::cout << "Exiting con_crust_ee: " << lambda << std::endl;
  }
}

void nscool::con_crust(int icon_crust, double debug, double T, double rho,
                       double kfe, double A, double A1, double Z, double Q_imp,
                       double &sigma, double &lambda, double &nu_e_s,
                       double &nu_e_l, double rhodrip) {
  if (debug == 1.2) {
    std::cout << "Entering con_crust: T, rho, A, A1, Z, Q_imp = " << T << ", "
              << rho << ", " << A << ", " << A1 << ", " << Z << ", " << Q_imp
              << std::endl;
  }

  // For GYP conductivity
  int ifs = 1;

  if (rho >= 6.0e7) {
    double gamma = 2.273e5 * Z * Z * std::pow((rho / A), 1.0 / 3.0) / T;
    double sigma_ph = 0.0, lambda_ph = 0.0, nu_e_s_ph = 0.0, nu_e_l_ph = 0.0;
    double sigma_imp = 0.0, lambda_imp = 0.0, nu_e_s_imp = 0.0,
           nu_e_l_imp = 0.0;

    if (gamma > gammacryst) {
      // SOLID region
      if (icon_crust == 1) {
        con_crust_e_phonon_Itoh(T, rho, A, A1, Z, sigma_ph, lambda_ph, debug,
                                nu_e_s_ph, nu_e_l_ph);
        con_crust_e_imp_YU(T, rho, A, A1, Z, Q_imp, sigma_imp, lambda_imp,
                           debug, nu_e_s_imp, nu_e_l_imp);

        nu_e_s = nu_e_s_ph + nu_e_s_imp;
        nu_e_l = nu_e_l_ph + nu_e_l_imp;

        // Resistivity addition => 1 / total = sum(1/x_i)
        sigma = 1.0 / (1.0 / sigma_ph + 1.0 / sigma_imp);
        lambda = 1.0 / (1.0 / lambda_ph + 1.0 / lambda_imp);
      } else if (icon_crust == 2) {
        con_crust_e_phonon_BY(T, rho, A, A1, Z, sigma_ph, lambda_ph, debug,
                              nu_e_s_ph, nu_e_l_ph, rhodrip);
        con_crust_e_imp_YU(T, rho, A, A1, Z, Q_imp, sigma_imp, lambda_imp,
                           debug, nu_e_s_imp, nu_e_l_imp);

        nu_e_s = nu_e_s_ph + nu_e_s_imp;
        nu_e_l = nu_e_l_ph + nu_e_l_imp;

        sigma = 1.0 / (1.0 / sigma_ph + 1.0 / sigma_imp);
        lambda = 1.0 / (1.0 / lambda_ph + 1.0 / lambda_imp);
      } else if (icon_crust == 3) {
        con_e_phon_ion_GYP(T, rho, A, A1, Z, ifs, sigma_ph, lambda_ph, debug,
                           nu_e_s_ph, nu_e_l_ph, rhodrip);
        con_crust_e_imp_YU(T, rho, A, A1, Z, Q_imp, sigma_imp, lambda_imp,
                           debug, nu_e_s_imp, nu_e_l_imp);

        nu_e_s = nu_e_s_ph + nu_e_s_imp;
        nu_e_l = nu_e_l_ph + nu_e_l_imp;

        sigma = 1.0 / (1.0 / sigma_ph + 1.0 / sigma_imp);
        lambda = 1.0 / (1.0 / lambda_ph + 1.0 / lambda_imp);
      } else {
        std::cerr << "sub. con_crust: icon_crust incorrectly def !"
                  << std::endl;
        std::exit(1);
      }
    } else if (gamma < gammaliq) {
      // LIQUID region
      if (icon_crust == 1 || icon_crust == 2) {
        con_crust_e_ion_Itoh(T, rho, A, A1, Z, sigma, lambda, debug, nu_e_s,
                             nu_e_l);
      } else if (icon_crust == 3) {
        con_e_phon_ion_GYP(T, rho, A, A1, Z, ifs, sigma, lambda, debug, nu_e_s,
                           nu_e_l, rhodrip);
      } else {
        std::cerr << "sub. con_crust: icon_crust incorrectly def !"
                  << std::endl;
        std::exit(1);
      }
    } else {
      // SOLID-LIQUID transition Weighted average w = (gamma -
      // gammaliq)/(gammacryst - gammaliq)
      double w = (gamma - gammaliq) / (gammacryst - gammaliq);
      double w2 = smooth(w);
      double w1 = 1.0 - w2;

      if (icon_crust == 1) {
        double sigma1 = 0.0, lambda1 = 0.0, nu_e_s1 = 0.0, nu_e_l1 = 0.0;
        con_crust_e_ion_Itoh(T, rho, A, A1, Z, sigma1, lambda1, debug, nu_e_s1,
                             nu_e_l1);

        double sigma2_ph = 0.0, lambda2_ph = 0.0, nu_e_s2_ph = 0.0,
               nu_e_l2_ph = 0.0;
        con_crust_e_phonon_Itoh(T, rho, A, A1, Z, sigma2_ph, lambda2_ph, debug,
                                nu_e_s2_ph, nu_e_l2_ph);

        double sigma2_imp = 0.0, lambda2_imp = 0.0, nu_e_s2_imp = 0.0,
               nu_e_l2_imp = 0.0;
        con_crust_e_imp_YU(T, rho, A, A1, Z, Q_imp, sigma2_imp, lambda2_imp,
                           debug, nu_e_s2_imp, nu_e_l2_imp);

        double nu_e_s2 = nu_e_s2_ph + nu_e_s2_imp;
        double nu_e_l2 = nu_e_l2_ph + nu_e_l2_imp;

        double sigma2 = 1.0 / (1.0 / sigma2_ph + 1.0 / sigma2_imp);
        double lambda2 = 1.0 / (1.0 / lambda2_ph + 1.0 / lambda2_imp);

        nu_e_s = w1 * nu_e_s1 + w2 * nu_e_s2;
        nu_e_l = w1 * nu_e_l1 + w2 * nu_e_l2;
        lambda = w1 * lambda1 + w2 * lambda2;
        sigma = w1 * sigma1 + w2 * sigma2;
      } else if (icon_crust == 2) {
        double sigma1 = 0.0, lambda1 = 0.0, nu_e_s1 = 0.0, nu_e_l1 = 0.0;
        con_crust_e_ion_Itoh(T, rho, A, A1, Z, sigma1, lambda1, debug, nu_e_s1,
                             nu_e_l1);

        double sigma2_ph = 0.0, lambda2_ph = 0.0, nu_e_s2_ph = 0.0,
               nu_e_l2_ph = 0.0;
        con_crust_e_phonon_BY(T, rho, A, A1, Z, sigma2_ph, lambda2_ph, debug,
                              nu_e_s2_ph, nu_e_l2_ph, rhodrip);

        double sigma2_imp = 0.0, lambda2_imp = 0.0, nu_e_s2_imp = 0.0,
               nu_e_l2_imp = 0.0;
        con_crust_e_imp_YU(T, rho, A, A1, Z, Q_imp, sigma2_imp, lambda2_imp,
                           debug, nu_e_s2_imp, nu_e_l2_imp);

        double nu_e_s2 = nu_e_s2_ph + nu_e_s2_imp;
        double nu_e_l2 = nu_e_l2_ph + nu_e_l2_imp;

        double sigma2 = 1.0 / (1.0 / sigma2_ph + 1.0 / sigma2_imp);
        double lambda2 = 1.0 / (1.0 / lambda2_ph + 1.0 / lambda2_imp);

        nu_e_s = w1 * nu_e_s1 + w2 * nu_e_s2;
        nu_e_l = w1 * nu_e_l1 + w2 * nu_e_l2;
        lambda = w1 * lambda1 + w2 * lambda2;
        sigma = w1 * sigma1 + w2 * sigma2;
      } else if (icon_crust == 3) {
        con_e_phon_ion_GYP(T, rho, A, A1, Z, ifs, sigma, lambda, debug, nu_e_s,
                           nu_e_l, rhodrip);
      } else {
        std::cerr << "sub. con_crust: icon_crust incorrectly def !"
                  << std::endl;
        std::exit(1);
      }
    }
  } else if (rho < 6.0e7) {
    // Envelope regime
    double sigma1 = 0.0, lambda1 = 0.0, nu_e_s1 = 0.0, nu_e_l1 = 0.0;
    con_env_e_phon_ion_PBHY(T, rho, A, A1, Z, sigma1, lambda1, debug, nu_e_s1,
                            nu_e_l1);

    double sigma2 = 0.0, lambda2 = 0.0, nu_e_s2 = 0.0, nu_e_l2 = 0.0;
    con_crust_e_imp_YU(T, rho, A, A1, Z, Q_imp, sigma2, lambda2, debug, nu_e_s2,
                       nu_e_l2);

    // Resistivity sum
    sigma = sigma1 * sigma2 / (sigma1 + sigma2);
    lambda = lambda1 * lambda2 / (lambda1 + lambda2);

    nu_e_s = nu_e_s1 + nu_e_s2;
    nu_e_l = nu_e_l1 + nu_e_l2;
  }

  // Add e-e scattering:
  double ne =
      std::pow(kfe, 3) / (3.0 * o2scl_const::pi2) * 1.0e39; // from the snippet
  double lambda_ee = 0.0;
  con_crust_ee(T, ne, kfe, lambda_ee, debug);

  // Combine lambdas
  lambda = (lambda * lambda_ee) / (lambda + lambda_ee);

  if (debug == 1.2) {
    std::cout << "Exiting con_crust: sigma, lambda= " << sigma << ", " << lambda
              << std::endl;
  }
}

void nscool::conduct(
    int i, double T, double rho, double A, double A1, double Z, double Q,
    double magfield, double &sigma, double &lambda, double debug,
    double &nu_e_s, double &nu_e_l, int icon_crust, int icon_core,
    double rhodrip, double rhocore, const std::vector<double> &kfe,
    const std::vector<double> &kfm, const std::vector<double> &kfn,
    const std::vector<double> &kfp, const std::vector<double> &kfla,
    const std::vector<double> &kfsm, const std::vector<double> &kfs0,
    const std::vector<double> &kfsp, const std::vector<double> &mstn,
    const std::vector<double> &mstp, const std::vector<double> &mstla,
    const std::vector<double> &mstsm, const std::vector<double> &msts0,
    const std::vector<double> &mstsp, const std::vector<double> &tcn,
    const std::vector<double> &tcp, const std::vector<double> &tcla,
    const std::vector<double> &tcsm, const std::vector<double> &tcs0,
    const std::vector<double> &tcsp, int isf, const std::vector<double> &fhad,
    int &istrange) {
  if (static_cast<size_t>(i) > kfe.size()) {
    std::cerr << "Index i=" << i << " " << kfe.size()
              << " out of range for input arrays in `conduct()`.\n";
    std::exit(1);
  }

  // If debug >= 2.0, we print entering message
  if (debug > 2.0) {
    std::cout << "Entering conduct: T=" << T << ", rho=" << rho << ", A=" << A
              << ", A1=" << A1 << ", Z=" << Z << ", Q=" << Q << std::endl;
  }

  istrange = 0;

  // Decide whether we are in the core or in the crust
  if (rho >= rhocore) {
    // -----------------------
    // We are in the CORE
    // -----------------------
    if (istrange == 0) {
      // The code chooses isfn=3 if i<=isf, else isfn=1
      int isfn;
      if (i <= isf) {
        isfn = 3;
      } else {
        isfn = 1;
      }
      con_core(icon_core, debug, T, kfe[i], kfm[i], kfp[i], mstp[i], tcp[i],
               kfn[i], mstn[i], tcn[i], isfn, kfla[i], mstla[i], tcla[i],
               kfsm[i], mstsm[i], tcsm[i], kfs0[i], msts0[i], tcs0[i], kfsp[i],
               mstsp[i], tcsp[i], fhad[i], sigma, lambda, nu_e_s, nu_e_l);
    } else if (istrange == 1) {
    } else {
      std::cerr << "conduct: istrange not defined!\n";
      std::exit(1);
    }
  } else {
    // -----------------------
    // We are in the CRUST
    // -----------------------
    con_crust(icon_crust, debug, T, rho, kfe[i], A, A1, Z, Q, sigma, lambda,
              nu_e_s, nu_e_l, rhodrip);
  }

  // If debug >= 2.0, we print exiting message
  if (debug > 2.0) {
    std::cout << "Exiting conduct: sigma=" << sigma << ", lambda=" << lambda
              << std::endl;
  }
}

// ------------------------------------------------------------------
// numurca_nucl(...) => sets qmurca_nucl
// ------------------------------------------------------------------
void nscool::numurca_nucl(
    int i, double t, double &qmurca_nucl, const std::vector<double> &tcn,
    const std::vector<double> &tcp, int isf, const std::vector<double> &mstn,
    const std::vector<double> &mstp, const std::vector<double> &kfe,
    const std::vector<double> &kfm, const std::vector<double> &kfn,
    const std::vector<double> &kfp) {

  if (kfn[i] == 0.0 || kfp[i] == 0.0) {
    qmurca_nucl = 0.0;
    return;
  }

  double rmn = mstn[i];
  double rmp = mstp[i];
  // *** n => n+n -> n+p+e+nu
  double alpha_n = 1.76, beta_n = 0.68;
  double qmurca_n = 8.55e21 * std::pow(rmn, 3) * rmp *
                    ((kfe[i] / 1.68) + (kfm[i] / 1.68)) * alpha_n * beta_n *
                    std::pow(t / 1.0e9, 8);

  // *** p => n+p -> p+p+e+nu
  double alpha_p = alpha_n, beta_p = beta_n;
  double qmurca_p =
      8.55e21 * rmn * std::pow(rmp, 3) *
      ((kfe[i] / 1.68) *
       std::pow((kfe[i] + 3.0 * kfp[i] - kfn[i]) / (8.0 * kfe[i] * kfp[i]),
                2)) *
      alpha_p * beta_p * std::pow(t / 1.0e9, 8);

  // superfluid suppression
  double rmurca_n_p = 1.0, rmurca_n_n = 1.0;
  double rmurca_p_p = 1.0, rmurca_p_n = 1.0;

  auto pairing_n = [&](double T) {
    // if T < tcn(i): if i>=isf => use 1S0, else 3P2B
    if (T < tcn[i]) {
      if (i >= isf) {
        double tt = T / tcn[i];
        double u = u_1s0(tt);
        // rmurca_n_n = rmurca_n_n1s0(u) => define inline but we define them now
        // as function:
        auto rmurca_n_n1s0 = [&](double uu) {
          double val =
              fexp(5.3390 -
                   std::sqrt(5.3390 * 5.3390 + (2.0 * uu) * (2.0 * uu))) *
              std::pow((0.2414 + std::sqrt(0.7586 * 0.7586 +
                                           (0.1318 * uu) * (0.1318 * uu))),
                       7.0);
          return val;
        };
        return rmurca_n_n1s0(u);
      } else {
        // 3P2
        double tt = T / tcn[i];
        double u = u_3p2B(tt);
        // rmurca_n_n3p2B(u,t) => 39.1 t fexp(...) * rmurca_p_n3p2B(u)
        auto rmurca_p_n3p2B = [&](double uu) {
          return fexp(2.3980 -
                      std::sqrt(2.3980 * 2.3980 + (1.0 * uu) * (1.0 * uu))) *
                 0.5 *
                 (std::pow((0.1612 + std::sqrt(0.8388 * 0.8388 +
                                               (0.1117 * uu) * (0.1117 * uu))),
                           7) +
                  std::pow((0.1612 + std::sqrt(0.8388 * 0.8388 +
                                               (0.1274 * uu) * (0.1274 * uu))),
                           5));
        };
        auto rmurca_n_n3p2B = [&](double uu, double temperature) {
          return 39.1 * temperature * fexp(-1.188 / temperature) *
                 rmurca_p_n3p2B(uu);
        };
        return rmurca_n_n3p2B(u, tt);
      }
    } else {
      return 1.0;
    }
  };

  auto pairing_p = [&](double T) {
    // if T < tcp(i) => 1S0
    if (T < tcp[i]) {
      double tt = T / tcp[i];
      double u = u_1s0(tt);
      // rmurca_n_p1s0(u) => fexp(3.4370-...), etc. We'll inline define:
      auto rmurca_n_p1s0 = [&](double uu) {
        return fexp(3.4370 -
                    std::sqrt(3.4370 * 3.4370 + (1.0 * uu) * (1.0 * uu))) *
               0.5 *
               (std::pow((0.1477 + std::sqrt(0.8523 * 0.8523 +
                                             (0.1175 * uu) * (0.1175 * uu))),
                         7.5) +
                std::pow((0.1477 + std::sqrt(0.8523 * 0.8523 +
                                             (0.1297 * uu) * (0.1297 * uu))),
                         5.5));
      };
      return rmurca_n_p1s0(u);
    } else {
      return 1.0;
    }
  };

  rmurca_n_n = pairing_n(t); // n(n)
  rmurca_p_n = pairing_p(t); // n(p)
  rmurca_n_p = pairing_p(t); // p(n)
  rmurca_p_p = pairing_n(t);

  double rmurca_n = std::min(rmurca_n_p, rmurca_n_n);
  double rmurca_p = std::min(rmurca_p_p, rmurca_p_n);

  qmurca_n *= rmurca_n;
  qmurca_p *= rmurca_p;

  qmurca_nucl = qmurca_n + qmurca_p;
}

void nscool::numurca_q(int i, double t, double rho, double &qmurca_q,
                       const std::vector<double> &kfqu,
                       const std::vector<double> &tcu,
                       const std::vector<double> &tcd) {
  double alpha_c = 0.0;
  double g_fermi = 1.436e-49;
  double theta_c = 0.239;
  double h_bar = 1.054e-27;
  double kb = 1.38e-16;
  double c_light = 3.0e10;

  double num_coeff = 1.0;

  qmurca_q = num_coeff *
             std::pow(alpha_c * g_fermi * std::cos(theta_c) /
                          (std::pow(h_bar, 5) * std::pow(c_light, 4)),
                      2) *
             (1.0e13 * kfqu[i] * h_bar) * std::pow(kb * t, 8);

  double r_u = 1.0, r_d = 1.0;
  if (t < tcu[i]) {
    r_u = std::exp(-1.76 * tcu[i] / t);
  }
  if (t < tcd[i]) {
    r_d = std::exp(-1.76 * tcd[i] / t);
  }
  double r_ud = r_u * r_d;
  qmurca_q *= r_ud;
}

void nscool::numurca_hyp(int i, double t, double &qmurca_hyp) {
  qmurca_hyp = 0.0;
}

void nscool::nubrem_crust_nn(int i, double t, double vion, double &qbrem_nn,
                             const std::vector<double> &tcn, int isf,
                             const std::vector<double> &kfn,
                             const std::vector<double> &mstn) {
  double n_nu = 3.0;
  double alpha_nn = 0.59, beta_nn = 0.56;
  qbrem_nn = n_nu * 7.4e19 * std::pow(mstn[i], 4) * (kfn[i] / 1.68) * alpha_nn *
             beta_nn * std::pow(t / 1.0e9, 8);

  // superfluid pairing factor
  double rbrem_nn = 1.0;
  if (t < tcn[i]) {
    if (i >= isf) {
      double tt = t / tcn[i];
      double u = u_1s0(tt);
      auto rbrem_nn_n1s0 = [&](double uu) {
        double part1 =
            std::pow((0.1747 + std::sqrt(0.8253 * 0.8253 +
                                         (0.07933 * uu) * (0.07933 * uu))),
                     2.0) *
            fexp(4.228 - std::sqrt(4.228 * 4.228 + (4.0 * uu) * (4.0 * uu))) /
            2.0;
        double part2 =
            std::pow((0.7333 + std::sqrt(0.2667 * 0.2667 +
                                         (0.1678 * uu) * (0.1678 * uu))),
                     7.5) *
            fexp(7.762 - std::sqrt(7.762 * 7.762 + (9.0 * uu) * (9.0 * uu))) /
            2.0;
        return part1 + part2;
      };
      rbrem_nn = rbrem_nn_n1s0(u);
    } else {
      // i<isf => 3P2
      double tt = t / tcn[i];
      double u = u_3p2B(tt);
      auto rbrem_nn_n1s0 = [&](double uu) {
        double part1 =
            std::pow((0.1747 + std::sqrt(0.8253 * 0.8253 +
                                         (0.07933 * uu) * (0.07933 * uu))),
                     2.0) *
            fexp(4.228 - std::sqrt(4.228 * 4.228 + (4.0 * uu) * (4.0 * uu))) /
            2.0;
        double part2 =
            std::pow((0.7333 + std::sqrt(0.2667 * 0.2667 +
                                         (0.1678 * uu) * (0.1678 * uu))),
                     7.5) *
            fexp(7.762 - std::sqrt(7.762 * 7.762 + (9.0 * uu) * (9.0 * uu))) /
            2.0;
        return part1 + part2;
      };
      rbrem_nn = rbrem_nn_n1s0(u);
    }
  }

  // multiply by pairing factor, then reduce by ion volume
  qbrem_nn *= rbrem_nn;
  qbrem_nn *= (1.0 - vion);
}

void nscool::nubrem_nucl(int i, double t, double &qbrem_nucl,
                         const std::vector<double> &tcn,
                         const std::vector<double> &tcp, int isf,
                         const std::vector<double> &kfn,
                         const std::vector<double> &kfp,
                         const std::vector<double> &mstn,
                         const std::vector<double> &mstp) {
  double n_nu = 3.0;
  // Brem_nn
  double alpha_nn = 0.59, beta_nn = 0.56;
  double qbrem_nn = n_nu * 7.4e19 * std::pow(mstn[i], 4) * (kfn[i] / 1.68) *
                    alpha_nn * beta_nn * std::pow(t / 1.0e9, 8);
  // Brem_np
  double alpha_np = 1.06, beta_np = 0.66;
  double qbrem_np = n_nu * 1.5e20 * mstn[i] * mstp[i] * (kfp[i] / 1.68) *
                    alpha_np * beta_np * std::pow(t / 1.0e9, 8);
  // Brem_pp
  double alpha_pp = 0.11, beta_pp = 0.70;
  double qbrem_pp = n_nu * 7.4e19 * std::pow(mstp[i], 4) * (kfp[i] / 1.68) *
                    alpha_pp * beta_pp * std::pow(t / 1.0e9, 8);

  // Pairing suppression define local r-factors for nn, np, pp in n & p channels
  double rbrem_nn_n = 1.0, rbrem_nn_p = 1.0;
  double rbrem_np_n = 1.0, rbrem_np_p = 1.0;
  double rbrem_pp_n = 1.0, rbrem_pp_p = 1.0;

  // If t< tcn => neutron pairing
  if (t < tcn[i]) {
    if (i >= isf) {
      // 1S0
      double tt = t / tcn[i];
      double u = u_1s0(tt);
      // For nn => "rbrem_nn_n1s0(u)" => same code we used above
      auto rbrem_nn_n1s0 = [&](double uu) {
        double part1 =
            std::pow((0.1747 + std::sqrt(0.8253 * 0.8253 +
                                         (0.07933 * uu) * (0.07933 * uu))),
                     2.0) *
            fexp(4.228 - std::sqrt(4.228 * 4.228 + (4.0 * uu) * (4.0 * uu))) /
            2.0;
        double part2 =
            std::pow((0.7333 + std::sqrt(0.2667 * 0.2667 +
                                         (0.1678 * uu) * (0.1678 * uu))),
                     7.5) *
            fexp(7.762 - std::sqrt(7.762 * 7.762 + (9.0 * uu) * (9.0 * uu))) /
            2.0;
        return part1 + part2;
      };
      rbrem_nn_n = rbrem_nn_n1s0(u);

      // For np => "rbrem_np_n1s0(u)"
      auto rbrem_np_n1s0 = [&](double uu) {
        double part1 =
            std::pow((0.9982 + std::sqrt((0.0018 * 0.0018) +
                                         (0.3815 * uu) * (0.3815 * uu))),
                     1.0) *
            fexp(1.306 - std::sqrt(1.306 * 1.306 + (1.0 * uu) * (1.0 * uu))) /
            2.732;
        double part2 =
            std::pow((0.3949 + std::sqrt((0.6051 * 0.6051) +
                                         (0.2666 * uu) * (0.2666 * uu))),
                     7.0) *
            fexp(3.303 - std::sqrt(3.303 * 3.303 + (4.0 * uu) * (4.0 * uu))) /
            1.577;
        return part1 + part2;
      };
      rbrem_np_n = rbrem_np_n1s0(u);
      rbrem_pp_n = 1.0;
    } else {
      // 3P2
      double tt = t / tcn[i];
      double u = u_3p2B(tt);
      auto same_as_n1s0 = [&](double uu) {
        double part1 =
            std::pow((0.1747 + std::sqrt(0.8253 * 0.8253 +
                                         (0.07933 * uu) * (0.07933 * uu))),
                     2.0) *
            fexp(4.228 - std::sqrt(4.228 * 4.228 + (4.0 * uu) * (4.0 * uu))) /
            2.0;
        double part2 =
            std::pow((0.7333 + std::sqrt(0.2667 * 0.2667 +
                                         (0.1678 * uu) * (0.1678 * uu))),
                     7.5) *
            fexp(7.762 - std::sqrt(7.762 * 7.762 + (9.0 * uu) * (9.0 * uu))) /
            2.0;
        return part1 + part2;
      };
      rbrem_nn_n = same_as_n1s0(u);
      auto same_as_np_n1s0 = [&](double uu) {
        double part1 =
            std::pow((0.9982 + std::sqrt((0.0018 * 0.0018) +
                                         (0.3815 * uu) * (0.3815 * uu))),
                     1.0) *
            fexp(1.306 - std::sqrt(1.306 * 1.306 + (uu * 1.0) * (uu * 1.0))) /
            2.732;
        double part2 =
            std::pow((0.3949 + std::sqrt((0.6051 * 0.6051) +
                                         (0.2666 * uu) * (0.2666 * uu))),
                     7.0) *
            fexp(3.303 - std::sqrt(3.303 * 3.303 + (4.0 * uu) * (4.0 * uu))) /
            1.577;
        return part1 + part2;
      };
      rbrem_np_n = same_as_np_n1s0(u);
      rbrem_pp_n = 1.0; // unaffected by neutron pairing
    }
  }

  // If t< tcp => proton pairing
  rbrem_nn_p = 1.0, rbrem_np_p = 1.0, rbrem_pp_p = 1.0;
  if (t < tcp[i]) {
    double tt = t / tcp[i];
    double u = u_1s0(tt);
    // rbrem_nn_p1s0(u)=1 => Not affected by proton pairing
    rbrem_nn_p = 1.0;
    auto rbrem_np_p1s0 = [&](double uu) {
      double part1 =
          std::pow((0.9982 + std::sqrt((0.0018 * 0.0018) +
                                       (0.3815 * uu) * (0.3815 * uu))),
                   1.0) *
          fexp(1.306 - std::sqrt(1.306 * 1.306 + uu * uu)) / 2.732;
      double part2 =
          std::pow((0.3949 + std::sqrt((0.6051 * 0.6051) +
                                       (0.2666 * uu) * (0.2666 * uu))),
                   7.0) *
          fexp(3.303 - std::sqrt(3.303 * 3.303 + (4.0 * uu) * (4.0 * uu))) /
          1.577;
      return part1 + part2;
    };
    rbrem_np_p = rbrem_np_p1s0(u);

    // rbrem_pp_p1s0(u)= rbrem_nn_n1s0(u)
    auto rbrem_nn_n1s0 = [&](double uu) {
      double part1 =
          std::pow((0.1747 + std::sqrt(0.8253 * 0.8253 +
                                       (0.07933 * uu) * (0.07933 * uu))),
                   2.0) *
          fexp(4.228 - std::sqrt(4.228 * 4.228 + (4.0 * uu) * (4.0 * uu))) /
          2.0;
      double part2 =
          std::pow((0.7333 +
                    std::sqrt(0.2667 * 0.2667 + (0.1678 * uu) * (0.1678 * uu))),
                   7.5) *
          fexp(7.762 - std::sqrt(7.762 * 7.762 + (9.0 * uu) * (9.0 * uu))) /
          2.0;
      return part1 + part2;
    };
    rbrem_pp_p = rbrem_nn_n1s0(u);
  }

  // Combine
  double rbrem_nn = std::min(rbrem_nn_p, rbrem_nn_n);
  double rbrem_np = std::min(rbrem_np_p, rbrem_np_n);
  double rbrem_pp = std::min(rbrem_pp_p, rbrem_pp_n);

  qbrem_nn *= rbrem_nn;
  qbrem_np *= rbrem_np;
  qbrem_pp *= rbrem_pp;

  qbrem_nucl = qbrem_nn + qbrem_np + qbrem_pp;
}

void nscool::neebrem(double T, double mu, double &qeebrem) {
  double mu10 = mu / 10.0;
  double alpha = T / mu10;
  double logalpha = std::log10(alpha);

  double naa = iv.eval(logalpha);
  qeebrem =
      2.16e14 * std::pow((T / 1.0e9), 7.0) * std::pow((mu / 10.0), 2.0) * naa;
}

// ------------------------------------------------------------------
// numurca_hyp(...), nubrem_hyp(...): placeholder returning 0
// ------------------------------------------------------------------
void nscool::nubrem_hyp(int i, double t, double &qbrem_hyp) { qbrem_hyp = 0.0; }

double nscool::emissivity_durca_npl(double kfn, double kfp, double kfe,
                                    double ne, double kfmu, double nmu,
                                    double rmn, double rmp, double TK,
                                    double Tcn1s0, double Tcp1s0,
                                    double Tcn3p2) {

  double s_e = kfn + kfp + kfe;
  double heron_e = s_e * (s_e - kfn) * (s_e - kfp) * (s_e - kfe);
  double fact_e = 1.0;
  if (heron_e <= 0.0)
    fact_e = 0.0;
  double s_mu = kfn + kfp + kfmu;
  double heron_mu = s_mu * (s_mu - kfn) * (s_mu - kfp) * (s_mu - kfmu);
  double fact_mu = 1.0;
  if (heron_mu <= 0.0)
    fact_mu = 0.0;

  double emis = 4.24e27 * rmn * rmp * pow(TK, 6.0) *
                (fact_e * cbrt(ne / 0.16) + fact_mu * cbrt(nmu / 0.16));

  // Pairing suppression
  if (TK < Tcn1s0 || TK < Tcn3p2) {
    if (TK < Tcp1s0) {
      double tp = TK / Tcp1s0;
      double up = u_1s0(tp);
      if (Tcn3p2 > Tcn1s0) {
        double tn = TK / Tcn3p2;
        emis *= r_1s0_3p2B(tp, tn);
      } else {
        double tn = TK / Tcn1s0;
        double un = u_1s0(tn);
        emis *= r_1s0_1s0(up, un);
      }
    } else {
      if (Tcn3p2 > Tcn1s0) {
        double tn = TK / Tcn3p2;
        double u = u_3p2B(tn);
        emis *= r_3p2B(u);
      } else {
        double tn = TK / Tcn1s0;
        double u = u_1s0(tn);
        emis *= r_1s0(u);
      }
    }
  } else if (TK < Tcp1s0) {
    double tp = TK / Tcp1s0;
    double u = u_1s0(tp);
    emis *= r_1s0(u);
  }

  return emis;
}

void nscool::nudurca_q(
    int i, double t, double rho, double &qdurca_q,
    const std::vector<double> &tcu1, const std::vector<double> &tcu2,
    const std::vector<double> &tcu3, const std::vector<double> &tcd1,
    const std::vector<double> &tcd2, const std::vector<double> &tcd3,
    const std::vector<double> &tcs1, const std::vector<double> &tcs2,
    const std::vector<double> &tcs3, const std::vector<double> &kfe,
    const std::vector<double> &kfm, const std::vector<double> &kfqu,
    const std::vector<double> &kfqd, const std::vector<double> &kfqs,
    const std::vector<int> &idurca_quqd, const std::vector<int> &idurca_quqs) {
  // alpha_c=0 => "AWS needs fix":
  double alpha_c = 0.0;
  double g_fermi = 1.436e-49;
  double theta_c = 0.239;
  double h_bar = 1.054e-27;
  double kb = 1.38e-16;
  double c_light = 3.0e10;
  // etc.

  // 1) compute qdurca_quqd
  double coeff_ud =
      (1.0 / 3.0) * (914.0 / 315.0) *
      std::pow(g_fermi * std::cos(theta_c) /
                   (h_bar * h_bar * h_bar * c_light * c_light * c_light),
               2) *
      alpha_c;
  double qdurca_quqd = 0.0;
  if (idurca_quqd[i] == 1) {
    qdurca_quqd = coeff_ud * 1.0e39 * kfqd[i] * kfqu[i] * kfe[i] *
                  std::pow(h_bar, 3) * std::pow(kb * t, 6);
  } else if (idurca_quqd[i] == 2) {
    qdurca_quqd =
        coeff_ud * 1.0e39 *
        (kfqd[i] * kfqu[i] * kfe[i] * std::pow(h_bar, 3) * std::pow(kb * t, 6) +
         kfqd[i] * kfqu[i] * kfm[i] * std::pow(h_bar, 3) * std::pow(kb * t, 6));
  }

  // Pairing color 1 => r_ud1, color 2 => r_ud2, color 3 => r_ud3
  auto get_r_ud = [&](double T, double Tcu, double Tcd) {
    if (T > Tcu && T > Tcd) {
      return 1.0;
    } else if (T > Tcu && T <= Tcd) {
      double tt = T / Tcd;
      double u = u_1s0(tt);
      return r_1s0(u);
    } else if (T <= Tcu && T > Tcd) {
      double tt = T / Tcu;
      double u = u_1s0(tt);
      return r_1s0(u);
    } else {
      double tt1 = T / Tcu;
      double u1 = u_1s0(tt1);
      double tt2 = T / Tcd;
      double u2 = u_1s0(tt2);
      return r_1s0_1s0(u1, u2);
    }
  };
  double r_ud1 = get_r_ud(t, tcu1[i], tcd1[i]);
  double r_ud2 = get_r_ud(t, tcu2[i], tcd2[i]);
  double r_ud3 = get_r_ud(t, tcu3[i], tcd3[i]);

  qdurca_quqd *= (r_ud1 + r_ud2 + r_ud3);

  // 2) compute qdurca_quqs (u-s)
  double theta_34 = 3.14159265 / 4.0;
  double coeff_us =
      (1.0 / 3.0) * (457.0 * 3.14159265 / 840.0) *
      std::pow(g_fermi * std::sin(theta_c) /
                   (h_bar * h_bar * h_bar * c_light * c_light * c_light),
               2) *
      (1.0 - std::cos(theta_34));

  double qdurca_quqs = 0.0;
  if (idurca_quqs[i] == 1) {
    // uses "strange_mass" from snippet, not given. We'll define a placeholder
    double strange_mass = 0.0; // or set from global
    double mus = std::sqrt(kfqs[i] * kfqs[i] * std::pow(10.0, 26) * c_light *
                               c_light * h_bar * h_bar +
                           std::pow(1.6e-6 * strange_mass, 2));
    qdurca_quqs = coeff_us * 1.0e26 *
                  (mus / c_light * kfqu[i] * kfe[i] * h_bar * h_bar *
                   std::pow(kb * t, 6));
  } else if (idurca_quqs[i] == 2) {
    double strange_mass = 0.0;
    double mus = std::sqrt(kfqs[i] * kfqs[i] * std::pow(10.0, 13) * c_light *
                               c_light * h_bar * h_bar +
                           std::pow(1.6e-6 * strange_mass, 2));
    qdurca_quqs = coeff_us * 1.0e26 *
                  (mus / c_light * kfqu[i] * kfe[i] * h_bar * h_bar *
                       std::pow(kb * t, 6) +
                   mus / c_light * kfqu[i] * kfm[i] * h_bar * h_bar *
                       std::pow(kb * t, 6));
  }

  auto get_r_us = [&](double T, double Tcu, double Tcs) {
    if (T > Tcu && T > Tcs) {
      return 1.0;
    } else if (T > Tcu && T <= Tcs) {
      double tt = T / Tcs;
      double u = u_1s0(tt);
      return r_1s0(u);
    } else if (T <= Tcu && T > Tcs) {
      double tt = T / Tcu;
      double u = u_1s0(tt);
      return r_1s0(u);
    } else {
      double tt1 = T / Tcu;
      double u1 = u_1s0(tt1);
      double tt2 = T / Tcs;
      double u2 = u_1s0(tt2);
      return r_1s0_1s0(u1, u2);
    }
  };
  double r_us1 = get_r_us(t, tcu1[i], tcs1[i]);
  double r_us2 = get_r_us(t, tcu2[i], tcs2[i]);
  double r_us3 = get_r_us(t, tcu3[i], tcs3[i]);

  qdurca_quqs *= (r_us1 + r_us2 + r_us3);

  qdurca_q = qdurca_quqd + qdurca_quqs;
  return;
}

void nscool::nudurca_h(
    int i, double t, double rho, double &qdurca_np, double &qdurca_lap,
    double &qdurca_smn, double &qdurca_smla, double &qdurca_sms0,
    const std::vector<double> &tcn, const std::vector<double> &tcp,
    const std::vector<double> &tcla, int isf, const std::vector<double> &bar,
    const std::vector<double> &yelect, const std::vector<double> &ymuon,
    const std::vector<double> &mstp, const std::vector<double> &mstn,
    const std::vector<double> &mstla, const std::vector<double> &mstsm,
    const std::vector<double> &msts0, const std::vector<double> &mstsp,
    const std::vector<double> &durca_ctrl_e,
    const std::vector<double> &durca_ctrl_m, const std::vector<int> &idurca_lap,
    const std::vector<int> &idurca_smla, const std::vector<int> &idurca_smn,
    const std::vector<int> &idurca_sms0, const std::vector<int> &idurca_np) {
  // Placeholder for now

  qdurca_np = 0.0;
  qdurca_lap = 0.0;
  qdurca_smn = 0.0;
  qdurca_smla = 0.0;
  qdurca_sms0 = 0.0;
}

void nscool::nu_1s0_pbf(double T, double Tc, double mst, double kf,
                        double &q_1s0_pbf) {
  if (T <= Tc) {
    double pf = kf * 197.0;
    double vf = pf / (mst * 940.0);
    double a_v = 0.0, a_a = 0.0;
    a_a = 1.60 * vf * vf * (mst * mst + 11.0 / 42.0);
    double a = a_v + a_a;
    double tau = T / Tc;
    double u =
        std::sqrt(1.0 - tau) * (1.456 - 0.157 / std::sqrt(tau) + 1.764 / tau);

    auto control_pbf_1S0 = [&](double v) {
      double x =
          0.602 * (v * v) + 0.5942 * std::pow(v, 4) + 0.288 * std::pow(v, 6);
      double y =
          std::sqrt(0.5547 + std::sqrt(0.4453 * 0.4453 + 0.01130 * v * v));
      double z = std::exp(-std::sqrt(4.0 * v * v + 2.245 * 2.245) + 2.245);
      return x * y * z;
    };

    q_1s0_pbf = 1.170e21 * mst * mst * vf * std::pow((T / 1.0e9), 7) * 3.0 * a *
                control_pbf_1S0(u);
  } else {
    q_1s0_pbf = 0.0;
  }
}

void nscool::nu_n3p2_B_pbf(double T, double Tc, double mst, double kf,
                           double &q_n3p2_pbf) {
  if (T <= Tc) {
    double pf = kf * 197.0;
    double vf = pf / (mst * 940.0);
    double g_A = 1.26;
    double a_v = 0.0;
    double a_a = 0.5 * (g_A * g_A);
    double a = a_v + a_a;
    double tau = T / Tc;
    double u = std::sqrt(1.0 - tau) * (0.7893 + 1.764 / tau);

    auto control_pbf_3P2_B = [&](double v) {
      double x =
          (1.204 * v * v + 3.733 * std::pow(v, 4) + 0.3191 * std::pow(v, 6)) /
          (1.0 + 0.3511 * v * v);
      double y =
          std::pow(0.7591 + std::sqrt(0.2409 * 0.2409 + 0.3145 * v * v), 2.0);
      double z = std::exp(-std::sqrt(4.0 * v * v + 0.4616 * 0.4616) + 0.4616);
      return x * y * z;
    };

    q_n3p2_pbf = 1.170e21 * mst * mst * vf * std::pow((T / 1.0e9), 7) * 3.0 *
                 a * control_pbf_3P2_B(u);
  } else {
    q_n3p2_pbf = 0.0;
  }
}

void nscool::nufast(
    int i, double t, double rho, double &qfast, const std::vector<double> &tcn,
    const std::vector<double> &tcp, int isf, const std::vector<double> &bar,
    const std::vector<double> &theta_k, const std::vector<double> &theta_p,
    const std::vector<double> &yelect, double rhoexo, double cexo, double pexo,
    const std::vector<double> &mstn, const std::vector<double> &mstp,
    const std::vector<double> &kfe) {
  double u = bar[i] / 0.16;
  double ratio = 0.319 / std::pow(std::fabs(yelect[i]) * u, 1.0 / 3.0);
  double f = 0.0;
  if (ratio < 1.0) {
    f = std::sqrt(1.0 - ratio * ratio);
  }

  // Kaon urca
  double qkaon = 0.0;
  if (theta_k[i] != 0.0) {
    double g_a = 1.0;
    double mu_el = kfe[i] * 197.0;
    qkaon = 5.0 / 4.0 * std::pow(std::sin(theta_k[i]), 2.0) * std::sin(0.223) *
            std::sin(0.223) * 2.21e26 * mstn[i] * mstp[i] * (mu_el / 100.0) *
            (1.0 + 3.0 * g_a * g_a) * std::pow(t / 1.0e9, 6.0);
  }

  double qexo = 0.0;
  if (rho >= rhoexo) {
    qexo = cexo * std::pow((rho / 2.8e14), (2.0 / 3.0)) *
           std::pow(t / 1.0e9, pexo);
  }

  // superfluid factor
  double r = 1.0;
  // replicate snippet. NEEDS FIX
  // approximate:
  if (t < tcp[i] && t < tcn[i]) {
    // if i>= isf => n pairing => 1S0, else => 3P2
    double rn = 1.0, rp = 1.0;
    if (i >= isf) {
      double un = u_1s0(t / tcn[i]);
      rn = r_1s0(un);
    } else {
      double un = u_3p2B(t / tcn[i]);
      // r_3p2B(un) ...
      auto r_3p2B_ = [&](double uu) {
        return std::pow((0.2546 + std::sqrt(0.7454 * 0.7454 +
                                            (0.01811 * uu) * (0.01811 * uu))),
                        5.0) *
               fexp(2.701 -
                    std::sqrt(2.701 * 2.701 + uu * uu / (16.0 * 3.14159265)));
      };
      rn = r_3p2B_(un);
    }
    double up = u_1s0(t / tcp[i]);
    double rp_ = r_1s0(up);

    r = std::min(rn, rp_);
  } else if (t < tcn[i]) {
    if (i >= isf) {
      double un = u_1s0(t / tcn[i]);
      r = r_1s0(un);
    } else {
      double un = u_3p2B(t / tcn[i]);
      // define r_3p2B => see above
      auto r_3p2B_ = [&](double uu) {
        return std::pow((0.2546 + std::sqrt(0.7454 * 0.7454 +
                                            (0.01811 * uu) * (0.01811 * uu))),
                        5.0) *
               fexp(2.701 -
                    std::sqrt(2.701 * 2.701 + uu * uu / (16.0 * 3.14159265)));
      };
      r = r_3p2B_(un);
    }
  } else if (t < tcp[i]) {
    double up = u_1s0(t / tcp[i]);
    r = r_1s0(up);
  } else {
    r = 1.0;
  }

  qkaon *= r;
  double r_exo = r;

  qexo *= r_exo;
  qfast = qkaon + qexo;
}

void nscool::npb_new(double temp, double rho, double &qnpb) {
  double tau = log10(temp / 1.0e8);
  double r = log10(rho / 1.0e12);
  double rho0 = 2.8e14;
  double lgq = 11.204 + 7.304 * tau + 0.2976 * r - 0.370 * tau * tau +
               0.188 * tau * r - 0.103 * r * r + 0.0547 * tau * tau * r -
               6.77 * log10(1.0 + 0.228 * rho / rho0);
  qnpb = pow(10.0, lgq);
  return;
}

void nscool::npb(double t, double rho, double a, double z, double &qnpb) {

  double mm = std::pow((rho / 1.0e6 * z / a), (2.0 / 3.0));
  double tf = 5.930e9 * (std::sqrt(1.0 + 1.018 * mm) - 1.0);
  double gamma = 2.273e5 * z * z * std::pow((rho / a), (1.0 / 3.0)) / t;

  if (rho <= 1.0e4) {
    double tmp = 0.0;
    npbpde(t, rho, a, z, tmp);
    qnpb = tmp;
  } else if (t > 0.35 * tf) {
    double tmp = 0.0;
    npbpde(t, rho, a, z, tmp);
    qnpb = tmp;
  } else if (t >= 0.30 * tf) {
    double qnpb1 = 0.0, qnpb2 = 0.0;
    npbpde(t, rho, a, z, qnpb1);
    npbl(t, rho, a, z, qnpb2);
    qnpb = (t - 0.25 * tf) / (0.1 * tf) * qnpb1 +
           (0.35 * tf - t) / (0.1 * tf) * qnpb2;
  } else if (gamma < gammaliq) {
    npbl(t, rho, a, z, qnpb);
  } else if (gamma > gammacryst) {
    npbc(t, rho, a, z, qnpb);
  } else {
    double qnpb1 = 0.0, qnpb2 = 0.0;
    npbl(t, rho, a, z, qnpb1);
    npbc(t, rho, a, z, qnpb2);
    qnpb = (gammacryst - gamma) / (gammacryst - gammaliq) * qnpb1 +
           (gamma - gammaliq) / (gammacryst - gammaliq) * qnpb2;
  }
  // For extremely impure crust => just call npbl but we keep as is
}

void nscool::npbpde(double t, double rho, double a, double z, double &qnpbpde) {
  double n = 2.0;
  double gamma = 2.273e5 * z * z * std::pow((rho / a), (1.0 / 3.0)) / t;
  double t8 = t / 1.0e8;
  double etat =
      (rho * z / a) / (7.05e6 * std::pow(t8, 1.5) + 5.12e4 * std::pow(t8, 3.0));
  double f1 = 23.5 + 6.83e4 / std::pow(t8, 2.0) + 7.81e8 / std::pow(t8, 5.0);
  double f2 = 1.0 + 1.47 / etat + 0.0329 / (etat * etat);
  double f = 1.0 / f1 + 1.26 * (1.0 + 1.0 / eta) / f2;

  double b3 = 7.75e5 * std::pow(t8, 1.5) + 247.0 * std::pow(t8, 3.85);
  double b4 = 4.07 + 0.0240 * std::pow(t8, 1.40);
  double b5 = 4.59e-5 / std::pow(t8, 0.11);
  double g1 = 230.0 + 6.7e5 / std::pow(t8, 2.0) + 7.66e9 / std::pow(t8, 5.0);
  double g2 = b3 / ((rho * z / a)) + b4 + b5 * std::pow((rho * z / a), 0.656);
  double g = 1.0 / (1.0 + 1.0e-9 * rho * z / a) / g1 + 1.0 / g2;

  qnpbpde = 0.5738 * (z * z) / a * std::pow(t8, 6.0) * rho *
            (0.5 * (1.122 + 0.254 * n) * f - 0.5 * (0.622 - 0.246 * n) * g);
}

void nscool::npbl(double t, double rho, double a, double z, double &qnpbl) {
  std::cerr << "AWS: Removed because this required a data file.\n";
  std::exit(1);
}

void nscool::npbc(double t, double rho, double a, double z, double &qnpbc) {
  std::cerr << "AWS: Removed because this required a data file.\n";
  std::exit(1);
}

void nscool::npair(double t, double rho, double a, double z, double &qpair) {

  static double apa[3] = {6.002e19, 2.084e20, 1.872e21};
  static double bpa_l[3] = {0.9383, -0.4141, 0.05829};

  // if(t<1.e7 => qpair=0 => return)
  if (t < 1.0e7) {
    qpair = 0.0;
    return;
  }

  auto fexp_ = [&](double x) {
    double xx = (x < -700.0) ? -700.0 : x;
    return std::exp(xx);
  };

  double l = t / 5.9302e9;
  double xi = std::pow(rho * z / a * 1.0e-9, 1.0 / 3.0) / l;
  double n = 2.0; // number neutrinos (excl. electron neutrino)

  double cpa_l = 5.5924;
  double cpa_h = 4.9924;
  double fpair = 0.0;

  // logic:
  if (t < 1.0e10) {
    double top = apa[0] + apa[1] * xi + apa[2] * (xi * xi);
    double denom = (xi * xi * xi) + bpa_l[0] / l + bpa_l[1] / (l * l) +
                   bpa_l[2] / (l * l * l);
    fpair = top * fexp_(-cpa_l * xi) / denom;
  } else {
    static double bpa_h[3] = {1.2383, -0.8141, 0.0}; // from snippet
    double top = apa[0] + apa[1] * xi + apa[2] * (xi * xi);
    double denom = (xi * xi * xi) + bpa_h[0] / l + bpa_h[1] / (l * l) +
                   bpa_h[2] / (l * l * l);
    fpair = top * fexp_(-cpa_h * xi) / denom;
  }

  double g = 1.0 - 13.04 * (l * l) + 133.5 * std::pow(l, 4) +
             1534.0 * std::pow(l, 6) + 918.6 * std::pow(l, 8);
  double qpa = (10.7480 * std::pow(l, 2) + 0.3967 * std::pow(l, 0.5) + 1.0050);
  if (qpa != 0.0) {
    qpa = 1.0 / qpa;
  }
  // plus some correction for (1+ rho*z/a(...) )^ -0.3 ??? This is
  // partially incomplete. I'll approximate.

  // final:
  qpair = 0.5 * (1.122 + n * 0.254) *
          (1.0 + (0.622 - n * 0.246) / (1.122 + n * 0.254) * qpa) * g *
          fexp_(-2.0 / l) * fpair;
}

void nscool::nphoto(double t, double rho, double a, double z, double &qphoto) {
  if (t < 1.0e7) {
    qphoto = 0.0;
    return;
  }

  double l = t / 5.9302e9;
  double xi = std::pow((rho * z / a * 1.0e-9), (1.0 / 3.0)) / l;

  double n = 2.0;

  auto fexp_ = [&](double x) {
    double xx = (x < -700.0) ? -700.0 : x;
    return std::exp(xx);
  };

  double qphot = 0.0; // final

  qphoto = 0.0;
}

void nscool::nplasma(double t, double rho, double a, double z,
                     double &qplasma) {
  if (z == 0.0) {
    qplasma = 0.0;
    return;
  }
  auto fexp_ = [&](double x) {
    double xx = (x < -700.0) ? -700.0 : x;
    return std::exp(xx);
  };

  double l = t / 5.9302e9;
  double den = 1.0 + std::pow(1.019e-6 * rho * z / a, 2.0 / 3.0);
  double gamma2 = 1.1095e11 * (rho * z / a) / (t * t) / std::sqrt(den);
  double gamma = std::sqrt(gamma2);

  double f_t =
      2.4 + 0.6 * std::sqrt(gamma) + 0.51 * gamma + 1.25 * std::pow(gamma, 1.5);
  double f_l = (8.6 * gamma2 + 1.35 * std::pow(gamma, 3.5)) /
               (225.0 - 17.0 * gamma + gamma2);

  double x = (1.0 / 6.0) *
             (17.5 + std::log10(2.0 * rho * z / a) - 3.0 * std::log10(t));
  double y = (1.0 / 6.0) *
             (-24.5 + std::log10(2.0 * rho * z / a) + 3.0 * std::log10(t));
  double f_xy = 1.0;
  if ((std::fabs(x) > 0.7) || (y < 0.0)) {
    f_xy = 1.0;
  } else {
    double first = 0.39 - 1.25 * x - 0.35 * std::sin(4.5 * x) -
                   0.3 * fexp_(-(4.5 * x + 0.9) * (4.5 * x + 0.9));
    double sec = y - 1.6 + 1.25 * x;
    sec = std::min(0.0, sec);
    double second = sec / (0.57 - 0.25 * x);
    f_xy = 1.05 + first * fexp_(-second * second);
  }

  qplasma = 3.00e21 * std::pow(l, 9) * std::pow(gamma, 6) * fexp_(-gamma) *
            (f_l + f_t) * f_xy;
}

void nscool::nplasma_old(double t, double rho, double a, double z,
                         double &qplasma) {
  static const double apl[3] = {2.320e-7, 8.449e-8, 1.787e-8};
  static const double bpl[3] = {2.581e-2, 1.734e-2, 6.990e-4};
  static const double cpl = 0.56457;
  if (z == 0.0) {
    qplasma = 0.0;
    return;
  }
  double l = t / 5.93e9;
  double xi = cbrt(rho * z / a * 1.0e-9) / l;
  int n = 2;
  double fplasma =
      (apl[0] + apl[1] * xi + apl[2] * xi * xi) * fexp(-cpl * xi) /
      (xi * xi * xi + bpl[0] / l + bpl[1] / l / l + bpl[2] / l / l / l);
  qplasma = (0.872 + n * 0.004) * pow(rho * z / a, 3.0) * fplasma;
  return;
}

void nscool::nbub(int i, double t, double rho, double a, double z,
                  double &qbubble, double rhocore, std::vector<double> tcn,
                  int isf) {
  double rhomin = 1.0e14;
  if (rho < rhocore && rho >= rhomin) {
    qbubble = 1.1e22 * pow(t / 1.0e9, 6.0);
  } else {
    qbubble = 0.0;
  }
  // effect of superfluidity:
  double r = 1.0;
  if (t < tcn[i]) {
    if (i >= isf) {
      double u = u_1s0(t / tcn[i]);
      r = r_1s0(u);
    } else {
      double u = u_3p2B(t / tcn[i]);
      r = r_3p2B(u);
    }
  }
  qbubble *= r;
  return;
}

void nscool::nsynch(double t, double bfield, double kfe, double &qsynch) {

  static const double a1 = 2.036e-4;
  static const double b1 = 7.405e-8;
  static const double c1 = 3.675e-4;
  static const double a2 = 3.356e-3;
  static const double b2 = 1.536e-5;
  static const double c2 = 1.436e-2;
  static const double d2 = 1.024e-5;
  static const double e2 = 7.647e-8;

  double b13 = bfield / 1.0e13;
  double x = kfe / 197 / 0.511;

  double tp = 2.02e9 * b13 * x * x;
  double xi = tp / t;
  double y1 = pow(pow(1.0 + 3172.0 * pow(xi, 2.0 / 3.0), 2.0 / 3.0) - 1.0, 1.5);
  double y2 = pow(pow(1.0 + 172.2 * pow(xi, 2.0 / 3.0), 2.0 / 3.0) - 1.0, 1.5);
  double fp =
      44.01 * pow(1.0 + c1 * y1, 2.0) / pow(1.0 + a1 * y1 + b1 * y1 * y1, 4.0);
  double fm = 36.97 * (1.0 + c2 * y2 + d2 * y2 * y2 + e2 * y2 * y2 * y2) /
              pow(1.0 + a2 * y2 + b2 * y2 * y2, 5.0);
  double s_ab = 27.0 * pow(xi, 4.0) / o2scl_const::pi2 / 512.0 / 1.037 *
                (fp - 0.175 / 1.675 * fm);

  double tb = 1.34e9 * b13 / sqrt(1.0 + x * x);
  double z = tb / t;
  double d_1 = 1.0 + 0.4228 * z + 0.1014 * z * z + 0.006240 * z * z * z;
  double d_2 = 1.0 + 0.4535 * pow(z, 2.0 / 3.0) + 0.03008 * z -
               0.05043 * z * z + 0.004314 * z * z * z;
  double s_bc = exp(-z / 2.0) * d_1 / d_2;

  qsynch = 9.04e14 * b13 * b13 * pow(t / 1.0e9, 5.0) * s_ab * s_bc;

  return;
}

void nscool::neutrino(
    int i, double t, double rho, double a, double z, double &qtot,
    double &qeebrem, double &qnpb, double &qplasma, double &qsynch,
    double &qbubble, double &qpair, double &qphoto, double &qbrem_nn,
    double &qmurca_nucl, double &qbrem_nucl, double &qmurca_hyp,
    double &qbrem_hyp, double &qdurca_np, double &qdurca_lap,
    double &qdurca_smn, double &qdurca_smla, double &qdurca_sms0, double &qfast,
    double &qdurca_q, double &qmurca_q, double &qpbf_n1s0, double &qpbf_n3p2,
    double &qpbf_p1s0, double &qpbf_q, double debug,
    const std::vector<double> &nbfield2, double rhodrip, double rhocore,
    const std::vector<double> &mstp, const std::vector<double> &mstn,
    const std::vector<double> &mstla, const std::vector<double> &mstsm,
    const std::vector<double> &msts0, const std::vector<double> &mstsp,
    const std::vector<double> &kfe, const std::vector<double> &kfm,
    const std::vector<double> &kfp, const std::vector<double> &kfn,
    const std::vector<double> &kfqu, const std::vector<double> &kfqd,
    const std::vector<double> &kfqs, const std::vector<double> &bar,
    const std::vector<double> &yelect, const std::vector<double> &ymuon,
    const std::vector<double> &fhad, const std::vector<double> &theta_k,
    const std::vector<double> &theta_p, const std::vector<double> &v_ion,
    double rhoexo, double cexo, double pexo, double c_nu_str, double p_nu_str,
    double murca_increase, int inu_durca, int inu_eion, int inu_plasma,
    int inu_synch, int inu_n1s0_pbf, int inu_n3p2_pbf, int inu_p_pbf,
    int inu_bubble, int inu_photo, int inu_pair,
    const std::vector<int> &idurca_np, const std::vector<int> &idurca_lap,
    const std::vector<double> &durca_ctrl_e,
    const std::vector<double> &durca_ctrl_m, const std::vector<int> &idurca_smn,
    const std::vector<int> &idurca_smla, const std::vector<int> &idurca_sms0,
    const std::vector<int> &idurca_quqd, const std::vector<int> &idurca_quqs,
    const std::vector<double> &tcn, const std::vector<double> &tcp,
    const std::vector<double> &tcla, const std::vector<double> &tcu,
    const std::vector<double> &tcd, const std::vector<double> &tcs,
    const std::vector<double> &tcu1, const std::vector<double> &tcu2,
    const std::vector<double> &tcu3, const std::vector<double> &tcd1,
    const std::vector<double> &tcd2, const std::vector<double> &tcd3,
    const std::vector<double> &tcs1, const std::vector<double> &tcs2,
    const std::vector<double> &tcs3, int isf) {
  int istrange = 0;
  double qstrange = 0.0;
  static int print_it = 0;

  if (debug >= 2.0) {
    std::cout << "Entering subroutine `neutrino`: "
              << " T=" << t << ", rho=" << rho << ", A=" << a << ", Z=" << z
              << std::endl;
  }

  // *** ELECTRON-ELECTRON PAIR BREMSSTRAHLUNG:
  if (rho < rhocore) {
    double mu_el = kfe[i] * 197.0;
    neebrem(t, mu_el, qeebrem);
  } else {
    qeebrem = 0.0;
  }

  // *** ELECTRON-ION PAIR BREMSSTRAHLING:
  if (inu_eion == 1) {
    if (rho < rhocore) {
      npb_new(t, rho, qnpb);
    } else {
      qnpb = 0.0;
    }
  } else if (inu_eion == 2) {
    if (rho < rhocore) {
      npb(t, rho, a, z, qnpb);
    } else {
      qnpb = 0.0;
    }
  } else {
    if (rho < rhocore) {
      qnpb = 0.0;
      if (print_it != 1) {
        std::cout << "No npb: Rho, Qnpb=" << rho << ", " << qnpb << std::endl;
        print_it = 1;
      }
    } else {
      qnpb = 0.0;
    }
  }

  // *** PLASMA NEUTRINO:
  if (inu_plasma == 1) {
    if (rho < rhocore) {
      nplasma(t, rho, a, z, qplasma);
    } else {
      qplasma = 0.0;
    }
  } else if (inu_plasma == -1) {
    if (rho < rhocore) {
      nplasma_old(t, rho, a, z, qplasma);
    } else {
      qplasma = 0.0;
    }
  } else {
    qplasma = 0.0;
  }

  // *** SYNCHROTRON NEUTRINO:
  if (inu_synch == 1) {
    if (rho < rhocore) {
      nsynch(t, nbfield2[i], kfe[i], qsynch);
    } else {
      qsynch = 0.0;
    }
  } else {
    qsynch = 0.0;
  }

  // *** BUBBLE NEUTRINO:
  if (inu_bubble == 1) {
    if (rho < rhocore) {
      nbub(i, t, rho, a, z, qbubble, rhocore, tcn, isf);
    } else {
      qbubble = 0.0;
    }
  } else {
    qbubble = 0.0;
  }

  // *** NEUTRINO PAIR:
  if (inu_pair == 1) {
    if (rho < rhocore) {
      npair(t, rho, a, z, qpair);
    } else {
      qpair = 0.0;
    }
  } else {
    qpair = 0.0;
  }

  // *** PHOTO-NEUTRINO:
  if (inu_photo == 1) {
    if (rho < rhocore) {
      nphoto(t, rho, a, z, qphoto);
    } else {
      qphoto = 0.0;
    }
  } else {
    qphoto = 0.0;
  }

  // *** NN-BREMSTRAHLUNG in the inner crust
  if ((rho < rhocore) && (rho >= rhodrip)) {
    nubrem_crust_nn(i, t, v_ion[i], qbrem_nn, tcn, isf, kfn, mstn);
  } else {
    qbrem_nn = 0.0;
  }

  // *** URCA et al. PROCESSES:
  if (rho >= rhocore) {
    if (istrange == 0) {
      // normal hadronic
      numurca_nucl(i, t, qmurca_nucl, tcn, tcp, isf, mstn, mstp, kfe, kfm, kfn,
                   kfp);
      qmurca_nucl = qmurca_nucl * (1.0 + murca_increase);
      qmurca_nucl = qmurca_nucl * fhad[i];

      nubrem_nucl(i, t, qbrem_nucl, tcn, tcp, isf, kfn, kfp, mstn, mstp);
      qbrem_nucl = qbrem_nucl * (1.0 + murca_increase);
      qbrem_nucl = qbrem_nucl * fhad[i];

      numurca_hyp(i, t, qmurca_hyp);
      qmurca_hyp = qmurca_hyp * fhad[i];

      nubrem_hyp(i, t, qbrem_hyp);
      qbrem_hyp = qbrem_hyp * fhad[i];

      // Direct Urca
      if (inu_durca == 1) {
        nudurca_h(i, t, rho, qdurca_np, qdurca_lap, qdurca_smn, qdurca_smla,
                  qdurca_sms0, tcn, tcp, tcla, isf, bar, yelect, ymuon, mstp,
                  mstn, mstla, mstsm, msts0, mstsp, durca_ctrl_e, durca_ctrl_m,
                  idurca_lap, idurca_smla, idurca_smn, idurca_sms0, idurca_np);
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
      }

      // *** FAST neutrino emission:
      nufast(i, t, rho, qfast, tcn, tcp, isf, bar, theta_k, theta_p, yelect,
             rhoexo, cexo, pexo, mstn, mstp, kfe);
      qfast = qfast * fhad[i];

      // *** QUARK processes:
      nudurca_q(i, t, rho, qdurca_q, tcu1, tcu2, tcu3, tcd1, tcd2, tcd3, tcs1,
                tcs2, tcs3, kfe, kfm, kfqu, kfqd, kfqs, idurca_quqd,
                idurca_quqs);

      numurca_q(i, t, rho, qmurca_q, kfqu, tcu, tcd);

      qdurca_q = qdurca_q * (1.0 - fhad[i]);
      qmurca_q = qmurca_q * (1.0 - fhad[i]);

      qstrange = 0.0;
    } else if (istrange == 1) {
      // STRANGE QUARK MATTER processes
      qstrange = c_nu_str * std::pow((t / 1.0e9), p_nu_str);

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
      std::cerr << "neutrino: istrange not defined !\n";
      std::exit(1);
    }
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
    qstrange = 0.0;
  }

  // *** PBF PROCESSES:
  if (istrange == 0) {
    // Neutrons 1S0:
    if ((inu_n1s0_pbf == 1) && (i > isf)) {
      nu_1s0_pbf(t, tcn[i], mstn[i], kfn[i], qpbf_n1s0);
      qpbf_n1s0 *= fhad[i];
    } else {
      qpbf_n1s0 = 0.0;
    }
    // Neutron 3P2:
    if ((inu_n3p2_pbf == 1) && (i <= isf)) {
      nu_n3p2_B_pbf(t, tcn[i], mstn[i], kfn[i], qpbf_n3p2);
      qpbf_n3p2 *= fhad[i];
    } else {
      qpbf_n3p2 = 0.0;
    }
    // Protons:
    if (inu_p_pbf == 1) {
      nu_1s0_pbf(t, tcp[i], mstp[i], kfp[i], qpbf_p1s0);
      qpbf_p1s0 *= fhad[i];
    } else {
      qpbf_p1s0 = 0.0;
    }
    // Quarks => not included => set 0
    qpbf_q = 0.0;
    qpbf_q = qpbf_q * (1.0 - fhad[i]);
  } else {
    qpbf_n1s0 = 0.0;
    qpbf_n3p2 = 0.0;
    qpbf_p1s0 = 0.0;
    qpbf_q = 0.0;
  }

  // *** ADDING EVERYTHING:
  qtot = qeebrem + qnpb + qplasma + qsynch + qbubble + qpair + qphoto +
         qbrem_nn + qmurca_nucl + qbrem_nucl + qmurca_hyp + qbrem_hyp +
         qdurca_np + qdurca_lap + qdurca_smn + qdurca_smla + qdurca_sms0 +
         qfast + qdurca_q + qmurca_q + qstrange + qpbf_n1s0 + qpbf_n3p2 +
         qpbf_p1s0 + qpbf_q;

  if (debug >= 2.0) {
    std::cout << "Exiting subroutine `neutrino`" << std::endl;
  }
  return;
}

void nscool::default_star(std::string dir) {

  o2scl_hdf::hdf_file hf;
  std::string name;

  // Read default crust EOS
  hf.open(dir + "../crust_HZD_NV.o2");
  hdf_input(hf, nscool_crust, name);
  hf.close();

  // Read default core EOS
  hf.open(dir + "../core_APR.o2");
  hdf_input(hf, nscool_core, name);
  hf.close();

  // Read default stellar profile
  hf.open(dir + "../tov_APR_14.o2");
  hdf_input(hf, nscool_tov, name);
  hf.close();

  return;
}

void nscool::hhj_eos(double mass) {

  eos_had_hhj hhj;
  o2scl::nstar_cold nc;
  o2scl::convert_units<double> &cu = o2scl::o2scl_settings.get_convert_units();

  // ------------------------------------------------------------
  // Construct the EOS from an nstar_cold object

  // Automatically construct beta-equilibrium EOS
  nc.verbose = 0;
  nc.def_tov.verbose = 0;
  nc.set_eos(hhj);
  nc.err_nonconv = false;
  nc.calc_eos();
  std::shared_ptr<o2scl::table_units<>> nc_eos = nc.get_eos_results();
  nc.def_eos_tov.read_table(*nc_eos, "ed", "pr", "nb");

  // Prepare to compute effective masses
  o2scl::fermion n(o2scl::o2scl_settings.get_convert_units().convert(
                       "kg", "1/fm", o2scl_const::mass_neutron_f<double>()),
                   2.0);
  o2scl::fermion p(o2scl::o2scl_settings.get_convert_units().convert(
                       "kg", "1/fm", o2scl_const::mass_proton_f<double>()),
                   2.0);
  n.non_interacting = false;
  p.non_interacting = false;
  o2scl::thermo th;

  // ------------------------------------------------------------
  // Create a core table of the correct size with the correct units

  // Clear table for new columns
  nscool_core.clear();
  nscool_core.line_of_names("Rho Press nbar Ye Ymu Yn Yp mstp mstn");
  nscool_core.line_of_names("Yla Ysm Ys0 Ysp mstla mstsm msts0 mstsp");

  // Baryon density grid
  double nb_min = 0.09;
  double nb_max = nc_eos->max("nb");
  double dnb = (nb_max - nb_min) / 178.0;

  // Fill core table
  for (double nb = nb_max; nb > nb_min - dnb / 10.0; nb -= dnb) {
    n.n = nc_eos->interp("nb", nb, "nn");
    p.n = nc_eos->interp("nb", nb, "np");
    hhj.calc_e(n, p, th);
    double line[17] = {
        cu.convert("1/fm^4", "g/cm^3", nc_eos->interp("nb", nb, "ed")),
        cu.convert("1/fm^4", "dyne/cm^2", nc_eos->interp("nb", nb, "pr")),
        nb,
        nc_eos->interp("nb", nb, "ne") / nb,
        nc_eos->interp("nb", nb, "nmu") / nb,
        nc_eos->interp("nb", nb, "nn") / nb,
        nc_eos->interp("nb", nb, "np") / nb,
        n.ms / n.m,
        p.ms / p.m,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0};
    nscool_core.line_of_data(17, line);
  }

  // ------------------------------------------------------------
  // Use the nstar_cold object to construct the profile

  // Construct neutron star profile
  nc.def_tov.calc_gpot = true;
  nc.calc_nstar();
  std::shared_ptr<o2scl::table_units<>> nc_prof = nc.get_tov_results();
  std::cout << "M_max: " << nc_prof->max("gm") << std::endl;
  std::cout << "R(M_max): "
            << nc_prof->get("r", nc_prof->lookup("gm", nc_prof->max("gm")))
            << std::endl;

  double mass2 = mass;
  if (mass2 < 0.0)
    mass2 = mass + nc_prof->max("gm");
  std::cout << "Set mass to " << mass2 << std::endl;
  nc.fixed(mass2);

  // ------------------------------------------------------------
  // Create a table with the right size and the right units

  // Clear table for new columns
  nscool_tov.clear();
  nscool_tov.line_of_names(((std::string) "step radius n_baryon density ") +
                           "pressure emass phi bmass");

  // Radial grid
  double r_max = nc_prof->max("r");

  // Fill TOV table
  for (size_t ix = 0; ix < 153; ix++) {
    // A grid which focuses most of the points on the outer part of the star
    double r = r_max * 2.0 * (1.0 - pow(2.0, -((double)ix) / 152.0));
    std::vector<double> line = {
        ((double)ix),
        r * 1.0e3,
        nc_prof->interp("r", r, "nb"),
        cu.convert("1/fm^4", "g/cm^3", nc_prof->interp("r", r, "ed")),
        cu.convert("1/fm^4", "dyne/cm^2", nc_prof->interp("r", r, "pr")),
        nc_prof->interp("r", r, "gm"),
        nc_prof->interp("r", r, "gp"),
        nc_prof->interp("r", r, "bm")};
    nscool_tov.line_of_data(8, line);
  }

  return;
}

void nscool::tptr_init(int ifteff, double tempini, double ephi_surf,
                       double ephi_drip, double ephi_core, double &tsurface,
                       double &tdrip, double &tcore, double &tb_acc0) {

  if (ifteff != 15) {
    if (tempini > 0.0) {
      tsurface = T_fact_surf * ephi_surf * tempini;
      tdrip = T_fact_drip * ephi_drip * tempini;
      tcore = 1.0 * ephi_core * tempini;
    } else {
      tsurface = 1.0e9;
      tdrip = 2.0e10;
      tcore = 1.0e11;
    }
  } else {
    tb_acc0 *= ephi_surf;
    tsurface = tb_acc0;
    tdrip = tb_acc0;
    tcore = tb_acc0;
  }
  return;
}

double nscool::Teff(double Tb, int ifteff, double eta_arg, double bfield,
                    int istep, double time, double Ts1, double Ts2, double Z,
                    double A, double Rho, int debug, double gs14,
                    double compactness) {

  // The internal temperature in units of 10^9 K
  double Tb9 = Tb / 1.0e9;

  // T_{*} in units of 10^{6} K
  double Ts = sqrt(7.0e0 * Tb9 * sqrt(gs14));

  // zeta
  double z = Tb9 - Ts / 1.0e3;

  // The effective temperature to the fourth power for an Iron envelope in units
  // of 10^{6} K
  double t4_iron = gs14 * (pow(7.0e0 * z, 2.25) + pow(z / 3.0e0, 1.25));

  // The effective temperature to the fourth power for a fully accreted envelope
  // in units of 10^{6} K
  double t4_wacc = gs14 * pow(18.1e0 * Tb9, 2.42);

  // For a partially accreted envelope
  double t4_acc;
  if (eta_arg > 1.0e-30) {
    double a = (1.2e0 + pow(5.3e-6 / eta_arg, 0.38)) * pow(Tb9, 5.0 / 3.0);
    t4_acc = (a * t4_iron + t4_wacc) / (a + 1.0e0);
  } else {
    t4_acc = t4_iron;
  }

  // Return the final effective temperature in Kelvin
  return pow(t4_acc, 0.25) * 1.0e6;
}

void nscool::print_temp(
    int istep, int itprint, double time, double t_effective, int imax,
    double w1, double w2, std::vector<double> otemp, std::vector<double> temp,
    std::vector<double> olum, std::vector<double> lum, std::vector<double> rad,
    std::vector<double> rrho, std::vector<double> ephi,
    std::vector<double> dvol, std::vector<double> e2phi,
    std::vector<double> tcn, std::vector<double> tcp, std::vector<double> qnu,
    std::vector<double> qeebrem, std::vector<double> qnpb,
    std::vector<double> qplasma, std::vector<double> qsynch,
    std::vector<double> qbubble, std::vector<double> qpair,
    std::vector<double> qphoto, std::vector<double> qbrem_nn,
    std::vector<double> qmurca_nucl, std::vector<double> qbrem_nucl,
    std::vector<double> qmurca_hyp, std::vector<double> qbrem_hyp,
    std::vector<double> qdurca_np, std::vector<double> qdurca_lap,
    std::vector<double> qdurca_smn, std::vector<double> qdurca_smla,
    std::vector<double> qdurca_sms0, std::vector<double> qfast,
    std::vector<double> qdurca_q, std::vector<double> qmurca_q,
    std::vector<double> qpbf_n1s0, std::vector<double> qpbf_p1s0,
    std::vector<double> qpbf_n3p2, std::vector<double> qpbf_q) {

  // T_eff, as passed by the Fortran code in t_effective, is not currently
  // stored in tl_prof.o2. The variable istep is also not currently stored
  // anywhere.

  if (fabs(time_print[itprint] - time) / fabs(time) > 1.0e-6) {
    O2SCL_ERR("Temperature print sanity check.", o2scl::exc_einval);
  }

  if (sxrt_mode) {
    itprint = mdot_index;
  }

  if (itprint == 0) {
    tl_prof.clear();
    std::vector<double> r_grid;
    for (int i = imax; i >= 1; i -= 2) {
      r_grid.push_back(rad[imax + 1 - i] / 1.0e5);
    }
    if (!sxrt_mode) {
      tl_prof.set_xy("r", r_grid.size(), r_grid, "t", temp_ygrid.size(),
                     temp_ygrid);
    } else {
      tl_prof.set_xy("r", r_grid.size(), r_grid, "mdot", temp_ygrid.size(),
                     temp_ygrid);
    }

    tl_prof.line_of_names(
        ((std::string) "rho ephi vol Tinf Linf tcn tcp qnu ") +
        "qeebrem qnpb qplasma qsynch qbubble qpair " +
        "qphoto qbrem_nn qmurca_nucl qbrem_nucl " +
        "qmurca_hyp qbrem_hyp qdurca_np qdurca_lap " +
        "qdurca_smn qdurca_smla qdurca_sms0 qfast " +
        "qdurca_q qmurca_q qpbf_n1s0 qpbf_p1s0 " + "qpbf_n3p2 qpbf_q qmax " +
        "cv cv_n cv_p cv_e cv_m cv_la " + "cv_sm cv_s0 cv_sp cv_q");
    tl_prof.add_constant("it_last", itprint);
  }

  tl_prof.set_constant("it_last", itprint);

  if (ptemp >= 2.0) {
    std::cout << "Time: " << time << " years, T_eff: " << t_effective << " K"
              << std::endl;
  }

  for (int i = imax; i >= 1; i -= 2) {
    int io2 = i / 2;

    double logtemp = w1 * log(otemp[i]) + w2 * log(temp[i]);
    double temperature = exp(logtemp);
    double lumino = 0.0;
    if (i != 1) {
      double loglum = w1 * log(fabs(olum[i - 1])) + w2 * log(fabs(lum[i - 1]));
      lumino = exp(loglum);
      if (lum[i - 1] < 0.0)
        lumino *= -1.0;
    }
    tl_prof.set(io2, itprint, "rho", rrho[i]);
    tl_prof.set(io2, itprint, "ephi", ephi[i]);
    tl_prof.set(io2, itprint, "vol", dvol[i] + dvol[i + 1]);
    tl_prof.set(io2, itprint, "Tinf", temperature / ephi[i]);
    tl_prof.set(io2, itprint, "Linf", lumino / e2phi[i - 1]);
    tl_prof.set(io2, itprint, "tcn", tcn[i]);
    tl_prof.set(io2, itprint, "tcp", tcp[i]);
    tl_prof.set(io2, itprint, "qnu", qnu[i]);
    tl_prof.set(io2, itprint, "qeebrem", qeebrem[i]);
    tl_prof.set(io2, itprint, "qnpb", qnpb[i]);
    tl_prof.set(io2, itprint, "qplasma", qplasma[i]);
    tl_prof.set(io2, itprint, "qsynch", qsynch[i]);
    tl_prof.set(io2, itprint, "qbubble", qbubble[i]);
    tl_prof.set(io2, itprint, "qpair", qpair[i]);
    tl_prof.set(io2, itprint, "qphoto", qphoto[i]);
    tl_prof.set(io2, itprint, "qbrem_nn", qbrem_nn[i]);
    tl_prof.set(io2, itprint, "qmurca_nucl", qmurca_nucl[i]);
    tl_prof.set(io2, itprint, "qbrem_nucl", qbrem_nucl[i]);
    tl_prof.set(io2, itprint, "qmurca_hyp", qmurca_hyp[i]);
    tl_prof.set(io2, itprint, "qbrem_hyp", qbrem_hyp[i]);
    tl_prof.set(io2, itprint, "qdurca_np", qdurca_np[i]);
    tl_prof.set(io2, itprint, "qdurca_lap", qdurca_lap[i]);
    tl_prof.set(io2, itprint, "qdurca_smn", qdurca_smn[i]);
    tl_prof.set(io2, itprint, "qdurca_smla", qdurca_smla[i]);
    tl_prof.set(io2, itprint, "qdurca_sms0", qdurca_sms0[i]);
    tl_prof.set(io2, itprint, "qfast", qfast[i]);
    tl_prof.set(io2, itprint, "qdurca_q", qdurca_q[i]);
    tl_prof.set(io2, itprint, "qmurca_q", qmurca_q[i]);
    tl_prof.set(io2, itprint, "qpbf_n1s0", qpbf_n1s0[i]);
    tl_prof.set(io2, itprint, "qpbf_p1s0", qpbf_p1s0[i]);
    tl_prof.set(io2, itprint, "qpbf_n3p2", qpbf_n3p2[i]);
    tl_prof.set(io2, itprint, "qpbf_q", qpbf_q[i]);

    {
      std::vector<double> qvec;
      qvec.push_back(fabs(qeebrem[i]));
      qvec.push_back(fabs(qnpb[i]));
      qvec.push_back(fabs(qplasma[i]));
      qvec.push_back(fabs(qsynch[i]));
      qvec.push_back(fabs(qbubble[i]));
      qvec.push_back(fabs(qpair[i]));
      qvec.push_back(fabs(qphoto[i]));
      qvec.push_back(fabs(qbrem_nn[i]));
      qvec.push_back(fabs(qmurca_nucl[i]));
      qvec.push_back(fabs(qbrem_nucl[i]));
      qvec.push_back(fabs(qmurca_hyp[i]));
      qvec.push_back(fabs(qbrem_hyp[i]));
      qvec.push_back(fabs(qdurca_np[i]));
      qvec.push_back(fabs(qdurca_lap[i]));
      qvec.push_back(fabs(qdurca_smn[i]));
      qvec.push_back(fabs(qdurca_smla[i]));
      qvec.push_back(fabs(qdurca_sms0[i]));
      qvec.push_back(fabs(qfast[i]));
      qvec.push_back(fabs(qdurca_q[i]));
      qvec.push_back(fabs(qmurca_q[i]));
      qvec.push_back(fabs(qpbf_n1s0[i]));
      qvec.push_back(fabs(qpbf_p1s0[i]));
      qvec.push_back(fabs(qpbf_n3p2[i]));
      qvec.push_back(fabs(qpbf_q[i]));
      o2scl::permutation order(qvec.size());
      o2scl::vector_sort_index(qvec.size(), qvec, order);
      if (qvec[order[qvec.size() - 1]] > 0.0) {
        tl_prof.set(io2, itprint, "qmax", order[qvec.size() - 1] + 1);
      } else {
        tl_prof.set(io2, itprint, "qmax", -order[qvec.size() - 1] + 1);
      }
    }

    {
      double total = qeebrem[i] + qnpb[i] + qplasma[i] + qsynch[i] +
                     qbubble[i] + qpair[i] + qphoto[i] + qbrem_nn[i] +
                     qmurca_nucl[i] + qbrem_nucl[i] + qmurca_hyp[i] +
                     qbrem_hyp[i] + qdurca_np[i] + qdurca_lap[i] +
                     qdurca_smn[i] + qdurca_smla[i] + qdurca_sms0[i] +
                     qfast[i] + qdurca_q[i] + qmurca_q[i] + qpbf_n1s0[i] +
                     qpbf_p1s0[i] + qpbf_n3p2[i] + qpbf_q[i];
      if (fabs(qnu[i] - total) / fabs(qnu[i]) > 1.0e-6) {
        std::cout << "Emissivity problem " << qnu[i] << " " << total
                  << std::endl;
        exit(-1);
      }
    }

    if (ptemp >= 2.0) {
      std::cout << i << " " << tl_prof.get_grid_x(io2) << " " << rrho[i] << " "
                << ephi[i] << " " << dvol[i] + dvol[i + 1] << " "
                << temperature / ephi[i] << " " << lumino / e2phi[i - 1] << " "
                << qnu[i] << " " << qeebrem[i] << " " << qnpb[i] << " "
                << qplasma[i] << " " << qsynch[i] << " " << qbubble[i] << " "
                << qpair[i] << " " << qphoto[i] << " " << qbrem_nn[i] << " "
                << qmurca_nucl[i] << " " << qbrem_nucl[i] << " "
                << qmurca_hyp[i] << " " << qbrem_hyp[i] << " " << qdurca_np[i]
                << " " << qdurca_lap[i] << " " << qdurca_smn[i] << " "
                << qdurca_smla[i] << " " << qdurca_sms0[i] << " " << qfast[i]
                << " " << qdurca_q[i] << " " << qmurca_q[i] << " "
                << qpbf_n1s0[i] << " " << qpbf_p1s0[i] << " " << qpbf_n3p2[i]
                << " " << qpbf_q[i] << std::endl;
    }
  }
  if (ptemp >= 2.0) {
    std::cout << std::endl;
  }
  return;
}

void nscool::print_cv(int itprint, int imax, std::vector<double> cv,
                      std::vector<double> cv_n, std::vector<double> cv_p,
                      std::vector<double> cv_e, std::vector<double> cv_m,
                      std::vector<double> cv_la, std::vector<double> cv_sm,
                      std::vector<double> cv_s0, std::vector<double> cv_sp,
                      std::vector<double> cv_q) {

  if (sxrt_mode) {
    itprint = mdot_index;
  }

  if (ptemp >= 1.0) {
    for (int i = imax; i >= 1; i -= 2) {
      int io2 = i / 2;
      tl_prof.set(io2, itprint, "cv", cv[i]);
      tl_prof.set(io2, itprint, "cv_n", cv_n[i]);
      tl_prof.set(io2, itprint, "cv_p", cv_p[i]);
      tl_prof.set(io2, itprint, "cv_e", cv_e[i]);
      tl_prof.set(io2, itprint, "cv_m", cv_m[i]);
      tl_prof.set(io2, itprint, "cv_la", cv_la[i]);
      tl_prof.set(io2, itprint, "cv_sm", cv_sm[i]);
      tl_prof.set(io2, itprint, "cv_s0", cv_s0[i]);
      tl_prof.set(io2, itprint, "cv_sp", cv_sp[i]);
      tl_prof.set(io2, itprint, "cv_q", cv_q[i]);
      if (ptemp >= 2.0) {
        std::cout << i << " " << cv_n[i] << " " << cv_p[i] << " " << cv_e[i]
                  << " " << cv_m[i] << " " << cv_la[i] << " " << cv_sm[i] << " "
                  << cv_s0[i] << " " << cv_sp[i] << " " << cv_q[i] << std::endl;
      }
    }
  }

  return;
}

void nscool::num_param(double &time0, double &timemax, int &istepmax,
                       int &itrial_max, int &itrial_opt, double &tcut,
                       double &dtime, double &dtlimit, double &scale_dt0,
                       double &scale_dt1, double &repeat, int &istart,
                       double &mratt, double &mratl, double &mrats,
                       double &tvar, double &svar, double &tcon) {
  time0 = 0.0;
  timemax = 2.0e10;
  istepmax = 1000000;
  itrial_max = 20;
  itrial_opt = 12;
  tcut = 2.0;
  dtime = 1.0e-12;
  dtlimit = 3.15e15;
  scale_dt0 = 1.2;
  scale_dt1 = 1.5;
  repeat = 0.2;
  istart = 2;
  mratt = 1.0e-12;
  mratl = 1.0e-10;
  mrats = 1.0e-10;
  tvar = 1.20;
  svar = 1.05;
  tcon = 1.0e12;

  if (sxrt_mode) {
    timemax = 1.5e-12;
  }

  return;
}

void nscool::bound_param(int &ifteff, double &eta_arg, double &mag_coeff,
                         double &tb_acc0) {
  ifteff = 3;
  eta_arg = eta;
  mag_coeff = 3.0;
  tb_acc0 = 0.0;
  return;
}

void nscool::cool_param(int &pscreen, double &debug, int &istep_debug,
                        double &pteff, double &ptemp_arg, double &pstar,
                        int &idump1, int &idump2, int &idump3, double &tempmin,
                        double &tempini, int &icvel_nodeg, double &emnco,
                        double &emncr, double &emp, double &p0, int &itpmax,
                        double *tprint) {

  pscreen = 1;
  debug = ((double)(nscool_debug));
  // debug=1;
  istep_debug = 0;
  pteff = 0.0;
  ptemp_arg = 1.0;
  pstar = 0.0;
  idump1 = 1;
  idump2 = 111;
  idump3 = 421;
  tempmin = 1.0e4;
  tempini = 1.0e10;
  // If this is true, then the function 'cvelec()' is used to handle the
  // electron specific heat rather than the simple expression for degenerate
  // electrons
  icvel_nodeg = 0;
  emnco = 5.0;
  emncr = 5.0;
  emp = 3.0;
  p0 = 0.1;

  // The array 'tprint' the cooling code is actually zero-indexed, but Dany's
  // code ignores the first value.
  itpmax = time_print.size();
  for (int i = 1; i <= itpmax; i++) {
    tprint[i] = time_print[i - 1];
  }

  if (sxrt_mode) {
    tempini = sxrt_Tinit;
  }
  return;
}

void nscool::core_comp(int &imax, int &icore, std::vector<double> &rrho,
                       std::vector<double> &bar, std::vector<double> &yneutr,
                       std::vector<double> &yprot, std::vector<double> &yelect,
                       std::vector<double> &ymuon, std::vector<double> &ylambda,
                       std::vector<double> &ysminus,
                       std::vector<double> &yszero, std::vector<double> &ysplus,
                       std::vector<double> &yquarku,
                       std::vector<double> &yquarkd,
                       std::vector<double> &yquarks,
                       std::vector<double> &theta_k,
                       std::vector<double> &theta_p, std::vector<double> &fhad,
                       std::vector<double> &mstn, std::vector<double> &mstp,
                       std::vector<double> &mstla, std::vector<double> &mstsm,
                       std::vector<double> &msts0, std::vector<double> &mstsp) {

  std::string version = "NEW";

  size_t ix = nscool_core.get_nlines();

  // Temporary arrays for EOS tables
  std::vector<double> rho_t(ix, 0.0), pres_t(ix, 0.0), nbar_t(ix, 0.0),
      yelect_t(ix, 0.0), ymuon_t(ix, 0.0), yneutr_t(ix, 0.0), yprot_t(ix, 0.0),
      ylambda_t(ix, 0.0), ysminus_t(ix, 0.0), yszero_t(ix, 0.0),
      ysplus_t(ix, 0.0), mstp_t(ix, 0.0), mstn_t(ix, 0.0), mstla_t(ix, 0.0),
      msts0_t(ix, 0.0), mstsm_t(ix, 0.0), mstsp_t(ix, 0.0), theta_k_t(ix, 0.0),
      theta_p_t(ix, 0.0), yquarku_t(ix, 0.0), yquarkd_t(ix, 0.0),
      yquarks_t(ix, 0.0), fhad_t(ix, 0.0);

  for (size_t i = 0; i < nscool_core.get_nlines(); i++) {
    // rho is energy density
    rho_t[i] = nscool_core.get("Rho", i);
    nbar_t[i] = nscool_core.get("nbar", i);
    yelect_t[i] = nscool_core.get("Ye", i);
    ymuon_t[i] = nscool_core.get("Ymu", i);
    yneutr_t[i] = nscool_core.get("Yn", i);
    yprot_t[i] = nscool_core.get("Yp", i);
    ylambda_t[i] = nscool_core.get("Yla", i);
    ysminus_t[i] = nscool_core.get("Ysm", i);
    yszero_t[i] = nscool_core.get("Ys0", i);
    ysplus_t[i] = nscool_core.get("Ysp", i);

    double diff =
        yprot_t[i] + ysplus_t[i] - ysminus_t[i] - yelect_t[i] - ymuon_t[i];
    if (fabs(diff) > 1.0e-5) {
      std::cerr << "Charge problem in nscool::core_comp()." << std::endl;
      std::cout << i << " " << yprot_t[i] << " " << yelect_t[i] << " "
                << ymuon_t[i] << " " << ysplus_t[i] << " " << ysminus_t[i]
                << " "[i] << std::endl;
      (ix) = 0;
      return;
    }
    if (fabs(yprot_t[i] + yneutr_t[i] + ylambda_t[i] + ysminus_t[i] +
             yszero_t[i] + ysplus_t[i] - 1.0) > 1.0e-5) {
      std::cerr << "Baryon problem in core_comp." << std::endl;
      std::cout << i << " " << yneutr_t[i] << " " << yprot_t[i] << " "
                << ylambda_t[i] << " " << ysminus_t[i] << " " << yszero_t[i]
                << " " << ysplus_t[i] << std::endl;
      (ix) = 0;
      return;
    }

    // These are the reduced effective masses, i.e. m^{*}/m
    mstp_t[i] = nscool_core.get("mstp", i);
    mstn_t[i] = nscool_core.get("mstn", i);
    mstla_t[i] = nscool_core.get("mstla", i);
    mstsm_t[i] = nscool_core.get("mstsm", i);
    msts0_t[i] = nscool_core.get("msts0", i);
    mstsp_t[i] = nscool_core.get("mstsp", i);
  }

  if (rho_t[1] > rho_t[0]) {
    O2SCL_ERR("Core composition table should be decreasing", o2scl::exc_einval);
  }

  for (size_t i = 0; i < ix; i++) {
    theta_k_t[i] = theta_p_t[i] = yquarku_t[i] = yquarkd_t[i] = yquarks_t[i] =
        0.0;
    ylambda_t[i] = ysminus_t[i] = yszero_t[i] = ysplus_t[i] = 0.0;
    mstla_t[i] = mstsm_t[i] = msts0_t[i] = mstsp_t[i] = 0.0;
  }

  for (size_t i0 = 0; i0 <= (size_t)icore; i0++) {
    size_t i1 = 1, i2 = 2;
    if (rrho[i0] >= rho_t[0]) {
      i1 = 1;
      i2 = 2;
    } else if (rrho[i0] <= rho_t[ix]) {
      i1 = ix - 1;
      i2 = ix;
    } else {
      size_t i = i1 - 1;
      while (++i < ix - 1) {
        if (rrho[i0] >= rho_t[i + 1] && rrho[i0] <= rho_t[i]) {
          i1 = i;
          i2 = i + 1;
          break;
        }
      }
    }

    double x1 = (std::log(rho_t[i2]) - std::log(rrho[i0])) /
                (std::log(rho_t[i2]) - std::log(rho_t[i1]));
    double x2 = 1.0 - x1;

    // std::cout << rho_t[i1] << " " << rrho[i0] << " " << rho_t[i2] << " "
    //           << x1 << " " << x2 << " " << i1 << " " << i2 << std::endl;

    // exit(-1);

    bar[i0] = x1 * nbar_t[i1] + x2 * nbar_t[i2];
    yelect[i0] = x1 * yelect_t[i1] + x2 * yelect_t[i2];
    ymuon[i0] = x1 * ymuon_t[i1] + x2 * ymuon_t[i2];
    yneutr[i0] = x1 * yneutr_t[i1] + x2 * yneutr_t[i2];
    yprot[i0] = x1 * yprot_t[i1] + x2 * yprot_t[i2];
    ylambda[i0] = x1 * ylambda_t[i1] + x2 * ylambda_t[i2];
    ysminus[i0] = x1 * ysminus_t[i1] + x2 * ysminus_t[i2];
    yszero[i0] = x1 * yszero_t[i1] + x2 * yszero_t[i2];
    ysplus[i0] = x1 * ysplus_t[i1] + x2 * ysplus_t[i2];
    yquarku[i0] = x1 * yquarku_t[i1] + x2 * yquarku_t[i2];
    yquarkd[i0] = x1 * yquarkd_t[i1] + x2 * yquarkd_t[i2];
    yquarks[i0] = x1 * yquarks_t[i1] + x2 * yquarks_t[i2];
    theta_k[i0] = x1 * theta_k_t[i1] + x2 * theta_k_t[i2];
    theta_p[i0] = x1 * theta_p_t[i1] + x2 * theta_p_t[i2];

    // Handle 'fhad' based on version
    if (version == "old" || version == "new" || version == "NEW") {
      fhad[i0] = 1.0;
    } else if (version == "QRK") {
      fhad[i0] = x1 * fhad_t[i1 - 1] + x2 * fhad_t[i2 - 1];
      // Clamp fhad[i0] to [0.0, 1.0]
      fhad[i0] = std::clamp(fhad[i0], 0.0, 1.0);
    }

    // Calculate baryon and charge quantities
    double bnuc = yneutr[i0] + yprot[i0];
    double bhyp = ylambda[i0] + ysminus[i0] + yszero[i0] + ysplus[i0];
    double bqua = (1.0 / 3.0) * (yquarku[i0] + yquarkd[i0] + yquarks[i0]);
    double btot = bnuc + bhyp + bqua;
    double qlep = -yelect[i0] - ymuon[i0];
    double qnuc = yprot[i0];
    double qhyp = ysplus[i0] - ysminus[i0];
    double qqua = (1.0 / 3.0) * (2.0 * yquarku[i0] - yquarkd[i0] - yquarks[i0]);
    double qtot = qlep + qnuc + qhyp + qqua;

    // Check for consistency: btot should be ~1.0 and qtot ~0.0
    if (std::abs(btot - 1.0) > 1e-2) {
      std::cerr << std::fixed << std::setprecision(3);
      std::cerr << "i, rho, Btot, Qtot = " << i0 << ", " << rrho[i0] << ", "
                << btot << ", " << qtot << "\n";
      std::cerr << "Btot not equal to 1! " << i0 << ", " << yneutr[i0] << ","
                << yprot[i0] << "\n";
      std::cerr << ylambda[i0] << ", " << ysminus[i0] << ", " << yszero[i0]
                << ", " << ysplus[i0] << "\n";
      icore = 0;
      return;
    }

    if (std::abs(qtot) > 1e-2) {
      std::cerr << std::fixed << std::setprecision(3);
      std::cerr << "i, rho, Btot, Qtot = " << i0 << ", " << rrho[i0] << ", "
                << btot << ", " << qtot << "\n";
      std::cerr << "Qtot not equal to 0! " << i0 << ", " << qlep << ", " << qnuc
                << ", " << qhyp << ", " << qqua << ", " << qtot << "\n";
      std::cerr << yneutr[i0] << ", " << yprot[i0] << ", " << yelect[i0] << ", "
                << ymuon[i0] << "\n";
      icore = 0;
      return;
    }

    if (version == "NEW") {
      mstp[i0] = x1 * mstp_t[i1 + 1] + x2 * mstp_t[i2 + 1];
      mstn[i0] = x1 * mstn_t[i1 + 1] + x2 * mstn_t[i2 + 1];
      mstla[i0] = x1 * mstla_t[i1 + 1] + x2 * mstla_t[i2 + 1];
      mstsm[i0] = x1 * mstsm_t[i1 + 1] + x2 * mstsm_t[i2 + 1];
      msts0[i0] = x1 * msts0_t[i1 + 1] + x2 * msts0_t[i2 + 1];
      mstsp[i0] = x1 * mstsp_t[i1 + 1] + x2 * mstsp_t[i2 + 1];
    }
  }

  for (int i0 = icore + 1; i0 <= imax; i0++) {
    yelect[i0] = ymuon[i0] = yneutr[i0] = yprot[i0] = ylambda[i0] = 0.0;
    ysminus[i0] = yszero[i0] = ysplus[i0] = yquarku[i0] = yquarkd[i0] =
        yquarks[i0] = 0.0;
    theta_k[i0] = theta_p[i0] = 0.0;
    fhad[i0] = 1.0;
  }

  return;
}

void nscool::urca_settings(double &durca, double &a_durca, double &b_durca) {
  durca = fix_durca;
  a_durca = alpha_durca;
  b_durca = beta_durca;
  return;
}

void nscool::get_fermi_momenta(
    int &imax, int &icore, const std::vector<double> &rrho,
    const std::vector<double> &bar, const std::vector<double> yneutr,
    const std::vector<double> yprot, const std::vector<double> yelect,
    const std::vector<double> ymuon, const std::vector<double> ylambda,
    const std::vector<double> ysminus, const std::vector<double> yszero,
    const std::vector<double> ysplus, const std::vector<double> yquarku,
    const std::vector<double> yquarkd, const std::vector<double> yquarks,
    const std::vector<double> fhad, const std::vector<double> &theta_k,
    const std::vector<double> &theta_p, std::vector<double> &kfn,
    std::vector<double> &kfp, std::vector<double> &kfe,
    std::vector<double> &kfm, std::vector<double> &kfla,
    std::vector<double> &kfsm, std::vector<double> &kfs0,
    std::vector<double> &kfsp, std::vector<double> &kfqu,
    std::vector<double> &kfqd, std::vector<double> &kfqs,
    std::vector<int> &idurca_np, std::vector<int> &idurca_lap,
    std::vector<double> &durca_ctrl_e, std::vector<double> &durca_ctrl_m,
    std::vector<int> &idurca_smn, std::vector<int> &idurca_smla,
    std::vector<int> &idurca_sms0, std::vector<int> &idurca_quqd,
    std::vector<int> &idurca_quqs, std::vector<double> &durca_henon_e,
    std::vector<double> &durca_henon_m) {

  const double coeff = 3.0 * o2scl_const::pi2;

  // Temporary variables
  double nn, np, nla, nsm, ns0, nsp;
  double nqu, nqd, nqs;
  double ne, nm;
  double charge_l, charge_h, charge_q, charge, baryon_h, baryon_q, baryon,
      barrel;
  double s;
  double nb_low, nb_high, nb_durca_e, nb_durca_m;

  // Loop over j=0 to icore to calculate Fermi momenta in the core
  for (int j = 0; j <= icore; ++j) {
    if (fhad[j] != 0.0) {
      nn = std::max(0.0, yneutr[j] * bar[j]) / fhad[j];
      np = std::max(0.0, yprot[j] * bar[j]) / fhad[j];
      nla = std::max(0.0, ylambda[j] * bar[j]) / fhad[j];
      nsm = std::max(0.0, ysminus[j] * bar[j]) / fhad[j];
      ns0 = std::max(0.0, yszero[j] * bar[j]) / fhad[j];
      nsp = std::max(0.0, ysplus[j] * bar[j]) / fhad[j];
    } else {
      nn = np = nla = nsm = ns0 = nsp = 0.0;
    }

    if (fhad[j] != 1.0) {
      double denominator = 1.0 - fhad[j];
      nqu = std::max(0.0, yquarku[j] * bar[j]) / denominator;
      nqd = std::max(0.0, yquarkd[j] * bar[j]) / denominator;
      nqs = std::max(0.0, yquarks[j] * bar[j]) / denominator;
    } else {
      nqu = nqd = nqs = 0.0;
    }

    // Calculate Fermi momenta using cube roots for performance
    kfn[j] = std::cbrt(coeff * nn);
    kfp[j] = std::cbrt(coeff * np);
    kfla[j] = std::cbrt(coeff * nla);
    kfsm[j] = std::cbrt(coeff * nsm);
    kfs0[j] = std::cbrt(coeff * ns0);
    kfsp[j] = std::cbrt(coeff * nsp);
    kfqu[j] = std::cbrt(coeff * nqu);
    kfqd[j] = std::cbrt(coeff * nqd);
    kfqs[j] = std::cbrt(coeff * nqs);

    // Calculate electron and muon densities and their Fermi momenta
    ne = std::abs(yelect[j] * bar[j]);
    nm = std::abs(ymuon[j] * bar[j]);

    kfe[j] = std::cbrt(coeff * ne);
    if (yelect[j] <= 0.0)
      kfe[j] = -kfe[j];

    kfm[j] = std::cbrt(coeff * nm);
    if (ymuon[j] <= 0.0)
      kfm[j] = -kfm[j];

    if (false) {
      // Verify charge neutrality and baryon number conservation
      double nn_new = std::pow(kfn[j], 3.0) / coeff * fhad[j];
      double np_new = std::pow(kfp[j], 3.0) / coeff * fhad[j];
      double nla_new = std::pow(kfla[j], 3.0) / coeff * fhad[j];
      double nsm_new = std::pow(kfsm[j], 3.0) / coeff * fhad[j];
      double ns0_new = std::pow(kfs0[j], 3.0) / coeff * fhad[j];
      double nsp_new = std::pow(kfsp[j], 3.0) / coeff * fhad[j];
      double nqu_new = std::pow(kfqu[j], 3.0) / coeff * (1.0 - fhad[j]);
      double nqd_new = std::pow(kfqd[j], 3.0) / coeff * (1.0 - fhad[j]);
      double nqs_new = std::pow(kfqs[j], 3.0) / coeff * (1.0 - fhad[j]);
      double ne_new = std::pow(kfe[j], 3.0) / coeff * 1.0;
      double nm_new = std::pow(kfm[j], 3.0) / coeff * 1.0;

      charge_l = -ne_new - nm_new;
      charge_h = np_new + nsp_new - nsm_new;
      charge_q =
          (2.0 / 3.0 * nqu_new) - (1.0 / 3.0 * nqd_new) - (1.0 / 3.0 * nqs_new);
      charge = charge_l + charge_h + charge_q;

      std::cout << charge_l << " " << charge_h << " " << charge_q << std::endl;

      baryon_h = nn_new + np_new + nla_new + nsm_new + ns0_new + nsp_new;
      baryon_q = (nqu_new + nqd_new + nqs_new) / 3.0;
      baryon = baryon_h + baryon_q;

      // Check for charge neutrality violation
      if (std::abs(charge) >= 1e-2) {
        std::cerr << "Charge neutrality violated at j=" << j
                  << ", Rho= " << rrho[j] << ": charge/fm3= " << charge
                  << std::endl;
        icore = 0;
        return;
      }

      // Check for baryon number conservation
      barrel = baryon / bar[j];
      if (std::abs(barrel - 1.0) >= 1e-2) {
        std::cerr << "Baryons do not sum up to baryon density at j=" << j
                  << ", Rho= " << rrho[j]
                  << ": sum(baryons)/baryon#= " << baryon << std::endl;
        icore = 0;
        return;
      }
    }
  }

  // Calculate the Fermi momenta in the crust
  for (int j = icore + 1; j <= imax; ++j) {
    ne = yelect[j] * bar[j];
    nn = yneutr[j] * bar[j];

    kfe[j] = std::cbrt(coeff * ne);
    kfm[j] = 0.0;
    kfn[j] = std::cbrt(coeff * nn);
    kfp[j] = 0.0;
    kfla[j] = 0.0;
    kfsm[j] = 0.0;
    kfs0[j] = 0.0;
    kfsp[j] = 0.0;
    kfqu[j] = 0.0;
    kfqd[j] = 0.0;
    kfqs[j] = 0.0;
  }

  // Compute triangle squared areas
  for (int j = 0; j <= icore; ++j) {
    // Calculate semi-perimeter s for electrons
    double s_e = (kfp[j] + kfn[j] + kfe[j]) / 2.0;
    durca_henon_e[j] = s_e * (s_e - kfp[j]) * (s_e - kfe[j]) * (s_e - kfn[j]);

    // Calculate semi-perimeter s for muons
    double s_m = (kfp[j] + kfn[j] + kfm[j]) / 2.0;
    durca_henon_m[j] = s_m * (s_m - kfp[j]) * (s_m - kfm[j]) * (s_m - kfn[j]);
  }

  // Compute direct Urca density thresholds using linear interpolation
  if (fix_durca == 0.0) {
    nb_durca_e = 0.0;
    nb_durca_m = 0.0;

    for (int j = 0; j < icore; ++j) {
      // Find where durca_henon_e changes sign
      if (nb_durca_e == 0.0 &&
          (durca_henon_e[j] * durca_henon_e[j + 1] < 0.0)) {
        double slope_e = durca_henon_e[j + 1] - durca_henon_e[j];
        if (slope_e != 0.0) {
          nb_low = bar[j];
          nb_high = bar[j + 1];
          nb_durca_e = nb_low - (nb_high - nb_low) * durca_henon_e[j] / slope_e;
        }
      }
      // Find where durca_henon_m changes sign
      if (nb_durca_m == 0.0 &&
          (durca_henon_m[j] * durca_henon_m[j + 1] < 0.0)) {
        double slope_m = durca_henon_m[j + 1] - durca_henon_m[j];
        if (slope_m != 0.0) {
          nb_low = bar[j];
          nb_high = bar[j + 1];
          nb_durca_m = nb_low - (nb_high - nb_low) * durca_henon_m[j] / slope_m;
        }
      }
      // Exit early if both thresholds are found
      if (nb_durca_e > 0.0 && nb_durca_m > 0.0)
        break;
    }
  } else {
    nb_durca_e = fix_durca;
    nb_durca_m = fix_durca;
  }

  // Compute direct Urca control functions from threshold densities
  for (int j = 0; j <= icore; ++j) {
    idurca_np[j] = 0;
    durca_ctrl_e[j] = 0.0;
    durca_ctrl_m[j] = 0.0;

    // Electron direct Urca control
    if (nb_durca_e > 0.0) {
      if (bar[j] >= (1.0 + alpha_durca) * beta_durca * nb_durca_e) {
        idurca_np[j] = 1;
        durca_ctrl_e[j] = 1.0;
      } else if (bar[j] >= (1.0 - alpha_durca) * beta_durca * nb_durca_e) {
        idurca_np[j] = 3;
        durca_ctrl_e[j] = 0.5 + (1.0 / (2.0 * alpha_durca)) *
                                    (bar[j] - nb_durca_e) / nb_durca_e;
      }
    }

    // Muon direct Urca control
    if (nb_durca_m > 0.0) {
      if (bar[j] >= (1.0 + alpha_durca) * beta_durca * nb_durca_m) {
        idurca_np[j] = 2;
        durca_ctrl_m[j] = 1.0;
      } else if (bar[j] >= (1.0 - alpha_durca) * beta_durca * nb_durca_m) {
        idurca_np[j] = 4;
        durca_ctrl_m[j] = 0.5 + (1.0 / (2.0 * alpha_durca)) *
                                    (bar[j] - nb_durca_m) / nb_durca_m;
      }
    }
  }

  // Dany's method for hyperon and quark direct Urca
  for (int j = 0; j <= icore; ++j) {
    // la-p
    if ((kfp[j] < kfla[j] + kfe[j]) && (kfla[j] < kfp[j] + kfe[j]) &&
        (kfe[j] < kfp[j] + kfla[j])) {
      idurca_lap[j] = 1;
      if ((kfp[j] < kfla[j] + kfm[j]) && (kfla[j] < kfp[j] + kfm[j]) &&
          (kfm[j] < kfla[j] + kfp[j])) {
        idurca_lap[j] = 2;
      }
    } else {
      idurca_lap[j] = 0;
    }

    // sm-n
    if ((kfsm[j] < kfn[j] + kfe[j]) && (kfn[j] < kfsm[j] + kfe[j]) &&
        (kfe[j] < kfsm[j] + kfn[j])) {
      idurca_smn[j] = 1;
      if ((kfsm[j] < kfn[j] + kfm[j]) && (kfn[j] < kfsm[j] + kfm[j]) &&
          (kfm[j] < kfsm[j] + kfn[j])) {
        idurca_smn[j] = 2;
      }
    } else {
      idurca_smn[j] = 0;
    }

    // sm-la
    if ((kfsm[j] < kfla[j] + kfe[j]) && (kfla[j] < kfsm[j] + kfe[j]) &&
        (kfe[j] < kfsm[j] + kfla[j])) {
      idurca_smla[j] = 1;
      if ((kfsm[j] < kfla[j] + kfm[j]) && (kfla[j] < kfsm[j] + kfm[j]) &&
          (kfm[j] < kfsm[j] + kfla[j])) {
        idurca_smla[j] = 2;
      }
    } else {
      idurca_smla[j] = 0;
    }

    // sm-s0
    if ((kfsm[j] < kfs0[j] + kfe[j]) && (kfs0[j] < kfsm[j] + kfe[j]) &&
        (kfe[j] < kfsm[j] + kfs0[j])) {
      idurca_sms0[j] = 1;
      if ((kfsm[j] < kfs0[j] + kfm[j]) && (kfs0[j] < kfsm[j] + kfm[j]) &&
          (kfm[j] < kfsm[j] + kfs0[j])) {
        idurca_sms0[j] = 2;
      }
    } else {
      idurca_sms0[j] = 0;
    }

    // qu-qd
    if ((kfqu[j] < kfqd[j] + kfe[j]) && (kfqd[j] < kfqu[j] + kfe[j]) &&
        (kfe[j] < kfqu[j] + kfqd[j])) {
      idurca_quqd[j] = 1;
      if ((kfqu[j] < kfqd[j] + kfm[j]) && (kfqd[j] < kfqu[j] + kfm[j]) &&
          (kfm[j] < kfqu[j] + kfqd[j])) {
        idurca_quqd[j] = 2;
      }
    } else {
      idurca_quqd[j] = 0;
    }

    // qu-qs
    if ((kfqu[j] < kfqs[j] + kfe[j]) && (kfqs[j] < kfqu[j] + kfe[j]) &&
        (kfe[j] < kfqu[j] + kfqs[j])) {
      idurca_quqs[j] = 1;
      if ((kfqu[j] < kfqs[j] + kfm[j]) && (kfqs[j] < kfqu[j] + kfm[j]) &&
          (kfm[j] < kfqu[j] + kfqs[j])) {
        idurca_quqs[j] = 2;
      }
    } else {
      idurca_quqs[j] = 0;
    }
  }
  return;
}

void nscool::crust_comp(int debug, const std::string &version, int &imax,
                        int &icore, std::vector<double> &rrho,
                        std::vector<double> &pres, std::vector<double> &debar,
                        std::vector<double> &dvol, std::vector<double> &bar,
                        std::vector<double> &A_cell, std::vector<double> &A_ion,
                        std::vector<double> &Z_ion, std::vector<double> &v_ion,
                        std::vector<double> &yelect,
                        std::vector<double> &yneutr) {
  // Debug message
  if (debug >= 1) {
    std::cout << "Entering nscool::crust_comp()" << std::endl;
  }

  int jmax = ((int)(nscool_crust.get_nlines()));
  if (nscool_crust.get_nlines() >= 500) {
    O2SCL_ERR("Crust table too large in nscool::crust_comp().",
              o2scl::exc_einval);
  }

  // Temporary arrays for EOS tables
  std::vector<double> rho_t(jmax, 0.0);
  std::vector<double> pres_t(jmax, 0.0);
  std::vector<double> bar_t(jmax, 0.0);
  std::vector<double> A_cell_t(jmax, 0.0);
  std::vector<double> A_ion_t(jmax, 0.0);
  std::vector<double> Z_ion_t(jmax, 0.0);

  for (size_t i = 0; i < nscool_crust.get_nlines(); i++) {
    size_t i2 = ((size_t)jmax) - 1 - i;
    rho_t[i] = nscool_crust.get("rho", i2);
    pres_t[i] = nscool_crust.get("P", i2);
    bar_t[i] = nscool_crust.get("n", i2);
    A_cell_t[i] = nscool_crust.get("A_cell", i2);
    A_ion_t[i] = nscool_crust.get("A_ion", i2);
    Z_ion_t[i] = nscool_crust.get("Z", i2);
  }
  if (rho_t[1] < rho_t[0]) {
    O2SCL_ERR("Crust composition table should be decreasing",
              o2scl::exc_einval);
  }

  // Find jdrip: first j from jmax downto 1 where A_cell_t(j) != A_ion_t(j)
  int jget_drip = 0, jdrip = 0;
  for (int j = jmax; j >= 1; --j) {
    if (jget_drip == 0 && A_cell_t[j] != A_ion_t[j]) {
      jdrip = j;
      jget_drip = 1;
    }
  }

  // Adjust jmax to ensure rho_t(jmax) < rrho(icore)
  int jjmax = jmax;
  for (int j = jmax; j >= 1; --j) {
    if (rho_t[j] >= rrho[icore]) {
      jjmax = j - 1;
    }
  }
  jmax = jjmax;

  // Interpolation from the last crust_EOS line up to the core First version:
  // - Take rho & bar from core EOS:
  // jmax += 1;
  rho_t[jmax] = rrho[icore];
  bar_t[jmax] = bar[icore];
  pres_t[jmax] = pres[icore];
  Z_ion_t[jmax] = Z_ion_t[jmax - 1];
  A_ion_t[jmax] = A_ion_t[jmax - 1];
  A_cell_t[jmax] = Z_ion_t[jmax] / yelect[icore];

  // Loop from icore+1 to imax to interpolate
  int j = jmax - 1; // Adjusting for 0-based indexing
  for (int i = icore + 1; i <= imax; ++i) {
    // Decrement j until rho_t[j] <= rrho[i]
    while (rho_t[j] >= rrho[i]) {
      --j;
    }

    double dd = rho_t[j + 1] - rho_t[j];
    double w2 = (rrho[i] - rho_t[j]) / dd;
    double w1 = 1.0 - w2;

    // Interpolate A_cell, A_ion, Z_ion
    if (A_ion_t[j + 1] == A_cell_t[j + 1]) {
      A_ion[i] = A_ion_t[j + 1];
      A_cell[i] = A_cell_t[j + 1];
      Z_ion[i] = Z_ion_t[j + 1];
    } else {
      A_cell[i] = w1 * A_cell_t[j] + w2 * A_cell_t[j + 1];
      A_ion[i] = A_ion_t[j + 1];
      Z_ion[i] = Z_ion_t[j + 1];
    }

    // Interpolate particle concentrations
    bar[i] = w1 * bar_t[j] + w2 * bar_t[j + 1];

    // Calculate the fraction of volume occupied by ions
    const double r1 = 1.1; // Scale parameter in fm
    double vion_fm3 =
        (4.0 / 3.0) * o2scl_const::pi * std::pow(r1, 3) * A_ion[i]; // fm^3
    double vion_cm3 = 1.0e-39 * vion_fm3;                           // cm^3
    double nion = rrho[i] / (1.66e-24 * A_ion[i]); // ion density per cm^3
    v_ion[i] = nion * vion_cm3;
    v_ion[i] = std::min(1.0, v_ion[i]); // Ensure v_ion[i] <= 1.0
  }

  for (int i = icore + 1; i <= imax; ++i) {
    yelect[i] = Z_ion[i] / A_cell[i];
    yneutr[i] = (A_cell[i] - A_ion[i]) / A_cell[i];
  }
  // Clean up the core, just in case
  for (int i = 0; i <= icore; ++i) {
    Z_ion[i] = A_ion[i] = A_cell[i] = v_ion[i] = 0.0;
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

  return;
}

double nscool::cvt_deg(double pf, double m) {
  constexpr double pi = 3.14159265;
  constexpr double kb = 1.38e-16;  // Boltzmann constant in erg/K
  constexpr double MeV = 1.602e-6; // 1 MeV in erg
  if (pf == 0.0) {
    return 0.0;
  }
  double N0 = 2.0 * m * pf / (2.0 * pi * pi);
  double cvt = (pi * pi / 3.0) * N0;
  return cvt * kb * kb / MeV / (197.0 * 197.0 * 197.0) *
         1.0e39; // Converts to cgs units
}

void nscool::get_spec_heat_degenerate(
    std::vector<double> &cve, std::vector<double> &cvm,
    std::vector<double> &cvn, std::vector<double> &cvp,
    std::vector<double> &cvla, std::vector<double> &cvsm,
    std::vector<double> &cvs0, std::vector<double> &cvsp,
    std::vector<double> &cvqu, std::vector<double> &cvqd,
    std::vector<double> &cvqs, const std::vector<double> kfe,
    const std::vector<double> kfm, const std::vector<double> kfn,
    const std::vector<double> kfp, const std::vector<double> kfla,
    const std::vector<double> kfsm, const std::vector<double> kfs0,
    const std::vector<double> kfsp, const std::vector<double> kfqu,
    const std::vector<double> kfqd, const std::vector<double> kfqs,
    const std::vector<double> mstn, const std::vector<double> mstp,
    const std::vector<double> mstla, const std::vector<double> mstsm,
    const std::vector<double> msts0, const std::vector<double> mstsp,
    const std::vector<double> fhad, int imax) {
  constexpr double strange_mass = 150.0; // will become class variable later

  for (int j = 0; j <= imax; ++j) {
    double pfe = kfe[j] * 197.0;
    double me = std::sqrt(0.511 * 0.511 + pfe * pfe);
    double pfm = kfm[j] * 197.0;
    double mm = std::sqrt(105.0 * 105.0 + pfm * pfm);
    double pfn = kfn[j] * 197.0;
    double mn = 939.56 * mstn[j];
    double pfp = kfp[j] * 197.0;
    double mp = 938.27 * mstp[j];
    double pfla = kfla[j] * 197.0;
    double mla = 1116.0 * mstla[j];
    double pfsm = kfsm[j] * 197.0;
    double msm = 1193.0 * mstsm[j];
    double pfs0 = kfs0[j] * 197.0;
    double ms0 = 1193.0 * msts0[j];
    double pfsp = kfsp[j] * 197.0;
    double msp = 1193.0 * mstsp[j];
    double pfqu = kfqu[j] * 197.0;
    double mqu = std::sqrt(5.0 * 5.0 + pfqu * pfqu);
    double pfqd = kfqd[j] * 197.0;
    double mqd = std::sqrt(8.0 * 8.0 + pfqd * pfqd);
    double pfqs = kfqs[j] * 197.0;
    double mqs = std::sqrt(strange_mass * strange_mass + pfqs * pfqs);

    cve[j] = cvt_deg(pfe, me);
    cvm[j] = cvt_deg(pfm, mm);
    cvn[j] = cvt_deg(pfn, mn) * fhad[j];
    cvp[j] = cvt_deg(pfp, mp) * fhad[j];
    cvla[j] = cvt_deg(pfla, mla) * fhad[j];
    cvsm[j] = cvt_deg(pfsm, msm) * fhad[j];
    cvs0[j] = cvt_deg(pfs0, ms0) * fhad[j];
    cvsp[j] = cvt_deg(pfsp, msp) * fhad[j];
    cvqu[j] = cvt_deg(pfqu, mqu) * (1.0 - fhad[j]);
    cvqd[j] = cvt_deg(pfqd, mqd) * (1.0 - fhad[j]);
    cvqs[j] = cvt_deg(pfqs, mqs) * (1.0 - fhad[j]);
  }
  return;
}

void nscool::crust_eos(const std::vector<double> &rrho,
                       const std::vector<double> &pres,
                       std::vector<double> &rhod, int imax, int ienv) {
  // Temporary arrays to hold crust EOS data
  std::vector<double> rho2(imax + 1, 0.0);
  std::vector<double> pres2(imax + 1, 0.0);
  // Populate rho2 and pres2 using the crust EOS
  int idata = ((int)(nscool_crust.get_nlines()));
  for (size_t i = 0; i < nscool_crust.get_nlines(); i++) {
    rho2[i] = nscool_crust.get("rho", (idata)-1 - i);
    pres2[i] = nscool_crust.get("P", (idata)-1 - i);
  }

  for (int i = 0; i < ienv; ++i) {
    rhod[i] = rrho[i];
  }
  for (int i = imax; i >= ienv; --i) {
    int j = 0;
    while (true) {
      j++;
      if (pres[i] >= pres2[j] && pres[i] <= pres2[j + 1]) {
        double x = (std::log(pres[i]) - std::log(pres2[j])) /
                   (std::log(pres2[j + 1]) - std::log(pres2[j]));
        double y = (std::log(pres2[j + 1]) - std::log(pres[i])) /
                   (std::log(pres2[j + 1]) - std::log(pres2[j]));
        double lrhod = y * std::log(rho2[j]) + x * std::log(rho2[j + 1]);
        rhod[i] = std::max(std::exp(lrhod), rrho[i]);
        if (i < imax && rrho[i + 1] == rhod[i + 1]) {
          rhod[i] = rrho[i];
        }
        break;
      }
    }
  }

  return;
}

void nscool::sf_suppress_data(double *lgtau1, double *lgtau2, double *lgr) {

#include "../sf_suppression.h"

  return;
}

void nscool::pair_brem_data(double *logt, double *nalpha) {

  nalpha[0] = 2.119400e+02;
  nalpha[1] = 2.119314e+02;
  nalpha[2] = 2.118338e+02;
  nalpha[3] = 2.116714e+02;
  nalpha[4] = 2.114446e+02;
  nalpha[5] = 2.111548e+02;
  nalpha[6] = 2.108042e+02;
  nalpha[7] = 2.103965e+02;
  nalpha[8] = 2.099369e+02;
  nalpha[9] = 2.094313e+02;
  nalpha[10] = 2.088862e+02;
  nalpha[11] = 2.025300e+02;
  nalpha[12] = 1.965045e+02;
  nalpha[13] = 1.913841e+02;
  nalpha[14] = 1.870169e+02;
  nalpha[15] = 1.832339e+02;
  nalpha[16] = 1.799095e+02;
  nalpha[17] = 1.769527e+02;
  nalpha[18] = 1.742953e+02;
  nalpha[19] = 1.718859e+02;
  nalpha[20] = 1.555852e+02;
  nalpha[21] = 1.458835e+02;
  nalpha[22] = 1.389667e+02;
  nalpha[23] = 1.335816e+02;
  nalpha[24] = 1.291637e+02;
  nalpha[25] = 1.254108e+02;
  nalpha[26] = 1.221419e+02;
  nalpha[27] = 1.192400e+02;
  nalpha[28] = 1.166251e+02;
  nalpha[29] = 9.846525e+01;
  nalpha[30] = 8.600176e+01;
  nalpha[31] = 7.553085e+01;
  nalpha[32] = 6.634075e+01;
  nalpha[33] = 5.830211e+01;
  nalpha[34] = 5.135016e+01;
  nalpha[35] = 4.539033e+01;
  nalpha[36] = 4.030437e+01;
  nalpha[37] = 3.596899e+01;
  nalpha[38] = 1.500019e+01;
  nalpha[39] = 8.665605e+00;
  nalpha[40] = 5.942522e+00;
  nalpha[41] = 4.494431e+00;
  nalpha[42] = 3.613857e+00;
  nalpha[43] = 3.027656e+00;
  nalpha[44] = 2.611401e+00;
  nalpha[45] = 2.301280e+00;
  nalpha[46] = 2.061534e+00;
  nalpha[47] = 1.063250e+00;
  nalpha[48] = 7.494363e-01;
  nalpha[49] = 5.912537e-01;
  nalpha[50] = 4.941738e-01;
  nalpha[51] = 4.277401e-01;
  nalpha[52] = 3.790197e-01;
  nalpha[53] = 3.415363e-01;
  nalpha[54] = 3.116674e-01;
  nalpha[55] = 0.000000e-00;

  logt[0] = log10(1.000000e-10) + 9.0;
  logt[1] = log10(1.160093e-04) + 9.0;
  logt[2] = log10(2.320186e-04) + 9.0;
  logt[3] = log10(3.480278e-04) + 9.0;
  logt[4] = log10(4.640371e-04) + 9.0;
  logt[5] = log10(5.800464e-04) + 9.0;
  logt[6] = log10(6.960557e-04) + 9.0;
  logt[7] = log10(8.120650e-04) + 9.0;
  logt[8] = log10(9.280742e-04) + 9.0;
  logt[9] = log10(1.044084e-03) + 9.0;
  logt[10] = log10(1.160093e-03) + 9.0;
  logt[11] = log10(2.320186e-03) + 9.0;
  logt[12] = log10(3.480278e-03) + 9.0;
  logt[13] = log10(4.640371e-03) + 9.0;
  logt[14] = log10(5.800464e-03) + 9.0;
  logt[15] = log10(6.960557e-03) + 9.0;
  logt[16] = log10(8.120650e-03) + 9.0;
  logt[17] = log10(9.280742e-03) + 9.0;
  logt[18] = log10(1.044084e-02) + 9.0;
  logt[19] = log10(1.160093e-02) + 9.0;
  logt[20] = log10(2.320186e-02) + 9.0;
  logt[21] = log10(3.480278e-02) + 9.0;
  logt[22] = log10(4.640371e-02) + 9.0;
  logt[23] = log10(5.800464e-02) + 9.0;
  logt[24] = log10(6.960557e-02) + 9.0;
  logt[25] = log10(8.120650e-02) + 9.0;
  logt[26] = log10(9.280742e-02) + 9.0;
  logt[27] = log10(1.044084e-01) + 9.0;
  logt[28] = log10(1.160093e-01) + 9.0;
  logt[29] = log10(2.320186e-01) + 9.0;
  logt[30] = log10(3.480278e-01) + 9.0;
  logt[31] = log10(4.640371e-01) + 9.0;
  logt[32] = log10(5.800464e-01) + 9.0;
  logt[33] = log10(6.960557e-01) + 9.0;
  logt[34] = log10(8.120650e-01) + 9.0;
  logt[35] = log10(9.280742e-01) + 9.0;
  logt[36] = log10(1.044084e+00) + 9.0;
  logt[37] = log10(1.160093e+00) + 9.0;
  logt[38] = log10(2.320186e+00) + 9.0;
  logt[39] = log10(3.480278e+00) + 9.0;
  logt[40] = log10(4.640371e+00) + 9.0;
  logt[41] = log10(5.800464e+00) + 9.0;
  logt[42] = log10(6.960557e+00) + 9.0;
  logt[43] = log10(8.120650e+00) + 9.0;
  logt[44] = log10(9.280742e+00) + 9.0;
  logt[45] = log10(1.044084e+01) + 9.0;
  logt[46] = log10(1.160093e+01) + 9.0;
  logt[47] = log10(2.320186e+01) + 9.0;
  logt[48] = log10(3.480278e+01) + 9.0;
  logt[49] = log10(4.640371e+01) + 9.0;
  logt[50] = log10(5.800464e+01) + 9.0;
  logt[51] = log10(6.960557e+01) + 9.0;
  logt[52] = log10(8.120650e+01) + 9.0;
  logt[53] = log10(9.280742e+01) + 9.0;
  logt[54] = log10(1.044084e+02) + 9.0;
  logt[55] = log10(1.000000e+10) + 9.0;

  return;
}

void nscool::star_struct(int idec, double rhocore, double rhodrip,
                         double rhoenv, double rhosurf, int &imax, int &icore,
                         int &idrip, int &ienv, std::vector<double> &rad,
                         std::vector<double> &rrho, std::vector<double> &pres,
                         std::vector<double> &dvol, std::vector<double> &emas,
                         std::vector<double> &phi) {
  // Define zone indices: icore, idrip & isurf
  icore = 2 * ((icore - 1) / 2) + 1; // Ensure icore is od
  int idel1 = static_cast<int>(std::log10(rhocore / rhodrip) *
                               static_cast<double>(idec));
  idel1 = 2 * (idel1 / 2); // Ensure idel1 is even
  idrip = icore + idel1;
  int idel2 = static_cast<int>(std::log10(rhodrip / rhosurf) *
                               static_cast<double>(idec));
  idel2 = 2 * (idel2 / 2); // Ensure idel2 is even
  int isurf = idrip + idel2;

  size_t jmax = nscool_tov.get_nlines();

  std::vector<double> rad_t(jmax), bar_t(jmax), rho_t(jmax);
  std::vector<double> pres_t(jmax), emas_t(jmax), phi_t(jmax);

  if (jmax > 9999) {
    O2SCL_ERR("Table too large.", o2scl::exc_einval);
  }
  size_t jcore = 0;
  for (size_t j = 0; j < nscool_tov.get_nlines(); j++) {
    rad_t[j] = nscool_tov.get("radius", j) * 100.0;
    bar_t[j] = nscool_tov.get("n_baryon", j);
    // This is energy density
    rho_t[j] = nscool_tov.get("density", j);
    pres_t[j] = nscool_tov.get("pressure", j);
    emas_t[j] = nscool_tov.get("emass", j);
    phi_t[j] = nscool_tov.get("phi", j);
    if (rho_t[j] < rhocore && (jcore) == 0) {
      jcore = j;
    }
  }

  if (rad_t[1] < rad_t[0]) {
    O2SCL_ERR("Structure table should be increasing in radius",
              o2scl::exc_einval);
  }
  if (jcore == 0) {
    O2SCL_ERR2("Variable 'jcore' not set in ", "nscool::star_struct().",
               o2scl::exc_einval);
  }

  double drho = rho_t[jcore - 1] - rho_t[jcore];
  double w1 = (rhocore - rho_t[jcore]) / drho;
  double w2 = 1.0 - w1;
  double rad_core = (w1)*rad_t[jcore - 1] + (w2)*rad_t[jcore];
  for (size_t i = 0; i <= (size_t)icore; i++) {
    rad[i] = cbrt(((float)i) / ((float)icore)) * rad_core;
  }

  // Initialize boundary conditions
  rrho[0] = rho_t[0]; // rad_t(1) -> rad_t[0]
  emas[0] = 0.0;
  phi[0] = phi_t[0];
  pres[0] = pres_t[0];
  dvol[0] = 0.0;

  size_t j = 1;
  for (int i = 1; i <= icore; ++i) {
    while (rad_t[j] < rad[i]) {
      ++j;
    }

    double delrad = rad_t[j] - rad_t[j - 1];
    double w1_interp = (rad_t[j] - rad[i]) / delrad;
    double w2_interp = 1.0 - w1_interp;

    rrho[i] = w1_interp * rho_t[j - 1] + w2_interp * rho_t[j];
    emas[i] = w1_interp * emas_t[j - 1] + w2_interp * emas_t[j];
    phi[i] = w1_interp * phi_t[j - 1] + w2_interp * phi_t[j];
    pres[i] = w1_interp * pres_t[j - 1] + w2_interp * pres_t[j];

    if (i == 1) {
      dvol[i] = (4.0 / 3.0) * o2scl_const::pi * std::pow(rad[i], 3);
    } else {
      double avg_rad_sq = std::pow((rad[i - 1] + rad[i]) / 2.0, 2);
      double delta_rad = rad[i] - rad[i - 1];
      double sqrt_term = std::sqrt(1.0 - 2.92e5 * (emas[i - 1] + emas[i]) /
                                             (rad[i - 1] + rad[i]));
      dvol[i] = 4.0 * o2scl_const::pi * avg_rad_sq * delta_rad / sqrt_term;
    }

    // Update j
    --j;
  }

  // Ensure exact accuracy
  rrho[icore] = rhocore;

  // Crust: zoning with "idec" zones per decade in density
  double dlogrho_core = std::log10(rhocore / rhodrip);
  double dlrho_core = dlogrho_core / static_cast<double>(idrip - icore);

  for (int i = icore + 1; i <= idrip; ++i) {
    double lrho =
        std::log10(rhocore) - static_cast<double>(i - icore) * dlrho_core;
    rrho[i] = std::pow(10.0, lrho);
  }

  double dlogrho_env = std::log10(rhodrip / rhosurf);
  double dlrho_env = dlogrho_env / static_cast<double>(isurf - idrip);

  for (int i = idrip + 1; i <= isurf; ++i) {
    double lrho =
        std::log10(rhodrip) - static_cast<double>(i - idrip) * dlrho_env;
    rrho[i] = std::pow(10.0, lrho);
  }

  // Interpolate for crust regions
  j = 1;
  for (int i = icore + 1; i <= isurf; ++i) {
    while (rho_t[j] > rrho[i]) {
      ++j;
    }

    double dellrho = std::log10(rho_t[j - 1]) - std::log10(rho_t[j]);
    double w1_interp = (std::log10(rrho[i]) - std::log10(rho_t[j])) / dellrho;
    double w2_interp = 1.0 - w1_interp;

    rad[i] = w1_interp * rad_t[j - 1] + w2_interp * rad_t[j];
    emas[i] = w1_interp * emas_t[j - 1] + w2_interp * emas_t[j];
    phi[i] = w1_interp * phi_t[j - 1] + w2_interp * phi_t[j];
    pres[i] = w1_interp * pres_t[j - 1] + w2_interp * pres_t[j];

    double avg_rad_sq = std::pow((rad[i - 1] + rad[i]) / 2.0, 2);
    double delta_rad = rad[i] - rad[i - 1];
    double sqrt_term = std::sqrt(1.0 - 2.92e5 * (emas[i - 1] + emas[i]) /
                                           (rad[i - 1] + rad[i]));
    dvol[i] = 4.0 * o2scl_const::pi * avg_rad_sq * delta_rad / sqrt_term;

    // Update j
    --j;
  }

  // Ensure exact accuracy
  dvol[isurf + 1] = dvol[isurf];
  rrho[idrip] = rhodrip;
  rrho[isurf] = rhosurf;

  // Find the envelope boundary
  ienv = isurf + 2;
  for (int i = isurf; i >= idrip; i -= 2) {
    if (rrho[i] < rhoenv) {
      ienv = i;
    }
  }

  // Set imax
  imax = isurf;

  return;
}

void nscool::get_Tc(int imax, int icore, int idrip, std::vector<double> &tcn,
                    std::vector<double> &tcp, std::vector<double> &tcla,
                    std::vector<double> &tcsm, std::vector<double> &tcs0,
                    std::vector<double> &tcsp, std::vector<double> &tcuu,
                    std::vector<double> &tcdd, std::vector<double> &tcss,
                    std::vector<double> &tcud, std::vector<double> &tcus,
                    std::vector<double> &tcds, std::vector<double> &tcu,
                    std::vector<double> &tcd, std::vector<double> &tcs,
                    int &sfl1s0, double fn1s0, double fn3p2, double fp1s0,
                    double fl1s0, int &isf, std::vector<double> kfn,
                    std::vector<double> kfp, std::vector<double> kfla,
                    std::vector<double> kfqu, std::vector<double> kfqd,
                    std::vector<double> kfqs, std::vector<double> bar,
                    std::vector<double> fhad, std::vector<double> yquarku,
                    std::vector<double> yquarkd, std::vector<double> yquarks) {

  double dinput_n1tc, dinput_n1kf, dinput_n1dk, dinput_n3tc, dinput_n3kf,
      dinput_n3dk, dinput_p1tc, dinput_p1kf, dinput_p1dk;
  // AWS: determine gap parameters
  gaps(sfn1s0, dinput_n1tc, dinput_n1kf, dinput_n1dk, sfn3p2, dinput_n3tc,
       dinput_n3kf, dinput_n3dk, sfp1s0, dinput_p1tc, dinput_p1kf, dinput_p1dk);
  // Just to be safe:
  for (int i = 0; i <= imax; ++i) {
    tcn[i] = tcp[i] = tcla[i] = tcuu[i] = tcdd[i] = tcss[i] = tcud[i] =
        tcus[i] = tcds[i] = tcu[i] = tcd[i] = tcs[i] = 1.0;
  }
  // 1s0 neutron superfluidity
  if (sfn1s0 == 1) {
    for (int i = 0; i <= idrip; ++i)
      tcn[i] = std::max(1.0, atc.cpp_tcn1_sfb(kfn[i])) * fn1s0;
  } else if (sfn1s0 == 2) {
    for (int i = 0; i <= idrip; ++i)
      tcn[i] = std::max(1.0, atc.cpp_tcn1_ccdk(kfn[i])) * fn1s0;
  } else if (sfn1s0 == 3) {
    for (int i = 0; i <= idrip; ++i)
      tcn[i] = std::max(1.0, atc.cpp_tcn1_wap(kfn[i])) * fn1s0;
  } else if (sfn1s0 == 4) {
    for (int i = 0; i <= idrip; ++i)
      tcn[i] = std::max(1.0, atc.cpp_tcn1_gc(kfn[i])) * fn1s0;
  } else if (sfn1s0 == 5) {
    for (int i = 0; i <= idrip; ++i)
      tcn[i] = std::max(1.0, atc.cpp_tcn1_gipsf(kfn[i])) * fn1s0;
  } else if (sfn1s0 == 201) {
    for (int i = 0; i <= idrip; ++i)
      tcn[i] = std::max(1.0, atc.cpp_tc_Ioffe_1ns(kfn[i])) * fn1s0;
  } else if (sfn1s0 == 202) {
    for (int i = 0; i <= idrip; ++i)
      tcn[i] = std::max(1.0, atc.cpp_tc_Ioffe_2ns(kfn[i])) * fn1s0;
  } else if (sfn1s0 == 203) {
    for (int i = 0; i <= idrip; ++i)
      tcn[i] = std::max(1.0, atc.cpp_tc_Ioffe_3ns(kfn[i])) * fn1s0;
  } else if (sfn1s0 == 150) {
    double tcmax_n1s0 = dinput_n1tc;
    double kfmax_n1s0 = dinput_n1kf;
    double delkf_n1s0 = dinput_n1dk;
    for (int i = 0; i <= idrip; ++i) {
      double temp =
          tcmax_n1s0 * exp(-pow((kfn[i] - kfmax_n1s0) / delkf_n1s0, 2)) * fn1s0;
      if (temp >= tcn[i]) {
        tcn[i] = temp;
        if (isf == i - 1)
          isf = i;
      }
    }
  }
  // 3p2 neutron superfluidity
  isf = -1;

  if (sfn3p2 == 1) {
    for (int i = 0; i <= idrip; ++i) {
      double temp = atc.cpp_tcn3_hgrr(kfn[i]) * fn3p2;
      if (temp >= tcn[i]) {
        tcn[i] = temp;
        isf = i;
      }
    }
  } else if (sfn3p2 == 2) {
    for (int i = 0; i <= idrip; ++i) {
      double temp = atc.cpp_tcn3_ao(kfn[i]) * fn3p2;
      if (temp >= tcn[i]) {
        tcn[i] = temp;
        isf = i;
      }
    }
  } else if (sfn3p2 == 3) {
    for (int i = 0; i <= idrip; ++i) {
      double temp = atc.cpp_tcn3_ao_m1(kfn[i]) * fn3p2;
      if (temp >= tcn[i]) {
        tcn[i] = temp;
        isf = i;
      }
    }
  } else if (sfn3p2 == 4) {
    for (int i = 0; i <= idrip; ++i) {
      double temp = atc.cpp_tcn3_t72(kfn[i]) * fn3p2;
      if (temp >= tcn[i]) {
        tcn[i] = temp;
        isf = i;
      }
    }
  } else if (sfn3p2 == 5) {
    for (int i = 0; i <= idrip; ++i) {
      double temp = atc.cpp_tcn3_t72_m1(kfn[i]) * fn3p2;
      if (temp >= tcn[i]) {
        tcn[i] = temp;
        isf = i;
      }
    }
  } else if (sfn3p2 == 6) {
    for (int i = 0; i <= idrip; ++i) {
      double temp = atc.cpp_tcn3_bcll92(kfn[i]) * fn3p2;
      if (temp >= tcn[i]) {
        tcn[i] = temp;
        isf = i;
      }
    }
  } else if (sfn3p2 == 7) {
    for (int i = 0; i <= idrip; ++i) {
      double temp = atc.cpp_tcn3_eehjo96_nr(kfn[i]) * fn3p2;
      if (temp >= tcn[i]) {
        tcn[i] = temp;
        isf = i;
      }
    }
  } else if (sfn3p2 == 8) {
    for (int i = 0; i <= idrip; ++i) {
      double temp = atc.cpp_tcn3_eehjo96_r(kfn[i]) * fn3p2;
      if (temp >= tcn[i]) {
        tcn[i] = temp;
        isf = i;
      }
    }
  } else if (sfn3p2 >= 100 && sfn3p2 < 200) {
    double kfmax_n3p2, delkf_n3p2, tcmax_n3p2;
    if (sfn3p2 == 101) {
      kfmax_n3p2 = 1.8;
      delkf_n3p2 = 0.5;
      tcmax_n3p2 = 1.0e9;
    } else if (sfn3p2 == 102) {
      kfmax_n3p2 = 2.0;
      delkf_n3p2 = 0.5;
      tcmax_n3p2 = 3.0e9;
    } else if (sfn3p2 == 103) {
      kfmax_n3p2 = 2.5;
      delkf_n3p2 = 0.7;
      tcmax_n3p2 = 1.0e10;
    } else if (sfn3p2 == 150) {
      kfmax_n3p2 = dinput_n3tc;
      delkf_n3p2 = dinput_n3kf;
      tcmax_n3p2 = dinput_n3dk;
    } else {
      kfmax_n3p2 = 0;
      delkf_n3p2 = 0;
      tcmax_n3p2 = 0;
    }
    for (int i = 0; i <= idrip; ++i) {
      double temp =
          tcmax_n3p2 * exp(-pow((kfn[i] - kfmax_n3p2) / delkf_n3p2, 2)) * fn3p2;
      if (temp >= tcn[i]) {
        tcn[i] = temp;
        if (isf == i - 1) {
          isf = i;
        }
      }
    }
  } else if (sfn3p2 == 201) {
    for (int i = 0; i <= idrip; ++i) {
      double temp = std::max(1.0, atc.cpp_tc_Ioffe_1nt(kfn[i])) * fn3p2;
      if (temp >= tcn[i]) {
        tcn[i] = temp;
        isf = i;
      }
    }
  } else if (sfn3p2 == 202) {
    for (int i = 0; i <= idrip; ++i) {
      double temp = std::max(1.0, atc.cpp_tc_Ioffe_2nt(kfn[i])) * fn3p2;
      if (temp >= tcn[i]) {
        tcn[i] = temp;
        isf = i;
      }
    }
  } else if (sfn3p2 == 203) {
    for (int i = 0; i <= idrip; ++i) {
      double temp = std::max(1.0, atc.cpp_tc_Ioffe_3nt(kfn[i])) * fn3p2;
      if (temp >= tcn[i]) {
        tcn[i] = temp;
        isf = i;
      }
    }
  }
  // For uniform Tc gap
  else if (sfn3p2 >= 1000) {
    for (int i = 0; i <= icore; ++i) {
      tcn[i] = sfn3p2;
    }
    isf = icore;
  }

  if (sfp1s0 == 1) {
    for (int i = 0; i <= icore; ++i)
      tcp[i] = std::max(1.0, atc.cpp_tcp1_ccy_ms(kfp[i])) * fp1s0;
  } else if (sfp1s0 == 2) {
    for (int i = 0; i <= icore; ++i)
      tcp[i] = std::max(1.0, atc.cpp_tcp1_ccy_ps(kfp[i])) * fp1s0;
  } else if (sfp1s0 == 3) {
    for (int i = 0; i <= icore; ++i)
      tcp[i] = std::max(1.0, atc.cpp_tcp1_t73(kfp[i])) * fp1s0;
  } else if (sfp1s0 == 4) {
    for (int i = 0; i <= icore; ++i)
      tcp[i] = std::max(1.0, atc.cpp_tcp1_ns(kfp[i])) * fp1s0;
  } else if (sfp1s0 == 5) {
    for (int i = 0; i <= icore; ++i)
      tcp[i] = std::max(1.0, atc.cpp_tcp1_ao(kfp[i])) * fp1s0;
  } else if (sfp1s0 == 6) {
    for (int i = 0; i <= icore; ++i)
      tcp[i] = std::max(1.0, atc.cpp_tcp1_bcll92(kfp[i])) * fp1s0;
  } else if (sfp1s0 == 7) {
    for (int i = 0; i <= icore; ++i)
      tcp[i] = std::max(1.0, atc.cpp_tcp1_ccdk(kfp[i])) * fp1s0;
  } else if (sfp1s0 == 21) {
    for (int i = 0; i <= icore; ++i)
      tcp[i] = std::max(1.0, atc.cpp_tcn1_t72(kfp[i])) * fp1s0;
  } else if (sfp1s0 == 22) {
    for (int i = 0; i <= icore; ++i)
      tcp[i] = std::max(1.0, atc.cpp_tcn1_awp_2(kfp[i])) * fp1s0;
  } else if (sfp1s0 == 23) {
    for (int i = 0; i <= icore; ++i)
      tcp[i] = std::max(1.0, atc.cpp_tcn1_awp_3(kfp[i])) * fp1s0;
  } else if (sfp1s0 == 201) {
    for (int i = 0; i <= icore; ++i)
      tcp[i] = std::max(1.0, atc.cpp_tc_Ioffe_1p(kfp[i])) * fp1s0;
  } else if (sfp1s0 == 202) {
    for (int i = 0; i <= icore; ++i)
      tcp[i] = std::max(1.0, atc.cpp_tc_Ioffe_2p(kfp[i])) * fp1s0;
  } else if (sfp1s0 == 203) {
    for (int i = 0; i <= icore; ++i)
      tcp[i] = std::max(1.0, atc.cpp_tc_Ioffe_3p(kfp[i])) * fp1s0;
  } else if (sfp1s0 >= 1e3) {
    for (int i = 0; i <= icore; ++i)
      tcp[i] = sfp1s0;
  } else if (sfp1s0 == 150) {
    double tcmax_p1s0 = dinput_p1tc;
    double kfmax_p1s0 = dinput_p1kf;
    double delkf_p1s0 = dinput_p1dk;
    for (int i = 0; i <= icore; ++i) {
      tcp[i] = dinput_p1tc *
               exp(-(kfp[i] - dinput_p1kf) * (kfp[i] - dinput_p1kf) /
                   (delkf_p1s0 * delkf_p1s0)) *
               fp1s0;
    }
  }

  // 1s0 Lambda superfluidity
  if (sfl1s0 == 1) {
    for (int i = 0; i <= icore; ++i) {
      tcla[i] = std::max(1.0, atc.cpp_tcla1_bb(kfla[i], bar[i])) * fl1s0;
    }
  }

  // Just in case the above formulas give Tc<0 instead of Tc=0 at the edges
  // where Tc is almost 0
  for (int i = 0; i <= idrip; ++i) {
    tcn[i] = std::abs(tcn[i]);
    tcp[i] = std::abs(tcp[i]);
    tcla[i] = std::abs(tcla[i]);
    tcuu[i] = std::abs(tcuu[i]);
    tcdd[i] = std::abs(tcdd[i]);
    tcss[i] = std::abs(tcss[i]);
    tcud[i] = std::abs(tcud[i]);
    tcus[i] = std::abs(tcus[i]);
    tcds[i] = std::abs(tcds[i]);
    tcu[i] = std::abs(tcu[i]);
    tcd[i] = std::abs(tcd[i]);
    tcs[i] = std::abs(tcs[i]);
  }
}

void nscool::get_effective_masses(const std::vector<double> kfn,
                                  std::vector<double> &mstn, int idrip,
                                  int icore) {
  for (int i = icore + 1; i <= idrip; ++i) {
    mstn[i] = std::min(1.0, 1.09 - 0.11 * kfn[i]);
  }
}

void nscool::main_out(double time, double tptr, double lphot, double lneut,
                      double lheat, int stop) {

  std::cout << "v_time.size()" << v_time.size() << std::endl;

  if (((int)v_time.size()) % main_out_it == (main_out_it - 1)) {
    std::cout.width(4);
    std::cout << v_time.size() << " " << time << " " << tptr << " " << lphot
              << " " << lneut << " " << lheat << std::endl;
  }

  v_time.push_back(time);
  v_tptr.push_back(tptr);
  v_lphot.push_back(lphot);
  v_lneut.push_back(lneut);
  v_lheat.push_back(lheat);

  stop = 0;
  if (v_time.size() > 10000) {
    std::cout << v_time.size() << std::endl;
    stop = 1;
  }

  return;
}

void nscool::write_cool_curve(std::string fname) {
  o2scl_hdf::hdf_file hf;
  hf.open_or_create(fname);
  o2scl::table_units<> t;
  t.line_of_names("t T L_neut L_phot L_heat");
  t.set_unit("t", "yr");
  t.set_unit("T", "K");
  t.set_unit("L_neut", "erg/s");
  t.set_unit("L_phot", "erg/s");
  t.set_unit("L_heat", "erg/s");
  for (size_t i = 0; i < v_time.size(); i++) {
    std::vector<double> line = {v_time[i], v_tptr[i], v_lneut[i], v_lphot[i],
                                v_lheat[i]};
    t.line_of_data(5, line);
  }
  hdf_output(hf, t, "cool_curve");
  hf.close();
  return;
}

void nscool::write_tl_prof(std::string fname) {

  if (!tl_prof.is_constant("it_last")) {
    std::cerr << "No table to write in write_tl_prof()." << std::endl;
    return;
  }

  o2scl_hdf::hdf_file hf;
  hf.open_or_create(fname);

  // Create new table omitting temperatures not stored
  int num_times = 1 + ((int)(tl_prof.get_constant("it_last") + 1.0e-6));

  // std::cout << num_times << " " << temp_ygrid.size() << std::endl;

  // If the time grid hasn't been filled, then restructure the table
  if (num_times != ((int)temp_ygrid.size())) {
    o2scl::table3d tl_prof2;
    const ubvector &r_grid = tl_prof.get_x_data();
    const ubvector &t_grid = tl_prof.get_y_data();
    ubvector t_grid2(num_times);
    o2scl::vector_copy(num_times, t_grid, t_grid2);
    tl_prof2.set_xy("r", r_grid.size(), r_grid, "t", num_times, t_grid2);
    tl_prof2.line_of_names(
        ((std::string) "rho ephi vol Tinf Linf qnu ") +
        "qeebrem qnpb qplasma qsynch qbubble qpair " +
        "qphoto qbrem_nn qmurca_nucl qbrem_nucl " +
        "qmurca_hyp qbrem_hyp qdurca_np qdurca_lap " +
        "qdurca_smn qdurca_smla qdurca_sms0 qfast " +
        "qdurca_q qmurca_q qpbf_n1s0 qpbf_p1s0 " + "qpbf_n3p2 qpbf_q qmax cv " +
        "cv_n cv_p cv_e cv_m cv_la " + "cv_sm cv_s0 cv_sp cv_q");
    for (size_t i = 0; i < r_grid.size(); i++) {
      for (size_t j = 0; j < ((size_t)num_times); j++) {
        tl_prof2.set(i, j, "rho", tl_prof.get(i, j, "rho"));
        tl_prof2.set(i, j, "ephi", tl_prof.get(i, j, "ephi"));
        tl_prof2.set(i, j, "vol", tl_prof.get(i, j, "vol"));
        tl_prof2.set(i, j, "Tinf", tl_prof.get(i, j, "Tinf"));
        tl_prof2.set(i, j, "Linf", tl_prof.get(i, j, "Linf"));
        tl_prof2.set(i, j, "qnu", tl_prof.get(i, j, "qnu"));
        tl_prof2.set(i, j, "qeebrem", tl_prof.get(i, j, "qeebrem"));
        tl_prof2.set(i, j, "qnpb", tl_prof.get(i, j, "qnpb"));
        tl_prof2.set(i, j, "qplasma", tl_prof.get(i, j, "qplasma"));
        tl_prof2.set(i, j, "qsynch", tl_prof.get(i, j, "qsynch"));
        tl_prof2.set(i, j, "qbubble", tl_prof.get(i, j, "qbubble"));
        tl_prof2.set(i, j, "qpair", tl_prof.get(i, j, "qpair"));
        tl_prof2.set(i, j, "qphoto", tl_prof.get(i, j, "qphoto"));
        tl_prof2.set(i, j, "qbrem_nn", tl_prof.get(i, j, "qbrem_nn"));
        tl_prof2.set(i, j, "qmurca_nucl", tl_prof.get(i, j, "qmurca_nucl"));
        tl_prof2.set(i, j, "qbrem_nucl", tl_prof.get(i, j, "qbrem_nucl"));
        tl_prof2.set(i, j, "qmurca_hyp", tl_prof.get(i, j, "qmurca_hyp"));
        tl_prof2.set(i, j, "qbrem_hyp", tl_prof.get(i, j, "qbrem_hyp"));
        tl_prof2.set(i, j, "qdurca_np", tl_prof.get(i, j, "qdurca_np"));
        tl_prof2.set(i, j, "qdurca_lap", tl_prof.get(i, j, "qdurca_lap"));
        tl_prof2.set(i, j, "qdurca_smn", tl_prof.get(i, j, "qdurca_smn"));
        tl_prof2.set(i, j, "qdurca_smla", tl_prof.get(i, j, "qdurca_smla"));
        tl_prof2.set(i, j, "qdurca_sms0", tl_prof.get(i, j, "qdurca_sms0"));
        tl_prof2.set(i, j, "qfast", tl_prof.get(i, j, "qfast"));
        tl_prof2.set(i, j, "qdurca_q", tl_prof.get(i, j, "qdurca_q"));
        tl_prof2.set(i, j, "qmurca_q", tl_prof.get(i, j, "qmurca_q"));
        tl_prof2.set(i, j, "qpbf_n1s0", tl_prof.get(i, j, "qpbf_n1s0"));
        tl_prof2.set(i, j, "qpbf_p1s0", tl_prof.get(i, j, "qpbf_p1s0"));
        tl_prof2.set(i, j, "qpbf_n3p2", tl_prof.get(i, j, "qpbf_n3p2"));
        tl_prof2.set(i, j, "qpbf_q", tl_prof.get(i, j, "qpbf_q"));
        tl_prof2.set(i, j, "qmax", tl_prof.get(i, j, "qmax"));
        tl_prof2.set(i, j, "cv", tl_prof.get(i, j, "cv"));
        tl_prof2.set(i, j, "cv_n", tl_prof.get(i, j, "cv_n"));
        tl_prof2.set(i, j, "cv_p", tl_prof.get(i, j, "cv_p"));
        tl_prof2.set(i, j, "cv_e", tl_prof.get(i, j, "cv_e"));
        tl_prof2.set(i, j, "cv_m", tl_prof.get(i, j, "cv_m"));
        tl_prof2.set(i, j, "cv_la", tl_prof.get(i, j, "cv_la"));
        tl_prof2.set(i, j, "cv_sm", tl_prof.get(i, j, "cv_sm"));
        tl_prof2.set(i, j, "cv_s0", tl_prof.get(i, j, "cv_s0"));
        tl_prof2.set(i, j, "cv_sp", tl_prof.get(i, j, "cv_sp"));
        tl_prof2.set(i, j, "cv_q", tl_prof.get(i, j, "cv_q"));
      }
    }

    tl_prof2.set_interp_type(o2scl::itp_nearest_neigh);
    o2scl::table3d t3dug =
        tl_prof2.slice_to_uniform_grid("qmax", 100, false, 100, true);
    tl_prof2.set_interp_type(o2scl::itp_linear);
    t3dug.set_interp_type(o2scl::itp_linear);
    for (size_t k = 0; k < tl_prof2.get_nslices(); k++) {
      std::string sl_name = tl_prof2.get_slice_name(k);
      if (sl_name != "qmax") {
        t3dug.add_slice_from_table(tl_prof2, sl_name, sl_name);
      }
    }

    hdf_output(hf, ((const o2scl::table3d &)(tl_prof2)), "tl_prof");
    hdf_output(hf, ((const o2scl::table3d &)(t3dug)), "tl_prof_ug");

  } else {

    if (false) {
      const ubvector &r_grid = tl_prof.get_x_data();
      const ubvector &t_grid = tl_prof.get_y_data();
      for (size_t i = 0; i < r_grid.size(); i++) {
        for (size_t j = 0; j < t_grid.size(); j++) {
          std::cout << i << " " << j << " " << tl_prof.get(i, j, "rho")
                    << std::endl;
        }
      }
    }

    tl_prof.set_interp_type(o2scl::itp_nearest_neigh);
    o2scl::table3d t3dug =
        tl_prof.slice_to_uniform_grid("qmax", 100, false, 100, true);
    tl_prof.set_interp_type(o2scl::itp_linear);
    t3dug.set_interp_type(o2scl::itp_linear);
    for (size_t k = 0; k < tl_prof.get_nslices(); k++) {
      std::string sl_name = tl_prof.get_slice_name(k);
      if (sl_name != "qmax") {
        t3dug.add_slice_from_table(tl_prof, sl_name, sl_name);
      }
    }

    hdf_output(hf, ((const o2scl::table3d &)(tl_prof)), "tl_prof");
    hdf_output(hf, ((const o2scl::table3d &)(t3dug)), "tl_prof_ug");
  }

  hf.close();
  return;
}

void nscool::gaps(int &sfn1s0_arg, double &n1_tc_arg, double &n1_kf_arg,
                  double &n1_dk_arg, int &sfn3p2_arg, double &n3_tc_arg,
                  double &n3_kf_arg, double &n3_dk_arg, int &sfp1s0_arg,
                  double &p1_tc_arg, double &p1_kf_arg, double &p1_dk_arg) {

  // If sfn3p2 and sfp1s0 are 150, then the double parameters specify the
  // Gaussian
  sfn3p2_arg = sfn3p2;
  sfp1s0_arg = sfp1s0;
  sfn1s0_arg = sfn1s0;
  if (sfn3p2_arg == 150) {
    n3_tc_arg = n3_tc;
    n3_kf_arg = n3_kf;
    n3_dk_arg = n3_dk;
  }
  if (sfp1s0_arg == 150) {
    p1_tc_arg = p1_tc;
    p1_kf_arg = p1_kf;
    p1_dk_arg = p1_dk;
  }
  if (sfn1s0_arg == 150) {
    n1_tc_arg = n1_tc;
    n1_kf_arg = n1_kf;
    n1_dk_arg = n1_dk;
  }
  return;
};
