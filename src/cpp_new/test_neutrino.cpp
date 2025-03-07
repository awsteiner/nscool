#include <iostream>
#include <vector>

#include "emissivities.h"  // Contains the C++ "neutrino(...)" declaration

// ------------------------------------------------------------------
// 1) Declaration for the Fortran neutrino_ subroutine, using extern "C"
//    so that we can call it from C++. Adjust the argument list if your
//    Fortran subroutine differs in naming or order.
// ------------------------------------------------------------------
extern "C" {

// This is a simplified version of the full Fortran signature. 
// You must include ALL pointer parameters if your Fortran code has them.
// We'll show a partial demonstration here. 
void neutrino_(
    int* irank,
    int* i,
    double* t,
    double* rho,
    double* a,
    double* z,
    double* qtot,
    double* qeebrem,
    double* qnpb,
    double* qplasma,
    double* qsynch,
    double* qbubble,
    double* qpair,
    double* qphoto,
    double* qbrem_nn,
    double* qmurca_nucl,
    double* qbrem_nucl,
    double* qmurca_hyp,
    double* qbrem_hyp,
    double* qdurca_np,
    double* qdurca_lap,
    double* qdurca_smn,
    double* qdurca_smla,
    double* qdurca_sms0,
    double* qfast,
    double* qdurca_q,
    double* qmurca_q,
    double* qpbf_n1s0,
    double* qpbf_n3p2,
    double* qpbf_p1s0,
    double* qpbf_q,
    double* debug,
    int* naa,
    std::vector<double>* nbfield2,   
    double* rhodrip,
    double* rhocore,
    std::vector<double>* mstp,
    std::vector<double>* mstn,
    std::vector<double>* mstla,
    std::vector<double>* mstsm,
    std::vector<double>* msts0,
    std::vector<double>* mstsp,
    std::vector<double>* kfe,
    std::vector<double>* kfm,
    std::vector<double>* kfp,
    std::vector<double>* kfn,
    std::vector<double>* kfqu,
    std::vector<double>* kfqd,
    std::vector<double>* kfqs,
    std::vector<double>* bar,
    std::vector<double>* yelect,
    std::vector<double>* ymuon,
    std::vector<double>* fhad,
    std::vector<double>* theta_k,
    std::vector<double>* theta_p,
    std::vector<double>* v_ion,
    double* rhoexo,
    double* cexo,
    double* pexo,
    double* c_nu_str,
    double* p_nu_str,
    double* murca_increase,
    int* inu_durca,
    int* inu_eion,
    int* inu_plasma,
    int* inu_synch,
    int* inu_n1s0_pbf,
    int* inu_n3p2_pbf,
    int* inu_p_pbf,
    int* inu_bubble,
    int* inu_photo,
    int* inu_pair,
    std::vector<int>* idurca_np,
    std::vector<int>* idurca_lap,
    std::vector<double>* durca_ctrl_e,
    std::vector<double>* durca_ctrl_m,
    std::vector<int>* idurca_smn,
    std::vector<int>* idurca_smla,
    std::vector<int>* idurca_sms0,
    std::vector<int>* idurca_quqd,
    std::vector<int>* idurca_quqs,
    std::vector<double>* tcn,
    std::vector<double>* tcp,
    std::vector<double>* tcla,
    std::vector<double>* tcu,
    std::vector<double>* tcd,
    std::vector<double>* tcs,
    std::vector<double>* tcu1,
    std::vector<double>* tcu2,
    std::vector<double>* tcu3,
    std::vector<double>* tcd1,
    std::vector<double>* tcd2,
    std::vector<double>* tcd3,
    std::vector<double>* tcs1,
    std::vector<double>* tcs2,
    std::vector<double>* tcs3,
    int* isf,
    std::vector<double>* neebrem_logt,
    std::vector<double>* neebrem_nalpha,
    std::vector<double>* neebrem_n2,
    std::vector<double>* sf_lgtau1,
    std::vector<double>* sf_lgtau2,
    std::vector<std::vector<double>>* sf_lgr,      // Flattened 2D array if necessary
    std::vector<std::vector<double>>* sf_lgr2      // Flattened 2D array if necessary
);

} // extern "C"


int main() {

  emissivities em;

  // -------------------------------------------------------------
  // Example: We define minimal parameters and arrays to call the 
  // C++ `neutrino(...)` vs. the Fortran `neutrino_()` side by side.
  // -------------------------------------------------------------
  
  int    irank      = 0;
  int    i          = 42;   // Some index
  double t          = 1.0e9; 
  double rho        = 1.0e14;
  double a          = 56.0;
  double z          = 26.0;
  double debug      = 0;
  int    naa        = 0;    // Not used, but Fortran might require it

  // We'll define minimal arrays with some size (or 1-element if you're testing).
  // In real usage, you'd fill these with the correct data.
  std::vector<double> nbfield2(100, 0.0); // For example, set everything to 0.0
  double rhodrip = 4.0e11;
  double rhocore = 2.0e14;

  // We also need outputs for neutrino calls:
  double qtot_cpp=0.0, qeebrem_cpp=0.0, qnpb_cpp=0.0, qplasma_cpp=0.0;
  double qsynch_cpp=0.0, qbubble_cpp=0.0, qpair_cpp=0.0, qphoto_cpp=0.0;
  double qbrem_nn_cpp=0.0, qmurca_nucl_cpp=0.0, qbrem_nucl_cpp=0.0;
  double qmurca_hyp_cpp=0.0, qbrem_hyp_cpp=0.0;
  double qdurca_np_cpp=0.0, qdurca_lap_cpp=0.0, qdurca_smn_cpp=0.0;
  double qdurca_smla_cpp=0.0, qdurca_sms0_cpp=0.0;
  double qfast_cpp=0.0, qdurca_q_cpp=0.0, qmurca_q_cpp=0.0;
  double qpbf_n1s0_cpp=0.0, qpbf_n3p2_cpp=0.0, qpbf_p1s0_cpp=0.0, qpbf_q_cpp=0.0;

  // We'll define minimal placeholders for the big arrays used by neutrino.
  // For example: 
  std::vector<double> mstp(100,1.0), mstn(100,1.0), mstla(100,0.0), mstsm(100,0.0),
                      msts0(100,0.0), mstsp(100,0.0),
                      kfe(100, 0.5), kfm(100, 0.3), kfp(100, 0.8), kfn(100, 1.0),
                      kfqu(100,0.0), kfqd(100,0.0), kfqs(100,0.0),
                      bar(100,0.16), yelect(100,0.05), ymuon(100,0.0), fhad(100,1.0),
                      theta_k(100,0.0), theta_p(100,0.0), v_ion(100,0.0);
  double rhoexo=1.0e25, cexo=1.0e25, pexo=0.0;
  double c_nu_str=1.0e20, p_nu_str=2.0;
  double murca_increase=0.0;
  int inu_durca=1, inu_eion=1, inu_plasma=1, inu_synch=0,
      inu_n1s0_pbf=1, inu_n3p2_pbf=1, inu_p_pbf=1,
      inu_bubble=0, inu_photo=0, inu_pair=0;

  // We'll define some integer arrays for idurca, etc. 
  std::vector<int> idurca_np(100,0), idurca_lap(100,0);
  std::vector<double> durca_ctrl_e(100,0.0), durca_ctrl_m(100,0.0);
  std::vector<int> idurca_smn(100,0), idurca_smla(100,0), idurca_sms0(100,0);
  std::vector<int> idurca_quqd(100,0), idurca_quqs(100,0);

  std::vector<double> tcn(100,1.0e9), tcp(100,1.0e9), tcla(100,0.0),
                      tcu(100,0.0), tcd(100,0.0), tcs(100,0.0),
                      tcu1(100,0.0), tcu2(100,0.0), tcu3(100,0.0),
                      tcd1(100,0.0), tcd2(100,0.0), tcd3(100,0.0),
                      tcs1(100,0.0), tcs2(100,0.0), tcs3(100,0.0);
  int isf=50;

  // If you have neebrem arrays, do something like:
  std::vector<double> neebrem_logt(56, 0.0), neebrem_nalpha(56, 0.0), neebrem_n2(56, 0.0);
  std::vector<double> sf_lgtau1(35,0.0), sf_lgtau2(35,0.0);
  std::vector<std::vector<double>> sf_lgr(35, std::vector<double>(35,0.0)),
                                   sf_lgr2(35, std::vector<double>(35,0.0));

  // 1) Call the C++ neutrino(...) version
  em.neutrino(
    irank, i, t, rho, a, z,
    qtot_cpp, qeebrem_cpp, qnpb_cpp, qplasma_cpp, qsynch_cpp,
    qbubble_cpp, qpair_cpp, qphoto_cpp, qbrem_nn_cpp,
    qmurca_nucl_cpp, qbrem_nucl_cpp, qmurca_hyp_cpp, qbrem_hyp_cpp,
    qdurca_np_cpp, qdurca_lap_cpp, qdurca_smn_cpp, qdurca_smla_cpp, qdurca_sms0_cpp,
    qfast_cpp, qdurca_q_cpp, qmurca_q_cpp,
    qpbf_n1s0_cpp, qpbf_n3p2_cpp, qpbf_p1s0_cpp, qpbf_q_cpp,
    debug, naa, nbfield2, rhodrip, rhocore,
    mstp, mstn, mstla, mstsm, msts0, mstsp,
    kfe, kfm, kfp, kfn, kfqu, kfqd, kfqs,
    bar, yelect, ymuon, fhad, theta_k, theta_p, v_ion,
    rhoexo, cexo, pexo, c_nu_str, p_nu_str,
    murca_increase, inu_durca, inu_eion, inu_plasma, inu_synch,
    inu_n1s0_pbf, inu_n3p2_pbf, inu_p_pbf,
    inu_bubble, inu_photo, inu_pair,
    idurca_np, idurca_lap, durca_ctrl_e, durca_ctrl_m,
    idurca_smn, idurca_smla, idurca_sms0,
    idurca_quqd, idurca_quqs,
    tcn, tcp, tcla, tcu, tcd, tcs,
    tcu1, tcu2, tcu3, tcd1, tcd2, tcd3, tcs1, tcs2, tcs3,
    isf,
    neebrem_logt, neebrem_nalpha, neebrem_n2,
    sf_lgtau1, sf_lgtau2, sf_lgr, sf_lgr2
  );

  // 2) Call the Fortran version neutrino_() with pointer-friendly copies
  int    irank_f= irank, i_f= i;
  double t_f= t, rho_f= rho, a_f= a, z_f= z;
  double debug_f= debug;
  int    naa_f= naa;
  
  double qtot_f=0.0, qeebrem_f=0.0, qnpb_f=0.0, qplasma_f=0.0;
  double qsynch_f=0.0, qbubble_f=0.0, qpair_f=0.0, qphoto_f=0.0;
  double qbrem_nn_f=0.0, qmurca_nucl_f=0.0, qbrem_nucl_f=0.0;
  double qmurca_hyp_f=0.0, qbrem_hyp_f=0.0;
  double qdurca_np_f=0.0, qdurca_lap_f=0.0, qdurca_smn_f=0.0;
  double qdurca_smla_f=0.0, qdurca_sms0_f=0.0;
  double qfast_f=0.0, qdurca_q_f=0.0, qmurca_q_f=0.0;
  double qpbf_n1s0_f=0.0, qpbf_n3p2_f=0.0, qpbf_p1s0_f=0.0, qpbf_q_f=0.0;

  // We'll also need pointers for the arrays, but for brevity we only show a minimal example 
  // (In reality, you'd pass them all as pointers).
  // For instance, double nbfield2_array[100]; copy data from vector nbfield2 into it, etc.
  // Then neutrino_(&irank_f, &i_f, &t_f, &rho_f, &a_f, &z_f, &qtot_f, &qeebrem_f, ... &debug_f, &naa_f, nbfield2_array, &rhodrip, &rhocore, ...)

  // This snippet is conceptual. 
  neutrino_(
    &irank_f, &i_f, &t_f, &rho_f, &a_f, &z_f,
    &qtot_f, &qeebrem_f, &qnpb_f, &qplasma_f, &qsynch_f,
    &qbubble_f, &qpair_f, &qphoto_f, &qbrem_nn_f,
    &qmurca_nucl_f, &qbrem_nucl_f, &qmurca_hyp_f, &qbrem_hyp_f,
    &qdurca_np_f, &qdurca_lap_f, &qdurca_smn_f, &qdurca_smla_f, &qdurca_sms0_f,
    &qfast_f, &qdurca_q_f, &qmurca_q_f,
    &qpbf_n1s0_f, &qpbf_n3p2_f, &qpbf_p1s0_f, &qpbf_q_f,
    &debug_f, &naa_f, &nbfield2, &rhodrip, &rhocore,
    &mstp, &mstn, &mstla, &mstsm, &msts0, &mstsp,
    &kfe, &kfm, &kfp, &kfn, &kfqu, &kfqd, &kfqs,
    &bar, &yelect, &ymuon, &fhad, &theta_k, &theta_p, &v_ion,
    &rhoexo, &cexo, &pexo, &c_nu_str, &p_nu_str,
    &murca_increase, &inu_durca, &inu_eion, &inu_plasma, &inu_synch,
    &inu_n1s0_pbf, &inu_n3p2_pbf, &inu_p_pbf,
    &inu_bubble, &inu_photo, &inu_pair,
    &idurca_np, &idurca_lap, &durca_ctrl_e, &durca_ctrl_m,
    &idurca_smn, &idurca_smla, &idurca_sms0,
    &idurca_quqd, &idurca_quqs,
    &tcn, &tcp, &tcla, &tcu, &tcd, &tcs,
    &tcu1, &tcu2, &tcu3, &tcd1, &tcd2, &tcd3, &tcs1, &tcs2, &tcs3,
    &isf,
    &neebrem_logt, &neebrem_nalpha, &neebrem_n2,
    &sf_lgtau1, &sf_lgtau2, &sf_lgr, &sf_lgr2
  );

  // 7) Print C++ vs Fortran results side by side
    std::cout << "\n=== Comparison of C++ vs. Fortran neutrino ===\n";
    
    std::cout << "qtot      : " << qtot_cpp      << " vs " << qtot_f      << "\n";
    std::cout << "qeebrem   : " << qeebrem_cpp   << " vs " << qeebrem_f   << "\n";
    std::cout << "qnpb      : " << qnpb_cpp      << " vs " << qnpb_f      << "\n";
    std::cout << "qplasma   : " << qplasma_cpp   << " vs " << qplasma_f   << "\n";
    std::cout << "qsynch    : " << qsynch_cpp    << " vs " << qsynch_f    << "\n";
    std::cout << "qbubble   : " << qbubble_cpp   << " vs " << qbubble_f   << "\n";
    std::cout << "qpair     : " << qpair_cpp     << " vs " << qpair_f     << "\n";
    std::cout << "qphoto    : " << qphoto_cpp    << " vs " << qphoto_f    << "\n";
    std::cout << "qbrem_nn  : " << qbrem_nn_cpp  << " vs " << qbrem_nn_f  << "\n";
    std::cout << "qmurca_nucl: " << qmurca_nucl_cpp << " vs " << qmurca_nucl_f << "\n";
    std::cout << "qbrem_nucl: " << qbrem_nucl_cpp << " vs " << qbrem_nucl_f << "\n";
    std::cout << "qmurca_hyp: " << qmurca_hyp_cpp << " vs " << qmurca_hyp_f << "\n";
    std::cout << "qbrem_hyp : " << qbrem_hyp_cpp  << " vs " << qbrem_hyp_f  << "\n";
    std::cout << "qdurca_np : " << qdurca_np_cpp  << " vs " << qdurca_np_f  << "\n";
    std::cout << "qdurca_lap: " << qdurca_lap_cpp << " vs " << qdurca_lap_f << "\n";
    std::cout << "qdurca_smn: " << qdurca_smn_cpp << " vs " << qdurca_smn_f << "\n";
    std::cout << "qdurca_smla: " << qdurca_smla_cpp << " vs " << qdurca_smla_f << "\n";
    std::cout << "qdurca_sms0: " << qdurca_sms0_cpp << " vs " << qdurca_sms0_f << "\n";
    std::cout << "qfast     : " << qfast_cpp     << " vs " << qfast_f     << "\n";
    std::cout << "qdurca_q  : " << qdurca_q_cpp  << " vs " << qdurca_q_f  << "\n";
    std::cout << "qmurca_q  : " << qmurca_q_cpp  << " vs " << qmurca_q_f  << "\n";
    std::cout << "qpbf_n1s0 : " << qpbf_n1s0_cpp << " vs " << qpbf_n1s0_f << "\n";
    std::cout << "qpbf_n3p2 : " << qpbf_n3p2_cpp << " vs " << qpbf_n3p2_f << "\n";
    std::cout << "qpbf_p1s0 : " << qpbf_p1s0_cpp << " vs " << qpbf_p1s0_f << "\n";
    std::cout << "qpbf_q    : " << qpbf_q_cpp    << " vs " << qpbf_q_f    << "\n\n";

  return 0;
}
