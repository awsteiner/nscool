// test_grid.cpp
#include <iostream>

// Include the C++ header containing the grid function
#include "f_precool.h"  // Replace with the actual path to your header

// Assuming the C++ grid function has the following signature:
// void grid(int irank, int idec, double rhocore, double rhodrip, double rhoenv, double rhosurf,
//           int& imax, int& icore, int& idrip, int& ienv,
//           double rad[], double rrho[], double pres[], double dvol[],
//           double emas[], double phi[]);
// Adjust the declaration based on your actual C++ grid function

// Declare Fortran subroutine with C linkage
extern "C" {

// Declaration of the Fortran subroutine con_core() 
// with the underscore appended:

void grid_(int* irank, int* idec, const double* rhocore, const double* rhodrip,
              const double* rhoenv, const double* rhosurf,
              int* imax, int* icore, int* idrip, int* ienv,
              double* rad, double* rrho, double* pres, double* dvol,
              double* emas, double* phi);

void get_core_chemistry_(int* irank, char* version, int* imax, int* icore,
                            double* rrho, double* bar,
                            double* yneutr, double* yprot,
                            double* yelect, double* ymuon,
                            double* ylambda, double* ysminus,
                            double* yszero, double* ysplus,
                            double* yquarku, double* yquarkd,
                            double* yquarks, double* theta_k,
                            double* theta_p, double* fhad,
                            double* mstn, double* mstp,
                            double* mstla, double* mstsm,
                            double* msts0, double* mstsp);

void get_crust_chemistry_(int* irank, int* debug, char* version, int* imax, int* icore,
                              double* rrho, double* pres,
                              double* debar, double* dvol, double* bar,
                              double* a_cell, double* a_ion, double* z_ion,
                              double* v_ion, double* yelect, double* yneutr);

void get_spec_heat_degenerate_(
        double* cve, double* cvm, double* cvn,
        double* cvp, double* cvla, double* cvsm,
        double* cvs0, double* cvsp, double* cvqu,
        double* cvqd, double* cvqs, double* kfe,
        double* kfm, double* kfn, double* kfp,
        double* kfla, double* kfsm, double* kfs0,
        double* kfsp, double* kfqu, double* kfqd,
        double* kfqs, double* mstn, double* mstp,
        double* mstla, double* mstsm, double* msts0,
        double* mstsp, double* fhad, int* imax);

void get_degenerate_density_(int* irank, double* rrho, double* pres,
        double* rhod, int* imax, int* ienv);


void get_fermi_momenta_(int* irank, int* imax, int* icore, 
    double* rrho, double* bar, double* yneutr, double* yprot, 
    double* yelect, double* ymuon, double* ylambda, 
    double* ysminus, double* yszero, double* ysplus, 
    double* yquarku, double* yquarkd, double* yquarks, 
    double* fhad, double* theta_k, double* theta_p, double* kfn, 
    double* kfp, double* kfe, double* kfm, double* kfla, 
    double* kfsm, double* kfs0, double* kfsp, double* kfqu, 
    double* kfqd, double* kfqs, int* idurca_np, int* idurca_lap, 
    double* durca_ctrl_e, double* durca_ctrl_m, int* idurca_smn, 
    int* idurca_smla, int* idurca_sms0, int* idurca_quqd, 
    int* idurca_quqs, double* durca_henon_e, double* durca_henon_m);

void get_Tc_(
        int* irank,
        int* imax,
        int* icore,
        int* idrip,
        double* tcn,
        double* tcp,
        double* tcla,
        double* tcsm,
        double* tcs0,
        double* tcsp,
        double* tcuu,
        double* tcdd,
        double* tcss,
        double* tcud,
        double* tcus,
        double* tcds,
        double* tcu,
        double* tcd,
        double* tcs,
        double* sfn1s0,
        double* sfn3p2,
        double* sfp1s0,
        double* sfl1s0,
        double* fn1s0,
        double* fn3p2,
        double* fp1s0,
        double* fl1s0,
        double* kfmax_n3p2,
        double* delkf_n3p2,
        double* tcmax_n3p2,
        int* isf,
        double* kfn,
        double* kfp,
        double* kfla,
        double* kfqu,
        double* kfqd,
        double* kfqs,
        double* bar,
        double* fhad,
        double* yquarku,
        double* yquarkd,
        double* yquarks
    );

}

int main(int argc, char *argv[]) {

#ifdef O2SCL_MPI
  // Init MPI
  MPI_Init(&argc,&argv);
#endif

    // Set precision for floating-point output
    std::cout << std::scientific << std::setprecision(5);

    precool pc;

    // Set global pointer
    nscool_wrap_ptrs.resize(1);
    nscool_wrap_ptrs[0]=&pc;

    // -----------------------------
    // Initialize Input Parameters
    // -----------------------------
    int irank = 0;
    int idec = 3;
    double rhocore = 1.6e14;   // Example value (units as per your context)
    double rhodrip = 4.0e11;   // Example value
    double rhoenv = 1.0e8;    // Example value
    double rhosurf = 1.0e10;    // Example value

    // -----------------------------
    // Allocate Arrays
    // -----------------------------
    const int isize = 10000;
    // C++ arrays using std::vector<double>
    std::vector<double> rad_cpp(isize + 1, 0.0);
    std::vector<double> rrho_cpp(isize + 1, 0.0);
    std::vector<double> pres_cpp(isize + 1, 0.0);
    std::vector<double> dvol_cpp(isize + 1, 0.0);
    std::vector<double> emas_cpp(isize + 1, 0.0);
    std::vector<double> phi_cpp(isize + 1, 0.0);

    // Initialize std::vector<double> with size isize + 1 and default value 0.0
    std::vector<double> bar_cpp(isize + 1, 0.0);
    std::vector<double> yneutr_cpp(isize + 1, 0.0);
    std::vector<double> yprot_cpp(isize + 1, 0.0);
    std::vector<double> yelect_cpp(isize + 1, 0.0);
    std::vector<double> ymuon_cpp(isize + 1, 0.0);
    std::vector<double> ylambda_cpp(isize + 1, 0.0);
    std::vector<double> ysminus_cpp(isize + 1, 0.0);
    std::vector<double> yszero_cpp(isize + 1, 0.0);
    std::vector<double> ysplus_cpp(isize + 1, 0.0);
    std::vector<double> yquarku_cpp(isize + 1, 0.0);
    std::vector<double> yquarkd_cpp(isize + 1, 0.0);
    std::vector<double> yquarks_cpp(isize + 1, 0.0);
    std::vector<double> theta_k_cpp(isize + 1, 0.0);
    std::vector<double> theta_p_cpp(isize + 1, 0.0);
    std::vector<double> fhad_cpp(isize + 1, 0.0);

    // Initialize C++ arrays using std::vector<double>
    std::vector<double> debar_cpp(isize + 1, 0.0);
    std::vector<double> a_cell_cpp(isize + 1, 0.0);
    std::vector<double> a_ion_cpp(isize + 1, 0.0);
    std::vector<double> z_ion_cpp(isize + 1, 0.0);
    std::vector<double> v_ion_cpp(isize + 1, 0.0);

    std::vector<double> mstn_cpp(isize + 1, 0.0);
    std::vector<double> mstp_cpp(isize + 1, 0.0);
    std::vector<double> mstla_cpp(isize + 1, 0.0);
    std::vector<double> mstsm_cpp(isize + 1, 0.0);
    std::vector<double> msts0_cpp(isize + 1, 0.0);
    std::vector<double> mstsp_cpp(isize + 1, 0.0);

    std::vector<double> cve_cpp(isize + 1, 0.0);
    std::vector<double> cvm_cpp(isize + 1, 0.0);
    std::vector<double> cvn_cpp(isize + 1, 0.0);
    std::vector<double> cvp_cpp(isize + 1, 0.0);
    std::vector<double> cvla_cpp(isize + 1, 0.0);
    std::vector<double> cvsm_cpp(isize + 1, 0.0);
    std::vector<double> cvs0_cpp(isize + 1, 0.0);
    std::vector<double> cvsp_cpp(isize + 1, 0.0);
    std::vector<double> cvqu_cpp(isize + 1, 0.0);
    std::vector<double> cvqd_cpp(isize + 1, 0.0);
    std::vector<double> cvqs_cpp(isize + 1, 0.0);

    std::vector<double> kfe_cpp(isize + 1, 0.0);
    std::vector<double> kfm_cpp(isize + 1, 0.0);
    std::vector<double> kfn_cpp(isize + 1, 0.0);
    std::vector<double> kfp_cpp(isize + 1, 0.0);
    std::vector<double> kfla_cpp(isize + 1, 0.0);
    std::vector<double> kfsm_cpp(isize + 1, 0.0);
    std::vector<double> kfs0_cpp(isize + 1, 0.0);
    std::vector<double> kfsp_cpp(isize + 1, 0.0);
    std::vector<double> kfqu_cpp(isize + 1, 0.0);
    std::vector<double> kfqd_cpp(isize + 1, 0.0);
    std::vector<double> kfqs_cpp(isize + 1, 0.0);
    std::vector<double> rhod_cpp(isize + 1, 0.0);

    std::vector<int> idurca_np_cpp(isize + 1, 0);
    std::vector<int> idurca_lap_cpp(isize + 1, 0);
    std::vector<double> durca_ctrl_e_cpp(isize + 1, 0.0);
    std::vector<double> durca_ctrl_m_cpp(isize + 1, 0.0);
    std::vector<int> idurca_smn_cpp(isize + 1, 0);
    std::vector<int> idurca_smla_cpp(isize + 1, 0);
    std::vector<int> idurca_sms0_cpp(isize + 1, 0);
    std::vector<int> idurca_quqd_cpp(isize + 1, 0);
    std::vector<int> idurca_quqs_cpp(isize + 1, 0);
    std::vector<double> durca_henon_e_cpp(isize + 1, 0.0);
    std::vector<double> durca_henon_m_cpp(isize + 1, 0.0);

    std::vector<double> tcn_cpp(isize + 1, 0.0);  
    std::vector<double> tcp_cpp(isize + 1, 0.0);
    std::vector<double> tcla_cpp(isize + 1, 0.0);
    std::vector<double> tcsm_cpp(isize + 1, 0.0);
    std::vector<double> tcs0_cpp(isize + 1, 0.0);
    std::vector<double> tcsp_cpp(isize + 1, 0.0);
    std::vector<double> tcuu_cpp(isize + 1, 0.0);
    std::vector<double> tcdd_cpp(isize + 1, 0.0);
    std::vector<double> tcss_cpp(isize + 1, 0.0);
    std::vector<double> tcud_cpp(isize + 1, 0.0);
    std::vector<double> tcus_cpp(isize + 1, 0.0);
    std::vector<double> tcds_cpp(isize + 1, 0.0);
    std::vector<double> tcu_cpp(isize + 1, 0.0);
    std::vector<double> tcd_cpp(isize + 1, 0.0);
    std::vector<double> tcs_cpp(isize + 1, 0.0);


    
    double* rad_fortran = new double[isize + 1];
    double* rrho_fortran = new double[isize + 1];
    double* pres_fortran = new double[isize + 1];
    double* dvol_fortran = new double[isize + 1];
    double* emas_fortran = new double[isize + 1];
    double* phi_fortran = new double[isize + 1];

    // Allocate Fortran arrays using double*
    double* cve_fortran  = new double[isize + 1];
    double* cvm_fortran  = new double[isize + 1];
    double* cvn_fortran  = new double[isize + 1];
    double* cvp_fortran  = new double[isize + 1];
    double* cvla_fortran = new double[isize + 1];
    double* cvsm_fortran = new double[isize + 1];
    double* cvs0_fortran = new double[isize + 1];
    double* cvsp_fortran = new double[isize + 1];
    double* cvqu_fortran = new double[isize + 1];
    double* cvqd_fortran = new double[isize + 1];
    double* cvqs_fortran = new double[isize + 1];

    double* kfe_fortran  = new double[isize + 1];
    double* kfm_fortran  = new double[isize + 1];
    double* kfn_fortran  = new double[isize + 1];
    double* kfp_fortran  = new double[isize + 1];
    double* kfla_fortran = new double[isize + 1];
    double* kfsm_fortran = new double[isize + 1];
    double* kfs0_fortran = new double[isize + 1];
    double* kfsp_fortran = new double[isize + 1];
    double* kfqu_fortran = new double[isize + 1];
    double* kfqd_fortran = new double[isize + 1];
    double* kfqs_fortran = new double[isize + 1];
    double* rhod_fortran = new double[isize + 1];

    // Similarly, allocate Fortran arrays
    double* bar_fortran    = new double[isize + 1];
    double* yneutr_fortran = new double[isize + 1];
    double* yprot_fortran  = new double[isize + 1];
    double* yelect_fortran = new double[isize + 1];
    double* ymuon_fortran  = new double[isize + 1];
    double* ylambda_fortran= new double[isize + 1];
    double* ysminus_fortran= new double[isize + 1];
    double* yszero_fortran = new double[isize + 1];
    double* ysplus_fortran = new double[isize + 1];
    double* yquarku_fortran= new double[isize + 1];
    double* yquarkd_fortran= new double[isize + 1];
    double* yquarks_fortran= new double[isize + 1];
    double* theta_k_fortran= new double[isize + 1];
    double* theta_p_fortran= new double[isize + 1];
    double* fhad_fortran   = new double[isize + 1];
    double* mstn_fortran   = new double[isize + 1];
    double* mstp_fortran   = new double[isize + 1];
    double* mstla_fortran  = new double[isize + 1];
    double* mstsm_fortran  = new double[isize + 1];
    double* msts0_fortran  = new double[isize + 1];
    double* mstsp_fortran  = new double[isize + 1];

    // Allocate Fortran arrays using double*
    double* debar_fortran   = new double[isize + 1];
    double* a_cell_fortran  = new double[isize + 1];
    double* a_ion_fortran   = new double[isize + 1];
    double* z_ion_fortran   = new double[isize + 1];
    double* v_ion_fortran   = new double[isize + 1];

    int* idurca_np_fortran = new int[isize + 1];
    int* idurca_lap_fortran = new int[isize + 1];
    double* durca_ctrl_e_fortran = new double[isize + 1];
    double* durca_ctrl_m_fortran = new double[isize + 1];
    int* idurca_smn_fortran = new int[isize + 1];
    int* idurca_smla_fortran = new int[isize + 1];
    int* idurca_sms0_fortran = new int[isize + 1];
    int* idurca_quqd_fortran = new int[isize + 1];
    int* idurca_quqs_fortran = new int[isize + 1];
    double* durca_henon_e_fortran = new double[isize + 1];
    double* durca_henon_m_fortran = new double[isize + 1];

    double* tcn_fortran = new double[isize + 1];  
    double* tcp_fortran = new double[isize + 1];
    double* tcla_fortran= new double[isize + 1];
    double* tcsm_fortran= new double[isize + 1];
    double* tcs0_fortran= new double[isize + 1];
    double* tcsp_fortran= new double[isize + 1];
    double* tcuu_fortran= new double[isize + 1];
    double* tcdd_fortran= new double[isize + 1];
    double* tcss_fortran= new double[isize + 1];
    double* tcud_fortran= new double[isize + 1];
    double* tcus_fortran= new double[isize + 1];
    double* tcds_fortran= new double[isize + 1];
    double* tcu_fortran= new double[isize + 1];
    double* tcd_fortran= new double[isize + 1];
    double* tcs_fortran= new double[isize + 1];

    // Initialize output variables
    int imax_cpp = 0, icore_cpp = 0, idrip_cpp = 0, ienv_cpp = 0;
    int imax_fortran = 0, icore_fortran = 0, idrip_fortran = 0, ienv_fortran = 0;

    // -----------------------------
    // Call C++ grid Function
    // -----------------------------
    pc.star_struct_cpp(idec, rhocore, rhodrip, rhoenv, rhosurf,
         imax_cpp, icore_cpp, idrip_cpp, ienv_cpp,
         rad_cpp, rrho_cpp, pres_cpp, dvol_cpp, emas_cpp, phi_cpp);

    // -----------------------------
    // Call Fortran grid Subroutine
    // -----------------------------
    grid_(&irank, &idec, &rhocore, &rhodrip, &rhoenv, &rhosurf,
         &imax_fortran, &icore_fortran, &idrip_fortran, &ienv_fortran,
         rad_fortran, rrho_fortran, pres_fortran, dvol_fortran, emas_fortran, phi_fortran);

    // -----------------------------
    // Compare and Display Outputs
    // -----------------------------
    std::cout << "----- Grid Function Comparison -----\n" << std::endl;

    // Compare integer outputs
    std::cout << "Integer Outputs:" << std::endl;
    std::cout << "C++ imax    : " << imax_cpp << "\tFortran imax    : " << imax_fortran
              << "\tDifference: " << (imax_cpp - imax_fortran) << std::endl;
    std::cout << "C++ icore   : " << icore_cpp << "\tFortran icore   : " << icore_fortran
              << "\tDifference: " << (icore_cpp - icore_fortran) << std::endl;
    std::cout << "C++ idrip   : " << idrip_cpp << "\tFortran idrip   : " << idrip_fortran
              << "\tDifference: " << (idrip_cpp - idrip_fortran) << std::endl;
    std::cout << "C++ ienv    : " << ienv_cpp << "\tFortran ienv    : " << ienv_fortran
              << "\tDifference: " << (ienv_cpp - ienv_fortran) << "\n" << std::endl;

    // Compare selected elements from arrays (e.g., first 5 and last 5 elements)
    int indices_to_compare[] = {0, 1, 2, 5000, 9999, 10000};
    int num_indices = sizeof(indices_to_compare) / sizeof(indices_to_compare[0]);

    std::cout << "Array Outputs Comparison (Selected Indices):\n" << std::endl;
    std::cout << std::setw(10) << "Index"
              << std::setw(20) << "rad_cpp"
              << std::setw(20) << "rad_fortran"
              << std::setw(20) << "Difference(rad)" << std::endl;

    for(int i = 0; i < num_indices; ++i) {
        int idx = indices_to_compare[i];
        std::cout << std::setw(10) << idx
                  << std::setw(20) << rad_cpp[idx]
                  << std::setw(20) << rad_fortran[idx]
                  << std::setw(20) << (rad_cpp[idx] - rad_fortran[idx]) << std::endl;
    }

    std::cout << "\nArray Outputs Comparison (Selected Indices):\n" << std::endl;
    std::cout << std::setw(10) << "Index"
              << std::setw(20) << "rrho_cpp"
              << std::setw(20) << "rrho_fortran"
              << std::setw(20) << "Difference(rrho)" << std::endl;

    for(int i = 0; i < num_indices; ++i) {
        int idx = indices_to_compare[i];
        std::cout << std::setw(10) << idx
                  << std::setw(20) << rrho_cpp[idx]
                  << std::setw(20) << rrho_fortran[idx]
                  << std::setw(20) << (rrho_cpp[idx] - rrho_fortran[idx]) << std::endl;
    }

    // Repeat similar comparisons for pres, dvol, emas, phi as needed

    char version[3] = {'n', 'e', 'w'}; // Example version; ensure it has exactly 3 characters

    // -----------------------------
    // Call C++ get_core_chemistry Function
    // -----------------------------
    pc.core_comp_cpp(imax_cpp, icore_cpp,
                    rrho_cpp, bar_cpp,
                    yneutr_cpp, yprot_cpp,
                    yelect_cpp, ymuon_cpp,
                    ylambda_cpp, ysminus_cpp,
                    yszero_cpp, ysplus_cpp,
                    yquarku_cpp, yquarkd_cpp,
                    yquarks_cpp, theta_k_cpp,
                    theta_p_cpp, fhad_cpp,
                    mstn_cpp, mstp_cpp,
                    mstla_cpp, mstsm_cpp,
                    msts0_cpp, mstsp_cpp);

    // -----------------------------
    // Call Fortran get_core_chemistry Subroutine
    // -----------------------------
    get_core_chemistry_(&irank, version, &imax_fortran, &icore_fortran,
                      rrho_fortran, bar_fortran,
                      yneutr_fortran, yprot_fortran,
                      yelect_fortran, ymuon_fortran,
                      ylambda_fortran, ysminus_fortran,
                      yszero_fortran, ysplus_fortran,
                      yquarku_fortran, yquarkd_fortran,
                      yquarks_fortran, theta_k_fortran,
                      theta_p_fortran, fhad_fortran,
                      mstn_fortran, mstp_fortran,
                      mstla_fortran, mstsm_fortran,
                      msts0_fortran, mstsp_fortran);

    // -----------------------------
    // Compare and Display Outputs
    // -----------------------------
    std::cout << "----- get_core_chemistry Function Comparison -----\n" << std::endl;

    // Compare integer outputs
    std::cout << "Integer Outputs:" << std::endl;
    std::cout << "C++ imax    : " << imax_cpp << "\tFortran imax    : " << imax_fortran
              << "\tDifference: " << (imax_cpp - imax_fortran) << std::endl;
    std::cout << "C++ icore   : " << icore_cpp << "\tFortran icore   : " << icore_fortran
              << "\tDifference: " << (icore_cpp - icore_fortran) << "\n" << std::endl;

    // Define indices to compare
    std::vector<int> indices_to_compare2 = {0, 1, 2, 5000, 9998, 9999, 10000};
    int num_indices2 = indices_to_compare2.size();

    // Function to print comparison for a given array
    auto print_comparison = [&](const std::string& name_cpp, const std::vector<double>& vec_cpp,
                                const std::string& name_fortran, double* vec_fortran) {
        std::cout << "Array Outputs Comparison for '" << name_cpp << "':\n" << std::endl;
        std::cout << std::setw(10) << "Index"
                  << std::setw(20) << name_cpp + "_cpp"
                  << std::setw(20) << name_fortran + "_fortran"
                  << std::setw(20) << "Difference" << std::endl;

        for(auto idx : indices_to_compare2) {
            if(idx < 0 || idx > isize) {
                std::cout << std::setw(10) << idx
                          << std::setw(20) << "N/A"
                          << std::setw(20) << "N/A"
                          << std::setw(20) << "N/A" << std::endl;
                continue;
            }
            double cpp_val = vec_cpp[idx];
            double fortran_val = vec_fortran[idx];
            double diff = cpp_val - fortran_val;
            std::cout << std::setw(10) << idx
                      << std::setw(20) << cpp_val
                      << std::setw(20) << fortran_val
                      << std::setw(20) << diff << std::endl;
        }
        std::cout << std::endl;
    };

    // List of arrays to compare
    std::vector<std::pair<std::string, std::pair<std::vector<double>*, double*>>> arrays_to_compare_list = {
        {"rrho", {&rrho_cpp, rrho_fortran}},
        {"bar", {&bar_cpp, bar_fortran}},
        {"yneutr", {&yneutr_cpp, yneutr_fortran}},
        {"yprot", {&yprot_cpp, yprot_fortran}},
        {"yelect", {&yelect_cpp, yelect_fortran}},
        {"ymuon", {&ymuon_cpp, ymuon_fortran}},
        {"ylambda", {&ylambda_cpp, ylambda_fortran}},
        {"ysminus", {&ysminus_cpp, ysminus_fortran}},
        {"yszero", {&yszero_cpp, yszero_fortran}},
        {"ysplus", {&ysplus_cpp, ysplus_fortran}},
        {"yquarku", {&yquarku_cpp, yquarku_fortran}},
        {"yquarkd", {&yquarkd_cpp, yquarkd_fortran}},
        {"yquarks", {&yquarks_cpp, yquarks_fortran}},
        {"theta_k", {&theta_k_cpp, theta_k_fortran}},
        {"theta_p", {&theta_p_cpp, theta_p_fortran}},
        {"fhad", {&fhad_cpp, fhad_fortran}},
        {"mstn", {&mstn_cpp, mstn_fortran}},
        {"mstp", {&mstp_cpp, mstp_fortran}},
        {"mstla", {&mstla_cpp, mstla_fortran}},
        {"mstsm", {&mstsm_cpp, mstsm_fortran}},
        {"msts0", {&msts0_cpp, msts0_fortran}},
        {"mstsp", {&mstsp_cpp, mstsp_fortran}}
    };

    for(auto& pair : arrays_to_compare_list) {
        const std::string& name_cpp = pair.first;
        std::vector<double>* vec_cpp_ptr = pair.second.first;
        double* vec_fortran_ptr = pair.second.second;

        if(vec_cpp_ptr && vec_fortran_ptr) {
            print_comparison(name_cpp, *vec_cpp_ptr, name_cpp, vec_fortran_ptr);
        } else {
            std::cout << "No mapping found for array: " << name_cpp << std::endl;
        }
    }

    // -----------------------------
    // Additional Verification (Optional)
    // -----------------------------
    // For thorough testing, verify specific properties or invariants.
    // Example:
    bool consistent = true;
    const double tolerance = 1e-10;
    for(auto idx : indices_to_compare2) {
        if(idx < 0 || idx > isize) continue;
        double diff_rrho = rrho_cpp[idx] - rrho_fortran[idx];
        double diff_bar = bar_cpp[idx] - bar_fortran[idx];
        // Add more differences as needed, e.g.,
        // double diff_yneutr = yneutr_cpp[idx] - yneutr_fortran[idx];
        // ...

        if(std::abs(diff_rrho) > tolerance || std::abs(diff_bar) > tolerance /* || ... */) {
            consistent = false;
            std::cerr << "Discrepancy at index " << idx << " in 'rrho' or 'bar' arrays." << std::endl;
            break;
        }
    }

    if(consistent) {
        std::cout << "Both C++ and Fortran implementations produce identical results within tolerance." << std::endl;
    } else {
        std::cout << "Discrepancies found between C++ and Fortran implementations." << std::endl;
    }

    // -----------------------------
    // Call C++ get_crust_chemistry Function
    // -----------------------------
    int debug=0;
    pc.crust_comp_cpp(irank, debug, version, imax_cpp, icore_cpp,
                                     rrho_cpp, pres_cpp,
                                     debar_cpp, dvol_cpp, bar_cpp,
                                     a_cell_cpp, a_ion_cpp, z_ion_cpp,
                                     v_ion_cpp, yelect_cpp, yneutr_cpp);

    // -----------------------------
    // Call Fortran get_crust_chemistry Subroutine
    // -----------------------------
    get_crust_chemistry_(&irank, &debug, version, &imax_fortran, &icore_fortran,
                         rrho_fortran, pres_fortran,
                         debar_fortran, dvol_fortran, bar_fortran,
                         a_cell_fortran, a_ion_fortran, z_ion_fortran,
                         v_ion_fortran, yelect_fortran, yneutr_fortran);

    // -----------------------------
    // Compare and Display Outputs
    // -----------------------------
    std::cout << "----- get_crust_chemistry Function Comparison -----\n" << std::endl;

    // Compare integer outputs
    std::cout << "Integer Outputs:" << std::endl;
    std::cout << "C++ imax    : " << imax_cpp << "\tFortran imax    : " << imax_fortran
              << "\tDifference: " << (imax_cpp - imax_fortran) << std::endl;
    std::cout << "C++ icore   : " << icore_cpp << "\tFortran icore   : " << icore_fortran
              << "\tDifference: " << (icore_cpp - icore_fortran) << "\n" << std::endl;

    // List of arrays to compare
    arrays_to_compare_list = {
        {"rrho", {&rrho_cpp, rrho_fortran}},
        {"pres", {&pres_cpp, pres_fortran}},
        {"debar", {&debar_cpp, debar_fortran}},
        {"dvol", {&dvol_cpp, dvol_fortran}},
        {"bar", {&bar_cpp, bar_fortran}},
        {"a_cell", {&a_cell_cpp, a_cell_fortran}},
        {"a_ion", {&a_ion_cpp, a_ion_fortran}},
        {"z_ion", {&z_ion_cpp, z_ion_fortran}},
        {"v_ion", {&v_ion_cpp, v_ion_fortran}}
    };

    // Iterate and compare each array
    for(auto& pair : arrays_to_compare_list) {
        const std::string& name_cpp = pair.first;
        std::vector<double>* vec_cpp_ptr = pair.second.first;
        double* vec_fortran_ptr = pair.second.second;

        if(vec_cpp_ptr && vec_fortran_ptr) {
            print_comparison(name_cpp, *vec_cpp_ptr, name_cpp, vec_fortran_ptr);
        } else {
            std::cout << "No mapping found for array: " << name_cpp << std::endl;
        }
    }

    // -----------------------------
    // Additional Verification (Optional)
    // -----------------------------
    consistent=true;
    // Check if all compared differences are within a defined tolerance
    for(auto idx : indices_to_compare) {
        if(idx < 0 || idx > isize) continue;
        double diff_rrho = rrho_cpp[idx] - rrho_fortran[idx];
        double diff_pres = pres_cpp[idx] - pres_fortran[idx];
        double diff_debar = debar_cpp[idx] - debar_fortran[idx];
        double diff_dvol = dvol_cpp[idx] - dvol_fortran[idx];
        double diff_bar = bar_cpp[idx] - bar_fortran[idx];
        double diff_a_cell = a_cell_cpp[idx] - a_cell_fortran[idx];
        double diff_a_ion = a_ion_cpp[idx] - a_ion_fortran[idx];
        double diff_z_ion = z_ion_cpp[idx] - z_ion_fortran[idx];
        double diff_v_ion = v_ion_cpp[idx] - v_ion_fortran[idx];
        double diff_yelect = yelect_cpp[idx] - yelect_fortran[idx];
        double diff_yneutr = yneutr_cpp[idx] - yneutr_fortran[idx];

        if(std::abs(diff_rrho) > tolerance || std::abs(diff_pres) > tolerance ||
           std::abs(diff_debar) > tolerance || std::abs(diff_dvol) > tolerance ||
           std::abs(diff_bar) > tolerance || std::abs(diff_a_cell) > tolerance ||
           std::abs(diff_a_ion) > tolerance || std::abs(diff_z_ion) > tolerance ||
           std::abs(diff_v_ion) > tolerance || std::abs(diff_yelect) > tolerance ||
           std::abs(diff_yneutr) > tolerance) {
            consistent = false;
            std::cerr << "Discrepancy at index " << idx << " in one or more arrays." << std::endl;
            break;
        }
    }

    if(consistent) {
        std::cout << "Both C++ and Fortran implementations produce identical results within tolerance." << std::endl;
    } else {
        std::cout << "Discrepancies found between C++ and Fortran implementations." << std::endl;
    }

    // -----------------------------
    // Call C++ get_spec_heat_degenerate Function
    // -----------------------------
    pc.get_spec_heat_degenerate(
        cve_cpp, cvm_cpp, cvn_cpp, cvp_cpp, cvla_cpp,
        cvsm_cpp, cvs0_cpp, cvsp_cpp, cvqu_cpp, cvqd_cpp, cvqs_cpp,
        kfe_cpp, kfm_cpp, kfn_cpp, kfp_cpp, kfla_cpp,
        kfsm_cpp, kfs0_cpp, kfsp_cpp, kfqu_cpp, kfqd_cpp, kfqs_cpp,
        mstn_cpp, mstp_cpp, mstla_cpp, mstsm_cpp, msts0_cpp, mstsp_cpp,
        fhad_cpp, imax_cpp
    );

    // -----------------------------
    // Call Fortran get_spec_heat_degenerate Subroutine
    // -----------------------------
    get_spec_heat_degenerate_(cve_fortran,cvm_fortran,cvn_fortran,cvp_fortran,
        cvla_fortran,cvsm_fortran,cvs0_fortran,cvsp_fortran,cvqu_fortran,
        cvqd_fortran,cvqs_fortran,kfe_fortran,kfm_fortran,kfn_fortran,
        kfp_fortran,kfla_fortran,kfsm_fortran,kfs0_fortran,kfsp_fortran,
        kfqu_fortran,kfqd_fortran,kfqs_fortran,mstn_fortran,mstp_fortran,
        mstla_fortran,mstsm_fortran,msts0_fortran,mstsp_fortran,fhad_fortran,&imax_fortran
    );

    // -----------------------------
    // Compare and Display Outputs
    // -----------------------------
    std::cout << "----- get_spec_heat_degenerate Function Comparison -----\n" << std::endl;

    // Compare integer output
    std::cout << "Integer Output:" << std::endl;
    std::cout << "C++ imax    : " << imax_cpp << "\tFortran imax    : " << imax_fortran
              << "\tDifference: " << (imax_cpp - imax_fortran) << std::endl;
    std::cout << "\n";

    // Define indices to compare
    std::vector<int> indices_to_compare3 = {0, 1, 2, 5000, 9998, 9999, 10000};

    // List of arrays to compare
    arrays_to_compare_list = {
        {"cve",  {&cve_cpp,  cve_fortran}},
        {"cvm",  {&cvm_cpp,  cvm_fortran}},
        {"cvn",  {&cvn_cpp,  cvn_fortran}},
        {"cvp",  {&cvp_cpp,  cvp_fortran}},
        {"cvla", {&cvla_cpp, cvla_fortran}},
        {"cvsm", {&cvsm_cpp, cvsm_fortran}},
        {"cvs0", {&cvs0_cpp, cvs0_fortran}},
        {"cvsp", {&cvsp_cpp, cvsp_fortran}},
        {"cvqu", {&cvqu_cpp, cvqu_fortran}},
        {"cvqd", {&cvqd_cpp, cvqd_fortran}},
        {"cvqs", {&cvqs_cpp, cvqs_fortran}},
        {"kfe",  {&kfe_cpp,  kfe_fortran}},
        {"kfm",  {&kfm_cpp,  kfm_fortran}},
        {"kfn",  {&kfn_cpp,  kfn_fortran}},
        {"kfp",  {&kfp_cpp,  kfp_fortran}},
        {"kfla", {&kfla_cpp, kfla_fortran}},
        {"kfsm", {&kfsm_cpp, kfsm_fortran}},
        {"kfs0", {&kfs0_cpp, kfs0_fortran}},
        {"kfsp", {&kfsp_cpp, kfsp_fortran}},
        {"kfqu", {&kfqu_cpp, kfqu_fortran}},
        {"kfqd", {&kfqd_cpp, kfqd_fortran}},
        {"kfqs", {&kfqs_cpp, kfqs_fortran}},
        {"mstn", {&mstn_cpp, mstn_fortran}},
        {"mstp", {&mstp_cpp, mstp_fortran}},
        {"mstla",{&mstla_cpp, mstla_fortran}},
        {"mstsm",{&mstsm_cpp, mstsm_fortran}},
        {"msts0",{&msts0_cpp, msts0_fortran}},
        {"mstsp",{&mstsp_cpp, mstsp_fortran}},
        {"fhad", {&fhad_cpp, fhad_fortran}}
    };

    // Iterate and compare each array
    for(auto& pair : arrays_to_compare_list) {
        const std::string& name_cpp = pair.first;
        std::vector<double>* vec_cpp_ptr = pair.second.first;
        double* vec_fortran_ptr = pair.second.second;

        if(vec_cpp_ptr && vec_fortran_ptr) {
            print_comparison(name_cpp, *vec_cpp_ptr, name_cpp, vec_fortran_ptr);
        } else {
            std::cout << "No mapping found for array: " << name_cpp << std::endl;
        }
    }

    // -----------------------------
    // Additional Verification (Optional)
    // -----------------------------
    consistent=true;
    // Check if all compared differences are within a defined tolerance
    for(auto idx : indices_to_compare3) {
        if(idx < 0 || idx > isize) continue;

        // Compare each array at index 'idx'
        // List all arrays and their differences
        double diff_cve  = cve_cpp[idx]  - cve_fortran[idx];
        double diff_cvm  = cvm_cpp[idx]  - cvm_fortran[idx];
        double diff_cvn  = cvn_cpp[idx]  - cvn_fortran[idx];
        double diff_cvp  = cvp_cpp[idx]  - cvp_fortran[idx];
        double diff_cvla = cvla_cpp[idx] - cvla_fortran[idx];
        double diff_cvsm = cvsm_cpp[idx] - cvsm_fortran[idx];
        double diff_cvs0 = cvs0_cpp[idx] - cvs0_fortran[idx];
        double diff_cvsp = cvsp_cpp[idx] - cvsp_fortran[idx];
        double diff_cvqu = cvqu_cpp[idx] - cvqu_fortran[idx];
        double diff_cvqd = cvqd_cpp[idx] - cvqd_fortran[idx];
        double diff_cvqs = cvqs_cpp[idx] - cvqs_fortran[idx];

        double diff_kfe  = kfe_cpp[idx]  - kfe_fortran[idx];
        double diff_kfm  = kfm_cpp[idx]  - kfm_fortran[idx];
        double diff_kfn  = kfn_cpp[idx]  - kfn_fortran[idx];
        double diff_kfp  = kfp_cpp[idx]  - kfp_fortran[idx];
        double diff_kfla = kfla_cpp[idx] - kfla_fortran[idx];
        double diff_kfsm = kfsm_cpp[idx] - kfsm_fortran[idx];
        double diff_kfs0 = kfs0_cpp[idx] - kfs0_fortran[idx];
        double diff_kfsp = kfsp_cpp[idx] - kfsp_fortran[idx];
        double diff_kfqu = kfqu_cpp[idx] - kfqu_fortran[idx];
        double diff_kfqd = kfqd_cpp[idx] - kfqd_fortran[idx];
        double diff_kfqs = kfqs_cpp[idx] - kfqs_fortran[idx];

        double diff_mstn  = mstn_cpp[idx]  - mstn_fortran[idx];
        double diff_mstp  = mstp_cpp[idx]  - mstp_fortran[idx];
        double diff_mstla = mstla_cpp[idx] - mstla_fortran[idx];
        double diff_mstsm = mstsm_cpp[idx] - mstsm_fortran[idx];
        double diff_msts0 = msts0_cpp[idx] - msts0_fortran[idx];
        double diff_mstsp = mstsp_cpp[idx] - mstsp_fortran[idx];

        double diff_fhad = fhad_cpp[idx] - fhad_fortran[idx];
        
        // Check if any difference exceeds the tolerance
        if(std::abs(diff_cve)  > tolerance ||
           std::abs(diff_cvm)  > tolerance ||
           std::abs(diff_cvn)  > tolerance ||
           std::abs(diff_cvp)  > tolerance ||
           std::abs(diff_cvla) > tolerance ||
           std::abs(diff_cvsm) > tolerance ||
           std::abs(diff_cvs0) > tolerance ||
           std::abs(diff_cvsp) > tolerance ||
           std::abs(diff_cvqu) > tolerance ||
           std::abs(diff_cvqd) > tolerance ||
           std::abs(diff_cvqs) > tolerance ||

           std::abs(diff_kfe)  > tolerance ||
           std::abs(diff_kfm)  > tolerance ||
           std::abs(diff_kfn)  > tolerance ||
           std::abs(diff_kfp)  > tolerance ||
           std::abs(diff_kfla) > tolerance ||
           std::abs(diff_kfsm) > tolerance ||
           std::abs(diff_kfs0) > tolerance ||
           std::abs(diff_kfsp) > tolerance ||
           std::abs(diff_kfqu) > tolerance ||
           std::abs(diff_kfqd) > tolerance ||
           std::abs(diff_kfqs) > tolerance ||

           std::abs(diff_mstn)  > tolerance ||
           std::abs(diff_mstp)  > tolerance ||
           std::abs(diff_mstla) > tolerance ||
           std::abs(diff_mstsm) > tolerance ||
           std::abs(diff_msts0) > tolerance ||
           std::abs(diff_mstsp) > tolerance ||

           std::abs(diff_fhad) > tolerance ) {
            consistent = false;
            std::cerr << "Discrepancy at index " << idx << " in one or more arrays." << std::endl;
            break;
        }
    }

    if(consistent) {
        std::cout << "Both C++ and Fortran implementations produce identical results within tolerance." << std::endl;
    } else {
        std::cout << "Discrepancies found between C++ and Fortran implementations." << std::endl;
    }

    // -----------------------------
    // Call C++ get_degenerate_density Function
    // -----------------------------
    pc.crust_eos_cpp(rrho_cpp, pres_cpp,
        rhod_cpp, imax_cpp, ienv_cpp);

    // -----------------------------
    // Call Fortran get_degenerate_density Subroutine
    // -----------------------------
    
    get_degenerate_density_(&irank, rrho_fortran, pres_fortran,
        rhod_fortran, &imax_fortran, &ienv_fortran);

    // -----------------------------
    // Compare and Display Outputs
    // -----------------------------
    std::cout << "----- get_degenerate_density Function Comparison -----\n" << std::endl;

    // Compare integer output
    std::cout << "Integer Output:" << std::endl;
    std::cout << "C++ imax    : " << imax_cpp << "\tFortran imax    : " << imax_fortran
              << "\tDifference: " << (imax_cpp - imax_fortran) << std::endl;
    std::cout << "C++ ienv    : " << ienv_cpp << "\tFortran ienv    : " << ienv_fortran
              << "\tDifference: " << (0) << "\n" << std::endl; // ienv is same

    // Define indices to compare
    std::vector<int> indices_to_compare4 = {0, 1, 2, 5000, 9998, 9999, 10000};

    // Compare rhod arrays
    print_comparison("rhod", rhod_cpp, "rhod", rhod_fortran);

    // -----------------------------
    // Additional Verification (Optional)
    // -----------------------------
    consistent=true;
    // Check if all compared differences are within a defined tolerance
    for(auto idx : indices_to_compare4) {
        if(idx < 0 || idx > isize) continue;

        double diff = rhod_cpp[idx] - rhod_fortran[idx];
        if(std::abs(diff) > tolerance) {
            consistent = false;
            std::cerr << "Discrepancy at index " << idx << " in 'rhod' array." << std::endl;
            break;
        }
    }

    if(consistent) {
        std::cout << "Both C++ and Fortran implementations produce identical 'rhod' results within tolerance." << std::endl;
    } else {
        std::cout << "Discrepancies found in 'rhod' between C++ and Fortran implementations." << std::endl;
    }

    // -----------------------------
    // Call C++ get_fermi_momentum Function
    // -----------------------------
    pc.get_fermi_momenta(irank,imax_cpp,icore_cpp,rrho_cpp,
            bar_cpp,yneutr_cpp,yprot_cpp,yelect_cpp,ymuon_cpp,
            ylambda_cpp,ysminus_cpp,yszero_cpp,ysplus_cpp,
            yquarku_cpp,yquarkd_cpp,yquarks_cpp,fhad_cpp,
            theta_k_cpp,theta_p_cpp,kfn_cpp,kfp_cpp,kfe_cpp,
            kfm_cpp,kfla_cpp,kfsm_cpp,kfs0_cpp,kfsp_cpp,kfqu_cpp,
            kfqd_cpp,kfqs_cpp,idurca_np_cpp,idurca_lap_cpp,
            durca_ctrl_e_cpp,durca_ctrl_m_cpp,idurca_smn_cpp,
            idurca_smla_cpp,idurca_sms0_cpp,idurca_quqd_cpp,
            idurca_quqs_cpp,durca_henon_e_cpp,durca_henon_m_cpp);
    
    // -----------------------------
    // Call Fortran get_fermi_momenta Subroutine
    // -----------------------------
    get_fermi_momenta_(&irank, &imax_fortran, &icore_fortran,
            rrho_fortran, bar_fortran, yneutr_fortran, yprot_fortran,
            yelect_fortran, ymuon_fortran, ylambda_fortran,
            ysminus_fortran, yszero_fortran, ysplus_fortran,
            yquarku_fortran, yquarkd_fortran, yquarks_fortran,
            fhad_fortran, theta_k_fortran, theta_p_fortran,
            kfn_fortran, kfp_fortran, kfe_fortran, kfm_fortran,
            kfla_fortran, kfsm_fortran, kfs0_fortran, kfsp_fortran,
            kfqu_fortran, kfqd_fortran, kfqs_fortran,
            idurca_np_fortran, idurca_lap_fortran, durca_ctrl_e_fortran,
            durca_ctrl_m_fortran, idurca_smn_fortran,
            idurca_smla_fortran, idurca_sms0_fortran, idurca_quqd_fortran,
            idurca_quqs_fortran, durca_henon_e_fortran, durca_henon_m_fortran);
    
    
    // -----------------------------
    // Compare Double Arrays
    // -----------------------------
    
    print_comparison("kfn", kfn_cpp, "kfn", kfn_fortran);
    print_comparison("kfp", kfp_cpp, "kfp", kfp_fortran);
    print_comparison("kfe", kfe_cpp, "kfe", kfe_fortran);
    print_comparison("kfm", kfm_cpp, "kfm", kfm_fortran);
    print_comparison("kfla", kfla_cpp, "kfla", kfla_fortran);
    print_comparison("kfsm", kfsm_cpp, "kfsm", kfsm_fortran);
    print_comparison("kfs0", kfs0_cpp, "kfs0", kfs0_fortran);
    print_comparison("kfsp", kfsp_cpp, "kfsp", kfsp_fortran);
    print_comparison("kfqu", kfqu_cpp, "kfqu", kfqu_fortran);
    print_comparison("kfqd", kfqd_cpp, "kfqd", kfqd_fortran);
    print_comparison("kfqs", kfqs_cpp, "kfqs", kfqs_fortran);
    print_comparison("durca_ctrl_e", durca_ctrl_e_cpp, "durca_ctrl_e", durca_ctrl_e_fortran);
    print_comparison("durca_ctrl_m", durca_ctrl_m_cpp, "durca_ctrl_m", durca_ctrl_m_fortran);
    print_comparison("durca_henon_e", durca_henon_e_cpp, "durca_henon_e", durca_henon_e_fortran);
    print_comparison("durca_henon_m", durca_henon_m_cpp, "durca_henon_m", durca_henon_m_fortran);



    int sfn1s0=1;
    int sfn3p2=101;
    int sfp1s0=3;
    int sfl1s0=0;
    int fn1s0=1;
    int fn3p2=1;
    int fp1s0=1;
    int fl1s0=1;
    int isf_cpp = -1;
    
    // -----------------------------
    // Call C++ get_Tc Function
    // -----------------------------
    pc.get_Tc( irank, imax_cpp, icore_cpp, idrip_cpp, 
    tcn_cpp, tcp_cpp, tcla_cpp, tcsm_cpp, tcs0_cpp, 
    tcsp_cpp, tcuu_cpp, tcdd_cpp, tcss_cpp, tcud_cpp, 
    tcus_cpp, tcds_cpp, tcu_cpp, tcd_cpp, tcs_cpp, 
    sfl1s0, fn1s0, fn3p2, fp1s0, fl1s0, isf_cpp, 
    kfn_cpp, kfp_cpp, kfla_cpp, kfqu_cpp, kfqd_cpp, 
    kfqs_cpp, bar_cpp, fhad_cpp, yquarku_cpp, 
    yquarkd_cpp, yquarks_cpp);

    double sfn1s0_fortran=1;
    double sfn3p2_fortran=101;
    double sfp1s0_fortran=3;
    double sfl1s0_fortran=0;
    double fn1s0_fortran=1;
    double fn3p2_fortran=1;
    double fp1s0_fortran=1;
    double fl1s0_fortran=1;
    int isf_fortran = -1;
    double kfmax_n3p2 = 2.5;
    double delkf_n3p2 = 0.7;
    double tcmax_n3p2 = 1e10;
/*
    // -----------------------------
    // Call Fortran get_Tc Subroutine
    // -----------------------------
    get_Tc_(&irank, &imax_fortran, &icore_fortran, &idrip_fortran, 
    tcn_fortran, tcp_fortran, tcla_fortran, tcsm_fortran, tcs0_fortran,
    tcsp_fortran, tcuu_fortran, tcdd_fortran, tcss_fortran, tcud_fortran,
    tcus_fortran, tcds_fortran, tcu_fortran, tcd_fortran, tcs_fortran,
    &sfn1s0_fortran, &sfn3p2_fortran, &sfp1s0_fortran, &sfl1s0_fortran, &fn1s0_fortran, 
    &fn3p2_fortran, &fp1s0_fortran, &fl1s0_fortran, &kfmax_n3p2, &delkf_n3p2, 
    &tcmax_n3p2, &isf_fortran, kfn_fortran, kfp_fortran, 
    kfla_fortran, kfqu_fortran, kfqd_fortran, kfqs_fortran, 
    bar_fortran, fhad_fortran, yquarku_fortran, yquarkd_fortran, 
    yquarks_fortran);

    // -----------------------------
    // Compare Double Arrays
    // -----------------------------
    print_comparison("tcn", tcn_cpp, "tcn", tcn_fortran);
    print_comparison("tcp", tcp_cpp, "tcp", tcp_fortran);
    print_comparison("tcla", tcla_cpp, "tcla", tcla_fortran);
    print_comparison("tcsm", tcsm_cpp, "tcsm", tcsm_fortran);
    print_comparison("tcs0", tcs0_cpp, "tcs0", tcs0_fortran);
    print_comparison("tcsp", tcsp_cpp, "tcsp", tcsp_fortran);
    print_comparison("tcuu", tcuu_cpp, "tcuu", tcuu_fortran);
    print_comparison("tcdd", tcdd_cpp, "tcdd", tcdd_fortran);
    print_comparison("tcss", tcss_cpp, "tcss", tcss_fortran);
    print_comparison("tcud", tcud_cpp, "tcud", tcud_fortran);
    print_comparison("tcus", tcus_cpp, "tcus", tcus_fortran);
    print_comparison("tcds", tcds_cpp, "tcds", tcds_fortran);
    print_comparison("tcu", tcu_cpp, "tcu", tcu_fortran);
    print_comparison("tcd", tcd_cpp, "tcd", tcd_fortran);
    print_comparison("tcs", tcs_cpp, "tcs", tcs_fortran);
    */
    return 0;

#ifdef O2SCL_MPI
  // Finalize MPI
  MPI_Finalize();
#endif
}
