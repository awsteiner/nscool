// test_cpp_fortran.cpp
#include <iostream>
#include <iomanip>
#include "f_tc.h"

// Declare Fortran subroutines and functions with C linkage
extern "C" {
    // solve_tc
    double tcn1_sfb_(double* k);
    double tcp1_t73_(double* k);
}

int main() {
    try {
        std::cout << std::fixed << std::setprecision(5);

        // -----------------------------
        // Test C++ function: solve_tc
        // -----------------------------
        // Sample data for C++ solve_tc
        const int imax_cpp = 5;
        double k0_cpp[5] = {0.1, 0.2, 0.3, 0.4, 0.5};

        // -----------------------------
        // Test Additional C++ Functions
        // -----------------------------
        // Example: cpp_tc_Ioffe_1p
        double kf = 1.0; // fm^-1
        double tcn1_sfb_c = cpp_tcn1_sfb(kf);
        double tcn1_sfb_f = tcn1_sfb_(&kf);
        kf=0.6;
        double tcp1_t73_c = cpp_tcp1_t73(kf);
        double tcp1_t73_f = tcp1_t73_(&kf);

        // Note: For Fortran counterparts of additional C++ functions,
        // similar steps need to be followed: expose via C bindings,
        // declare in C++ with extern "C", and call accordingly.

        // Display the result
        std::cout << "tcn-SFB Calculation Comparison:" << std::endl;
        std::cout << "c++, fort: " << tcn1_sfb_c << ", " << tcn1_sfb_f << std::endl;
        std::cout << "================================" << std::endl;
        std::cout << "tcp-T73 Calculation Comparison:" << std::endl;
        std::cout << "c++, fort: " << tcp1_t73_c << ", " << tcp1_t73_f << std::endl;

        // Add more function calls and comparisons as needed

    }
    catch (const std::exception& e) {
        std::cerr << "Exception caught in C++ test program: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
