/*
  -------------------------------------------------------------------
  
  Copyright (C) 2015-2019, Andrew W. Steiner
  
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
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

#include <o2scl/test_mgr.h>

#include "nscool_wrap.h"

#ifdef O2SCL_MPI
#include "mpi.h"
#endif

using namespace std;

/** \brief Class for testing neutron star cooling code
 */
class nscool_wrap_test : public nscool_wrap {

public:

  nscool_wrap_test() : nscool_wrap("./") {
  }
  
  /** \brief Desc
   */
  virtual void test(o2scl::test_mgr &t) {
#ifdef O2SCL_SET_OPENMP
      size_t i_thread=omp_get_thread_num();
#else
      size_t i_thread=0;
#endif
    if (true) {
      std::cout << "Running on thread: " << i_thread << std::endl;
      run(0);
      size_t n=v_time.size();
      if (n>0) {
	double last_time=v_time[n-1];
	double last_tptr=v_tptr[n-1];
	t.test_rel(last_time,1.482423e6,1.0e-2,"tptr 0");
	t.test_rel(last_tptr,9.675450e3,1.0e-2,"time 0");
      } else {
	t.test_rel(0.0,1.482423e6,1.0e-2,"tptr 0");
	t.test_rel(0.0,9.675450e3,1.0e-2,"time 0");
      }

    }
    
    if (true) {
      sfn1s0=1;
      sfn3p2=150;
      sfp1s0=150;
      n3_tc=1.055e9;
      n3_kf=1.88628;
      n3_dk=0.486867;
      p1_tc=4.99188e9;
      p1_kf=1.22489;
      p1_dk=0.53767;

      ptemp=1.0;

      run(0);

      double match_time=6.006e5;
      double match_tptr=5.086674e4;
      size_t n=v_time.size();
      if (n>0) {
	for(int i=0;i<((int)n)-1;i++) {
	  if (v_time[i]<match_time && v_time[i+1]>match_time) {
	    double interp=v_tptr[i]+(match_time-v_time[i])/
	      (v_time[i+1]-v_time[i])*(v_tptr[i+1]-v_tptr[i]);
	    double fact=fabs(interp-match_tptr)/match_tptr;
	    t.test_rel(interp,match_tptr,5.0e-3,"tptr 1");
	    i=n;
	  }
	}
	
	//write_tl_prof("tl_prof.o2");
      }
    }

    if (true) {
      sfn3p2=0;
      sfp1s0=3;

      ptemp=0.0;

      hhj_eos(-0.01);

      run(0);

      size_t n=v_time.size();
      if (n>0) {
	double last_time=v_time[n-1];
	double last_tptr=v_tptr[n-1];
	// These tests are less stringent because small
	// differences in the TOV solver lead to different
	// results
	t.test_rel(last_time,2.582910e6,5.0e-2,"tptr 2");
	t.test_rel(last_tptr,8.356029e3,2.0e-1,"time 2");
      } else {
	t.test_rel(0.0,2.582910e6,5.0e-2,"tptr 2");
	t.test_rel(0.0,8.356029e3,2.0e-1,"time 2");
      }
    }
    
    return;
  }
  
};

int main(int argc, char *argv[]) {

  cout.setf(ios::scientific);

#ifdef O2SCL_MPI
  MPI_Init(&argc, &argv);
#endif

  int n_threads = 1; // Number of threads

  // Parse command-line arguments
  for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]) == "-threads" && i + 1 < argc) {
          n_threads = std::stoi(argv[i + 1]);
          ++i; // Skip the next argument since it's the value
      }
  }

#ifdef O2SCL_SET_OPENMP
  omp_set_num_threads(n_threads);
#else
  n_threads = 1;
#endif

  std::cout << "Thread count: " << n_threads << std::endl;

  // Vector of test objects
  std::vector<nscool_wrap_test> nscool_wrap_tests(n_threads);

  // Resize and initialize nscool_wrap_ptrs with addresses of test objects
  nscool_wrap_ptrs.resize(1);
  for (int k = 0; k < n_threads; ++k) {
    nscool_wrap_ptrs[0] = &nscool_wrap_tests[k];
  }

  o2scl::test_mgr t;
  t.set_output_level(2);

  // Parallel execution of tests
  #pragma omp parallel for
  for (int k = 0; k < n_threads; ++k) {
    nscool_wrap_tests[k].test(t);
  }

  t.report();

#ifdef O2SCL_MPI
  MPI_Finalize();
#endif

  return 0;
}
