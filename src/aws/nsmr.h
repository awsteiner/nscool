#ifndef _NSMR_H_
#define _NSMR_H_

#include <vector>
#include <string>
#include <o2scl/tensor_grid.h>
#include <o2scl/hdf_io.h>
#include <o2scl/misc.h>

#ifndef NO_OPENMP
#include <omp.h>
#endif
#ifndef NO_MPI
#include <mpi.h>
#endif

typedef struct {
  std::string name;
} nsmr;

class nsmr_data {

public:

  std::vector<nsmr> list;

  // The neutron star mass and radius data
  std::vector<o2scl::table3d> MR_data;

  nsmr_data();
  
  void init_data(o2scl::vec_index &pvi,
		 std::vector<std::string> &param_units);

  // Copy constructor nsmr_data sd1(sd2);
  nsmr_data(const nsmr_data &nd) {
    list = nd.list;
  }
  
  // Copy constructor sxrt_data sd1=sd2;
  const nsmr_data &operator=(const nsmr_data &nd) {
    if (this != &nd) {
      list = nd.list;
    }
    return *this;
  }
  
};

#endif
