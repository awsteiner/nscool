#include <o2scl/mcmc_para.h>
#include <o2scl/vec_stats.h>
#include <o2scl/test_mgr.h>
#include <o2scl/hdf_io.h>
#include <o2scl/inte_qag_gsl.h>
#include <o2scl/set_openmp.h>
#include <o2scl/kde_python.h>

using namespace std;
using namespace o2scl;
using namespace o2scl_hdf;

/// Convenient typedefs

typedef boost::numeric::ublas::vector<double> ubvector;

typedef boost::numeric::ublas::matrix<double> ubmatrix;

int main(int argc, char *argv[]) {
  
  cout.setf(ios::scientific);

  // Read the preliminary data from a file
  hdf_file hf;
  hf.open("../data/noomma_tot_0_fix_out");
  table_units<> tab_in;
  hdf_input(hf,tab_in,"markov_chain_0");
  hf.close();

  // Remove empty rows from table.
  tab_in.delete_rows_func("mult<0.5");

  // Compute the autocorrelation length
  std::vector<double> ac, ftom;
  o2scl::vector_autocorr_vector_mult(tab_in.get_nlines(),
                                     tab_in["b"],tab_in["mult"],ac);
  size_t ac_len=o2scl::vector_autocorr_tau(ac,ftom);
  
  // Create a separate table of statistically independent samples
  table_units<> indep;
  copy_table_thin_mcmc(ac_len,tab_in,indep,"mult");
  
  cout << "Autocorrelation length, effective sample size: "
       << ac_len << " " << indep.get_nlines() << endl;
  
  // Write these samples to a file
  hf.open_or_create("../data/ex_mcmc_kde.o2");
  hdf_output(hf,tab_in,"mcmc");
  hdf_output(hf,indep,"indep");
  // hf.setd_vec("q_next",local_stepper->vq_next);
  // hf.setd_vec("w_next",local_stepper->vw_next);
  hf.close();

  // Compute the average of the correlated samples for comparison
  double avg2=vector_mean(tab_in.get_nlines(),(tab_in)["b"]);
  cout << "Average of correlated samples: " << avg2 << endl;
  
  // Use the independent samples to compute the final integral and
  // compare to the exact result. Note that we must specify the
  // number of elements in the vector, indep["x2"], because the
  // table_units object often has space at the end to add extra rows.
  
  double avg=vector_mean(indep.get_nlines(),indep["b"]);
  double std=vector_stddev(indep.get_nlines(),indep["b"]);
  cout << "Average and std. dev. of uncorrelated samples: "
       << avg << " " << std << endl;
  // cout << "Absolute difference: " << fabs(avg-exact) << endl;
  cout << "Uncertainty in the average: "
       << std/sqrt(indep.get_nlines()) << endl;

}