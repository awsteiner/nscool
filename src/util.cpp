#include <algorithm>
#include <o2scl/mcmc_para.h>
#include <o2scl/table_units.h>
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

typedef std::function<int(size_t,const ubvector &,double &,
  std::vector<double> &)> point_funct;

typedef std::function<int(const ubvector &,double,std::vector<double> &,
  std::vector<double> &)> fill_funct;
/// The MCMC object
mcmc_para_table<point_funct,fill_funct,std::vector<double>,ubvector> mct;

/** \brief A demonstration class for the MCMC example. This example
    could have been written with global functions, but we put them
    in a class to show how it would work in that case.
*/
class exc {

public:

  /** \brief A one-dimensional bimodal distribution

      Here, the variable 'log_weight' stores the natural logarithm of
      the objective function based on the parameters stored in \c pars.
      The object 'dat' stores any auxillary quantities which can be
      computed at every point in parameter space.
  */
  int bimodal(size_t nv, const ubvector &pars, double &log_weight,
        std::vector<double> &dat) {

    double x=pars[0];
    log_weight=log(exp(-x*x)*(sin(x-1.4)+1.0));
    dat[0]=x*x;
    return 0;
  }

  /** \brief Add auxillary quantities to 'line' so they can be
      stored in the table
  */
  int fill_line(const ubvector &pars, double log_weight, 
    std::vector<double> &line, std::vector<double> &dat) {
    line.push_back(dat[0]);
    return 0;
  }

  exc() {
  }

};

int main(int argc, char *argv[]) {
  
  cout.setf(ios::scientific);
  std::string filename = argv[1];
  std::string param = argv[2];

  exc e;

  // Parameter limits and initial points

  ubvector low_bimodal(1), high_bimodal(1);
  low_bimodal[0]=-5.0;
  high_bimodal[0]=5.0;

  // Function objects for the MCMC object
  point_funct bimodal_func=std::bind
    (std::mem_fn<int(size_t,const ubvector &,double &,
		     std::vector<double> &)>(&exc::bimodal),&e,
     std::placeholders::_1,std::placeholders::_2,std::placeholders::_3,
     std::placeholders::_4);
  fill_funct fill_func=std::bind
    (std::mem_fn<int(const ubvector &,double,std::vector<double> &,
		     std::vector<double> &)>(&exc::fill_line),&e,
     std::placeholders::_1,std::placeholders::_2,std::placeholders::_3,
     std::placeholders::_4);

  // Create function object vectors
  vector<point_funct> bimodal_vec;
  bimodal_vec.push_back(bimodal_func);
  vector<fill_funct> fill_vec;
  fill_vec.push_back(fill_func);
  
  mct.aff_inv=true;
  mct.n_walk=3;

  // Create and allocate data objects
  size_t vdat_size = 2*mct.n_walk*mct.n_threads;
  vector<std::vector<double> > data_vec(vdat_size);
  cout << vdat_size << endl;
  //data_vec[0].resize(mct.n_walk);
  //data_vec[1].resize(mct.n_walk);

  // Set parameter names and units
  vector<string> pnames={"x"};
  vector<string> punits={""};
  vector<string> dnames={"x2"};
  vector<string> dunits={""};
  mct.set_names_units(pnames,punits,dnames,dunits);

  // Set MCMC parameters
  
  mct.def_stepper->step_fac.resize(1);
  mct.def_stepper->step_fac[0]=2.0;
  mct.max_iters=20000;
  mct.prefix="ex_mcmc";
  //mct.table_prealloc=mct.max_iters/3;
  cout << "working till here?" << endl;
  // Perform MCMC
  mct.mcmc_fill(1,low_bimodal,high_bimodal,bimodal_vec,fill_vec,
                data_vec);

  // Output acceptance and rejection rate
  cout << "n_accept, n_reject: " << mct.n_accept[0] << " "
       << mct.n_reject[0] << endl;

  // Get the MCMC results
  shared_ptr<table_units<> > t=mct.get_table();

  if(false){

  // Read the preliminary data from a file
  hdf_file hf;
  hf.open(filename);
  table_units<> tab_in;
  size_t nw;
  hdf_input(hf,tab_in,"markov_chain_0");
  hf.get_szt("n_walk",nw);
  hf.close();

  vector<vector<double>> xall;
  xall.resize(nw);
  cout << "number of walkers: " << nw << endl;
  table_units<> gr_tab;
  gr_tab.set_nlines(tab_in.get_nlines()/nw);
  gr_tab.line_of_names("gelman_rubin_stat");
  
  for(size_t j=0;j<tab_in.get_nlines()/nw;j++) {
    for(size_t i=0;i<nw;i++) {
      xall[i].push_back(tab_in.get(param,j+i));
    }
    if(j>2) {
      double gr=mult_vector_gelman_rubin<vector<double>,double>(xall,0);
      gr_tab.set("gelman_rubin_stat", j-3, gr);
      // std::cout << "Gelman rubin statistics: " << gr << std::endl;
    }
  }

  hf.open_or_create("gr.o2");
  hdf_output(hf, gr_tab, "gr_tab");
  hf.close();

  // Remove empty rows from table.
  tab_in.delete_rows_func("mult<0.5");

  // Compute the autocorrelation length
  std::vector<double> ac, ftom;
  o2scl::vector_autocorr_vector_mult(tab_in.get_nlines(),
                                     tab_in[param],tab_in["mult"],ac);
  size_t ac_len=o2scl::vector_autocorr_tau(ac,ftom);
  
  // Create a separate table of statistically independent samples
  table_units<> indep;
  copy_table_thin_mcmc(ac_len,tab_in,indep,"mult");
  
  cout << "Autocorrelation length, effective sample size: "
       << ac_len << " " << indep.get_nlines() << endl;
  
  // Write these samples to a file
  hf.open_or_create("../data/util.o2");
  hdf_output(hf,tab_in,"mcmc");
  hdf_output(hf,indep,"indep");
  // hf.setd_vec("q_next",local_stepper->vq_next);
  // hf.setd_vec("w_next",local_stepper->vw_next);
  hf.close();

  // Compute the average of the correlated samples for comparison
  double avg2=vector_mean(tab_in.get_nlines(),(tab_in)[param]);
  cout << "Average of correlated samples: " << avg2 << endl;
  
  // Use the independent samples to compute the final integral and
  // compare to the exact result. Note that we must specify the
  // number of elements in the vector, indep["x2"], because the
  // table_units object often has space at the end to add extra rows.
  
  double avg=vector_mean(indep.get_nlines(),indep[param]);
  double std=vector_stddev(indep.get_nlines(),indep[param]);
  cout << "Average and std. dev. of uncorrelated samples: "
       << avg << " " << std << endl;
  // cout << "Absolute difference: " << fabs(avg-exact) << endl;
  cout << "Uncertainty in the average: "
       << std/sqrt(indep.get_nlines()) << endl;
  }

}