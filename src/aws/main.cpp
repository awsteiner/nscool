#include "rs.h"

#include <o2scl/cli_readline.h>

using namespace std;
using namespace o2scl;

int main(int argc, char *argv[]) {
  
  cout.setf(ios::scientific);

#ifndef NO_MPI
  // Init MPI
  MPI_Init(&argc,&argv);
#endif

  // Create a new error handler for each MPI rank
  o2scl::err_hnd_cpp error_handler;
  o2scl::err_hnd=&error_handler;

  mcmc_wrapper mw;

  mw.n_threads=1;
  mw.mct.n_walk=1;
  mw.mct.n_threads=mw.n_threads;

  // Command-line interface object
  cli_readline cl;

  static const int nopt=14;
  o2scl::comm_option_s options[nopt]=
    {{'s',"steady-state","Steady-state heating curve",1,1,"<filename>",
      "",new o2scl::comm_option_mfptr<mcmc_wrapper>
      (&mw,&mcmc_wrapper::steady_state),
      o2scl::cli::comm_option_both},
    {'m',"mcmc","MCMC.",0,0,"","",
     new o2scl::comm_option_mfptr<mcmc_wrapper>(&mw,&mcmc_wrapper::mcmc),
     o2scl::cli::comm_option_both},
    {0,"initial-point-last","Initial point from last point in file.",1,2,
     "<filename>","Long. desc.",
     new o2scl::comm_option_mfptr<mcmc_wrapper>
     (&mw,&mcmc_wrapper::initial_point_last),
     o2scl::cli::comm_option_both},
    {0,"initial-point-first","Initial point from first point in file.",1,1,
     "<filename>","Long. desc.",
     new o2scl::comm_option_mfptr<mcmc_wrapper>
     (&mw,&mcmc_wrapper::initial_point_first),
     o2scl::cli::comm_option_both},
    {'b',"initial-point-best","Initial point from best point in file.",1,3,
     "<filename>","Long. desc.",
     new o2scl::comm_option_mfptr<mcmc_wrapper>
     (&mw,&mcmc_wrapper::initial_point_best),
     o2scl::cli::comm_option_both},
    {0,"initial-point-rand","Initial point from random point in file.",1,1,
     "<filename>","Long. desc.",
     new o2scl::comm_option_mfptr<mcmc_wrapper>
     (&mw,&mcmc_wrapper::initial_point_rand),
     o2scl::cli::comm_option_both},
    {0,"make-gaussian","Make a gaussian from previous results.",2,2,
     "<in file> <out file>","Long. desc.",
     new o2scl::comm_option_mfptr<mcmc_wrapper>
     (&mw,&mcmc_wrapper::make_gaussian),
     o2scl::cli::comm_option_both},
    {0,"make-gaussian2","Make a gaussian from previous results.",2,2,
     "<in file> <out file>","Long. desc.",
     new o2scl::comm_option_mfptr<mcmc_wrapper>
     (&mw,&mcmc_wrapper::make_gaussian2),
     o2scl::cli::comm_option_both},
    {0,"fit-gaussian","Fit results to a Gaussian.",0,0,
     "","Long. desc.",
     new o2scl::comm_option_mfptr<mcmc_wrapper>
     (&mw,&mcmc_wrapper::fit_gaussian),
     o2scl::cli::comm_option_both},
    {0,"fit-gaussian2","Fit results to a Gaussian.",0,0,
     "","Long. desc.",
     new o2scl::comm_option_mfptr<mcmc_wrapper>
     (&mw,&mcmc_wrapper::fit_gaussian2),
     o2scl::cli::comm_option_both},
    {'t',"threads","Set the number of threads.",1,1,
     "<numbers of threads>","Set the number of threads.",
     new o2scl::comm_option_mfptr<mcmc_wrapper>
     (&mw,&mcmc_wrapper::threads),
     o2scl::cli::comm_option_both},
    {'c',"clean","",1,2,"<input file> [output file]",
     "Clean a table, removing duplicate and out-of-bounds points.",
     new o2scl::comm_option_mfptr<mcmc_wrapper>
     (&mw,&mcmc_wrapper::clean),
     o2scl::cli::comm_option_both},
    {0,"emu-aws","emu-aws",
     -1,-1,"<filename>","",
     new o2scl::comm_option_mfptr<mcmc_wrapper>
     (&mw,&mcmc_wrapper::emu_aws),
     o2scl::cli::comm_option_both},
    {0,"check-limits","check-limits",
     -1,-1,"<filename>","",
     new o2scl::comm_option_mfptr<mcmc_wrapper>
     (&mw,&mcmc_wrapper::check_limits),
     o2scl::cli::comm_option_both}
  };
  
  cl.set_comm_option_vec(nopt,options);
  cl.gnu_intro=false;
  
  o2scl::cli::parameter_int p_verbose;
  p_verbose.i=&mw.verbose;
  p_verbose.help="Verbose parameter (default 1)";
  cl.par_list.insert(make_pair("verbose",&p_verbose));

  o2scl::cli::parameter_double p_max_time;
  p_max_time.d=&mw.mct.max_time;
  p_max_time.help="max time";
  cl.par_list.insert(make_pair("max_time",&p_max_time));

  o2scl::cli::parameter_size_t p_max_iters;
  p_max_iters.s=&mw.mct.max_iters;
  p_max_iters.help="max time";
  cl.par_list.insert(make_pair("max_iters",&p_max_iters));

  o2scl::cli::parameter_size_t p_n_walk;
  p_n_walk.s=&mw.mct.n_walk;
  p_n_walk.help="number of walkers";
  cl.par_list.insert(make_pair("n_walk",&p_n_walk));

  o2scl::cli::parameter_double p_var_dec_factor;
  p_var_dec_factor.d=&mw.var_dec_factor;
  p_var_dec_factor.help="max time";
  cl.par_list.insert(make_pair("var_dec_factor",&p_var_dec_factor));

  o2scl::cli::parameter_double p_covar_dec_factor;
  p_covar_dec_factor.d=&mw.covar_dec_factor;
  p_covar_dec_factor.help="max time";
  cl.par_list.insert(make_pair("covar_dec_factor",&p_covar_dec_factor));

  o2scl::cli::parameter_double p_step_fac;
  p_step_fac.d=&mw.mct.step_fac;
  p_step_fac.help="step fac";
  cl.par_list.insert(make_pair("step_fac",&p_step_fac));

  o2scl::cli::parameter_double p_r14_max;
  p_r14_max.d=&mw.r14_max;
  p_r14_max.help="step fac";
  cl.par_list.insert(make_pair("R14_max",&p_r14_max));

  o2scl::cli::parameter_bool p_aff_inv;
  p_aff_inv.b=&mw.mct.aff_inv;
  p_aff_inv.help="affine-invariant sampling";
  cl.par_list.insert(make_pair("aff_inv",&p_aff_inv));

  o2scl::cli::parameter_bool p_meas_for_initial;
  p_meas_for_initial.b=&mw.mct.meas_for_initial;
  p_meas_for_initial.help="Call the measure function for the initial point";
  cl.par_list.insert(make_pair("meas_for_initial",&p_meas_for_initial));

  o2scl::cli::parameter_double p_file_update_time;
  p_file_update_time.d=&mw.mct.file_update_time;
  p_file_update_time.help="file update time";
  cl.par_list.insert(make_pair("file_update_time",&p_file_update_time));

  o2scl::cli::parameter_size_t p_file_update_iters;
  p_file_update_iters.s=&mw.mct.file_update_iters;
  p_file_update_iters.help="file update iters";
  cl.par_list.insert(make_pair("file_update_iters",&p_file_update_iters));

  o2scl::cli::parameter_int p_mcmc_verbose;
  p_mcmc_verbose.i=&mw.mct.verbose;
  p_mcmc_verbose.help="MCMC verbose parameter (default 1)";
  cl.par_list.insert(make_pair("mcmc_verbose",&p_mcmc_verbose));

  o2scl::cli::parameter_string p_prefix;
  p_prefix.str=&mw.mct.prefix;
  p_prefix.help="Output file prefix (default \"newmcmc\")";
  cl.par_list.insert(make_pair("prefix",&p_prefix));

  o2scl::cli::parameter_string p_ptype;
  p_ptype.str=&mw.ptype;
  p_ptype.help="Proposal type (default \"none\")";
  cl.par_list.insert(make_pair("ptype",&p_ptype));

  o2scl::cli::parameter_bool p_w_tl_prof;
  p_w_tl_prof.b=&mw.w_tl_prof;
  p_w_tl_prof.help="Write Temp-Lum profiles";
  cl.par_list.insert(make_pair("w_tl_prof",&p_w_tl_prof));

  o2scl::cli::parameter_bool p_no_nuclei;
  p_no_nuclei.b=&mw.no_nuclei;
  p_no_nuclei.help="no_nuclei";
  cl.par_list.insert(make_pair("no_nuclei",&p_no_nuclei));

  o2scl::cli::parameter_bool p_no_ins;
  p_no_ins.b=&mw.no_ins;
  p_no_ins.help="no_ins";
  cl.par_list.insert(make_pair("no_ins",&p_no_ins));

  o2scl::cli::parameter_bool p_debug_ins;
  p_debug_ins.b=&mw.debug_ins;
  p_debug_ins.help="debug_ins";
  cl.par_list.insert(make_pair("debug_ins",&p_debug_ins));

  o2scl::cli::parameter_bool p_no_sxrt;
  p_no_sxrt.b=&mw.no_sxrt;
  p_no_sxrt.help="no_sxrt";
  cl.par_list.insert(make_pair("no_sxrt",&p_no_sxrt));
  
  cl.run_auto(argc,argv);

#ifndef NO_MPI
  // Finalize MPI
  MPI_Finalize();
#endif
  
  return 0;
}

