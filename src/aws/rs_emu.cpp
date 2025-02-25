#include <o2scl/hdf_eos_io.h>
#include <o2scl/anneal_gsl.h>
#include <o2scl/kde_python.h>
//#include <o2scl/o2scl_linalg.h>
//#include <eigen3/Eigen>

#include "rs.h"

using namespace std;
using namespace o2scl;
using namespace o2scl_hdf;
using namespace o2scl_const;

/** \brief The interpolation estimate objects
 */

emulator_rs::emulator_rs() {
  
  mpi_rank=0;
  mpi_size=1;
  
#ifndef NO_MPI    
  // Get MPI rank, etc.
  MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
#endif
  
  unsigned long int seed=time(0);
  seed*=(mpi_rank+1);
  r.set_seed(seed);
}

void emulator_rs::train(o2scl::vec_index &pvii,
                        o2scl::vec_index &dvii, data_eval *depi) {
    
  dep=depi;
  dvi=dvii;
  pvi=pvii;

  std::vector<std::string> list_in, list_out;
  list.clear();
  for(size_t j=0;j<pvi.size();j++) {
    list.push_back(pvi[j]);
    list_in.push_back(pvi[j]);
  }
  // 0
  list.push_back("log_wgt");
  list.push_back("n0");
  list.push_back("EoA");
  list.push_back("msom");
  list.push_back("S");
  list.push_back("S2");
  list.push_back("K");
  list.push_back("L");
  // 8
  list.push_back("Mns_max");
  list.push_back("cs2_max");
  list.push_back("kfn_min");
  list.push_back("kfn_max");
  list.push_back("kfp_min");
  list.push_back("kfp_max");
  list.push_back("lw_nuc");
  list.push_back("lw_prex");
  list.push_back("lw_qlmxb");
  list.push_back("lw_ligo");
  list.push_back("lw_ins");
  list.push_back("lw_sxrt");
  // 20
  list.push_back("R_55");
  list.push_back("L_1905");

  // 0
  list_out.push_back("log_wgt");
  list_out.push_back("n0");
  list_out.push_back("EoA");
  list_out.push_back("msom");
  list_out.push_back("S");
  list_out.push_back("S2");
  list_out.push_back("K");
  list_out.push_back("L");
  // 8
  list_out.push_back("Mns_max");
  list_out.push_back("cs2_max");
  list_out.push_back("kfn_min");
  list_out.push_back("kfn_max");
  list_out.push_back("kfp_min");
  list_out.push_back("kfp_max");
  list_out.push_back("lw_nuc");
  list_out.push_back("lw_prex");
  list_out.push_back("lw_qlmxb");
  list_out.push_back("lw_ligo");
  list_out.push_back("lw_ins");
  list_out.push_back("lw_sxrt");
  // 20
  list_out.push_back("R_55");
  list_out.push_back("L_1905");
  
  cout << "Training column list (size " << list.size() << "): "; 
  o2scl::vector_out(std::cout,list,true);

  np=pvi.size();
  nout=list.size()-pvi.size();
  cout << "np: " << np << " nout: " << nout << endl;

  if (!jan_emu) {
  
    table.clear();
    for(size_t j=0;j<list.size();j++) {
      table.new_column(list[j]);
    }
  
    for(size_t k=0;k<files.size();k++) {

      size_t pos=files[k].find("<rank>");
      if (pos!=std::string::npos) {
        files[k].replace(pos,6,o2scl::itos(mpi_rank));
      }
    
#ifndef NO_MPI    
      // Ensure that multiple MPI ranks are not writing to the 
      // filesystem at the same time
      int tag=0, buffer=0;
      if (mpi_size>1 && mpi_rank>=1) {
        MPI_Recv(&buffer,1,MPI_INT,mpi_rank-1,
                 tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      }
#endif
    
      std::cout << "Rank " << mpi_rank << " reading training table "
                << "with index " << k << " from file "
                << files[k] << std::endl;
      table_units<> tab_k;
      hdf_file hf;
      hf.open(files[k]);
      hdf_input(hf,tab_k);
      hf.close();
      cout << "Rank " << mpi_rank << ": table has "
           << tab_k.get_nlines() << " lines." << endl;
    
#ifndef NO_MPI
      // Send a message to the next MPI rank
      if (mpi_size>1 && mpi_rank<mpi_size-1) {
        MPI_Send(&buffer,1,MPI_INT,mpi_rank+1,
                 tag,MPI_COMM_WORLD);
      }
#endif

      for(size_t j=0;j<list.size();j++) {
        if (tab_k.is_column(list[j])==false) {
          cout << "Table in file " << files[k] << " does not have "
               << "column " << list[j] << "." << endl;
          exit(-1);
        }
      }

      // Delete any rows with a small log_wgt, which are emulated, or
      // have a value of "mult" which is zero

      if (tab_k.is_column("emulated")) {
        if (tab_k.is_column("mult")) {
          if (jan_emu) {
            tab_k.delete_rows_func(((string)"emulated>0.5 || ")+
                                   "abs(mult)<0.5");
          } else {
            tab_k.delete_rows_func(((string)"emulated>0.5 || ")+
                                   "log_wgt<(-700) || abs(mult)<0.5");
          }
        } else {
          tab_k.delete_rows_func("emulated>0.5 || log_wgt<(-700)");
        }
      } else {
        if (tab_k.is_column("mult")) {
          if (jan_emu) {
            tab_k.delete_rows_func("abs(mult)<0.5");
          } else {
            tab_k.delete_rows_func("log_wgt<(-700) || abs(mult)<0.5");
          }
        } else {
          tab_k.delete_rows_func("log_wgt<(-700)");
        }
      }
      cout << "Rank " << mpi_rank << ": table now has "
           << tab_k.get_nlines() << " lines." << endl;

      // Add this table to the combined table
      for(size_t i=0;i<tab_k.get_nlines();i++) {
        vector<double> line;
        for(size_t j=0;j<list.size();j++) {
          line.push_back(tab_k.get(list[j],i));
        }
        table.line_of_data(line.size(),line);
      }
    
    }

    cout << "Rank " << mpi_rank << ": combined table has "
         << table.get_nlines() << " lines." << endl;
  
    // Go through the combined table and delete nearly equal rows
    double tol_abs=1.0e-12;
    double tol_rel=1.0e-12;

    vector<size_t> row_list;
  
    for(size_t i=0;i<table.get_nlines();i++) {
    
      if (i%500==499) {
        std::cout << "Rank " << mpi_rank
                  << " progress: i+1= " << i+1 << " of "
                  << table.get_nlines() << endl;
      }
      // Check for duplicates
      for(size_t j=i+1;j<table.get_nlines();j++) {
        bool match=true;
        for(size_t k=0;k<np && match==true;k++) {
          if (fabs(table.get(list[k],i))>tol_abs ||
              fabs(table.get(list[k],j))>tol_abs) {
            if (fabs(table.get(list[k],i)-table.get(list[k],j))/
                fabs(table.get(list[k],i)+table.get(list[k],j))>tol_rel) {
              match=false;
            }
          }
        }
        if (match==true) {
          row_list.push_back(j);
          if (false) {
            std::cout << "Match between rows " << i << " and " << j
                      << " " << table.get(list[0],i)
                      << " " << table.get(list[0],j)
                      << " " << table.get(list[1],i)
                      << " " << table.get(list[1],j)
                      << std::endl;
          }
        }
      }
    }
  
    table.delete_rows_list(row_list);
  
    for(size_t i=0;i<table.get_nlines();i+=table.get_nlines()/100) {
      cout << i << " " << table.get(7,i) << " "
           << table.get("log_wgt",i) << endl;
    }
  
    cout << "Rank " << mpi_rank << ": combined table now has "
         << table.get_nlines() << " lines." << endl;
    table.summary(&std::cout);

  } else {

    hdf_file hf;
    hf.open("train_0_out");
    hdf_input(hf,table);
    hf.close();
  }

  const vector<double> &lw_col=table["log_wgt"];
  std::vector<double> lw_col2(table.get_nlines());;
  vector_copy(table.get_nlines(),lw_col,lw_col2);
  
  vector_sort_double(table.get_nlines(),lw_col2);
  lw_10=lw_col2[lw_col2.size()*0.95];
  std::cout << "lw_10: " << lw_col2.size() << " " << lw_10 << std::endl;
  
#ifndef NO_MPI    
  // Ensure that multiple MPI ranks are not writing to the 
  // filesystem at the same time
  int tag=0, buffer=0;
  if (mpi_size>1 && mpi_rank>=1) {
    MPI_Recv(&buffer,1,MPI_INT,mpi_rank-1,
             tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  }
#endif

  if (!jan_emu) {
    std::cout << "Rank " << mpi_rank << " writing final training table."
              << std::endl;
    hdf_file hf2;
    hf2.open_or_create(((string)"train_")+o2scl::itos(mpi_rank)+"_out");
    hdf_output(hf2,table,"train");
    hf2.close();
  }
  
#ifndef NO_MPI
  // Send a message to the next MPI rank
  if (mpi_size>1 && mpi_rank<mpi_size-1) {
    MPI_Send(&buffer,1,MPI_INT,mpi_rank+1,
             tag,MPI_COMM_WORLD);
  }
#endif

  cout << "Going to em1." << endl;
  em1.set(np,nout,0,table,list_in,list_out);
  /*
    if (true) {
    em2.iko.verbose=2;
    em2.iko.timing=true;
    table_units<> table2=table;
    cout << "table lines: " << table2.get_nlines() << endl;
    cout << "Going to em2." << endl;
    em2.set(np,nout,0,table2,list);
    }
  */
  
  if (false) {
    
    std::cout << "Starting python execution." << std::endl;
    if (o2scl_settings.py_initialized==false) {
      std::cout << "Running py_init() at rank: " << mpi_rank << std::endl;
      o2scl_settings.py_init();
      std::cout << "Done with py_init()." << std::endl;
    }
    cout << "Adding path." << endl;
    o2scl_settings.add_python_path("./");
  }
  
  //cout << "Going to em3." << endl;
  //em3.verbose=2;
  //em3.set("emu_sklearn","es","train","point",np,
  //((string)"train_")+o2scl::itos(mpi_rank)+"_out",
  //"log_wgt",list,false);
  
  cout << "Rank " << mpi_rank << " done setting emulators." << endl;
  
  return;
}

int emulator_rs::eval(size_t n, const ubvector &p, double &log_wgt,
                      std::array<double,ndat> &dat) {

  if (false) {
    std::cout << "p: ";
    o2scl::vector_out(std::cout,p,true);
  }
  
  //std::cout << "Here. " << nout << " " << ndat << " "
  //<< p.size() << std::endl;
  
  // Show that we can use the o2scl emulator or the full function
  std::vector<double> x(nout);
  double log_wgt_unc;
  std::vector<double> x_unc(nout);
  int em1_ret=em1.eval_unc(n,p,log_wgt,log_wgt_unc,x,x_unc);

  if (false) {
    std::cout << "x: ";
    o2scl::vector_out(std::cout,x,true);
  }
  std::cout << "em1_ret, log_wgt, unc, Mns_max: " << em1_ret << " "
            << log_wgt << " " << log_wgt_unc << " " << x[8] 
            << std::endl;
  
  /*
    if (true) {
    int em2_ret=em2.eval_unc(n,p,log_wgt,log_wgt_unc,x,x_unc);
    std::cout << "log_wgt, unc, Mns_max: " << em1_ret << " " << log_wgt << " "
    << log_wgt_unc << " " << x[8] << std::endl;
    exit(-1);
    }
  */
  
  if (!std::isfinite(log_wgt)) {
    std::cout << "x: ";
    o2scl::vector_out(std::cout,x,true);
    exit(-1);
  }

  // Clear the dat array
  for(size_t i=0;i<ndat;i++) {
    dat[i]=0.0;
  }
    
  // Translate the emulated data into the 'dat' array used
  // by the data_eval point function. Skip k=0 because
  // it's already stored in log_wgt
  if (jan_emu==false) {
    for(size_t k=1;k<nout;k++) {
      if (false) {
        std::cout << "Mapping: " << k << " " << list[np+k] << std::endl;
      }
      dat[dvi[list[np+k]]]=x[k];
    }
  }

  dat[dvi["emulated"]]=1.0;
  if (jan_emu) {
    dat[dvi["log_wgt_unc"]]=0.0;
  } else {
    dat[dvi["log_wgt_unc"]]=log_wgt_unc;
  }

  if (false) {
    std::cout << "p: ";
    o2scl::vector_out(std::cout,p,true);
  }

  /*
    list.push_back("log_wgt");
    list.push_back("n0");
    list.push_back("EoA");
    list.push_back("msom");
    list.push_back("S");
    list.push_back("S2");
    list.push_back("K");
    list.push_back("L");
    // 8
    list.push_back("Mns_max");
    list.push_back("cs2_max");
    list.push_back("kfn_min");
    list.push_back("kfn_max");
    list.push_back("kfp_min");
    list.push_back("kfp_max");
    list.push_back("lw_nuc");
    list.push_back("lw_prex");
    list.push_back("lw_qlmxb");
    list.push_back("lw_ligo");
    list.push_back("lw_ins");
    list.push_back("lw_sxrt");
    // 20
    list.push_back("R_55");
    list.push_back("L_1905");
  */

  double n0=x[1];
  double EoA=x[2];
  double msom=x[3];
  double S=x[4];
  double S2=x[5];
  double K=x[6];
  double L=x[7];
  double cs2_max=x[9];
  double Mns_max=x[8];
  double kfn_min=x[10];
  double kfn_max=x[11];
  double kfp_min=x[12];
  double kfp_max=x[13];
  double R_55=x[20];
  double L_1905=x[21];

  double n0_unc=x_unc[1];
  double EoA_unc=x_unc[2];
  double S_unc=x_unc[4];
  double S2_unc=x_unc[5];
  double K_unc=x_unc[6];
  double L_unc=x_unc[7];
  double cs2_max_unc=x_unc[9];
  double Mns_max_unc=x_unc[8];
  double kfn_min_unc=x_unc[10];
  double kfn_max_unc=x_unc[11];
  double kfp_min_unc=x_unc[12];
  double kfp_max_unc=x_unc[13];
  double R_55_unc=x_unc[20];
  double L_1905_unc=x_unc[21];
  
  //cout << "L_1905, L_1905_unc: " << L_1905 << " " << L_1905_unc << endl;
  //exit(-1);
  /*  if (jan_emu && em1.ii.extrap[0]>2.0) {
    cout << "Emulator extrap too large" << endl;
    cout << em1.ii.extrap[0] << endl;
    //exit(-1);
    log_wgt=-788.0;
    return 0;
  }
  */
  if (Mns_max+Mns_max_unc<2.0) {
    cout << "Emulator predicted small maximum mass." << endl;
    cout << Mns_max << " " << Mns_max_unc << endl;
    //exit(-1);
    log_wgt=-799.0;
    return 0;
  }
  if (n0+n0_unc<0.14 || n0-n0_unc>0.17) {
    cout << "Emulator predicted unphysical saturation density." << endl;
    cout << n0 << " " << n0_unc << endl;
    //exit(-1);
    log_wgt=-798.0;
    return 0;
  }
  if (EoA+EoA_unc<-17.0 || EoA-EoA_unc>-15.0) {
    cout << "Emulator predicted unphysical binding energy." << endl;
    cout << EoA << " " << EoA_unc << endl;
    //exit(-1);
    log_wgt=-797.0;
    return 0;
  }
  if (S2+S2_unc<20.0 || S2-S2_unc>45.0) {
    cout << "Emulator predicted unphysical symmetry energy." << endl;
    cout << S2 << " " << S2_unc << endl;
    //exit(-1);
    log_wgt=-796.0;
    return 0;
  }
  if (S-S_unc>36.0 || L-L_unc>120.0) {
    cout << "Emulator predicted unphysical S or L." << endl;
    cout << S << " " << S_unc << endl;
    cout << L << " " << L_unc << endl;
    //exit(-1);
    log_wgt=-795.0;
    return 0;
  }
  if (K+K_unc<220.0 || K-K_unc>260.0) {
    cout << "EMulator predicted unphysical compressibility." << endl;
    cout << K << " " << K_unc << endl;
    //exit(-1);
    log_wgt=-794.0;
    return 0;
  }
  if (p[pvi["k_maxn"]]<kfn_min+kfn_min_unc ||
      p[pvi["k_maxn"]]>kfn_max+kfn_max_unc) {
    cout << "Emulator predicted kfn out of range." << endl;
    cout << kfn_min << " " << kfn_min_unc << endl;
    cout << kfn_max << " " << kfn_max_unc << endl;
    cout << p[pvi["k_maxn"]] << endl;
    //exit(-1);
    log_wgt=-793.0;
    return 0;
  }
  if (p[pvi["k_maxp"]]<kfp_min+kfp_min_unc ||
      p[pvi["k_maxp"]]>kfp_max+kfp_max_unc) {
    cout << "Emulator predicted kfp out of range." << endl;
    cout << kfp_min << " " << kfp_min_unc << endl;
    cout << kfp_max << " " << kfp_max_unc << endl;
    cout << p[pvi["k_maxp"]] << endl;
    //exit(-1);
    log_wgt=-792.0;
    return 0;
  }
  if (false && cs2_max-cs2_max_unc>1.0) {
    cout << "Emulator predicted unphysical speed of sound." << endl;
    cout << cs2_max << " " << cs2_max_unc << endl;
    //exit(-1);
    log_wgt=-791.0;
    return 0;
  }
  if (R_55-R_55_unc>r14_max) {
    cout << "Emulator predicted R14 beyond limit." << endl;
    cout << R_55 << " " << R_55_unc << " " << r14_max << endl;
    //exit(-1);
    log_wgt=-790.0;
    return 0;
  }
  if (L_1905-L_1905_unc>2.0e31) {
    cout << "Emulator predicted unphysical 1905 luminosity." << endl;
    cout << L_1905 << " " << L_1905_unc << endl;
    //exit(-1);
    log_wgt=-789.0;
    return 0;
  }

  if (jan_emu==false) {
    double xrand=r.random();
    if (log_wgt_unc>1.0e-2 && 
        (log_wgt+2*log_wgt_unc>lw_10 || x[8]>2.15 || xrand<0.02)) {
      if (log_wgt+2*log_wgt_unc>lw_10) {
        std::cout << "High log_wgt." << std::endl;
      }
      if (xrand<0.02) {
        std::cout << "Random." << std::endl;
      }
      if (x[8]>2.15) {
        std::cout << "High M_max." << std::endl;
      }
      
      double log_wgt_old=log_wgt;
      
      int iret=dep->point(n,p,log_wgt,dat);
      
      dat[dvi["log_wgt_unc"]]=fabs(log_wgt_old-log_wgt);
      
      std::cout << "log_wgt_old, log_wgt: " << log_wgt_old << " "
                << log_wgt << std::endl;
      
      if (false) {
        std::cout << "Character:" << std::endl;
        char ch;
        std::cin >> ch;
      }
      
      return iret;
    }
  }
    
  return 0;
}

int emulator_rs::eval_unc(size_t n, const ubvector &p, double &log_wgt,
                          double &lw_unc, 
                          std::array<double,ndat> &dat,
                          std::array<double,ndat> &dat_unc) {
  return eval(n,p,log_wgt,dat);
}

int mcmc_wrapper::emu_aws(std::vector<std::string> &sv, bool itive_com) {
  if (sv.size()<2) {
    cout << "No filename in emu-aws." << endl;
    exit(-1);
  }

  flag_emu_aws=true;
  ers.files.clear();
  for(size_t k=1;k<sv.size();k++) {
    ers.files.push_back(sv[k]);
  }

  return 0;
}

mcmc_wrapper::mcmc_wrapper() {
  covar_dec_factor=100;
  var_dec_factor=1.0;
  step_fac=10.0;
  r14_max=20.0;
  mct.verbose=1;
    
  mpi_rank=0;
  mpi_size=1;

#ifndef NO_MPI    
  // Get MPI rank, etc.
  MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
#endif

  // Filename prefix
  mct.prefix="newmcmc";

  // Set start time
#ifndef NO_MPI
  mct.mpi_start_time=MPI_Wtime();
#else
  mct.mpi_start_time=time(0);
#endif
    
  // This is a relatively clear way to set parameters and units
  // because it is obvious that each parameter is matched to a unit.
  // This should be the authoritative list and other objects should
  // derive from it.
  std::vector<std::string> temp_names_units=
    {"log10_Tcn","K","k_maxn","1/fm",
     "del_kn","1/fm","log10_Tcp","K",
     "k_maxp","1/fm","del_kp","1/fm",
     "b","",
     "c","","c_r","fm",
     "m_s","1/fm","zeta","",
     "c_s","fm","c_w","fm",
     "b1","","xi","",
     "a1","1/fm","a2","",
     "a3","fm","a4","fm^2",
     "a5","fm^3","a6","fm^4",
     "b2","fm^2","b3","fm^4",
     "alpha","1/fm","Q","MeV",
     "M_chirp","Msun","q","","Lambdat",""};
  
  for(size_t i=0;i<temp_names_units.size()/2;i++) {
    pvi.append(temp_names_units[i*2]);
    param_units.push_back(temp_names_units[i*2+1]);
  }

  id.init_data(pvi,param_units);
  nd.init_data(pvi,param_units);
  sd.init_data(pvi,param_units);
    
  nparam=pvi.size();

  de_size=0;
    
  low.resize(nparam);
  high.resize(nparam);

  low[pvi["log10_Tcn"]]=7.0;
  high[pvi["log10_Tcn"]]=10.0;

  low[pvi["k_maxn"]]=1.0;
  high[pvi["k_maxn"]]=2.5;

  /// AWS: changed lower limit from 1 to 0 on 11/21/2020
  low[pvi["del_kn"]]=0.0;
  high[pvi["del_kn"]]=2.5;

  low[pvi["log10_Tcp"]]=7.0;
  high[pvi["log10_Tcp"]]=10.0;

  low[pvi["k_maxp"]]=0.2;
  high[pvi["k_maxp"]]=1.8;

  low[pvi["del_kp"]]=0.0;
  high[pvi["del_kp"]]=2.5;

  // rmfp entries

  low[pvi["b"]]=-0.1;
  high[pvi["b"]]=0.1;
    
  low[pvi["c"]]=-0.1;
  high[pvi["c"]]=0.1;

  low[pvi["c_r"]]=1.0;
  high[pvi["c_r"]]=5.0;
  
  low[pvi["m_s"]]=1.0;
  high[pvi["m_s"]]=5.0;

  low[pvi["zeta"]]=-0.1;
  high[pvi["zeta"]]=0.1;
  
  low[pvi["c_s"]]=1.0;
  high[pvi["c_s"]]=5.0;
 
  low[pvi["c_w"]]=1.0;
  high[pvi["c_w"]]=5.0;
  
  low[pvi["b1"]]=-10.0;
  high[pvi["b1"]]=10.0;

  // Used values from page 10 of Steiner05 "Isospin Asymmetry in
  // Nuclei" Table for initial values
 
  low[pvi["xi"]]=-3.0351*10.0;
  high[pvi["xi"]]=3.0351*10.0;
    
  low[pvi["a1"]]=-35.470/hc_mev_fm*10.0;
  high[pvi["a1"]]=35.470/hc_mev_fm*10.0;
    
  low[pvi["a2"]]=-0.73588*10.0;
  high[pvi["a2"]]=0.73588*10.0;
    
  low[pvi["a3"]]=-7.5995e-03*hc_mev_fm;
  high[pvi["a3"]]=7.5995e-03*hc_mev_fm;
    
  low[pvi["a4"]]=-1.0686e-03*pow(hc_mev_fm,2.0);
  high[pvi["a4"]]=1.0686e-03*pow(hc_mev_fm,2.0);
    
  low[pvi["a5"]]=-6.7880e-06*pow(hc_mev_fm,3.0);
  high[pvi["a5"]]=6.7880e-06*pow(hc_mev_fm,3.0);
    
  low[pvi["a6"]]=-6.8908e-07*pow(hc_mev_fm,4.0);
  high[pvi["a6"]]=6.8908e-07*pow(hc_mev_fm,4.0);
    
  low[pvi["b2"]]=-10.0;
  high[pvi["b2"]]=1.0;
    
  low[pvi["b3"]]=-9.8907e-10*pow(hc_mev_fm,4.0)*1.0e3;
  high[pvi["b3"]]=9.8907e-10*pow(hc_mev_fm,4.0)*1.0e3;

  low[pvi["alpha"]]=0.01;
  high[pvi["alpha"]]=0.75;
    
  low[pvi["Q"]]=1.0;
  high[pvi["Q"]]=2.0;

  low[pvi["M_chirp"]]=1.197;
  high[pvi["M_chirp"]]=1.1993;

  low[pvi["q"]]=0.5;
  high[pvi["q"]]=1.0;

  low[pvi["Lambdat"]]=0.0;
  high[pvi["Lambdat"]]=2000.0;

  // INS parameters
  for (size_t i=0; i<id.list.size();i++) {
    cooling_ns &c=id.list[i];
    low[pvi[(string("mf_"))+c.name]]=0.0;
    high[pvi[(string("mf_"))+c.name]]=1.0;
    if (c.eta) {
      low[pvi[(string("eta_"))+c.name]]=-17.0;
      high[pvi[(string("eta_"))+c.name]]=-7.0;
    }
    // If it's an age upper limit star, then the age is a parameter
    // and we allow it to vary from 100 years to the upper limit
    if (c.tk==0.0 && c.tk_lo==0.0) {
      low[pvi[((string)"t_")+c.name]]=100.0;
      high[pvi[((string)"t_")+c.name]]=c.tc;
    }
    //if(c.L==0.0 && c.L_lo==0.0){
    // These luminosities are in units of 1.0e33 erg/s
    //  low[pvi[((string)"L_")+c.name]]=0.0;
    //  high[pvi[((string)"L_")+c.name]]=c.L_hi;
    //}
  }

  // NSMR parameters
  for (size_t i=0; i<nd.list.size(); i++){
    nsmr &c=nd.list[i];
    low[pvi[(string("mf_"))+c.name]]=0.0;
    high[pvi[(string("mf_"))+c.name]]=1.0;
    low[pvi[(string("eta_"))+c.name]]=0.0;
    high[pvi[(string("eta_"))+c.name]]=1.0;
  }
    
  // SXRT parameters
  for (size_t i=0; i<sd.list.size(); i++){
    sxrt &c=sd.list[i];
    low[pvi[(string("mf_"))+c.name]]=0.0;
    high[pvi[(string("mf_"))+c.name]]=1.0;
    low[pvi[(string("eta_"))+c.name]]=-17.0;
    high[pvi[(string("eta_"))+c.name]]=-7.0;
    if (c.mdot_ul) {
      low[pvi[(string("log10_mdot_"))+c.name]]=-13.0;
      high[pvi[(string("log10_mdot_"))+c.name]]=log10(c.mdot);
    }
    //if (c.L_ul) {
    //low[pvi[(string("L_"))+c.name]]=0.0;
    //high[pvi[(string("L_"))+c.name]]=c.L;
    //}
  }
    
  ptype="none";
  verbose=1;

  no_nuclei=false;
  no_sxrt=false;
  no_ins=false;
  debug_ins=false;
  flag_emu_aws=false;
  
  // End of mcmc_wrapper() constructor
}

int mcmc_wrapper::steady_state(std::vector<std::string> &sv, bool itive_com) {

  if (sv.size()<2) {
    cerr << "No output file specified." << endl;
    exit(-1);
  }
  std::string out_file=sv[1];

  n_threads=1;
  if (de_size>0) delete de;
  de=new data_eval[n_threads];

  nscool_wrap_ptrs.resize(n_threads);
  for(size_t k=0;k<((size_t)n_threads);k++) {
    nscool_wrap_ptrs[k]=&de[k].nw;
  }
    
  table_units<> t;
  t.clear_table();
  t.line_of_names("Mdot T_core L_phot_inf L_neut_inf L_heat_inf");
  t.set_unit("Mdot","Msun/yr");
  t.set_unit("T_core","K");
  t.set_unit("L_phot_inf","erg/s");
  t.set_unit("L_neut_inf","erg/s");
  t.set_unit("L_heat_inf","erg/s");

  // Put in fiducial values for now
  double Q_heat=1.45;
  double mass=1.4;
  double logT=8.0;
  double rad=11.0;

  cout << "Small Mdot values: " << endl;
  cout << "M_dot        T_core       L_phot       L_neut"
       << "       L_heat" << endl;
  for(double Mdot=6.0e-12;Mdot<1.0e-8;Mdot*=1.2) {
    de[0].ss_acc_solve(Mdot,Q_heat,mass,rad,logT);
    double lphot=0.0, lneut=0.0, lheat=0.0;
    int iret=de[0].acc_compute(logT,Q_heat,mass,rad,Mdot,
			       lphot,lneut,lheat);
    if (iret!=0) {
      cout << "Function acc_compute() failed." << endl;
    }
    double line[5]={Mdot,pow(10.0,logT),lphot,lneut,lheat};
    cout << line[0] << " " << line[1] << " "
	 << line[2] << " " << line[3] << " " << line[4] << endl;
    t.line_of_data(5,line);
  }

  logT=8.1;
    
  cout << "Large Mdot values: " << endl;
  cout << "M_dot        T_core       L_phot       L_neut"
       << "       L_heat" << endl;
  for(double Mdot=6.0e-12/1.2;Mdot>3.0e-15;Mdot/=1.2) {
    de[0].ss_acc_solve(Mdot,Q_heat,mass,rad,logT);
    double lphot=0.0, lneut=0.0, lheat=0.0;
    int iret=de[0].acc_compute(logT,Q_heat,mass,rad,Mdot,
			       lphot,lneut,lheat);
    if (iret!=0) {
      cout << "Function acc_compute() failed." << endl;
    }
    double line[5]={Mdot,pow(10.0,logT),lphot,lneut,lheat};
    cout << line[0] << " " << line[1] << " "
	 << line[2] << " " << line[3] << " " << line[4] << endl;
    t.line_of_data(5,line);
  }

  t.sort_table("Mdot");
  t.add_constant("mass",mass);
    
  o2scl_hdf::hdf_file hf;
  hf.open_or_create(out_file);
  hdf_output(hf,t,"sxrt");
  hf.close();

  return 0;
}

int mcmc_wrapper::clean(std::vector<std::string> &sv, bool itive_com) {

  table_units<> t;
  o2scl_hdf::hdf_file hf;
  hf.open(sv[1]);
  std::string name;
  hdf_input(hf,t,name);
  hf.close();
  std::cout << "Read " << t.get_nlines() << " lines." << std::endl;

  vector<size_t> list;

  double tol_abs=1.0e-12;
  double tol_rel=1.0e-12;
    
  for(size_t i=0;i<t.get_nlines();i++) {
    if (i%500==499) {
      std::cout << "Progress: i+1= " << i+1 << " of "
		<< t.get_nlines() << endl;
    }
    // Check if out of bounds
    bool out_of_bounds=false;
    for(size_t j=0;j<nparam && out_of_bounds==false;j++) {
      if (t.get(j+5,i)<low[j]) {
	cout << "Parameter " << j << " too low in row " << i << endl;
	cout << low[j] << " " << t.get(j+5,i) << " "
	     << high[j] << endl;
	out_of_bounds=true;
	list.push_back(i);
      }
      if (out_of_bounds==false && t.get(j+5,i)>high[j]) {
	cout << "Parameter " << j << " too high in row " << i << endl;
	cout << low[j] << " " << t.get(j+5,i) << " "
	     << high[j] << endl;
	out_of_bounds=true;
	list.push_back(i);
      }
    }
    if (out_of_bounds==false && fabs(t.get("mult",i))<0.1) {
      // Check if row has no data
      std::cout << "Row " << i << " has no data." << std::endl;
      list.push_back(i);
    } else if (out_of_bounds==false) {
      // Check for duplicates
      for(size_t j=i+1;j<t.get_nlines();j++) {
	bool match=true;
	if (i<t.get_nlines() && j<t.get_nlines() && j>i) {
	  for(size_t k=5;k<nparam+5 && match==true;k++) {
	    if (fabs(t.get(k,i))>tol_abs ||
		fabs(t.get(k,j))>tol_abs) {
	      if (fabs(t.get(k,i)-t.get(k,j))/
		  fabs(t.get(k,i)+t.get(k,j))>tol_rel) {
		match=false;
	      }
	    }
	  }
	}
	if (match==true) {
	  list.push_back(j);
	  std::cout << "Match between rows " << i << " and " << j
		    << std::endl;
	  for(size_t k=0;k<5;k++) {
	    std::cout << t.get_column_name(k+4) << " ";
	  }
	  std::cout << endl;
	  for(size_t k=0;k<5;k++) {
	    std::cout << t.get(k+4,i) << " ";
	  }
	  std::cout << endl;
	  for(size_t k=0;k<5;k++) {
	    std::cout << t.get(k+4,j) << " ";
	  }
	  std::cout << endl;
	}
      }
    }
  }
    
  if (list.size()>0) {
    t.delete_rows_list(list);
  } else if (verbose>0) {
    std::cout << "No rows to delete." << std::endl;
  }

  if (sv.size()>2) {
    hf.open_or_create(sv[2]);
    hdf_output(hf,t,"mcmc");
    hf.close();
    std::cout << "Wrote " << t.get_nlines() << " lines to file "
	      << sv[2] << std::endl;
  }
    
  return 0;
}

int mcmc_wrapper::mcmc(std::vector<std::string> &sv, bool itive_com) {

  // Instance of class for each OpenMP thread. The
  // init() functions are called later.
  if (((int)de_size)<n_threads) {
    cout << "Creating de object for " << n_threads
	 << " threads in mcmc()." << endl;
    if (de_size>0) delete de;
    de=new data_eval[n_threads];
  }

  // Parameter limits and initial point of MCMC

  // Set global pointers for nscool_wrap. Note that the data_eval::init()
  // functions are called later below.
  nscool_wrap_ptrs.resize(n_threads);
  for(size_t k=0;k<((size_t)n_threads);k++) {
    nscool_wrap_ptrs[k]=&de[k].nw;
  }

  // Parameter limits
  ubvector init(nparam), step(nparam);

  init[pvi["log10_Tcn"]]=9.0;
  step[pvi["log10_Tcn"]]=1.0;

  init[pvi["k_maxn"]]=1.98;
  step[pvi["k_maxn"]]=0.2;

  init[pvi["del_kn"]]=2.03;
  step[pvi["del_kn"]]=0.2;

  init[pvi["log10_Tcp"]]=9.0;
  step[pvi["log10_Tcp"]]=1.0;

  init[pvi["k_maxp"]]=0.929;
  step[pvi["k_maxp"]]=0.2;

  init[pvi["del_kp"]]=0.114;
  step[pvi["del_kp"]]=0.2;

  double fact=1.0;
    
  // rmfp entries

  init[pvi["b"]]=1.800213e-03;
  step[pvi["b"]]=2.6e-4/fact;

  init[pvi["c"]]=4.933333e-05;
  step[pvi["c"]]=4.7e-4/fact;

  init[pvi["c_r"]]=3.514640e+00;
  step[pvi["c_r"]]=1.7e-2/fact;
  
  init[pvi["m_s"]]=2.490790;
  step[pvi["m_s"]]=0.02/fact;

  init[pvi["zeta"]]=3.000000e-02;
  step[pvi["zeta"]]=7.6e-4/fact;
  
  init[pvi["c_s"]]=4.003264e+00;
  step[pvi["c_s"]]=1.6e-2/fact;
 
  init[pvi["c_w"]]=3.286363e+00;
  step[pvi["c_w"]]=1.0e-2/fact;
  
  init[pvi["b1"]]=7.812405e+00;
  step[pvi["b1"]]=7.0e-2/fact;

  // Remaining RMF parameters

  init[pvi["xi"]]=0.0;
  step[pvi["xi"]]=fabs(high[pvi["xi"]])/100.0;

  init[pvi["a1"]]=0.0;
  step[pvi["a1"]]=fabs(high[pvi["a1"]])/100.0;

  init[pvi["a2"]]=0.0;
  step[pvi["a2"]]=fabs(high[pvi["a2"]])/100.0;

  init[pvi["a3"]]=0.0;
  step[pvi["a3"]]=fabs(high[pvi["a3"]])/100.0;

  init[pvi["a4"]]=0.0;
  step[pvi["a4"]]=fabs(high[pvi["a4"]])/100.0;

  init[pvi["a5"]]=0.0;
  step[pvi["a5"]]=fabs(high[pvi["a5"]])/100.0;

  init[pvi["a6"]]=0.0;
  step[pvi["a6"]]=fabs(high[pvi["a6"]])/100.0;

  init[pvi["b2"]]=0.0;
  step[pvi["b2"]]=fabs(high[pvi["b2"]])/100.0;

  init[pvi["b3"]]=0.0;
  step[pvi["b3"]]=fabs(high[pvi["b3"]])/100.0;

  init[pvi["alpha"]]=0.02;
  step[pvi["alpha"]]=0.2;

  init[pvi["Q"]]=1.0;
  step[pvi["Q"]]=0.2;
    
  init[pvi["M_chirp"]]=1.1975;
  step[pvi["M_chirp"]]=0.0001;

  init[pvi["q"]]=0.9;
  step[pvi["q"]]=0.1;

  init[pvi["Lambdat"]]=1000.0;
  step[pvi["Lambdat"]]=20.0;

  // INS parameters
  for (size_t i=0; i<id.list.size(); i++){
    cooling_ns &c=id.list[i];
    init[pvi[(string("mf_"))+c.name]]=0.4;
    step[pvi[(string("mf_"))+c.name]]=0.1;
    if (c.eta){
      init[pvi[(string("eta_"))+c.name]]=-8.0;
      step[pvi[(string("eta_"))+c.name]]=3.0;
    }
    if(c.tk==0.0 && c.tk_lo==0.0){
      init[pvi[((string)"t_") + c.name]]=c.tc/2.0;
      step[pvi[((string)"t_") + c.name]]=c.tc/20.0;
    }
    //if(c.L==0.0 && c.L_lo==0.0){
    //init[pvi[((string)"L_") + c.name]]=c.L_hi/2.0;
    //step[pvi[((string)"L_") + c.name]]=c.L_hi/20.0;
    //}
  }

  // NSMR parameters
  for(size_t i=0;i<nd.list.size();i++) {
    nsmr &c=nd.list[i];
    init[pvi[((string)"mf_")+c.name]]=0.4;
    step[pvi[((string)"mf_")+c.name]]=0.1;
    init[pvi[((string)"eta_")+c.name]]=0.1;
    step[pvi[((string)"eta_")+c.name]]=0.5;
  }

  // SXRT parameters
  for (size_t i=0; i<sd.list.size(); i++) {
    sxrt &c=sd.list[i];
    init[pvi[(string("mf_"))+c.name]]=0.4;
    step[pvi[(string("mf_"))+c.name]]=0.1;
    init[pvi[(string("eta_"))+c.name]]=-10.0;
    step[pvi[(string("eta_"))+c.name]]=3.0;
    if (c.mdot_ul) {
      init[pvi[(string("log10_mdot_"))+c.name]]=log10(c.mdot/2.0);
      step[pvi[(string("log10_mdot_"))+c.name]]=
	(high[pvi[(string("log10_mdot_"))+c.name]]-
	 low[pvi[(string("log10_mdot_"))+c.name]])/20.0;
    }
    //if (c.L_ul) {
    //init[pvi[(string("L_"))+c.name]]=c.L/2.0;
    //step[pvi[(string("L_"))+c.name]]=c.L/20000;
    //}
  }

  // If initial points not already specified, use the 'init'
  // array to specify initial points
    
  if (mct.initial_points.size()==0) {
    for(size_t k=0;k<((size_t)n_threads);k++) {
      mct.initial_points.push_back(init);
    }
  }
    
  if (false) {
    // Reset initial points to that from IUFSU model
    eos_had_rmf rmf;
    o2scl_hdf::rmf_load(rmf,"IUFSU");
    mct.initial_points[0][pvi["b"]]=rmf.b;
    mct.initial_points[0][pvi["c"]]=rmf.c;
    mct.initial_points[0][pvi["c_r"]]=rmf.cr;
    mct.initial_points[0][pvi["m_s"]]=rmf.ms;
    mct.initial_points[0][pvi["zeta"]]=rmf.zeta;
    mct.initial_points[0][pvi["c_s"]]=rmf.cs;
    mct.initial_points[0][pvi["c_w"]]=rmf.cw;
    mct.initial_points[0][pvi["b1"]]=rmf.b1;
    mct.initial_points[0][pvi["xi"]]=rmf.xi;
    mct.initial_points[0][pvi["a1"]]=rmf.a1;
    mct.initial_points[0][pvi["a2"]]=rmf.a2;
    mct.initial_points[0][pvi["a3"]]=rmf.a3;
    mct.initial_points[0][pvi["a4"]]=rmf.a4;
    mct.initial_points[0][pvi["a5"]]=rmf.a5;
    mct.initial_points[0][pvi["a6"]]=rmf.a6;
    mct.initial_points[0][pvi["b2"]]=rmf.b2;
    mct.initial_points[0][pvi["b3"]]=rmf.b3;
  }  

  // Function objects for the MCMC class and for each OpenMP thread
  vector<point_funct> vpf(n_threads);
  vector<fill_funct> vff(n_threads);
  for(size_t k=0;k<((size_t)n_threads);k++) {
    if (flag_emu_aws) {
      vpf[k]=std::bind
        (std::mem_fn<int(size_t,const ubvector &,double &,
                         std::array<double,ndat> &)>(&emulator_rs::eval),
         &ers,std::placeholders::_1,std::placeholders::_2,
         std::placeholders::_3,std::placeholders::_4);
    } else {
      vpf[k]=std::bind
        (std::mem_fn<int(size_t,const ubvector &,double &,
                         std::array<double,ndat> &)>(&data_eval::point),
         &de[k],std::placeholders::_1,std::placeholders::_2,
         std::placeholders::_3,std::placeholders::_4);
    }
    vff[k]=std::bind
      (std::mem_fn<int(const ubvector &,double,std::vector<double> &,
		       std::array<double,ndat> &)>(&data_eval::fill_line),
       &de[k],std::placeholders::_1,std::placeholders::_2,
       std::placeholders::_3,std::placeholders::_4);
  }

  // This code constructs the parameter and output names and units
  // for use by the MCMC class, pnames and punits. This should be
  // the authoritative list of output names and units, but the
  // parameter names and units are initialized separately in the
  // constructor for this class.

  vector<string> pnames, punits;

  // Start with parameter names and units
  for(size_t i=0;i<nparam;i++) {
    pnames.push_back(pvi[i]);
    punits.push_back(param_units[i]);
  }

  if (!jan_emu) {
    
    vector<string> temp_names_units2=
      {"etot_Pb208","MeV","rcharge_Pb208","fm",
       "rnrp_Pb208","fm","etot_Ca40","MeV",
       "rcharge_Ca40","fm","rnrp_Ca40","fm",
       "etot_Zr90","MeV","rcharge_Zr90","fm",
       "rnrp_Zr90","fm","n0","1/fm^3",
       "EoA","MeV","msom","",
       "S","MeV","K","MeV",
       "L","Mev","S2","MeV","eta","","Mns_max","Msun",
       "Rns_max","km","nb_max","1/fm^3",
       "e_max","1/fm^4","cs2_max","",
       "ins_fail","","sxrt_fail","","tov_fail","",
       "nuc_fail","","sat_fail","","eos_fail","","ligo_fail","",
       "nsmr_fail","","prob_fail","","other_fail","",
       "kfn_min","1/fm","kfn_max","1/fm","kfp_min","1/fm","kfp_max","1/fm",
       "lw_nuc","","lw_prex","",
       "lw_qlmxb","","lw_ligo","","lw_ins","","lw_sxrt","",
       "emulated","","log_wgt_unc",""};
  
    for(size_t i=0;i<id.list.size();i++) {
      cooling_ns &c=id.list[i];
      temp_names_units2.push_back(((string)"Rns_")+c.name);
      temp_names_units2.push_back("km");
      temp_names_units2.push_back(((string)"log_wgt_")+c.name);
      temp_names_units2.push_back("");
      if (c.L==0.0 && c.L_lo==0.0) {
        temp_names_units2.push_back(((string)"L_")+c.name);
        temp_names_units2.push_back("ergs/s");
      }
    }
    
    for(size_t i=0;i<nd.list.size();i++) {
      nsmr &c=nd.list[i];
      temp_names_units2.push_back(((string)"Rns_")+c.name);
      temp_names_units2.push_back("km");
      temp_names_units2.push_back(((string)"log_wgt_")+c.name);
      temp_names_units2.push_back("");
    }

    for(size_t i=0;i<sd.list.size();i++) {
      sxrt &c=sd.list[i];
      temp_names_units2.push_back(((string)"Rns_")+c.name);
      temp_names_units2.push_back("km");
      temp_names_units2.push_back(((string)"log_wgt_")+c.name);
      temp_names_units2.push_back("");
      if (c.L_ul) {
        temp_names_units2.push_back(((string)"L_")+c.name);
        temp_names_units2.push_back("ergs/s");
      }
      //temp_names_units.push_back(((string)"logT_")+sxrt_names[i]);
      //temp_names_units.push_back("log(K)");
    }
    
    cout << "X1: " << pvi.size() << " " << pnames.size() << " "
         << punits.size() << " " << nparam << endl;

    // Then continue with the output names and units
    for(size_t i=0;i<temp_names_units2.size();i+=2) {
      pnames.push_back(temp_names_units2[i]);
      punits.push_back(temp_names_units2[i+1]);
    }

    for(size_t i=0;i<ngrid;i++) {
      pnames.push_back(((std::string)"R_")+o2scl::szttos(i));
      punits.push_back("km");
    }
    for(size_t i=0;i<ngrid;i++) {
      pnames.push_back(((std::string)"P_")+o2scl::szttos(i));
      punits.push_back("1/fm^4");
    }
    for(size_t i=0;i<ngrid;i++) {
      pnames.push_back(((std::string)"PM_")+o2scl::szttos(i));
      punits.push_back("1/fm^4");
    }
    for(size_t i=0;i<ngrid;i++) {
      pnames.push_back(((std::string)"Enb_")+o2scl::szttos(i));
      punits.push_back("1/fm^4");
    }
    for(size_t i=0;i<ngrid;i++) {
      pnames.push_back(((std::string)"Pnb_")+o2scl::szttos(i));
      punits.push_back("1/fm^4");
    }
    for(size_t i=0;i<ngrid;i++) {
      pnames.push_back(((std::string)"Pb_rhop_")+o2scl::szttos(i));
      punits.push_back("1/fm^3");
    }
    for(size_t i=0;i<ngrid;i++) {
      pnames.push_back(((std::string)"Pb_rhon_")+o2scl::szttos(i));
      punits.push_back("1/fm^3");
    }
    for(size_t i=0;i<ngrid;i++) {
      pnames.push_back(((std::string)"Zr_rhop_")+o2scl::szttos(i));
      punits.push_back("1/fm^3");
    }
    for(size_t i=0;i<ngrid;i++) {
      pnames.push_back(((std::string)"Zr_rhon_")+o2scl::szttos(i));
      punits.push_back("1/fm^3");
    }
    for(size_t i=0;i<ngrid;i++) {
      pnames.push_back(((std::string)"Ca_rhop_")+o2scl::szttos(i));
      punits.push_back("1/fm^3");
    }
    for(size_t i=0;i<ngrid;i++) {
      pnames.push_back(((std::string)"Ca_rhon_")+o2scl::szttos(i));
      punits.push_back("1/fm^3");
    }
    for(size_t ieta=0;ieta<3;ieta++) {
      for(size_t imass=0;imass<6;imass++) {
        for(size_t i=0;i<ngrid;i++) {
          pnames.push_back(((std::string)"cT_")+
                           o2scl::szttos(ieta)+"_"+
                           o2scl::szttos(imass)+"_"+
                           o2scl::szttos(i));
          punits.push_back("K");
        }
      }
    }
    for(size_t ieta=0;ieta<3;ieta++) {
      for(size_t imass=0;imass<6;imass++) {
        for(size_t i=0;i<ngrid;i++) {
          pnames.push_back(((std::string)"cL_")+
                           o2scl::szttos(ieta)+"_"+
                           o2scl::szttos(imass)+"_"+
                           o2scl::szttos(i));
          punits.push_back("erg/s");
        }
      }
    }
    for(size_t ieta=0;ieta<3;ieta++) {
      for(size_t imass=0;imass<6;imass++) {
        for(size_t i=0;i<30;i++) {
          pnames.push_back(((std::string)"sL_")+
                           o2scl::szttos(ieta)+"_"+
                           o2scl::szttos(imass)+"_"+
                           o2scl::szttos(i));
          punits.push_back("erg/s");
        }
      }
    }
    for(size_t i=0;i<ngrid;i++) {
      pnames.push_back(((std::string)"xp_")+o2scl::szttos(i));
      punits.push_back("");
    }
    for(size_t i=0;i<ngrid;i++) {
      pnames.push_back(((std::string)"Esym_")+o2scl::szttos(i));
      punits.push_back("MeV");
    }
    for(size_t i=0;i<ngrid;i++) {
      pnames.push_back(((std::string)"Esym2_")+o2scl::szttos(i));
      punits.push_back("MeV");
    }
    for(size_t i=0;i<ngrid;i++) {
      pnames.push_back(((std::string)"cs2_")+o2scl::szttos(i));
      punits.push_back("");
    }


    //vec_index dvi;
    
    for(size_t kk=nparam;kk<pnames.size();kk++) {
      dvi.append(pnames[kk]);
    }
    
    /*
      for(size_t i=0; i<pnames.size(); i++) {
      cout << pnames[i] << endl;
      //cout << "low" << "  " << low[i] << "  " << "high" << "  " 
      //<< high[i] << endl;
      //cout << "init" << "  " << init[i] << "  " << "step" << "  " 
      //<< step[i] << endl;
      }
    
      for(size_t i=0; i<init.size(); i++) {
      cout << init[i] << "  ";
      }
    */

  } else {
    vector<string> temp_names_units2=
      {"emulated","","log_wgt_unc",""};
    // Then continue with the output names and units
    for(size_t i=0;i<temp_names_units2.size();i+=2) {
      pnames.push_back(temp_names_units2[i]);
      punits.push_back(temp_names_units2[i+1]);
    }
    for(size_t kk=nparam;kk<pnames.size();kk++) {
      dvi.append(pnames[kk]);
    }
  }    

  mct.set_names_units(pnames,punits);

  if (pnames.size()!=nparam+ndat ||
      punits.size()!=nparam+ndat) {
    vector_out(cout,pnames,true);
    cout << "pnames.size: " << pnames.size() << endl;
    cout << "punits.size: " << punits.size() << endl;
    cout << "nparam,ndat: " << nparam << " " << ndat << endl;
    cout << "nparam+ndat: " << nparam+ndat << endl;
    for(size_t k=0;k<pnames.size();k++) {
      if (k>=nparam) {
	cout << k-nparam << " ";
      } else {
	cout << k << " ";
      }
      if (k<pvi.size()) {
	cout << pvi[k] << " ";
      }
      cout << pnames[k] << " " << punits[k];
      if (k<nparam) {
	cout << " " << init[k] << endl;
      } else {
	cout << endl;
      }
    }
    O2SCL_ERR("Mismatch between number of parameters and names or units.",
	      o2scl::exc_einval);
  }

  // Init all data_eval objects
  for(size_t k=0;k<((size_t)n_threads);k++) {
    de[k].init(nparam,dvi,pvi,verbose,id,nd,sd);
    de[k].no_nuclei=no_nuclei;
    de[k].no_ins=no_ins;
    de[k].debug_ins=debug_ins;
    de[k].no_sxrt=no_sxrt;
    de[k].w_tl_prof=w_tl_prof;
    de[k].r14_max=r14_max;
  }

  if (flag_emu_aws) {
    cout << "Training." << endl;
    ers.r14_max=r14_max;
    ers.train(pvi,dvi,&(de[0]));
    cout << "Training done." << endl;
    for(size_t k=0;k<((size_t)n_threads);k++) {
      de[k].flag_emu_aws=flag_emu_aws;
    }
  }
  
  if (verbose>1) {
    std::cout << "Initialize all data_eval objects" << endl;
  }
  
#ifndef NO_MPI
  // Get MPI rank, etc.
  MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
  
  // Ensure that multiple MPI ranks are not writing to the 
  // filesystem at the same time
  int tag=0, buffer=0;
  if (mpi_size>1 && mpi_rank>=1) {
    MPI_Recv(&buffer,1,MPI_INT,mpi_rank-1,
             tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  }
#endif
  
  // Initialize the file with settings, etc. (must be done
  // after de[k].init() above)
  string fname=mct.prefix+"_"+o2scl::itos(mpi_rank)+"_out";
  hdf_file hf2;
  hf2.open_or_create(fname);
  //hf2.seti("fit_cool_tptr",de[0].false);
  hf2.setd_vec("t_grid",de[0].t_grid);
  hf2.setd_vec("log_eta_grid",de[0].log_eta_grid);
  hf2.setd_vec("mass_grid",de[0].mass_grid);
  hf2.setd_vec("log_mdot_grid",de[0].log_mdot_grid);
  //hf2.sets_vec("sxrt_names",de[0].c.name);
  //hf2.sets_vec("nsmr_names",de[0].nsmr_names);
  //hf2.sets_vec("ins_names",de[0].ins_names);
  //hf2.sets_vec("ins_light_names",de[0].ins_light_names);
  hf2.sets_vec_copy("param_list",pvi.list());
  hf2.sets_vec_copy("data_list",dvi.list());
  hdf_output(hf2,de[0].mvsr_mass_grid,"mvsr_mass_grid");
  hdf_output(hf2,de[0].eden_grid,"eden_grid");
  hdf_output(hf2,de[0].nb_grid,"nb_grid");
  hdf_output(hf2,de[0].nucprof_grid,"nucprof_grid");
  hf2.close();
    
#ifndef NO_MPI
  // Send a message to the next MPI rank
  if (mpi_size>1 && mpi_rank<mpi_size-1) {
    MPI_Send(&buffer,1,MPI_INT,mpi_rank+1,
             tag,MPI_COMM_WORLD);
  }
#endif

  // Proposal distributions
  //vector<prob_cond_mdim_fixed_step<ubvector> > pcmrw;
  //vector<prob_cond_mdim_indep<> *> pcmi;
  //vector<prob_cond_mdim_gaussian<> *> pcmg;
  //vector<prob_dens_mdim_gmm<>> pgmm;
  vector<kde_python<ubvector>> pkde(1);

#ifdef RS_NEVER_DEFINED
  
  // The probability density vector type is ubvector,
  //but the vector type for the table is vector<double> .

  if (false) {
    
#ifndef NO_MPI
    // Get MPI rank, etc.
    MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
    
    // Ensure that multiple MPI ranks are not writing to the 
    // filesystem at the same time
    tag=0;
    buffer=0;
    if (mpi_size>1 && mpi_rank>=1) {
      MPI_Recv(&buffer,1,MPI_INT,mpi_rank-1,
	       tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
#endif
    
    table_units<> g1, g2, out;
    hdf_file hfy;
    std::string stemp;
    rng<> r;
    cout << mpi_rank << " reading: data4/all_nofail_out" << endl;
    hfy.open("data4/all_nofail_out");
    hdf_input(hfy,g1,stemp);
    hfy.close();
    cout << mpi_rank << " reading: wo_vela_out_large_wgt" << endl;
    hfy.open("wo_vela_out_large_wgt");
    hdf_input(hfy,g2,stemp);
    hfy.close();

#ifndef NO_MPI
    // Send a message to the next MPI rank
    if (mpi_size>1 && mpi_rank<mpi_size-1) {
      MPI_Send(&buffer,1,MPI_INT,mpi_rank+1,
               tag,MPI_COMM_WORLD);
    }
#endif
  
    // Create the output columns
    out.line_of_names("rank thread walker mult log_wgt");
    // pnames stores the parameters and the output data
    for(size_t i=0;i<pnames.size();i++) {
      out.new_column(pnames[i]);
    }
    
    ubvector pars(nparam);
    std::array<double,ndat> dat;
    double log_wgt;

    for(size_t jj=0;jj<10000;jj++) {
      for(size_t i=0;i<g1.get_nlines();i++) {
	// Set up the parameters from the rs_emu file
	for(size_t j=0;j<nparam;j++) {
	  pars[j]=g1.get(j+5,i);
	}
	cout << "original log_wgt: " << g1.get(4,i) << endl;
	
	// random row in the beloin data
	size_t row=r.random()*g2.get_nlines();
	cout << "row,log_wgt: " << row << " " << g2.get("log_wgt",row)
	     << endl;
	
	/*
	  11. b []          40
	  12. c []          41
	  13. c_r [fm]      42
	  14. m_s [1/fm]    43
	  15. zeta []       44
	  16. c_s [fm]      45
	  17. c_w [fm]      46
	  18. b1 []         47
	  19. xi []         64
	  20. a1 [1/fm]     65
	  21. a2 []         66
	  22. a3 [fm]       67
	  23. a4 [fm^2]     68
	  24. a5 [fm^3]     69
	  25. a6 [fm^4]     70
	  26. b2 [fm^2]     71
	  27. b3 [fm^4]     72
	*/
	// copy RMF parameters from beloin row
	if (true) {
	  for(size_t j=11;j<=27;j++) {
	    if (j<19) {
	      cout << out.get_column_name(j) << " " << g2.get_column_name(j+29)
		   << " " << g2.get(j+29,row) << endl;
	      pars[j-5]=g2.get(j+29,row);
	    } else {
	      cout << out.get_column_name(j) << " " << g2.get_column_name(j+45)
		   << " " << g2.get(j+45,row) << endl;
	      pars[j-5]=g2.get(j+45,row);
	    }
	  }
	}
	
	// Evaluate point
	int ret=de[0].point(nparam,pars,log_wgt,dat);
	
	cout << "Rank " << mpi_rank << " done: " << ret << " "
	     << pars[0] << " " << log_wgt << endl;
	
	if (ret==0) {
	  vector<double> line={((double)mpi_rank),0,0,1};
	  line.push_back(log_wgt);
	  for(size_t k=0;k<nparam;k++) line.push_back(pars[k]);
	  for(size_t k=0;k<ndat;k++) line.push_back(dat[k]);
	  out.line_of_data(line.size(),line);
	}
	
	if (i%20==0) {
	  cout << "Writing to file." << endl;
	  hdf_file hfz;
	  string out_filename="list_";
	  out_filename+=o2scl::itos(mpi_rank)+"_out";
	  hfz.open_or_create(out_filename);
	  hdf_output(hfz,out,"markov_chain0");
	  hfz.close();
	}
      }
    }
    exit(-1);
  }
  
  // Set random-walk step proposal distribution 
  if (ptype==((std::string)"step")) {

    pcmrw.resize(n_threads);
      
    unsigned long int seed=time(0);
    //for(size_t i=0;i<nparam;i++) {
    //step[i]/=step_fac;
    //}

    if (true) {
      for(size_t i=0;i<nparam;i++) {
	/*
          if (pnames[i].substr(0,3)!="mf_" &&
          pnames[i].substr(0,4)!="eta_" &&
          pnames[i].substr(0,2)!="t_" &&
          pnames[i].substr(0,11)!="log10_mdot_" &&
          pnames[i]!="log10_Tcn" && 
          pnames[i]!="k_maxn" && 
          pnames[i]!="del_kn" && 
          pnames[i]!="log10_Tcp" && 
          pnames[i]!="k_maxp" && 
          pnames[i]!="del_kp" &&
          pnames[i]!="alpha" &&
          pnames[i]!="Q" &&
          pnames[i]!="M_chirp" &&
          pnames[i]!="Lambdat") {
	  step[i]/=step_fac;
          }
	*/
	step[i]/=step_fac;
      }
    }
    
    for(size_t it=0;it<((size_t)n_threads);it++) {
      pcmrw[it].set(step,low,high);
	
      seed*=(mpi_rank*n_threads+it+1);
      pcmrw[it].set_seed(seed);
    }
    mct.set_proposal(pcmrw);

    if (false) {
      cout.setf(ios::showpos);
      for(size_t i=0;i<nparam;i++) {
	cout.width(15);
	cout << pnames[i] << " ";
	cout.width(4);
	cout << punits[i] << " " << low[i] << " "
	     << high[i] << " " << step[i] << endl;
      }
      cout.unsetf(ios::showpos);
      exit(-1);
    }

  } else if (ptype==((std::string)"gauss")) {

    // Set gaussian proposal distribution
      
    pcmg.resize(n_threads);
      
    table_units<> tprev;
    std::string name;
      
#ifndef NO_MPI
    // Get MPI rank, etc.
    MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
      
    // Ensure that multiple MPI ranks aren't reading from the
    // filesystem at the same time
    //int tag=0, buffer=0;
    if (mpi_size>1 && mpi_rank>=1) {
      MPI_Recv(&buffer,1,MPI_INT,mpi_rank-1,
	       tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
#endif
      
    string fnamex="data/data11/andrew7_0_out";
    cout << "Rank " << mpi_rank << " reading file "
	 << fnamex << "." << endl;
      
    // Read old peak and covariance matrix from data
    hdf_file hf;
    hf.open(fnamex);
    hdf_input(hf,tprev,name);
    cout << "Rank " << mpi_rank << " read previous table with "
	 << tprev.get_nlines() << " lines." << endl;
    hf.close();

    // Convert log_wgts to wgt to work with wvector_ functions
    tprev.function_column("exp(log_wgt)","wgt");
    cout << "Before: " << tprev.get_nlines() << endl;
    tprev.delete_rows_func("log_wgt< -300");
    cout << "After: " << tprev.get_nlines() << endl;

#ifndef NO_MPI
    // Send a message to the next MPI rank
    if (mpi_size>1 && mpi_rank<mpi_size-1) {
      MPI_Send(&buffer,1,MPI_INT,mpi_rank+1,
	       tag,MPI_COMM_WORLD);
    }
#endif

    ubvector std(nparam), avg(nparam);
    cout << "Rank " << mpi_rank <<endl;
    cout << "j param,avg,std: " << endl;
    for(size_t j=0;j<nparam;j++) {
      avg[j]=wvector_mean(tprev.get_nlines(),tprev[j+5],
			  tprev["wgt"]);
      std[j]=wvector_stddev(tprev.get_nlines(),tprev[j+5],
			    tprev["wgt"]);
      cout << j << " " << pvi[j] << " " << avg[j] 
           << " " << std[j] << endl;
    }
    
    ubmatrix covar(nparam,nparam);
    for(size_t i=0;i<nparam;i++) {
      for(size_t j=0;j<nparam;j++) {
        if (i==j) {
          covar(i,j)=std[j]*std[j];
          covar(i,j)/=var_dec_factor;
        } else {
          //covar(i,j)=0.0;
          covar(i,j)=wvector_covariance(tprev.get_nlines(),tprev[i+5],
                                        tprev[j+5],tprev["wgt"]);
          covar(i,j)/=covar_dec_factor;
        }
      }
    }
      
    // Set probability distribution
    for(size_t it=0;it<((size_t)n_threads);it++) {
      cout << "Setting pcmg." << endl;
      pcmg[it]=new prob_cond_mdim_gaussian<>;
      pcmg[it]->set(nparam,covar);
      cout << "Done setting pcmg." << endl;
    }
    unsigned long int seed2=time(0);
    for(size_t it=0;it<((size_t)n_threads);it++) {
      seed2*=(mpi_rank*n_threads+it+1);
      pcmg[it]->set_seed(seed2);
    }
    mct.set_proposal_ptrs(pcmg);

  } else if (ptype==((std::string)"gmm")) {

    // Set gaussian proposal distribution
      
    //vector<prob_cond_mdim_indep<> *> pcmi;
    //vector<prob_cond_mdim_gmm<> *> pgmm;
    pcmi.resize(n_threads);
    pgmm.resize(n_threads);

    std::string name;
      
#ifndef NO_MPI
    // Get MPI rank, etc.
    MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
      
    // Ensure that multiple MPI ranks aren't reading from the
    // filesystem at the same time
    //int tag=0, buffer=0;
    if (mpi_size>1 && mpi_rank>=1) {
      MPI_Recv(&buffer,1,MPI_INT,mpi_rank-1,
	       tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
#endif
      
    string fnamex="data/gmm10_out";
    cout << "Rank " << mpi_rank << " reading file "
	 << fnamex << "." << endl;
      
    // Read old peak and covariance matrix from data
    hdf_file hf;
    hf.open(fnamex);
    hdf_input(hf,pgmm[0]);
    cout << "Rank " << mpi_rank << " read gmm with " << endl;
    hf.close();


#ifndef NO_MPI
    // Send a message to the next MPI rank
    if (mpi_size>1 && mpi_rank<mpi_size-1) {
      MPI_Send(&buffer,1,MPI_INT,mpi_rank+1,
	       tag,MPI_COMM_WORLD);
    }
#endif

    /*
    pcmi[0]=new prob_cond_mdim_indep<>(pgmm[0]);
    
    unsigned long int seed2=time(0);
    for(size_t it=0;it<((size_t)n_threads);it++) {
      seed2*=(mpi_rank*n_threads+it+1);
      cout << "Setting seed: " << seed2 << endl;
      pgmm[0].r2.set_seed(seed2);
      unsigned long int seed3=seed2*2+1;
      for(size_t j=0;j<pgmm[0].pdmg.size();j++) {
	pgmm[0].pdmg[j].pdg.set_seed(seed3);
	seed3=seed3*2+1;
      }
    }
    mct.set_proposal_ptrs(pcmi);
    */
    // this section needs to be reworked for the new mcmc class
    exit(-1);

  }

#endif
    
  if (ptype==((std::string)"kde")) {
    
    std::string name;
    
#ifndef NO_MPI
    // Get MPI rank, etc.
    MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
      
    // Ensure that multiple MPI ranks aren't reading from the
    // filesystem at the same time
    //int tag=0, buffer=0;
    if (mpi_size>1 && mpi_rank>=1) {
      MPI_Recv(&buffer,1,MPI_INT,mpi_rank-1,
	       tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
#endif
      
    string fnamex="combined_emu";
    cout << "Rank " << mpi_rank << " reading file "
	 << fnamex << "." << endl;
    table_units<> tab_kde;
      
    // Read old peak and covariance matrix from data
    hdf_file hf;
    hf.open(fnamex);
    hdf_input(hf,tab_kde);
    cout << "Rank " << mpi_rank << " read kde table with " << endl;
    hf.close();

    tab_kde.delete_rows_func(((std::string)"abs(mult)<0.5 || ")+
                             "log_wgt<(-300) || Mns_max<2.035");
    
    tensor<> tin;
    vector<size_t> in_size={tab_kde.get_nlines(),nparam};
    tin.resize(2,in_size);
    cout << "Parameters: ";
    for(size_t i=0;i<nparam;i++) {
      cout << tab_kde.get_column_name(i+6) << " ";
    }
    cout << endl;
    for(size_t j=0;j<tab_kde.get_nlines();j++) {
      vector<size_t> ix;
      for(size_t i=0;i<nparam;i++) {
        ix={j,i};
        tin.get(ix)=tab_kde.get(i+6,j);
      }
    }

    // Train the KDE
    vector<double> weights;
    std::shared_ptr<kde_python<ubvector>> kp(new kde_python<ubvector>);
    kp->set_function("o2sclpy",tin,
                     weights,"verbose=0","kde_scipy");
    
#ifndef NO_MPI
    // Send a message to the next MPI rank
    if (mpi_size>1 && mpi_rank<mpi_size-1) {
      MPI_Send(&buffer,1,MPI_INT,mpi_rank+1,
	       tag,MPI_COMM_WORLD);
    }
#endif
    
    typedef mcmc_stepper_mh<point_funct,std::array<double,ndat>,ubvector,
                            ubmatrix,prob_cond_mdim_indep<>> stepper_t;
    shared_ptr<stepper_t> local_stepper(new stepper_t);
    mct.stepper=local_stepper;
    local_stepper->proposal.resize(1);
    local_stepper->proposal[0].set_base(kp);
    
  }

  // Maximum number of iterations (leaving this to the
  // default value, which is zero, allows an infinite number of
  // iterations and only stops the code after max_time).
  //mct.max_iters=50000;

  // Set number of OpenMP threads
  mct.n_threads=n_threads;

  if (ptype!=((std::string)"gmm") && ptype!="kde") {
    mct.store_rejects=true;
  }
  mct.table_sequence=false;

  if (false) {
    // AWS, 11/21/2020: temporary code to check the init, low, and
    // high values for all of the parameters.
    for(size_t i=0;i<nparam;i++) {
      cout.setf(ios::showpos);
      cout.width(15);
      cout << pvi[i] << " " << init[i] << " " << low[i] << " "
	   << high[i] << endl;
      cout.unsetf(ios::showpos);
    }
    exit(-1);
  }

  size_t vdat_size=2*mct.n_walk*mct.n_threads;
  vector<std::array<double,ndat>> vdat(vdat_size);
  cout << "Going to mcmc_fill() " << nparam << " "
       << low.size() << " " << high.size() << " " << vpf.size() << " "
       << vff.size() << " " << vdat.size() << " " << ndat << endl;
  mct.mcmc_fill(nparam,low,high,vpf,vff,vdat);

  return 0;
}

int mcmc_wrapper::initial_point_last(std::vector<std::string> &sv,
				     bool itive_com) {

  string fname=sv[1];
  size_t pos=fname.find("<rank>");
  if (pos!=std::string::npos) {
    fname.replace(pos,6,o2scl::itos(mpi_rank));
  }

  size_t offset=5;
  if (sv.size()>=3) offset=o2scl::stoszt(sv[2]);
  
  cout << "Rank " << mpi_rank
       << " loading last initial points from file '" << fname
       << "' with offset " << offset << "." << endl;
  
  mct.initial_points_file_last(fname,nparam,offset);

  return 0;
}
  
int mcmc_wrapper::initial_point_first(std::vector<std::string> &sv,
                                      bool itive_com) {

  string fname=sv[1];
  size_t pos=fname.find("<rank>");
  if (pos!=std::string::npos) {
    fname.replace(pos,6,o2scl::itos(mpi_rank));
  }
  cout << "Rank " << mpi_rank
       << " loading first point from file '" << fname
       << "'." << endl;

  o2scl::table_units<> tip;
    
#ifndef NO_MPI
  // Ensure that multiple threads aren't reading from the
  // filesystem at the same time
  int tag=0, buffer=0;
  if (mpi_size>1 && mpi_rank>0) {
    MPI_Recv(&buffer,1,MPI_INT,mpi_rank-1,
	     tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  }
#endif
    
  o2scl_hdf::hdf_file hf;
  hf.open(fname);
  std::string tname;
  hdf_input(hf,tip,tname);
  hf.close();
    
#ifndef NO_MPI
  if (mpi_size>1 && mpi_rank<mpi_size-1) {
    MPI_Send(&buffer,1,MPI_INT,mpi_rank+1,
	     tag,MPI_COMM_WORLD);
  }
#endif
    
  // Copy the entries from this row into the initial_points object
  mct.initial_points.resize(1);
  int row=0;
  if (verbose>0) {
    std::cout << "Initial point at row "
	      << row << " has log_weight= "
	      << tip.get("log_wgt",row) << std::endl;
  }
  mct.initial_points[0].resize(nparam);
  for(size_t ip=0;ip<nparam;ip++) {
    mct.initial_points[0][ip]=tip.get(ip+5,row);
  }
  return 0;
}
  
int mcmc_wrapper::initial_point_best(std::vector<std::string> &sv,
				     bool itive_com) {
  
  string fname=sv[1];
  size_t pos=fname.find("<rank>");
  if (pos!=std::string::npos) {
    fname.replace(pos,6,o2scl::itos(mpi_rank));
  }

  double prec=1.0e-6;
  if (sv.size()>=3) prec=o2scl::stod(sv[2]);
  size_t offset=5;
  if (sv.size()>=4) offset=o2scl::stoszt(sv[3]);
  
  cout << "Rank " << mpi_rank
       << " loading best initial points from file '" << fname
       << "'." << endl;
  cout << "Number of parameters is " << nparam << " precision is "
       << prec << " and offset is " << offset << endl;
  mct.initial_points_file_best(fname,nparam,prec,offset);
  cout << "Done in mcmc_para::initial_points_file_best()." << endl;
  
  //mct.meas_for_initial=true;
  return 0;
}

int mcmc_wrapper::initial_point_rand(std::vector<std::string> &sv,
				     bool itive_com) {
  string fname=sv[1];
  size_t pos=fname.find("<rank>");
  if (pos!=std::string::npos) {
    fname.replace(pos,6,o2scl::itos(mpi_rank));
  }
  cout << "Rank " << mpi_rank
       << " loading random initial point from file '" << fname
       << "'." << endl;

  o2scl::table_units<> tip;
    
#ifndef NO_MPI    
  MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);

  // Ensure that multiple threads aren't reading from the
  // filesystem at the same time
  int tag=0, buffer=0;
  if (mpi_size>1 && mpi_rank>0) {
    MPI_Recv(&buffer,1,MPI_INT,mpi_rank-1,
	     tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  }
#endif
    
  o2scl_hdf::hdf_file hf;
  hf.open(fname);
  std::string tname;
  hdf_input(hf,tip,tname);
  hf.close();
    
#ifndef NO_MPI
  if (mpi_size>1 && mpi_rank<mpi_size-1) {
    MPI_Send(&buffer,1,MPI_INT,mpi_rank+1,
	     tag,MPI_COMM_WORLD);
  }
#endif

  // Determine number of points
  size_t n_points=mct.n_walk*mct.n_threads;

  if (mct.verbose>0) {
    std::cout << "Initial points: Finding " << n_points
	      << " random points from file named "
	      << fname << " ." << std::endl;
  }

  vector<size_t> rows(n_points);
  rng<> rg;
  unsigned long int seed=time(0);
  unsigned long int seed2=seed*(1+mpi_rank);
  rg.set_seed(seed2);
  for(size_t k=0;k<n_points;k++) {
    rows[k]=rg.random()*tip.get_nlines();
  }
    
  // Copy the entries from this row into the initial_points object
  mct.initial_points.resize(n_points);
  for(size_t k=0;k<n_points;k++) {
    if (mct.verbose>0) {
      std::cout << "Initial point " << k << " at row "
		<< rows[k] << " has log_weight= "
		<< tip.get("log_wgt",rows[k]) << std::endl;
    }
    mct.initial_points[k].resize(nparam);
    for(size_t ip=0;ip<nparam;ip++) {
      mct.initial_points[k][ip]=tip.get(ip+5,rows[k]);
    }
  }

  mct.meas_for_initial=true;
  return 0;
}

int mcmc_wrapper::threads(std::vector<std::string> &sv, bool itive_com) {
  if (sv.size()<2) {
    cerr << "Need to specify number of threads." << endl;
    return 1;
  }
  n_threads=o2scl::stoszt(sv[1]);
  mct.n_threads=n_threads;
  cout << "Set number of OpenMP threads to: " << n_threads << endl;
  return 0;
}
  
int mcmc_wrapper::make_gaussian(std::vector<std::string> &sv, bool itive_com) {

  if (sv.size()<3) {
    cout << "Not enough arguments." << endl;
    exit(-1);
  }
  std::string in_file=sv[1];
  std::string out_file=sv[2];

  // Read previous results
  table_units<> t;
  hdf_file hf;
  hf.open(in_file);
  std::string name;
  hdf_input(hf,t,name);
  hf.close();

  bool weighted=false;
    
  // Clear these columns to sort 
  for(size_t k=0;k<t.get_nlines();k++) {
    t.set("rank",k,0.0);
    t.set("thread",k,0.0);
    t.set("walker",k,0.0);
    t.set("mult",k,1.0);
  }
  t.sort_table("log_wgt");

  // Compute weights
  if (weighted) {
    t.function_column("exp(log_wgt)","weight");
  }

  // List of parameter names
  vector<std::string> cols;
  for(size_t k=0;k<nparam;k++) {
    cols.push_back(t.get_column_name(k+5));
  }
  const_matrix_view_table<std::vector<double> > mvt(t,cols);

  // Peak and covariance matrix
  ubvector peak(nparam);
  ubmatrix covar(nparam,nparam);
  ubmatrix correl(nparam,nparam);
  typedef matrix_column_gen<const_matrix_view_table<std::vector<double> > >
    mat_col_t;
    
  // Set peak with average and diagonal elements in covariance
  // matrix with variance
  for(size_t i=0;i<nparam;i++) {
    mat_col_t col(mvt,i);
    if (weighted) {
      peak[i]=o2scl::wvector_mean<mat_col_t>(t.get_nlines(),col,t["weight"]);
      // Square standard deviation
      covar(i,i)=o2scl::wvector_stddev<mat_col_t>
	(t.get_nlines(),col,t["weight"]);
      cout << cols[i] << " " << peak[i] << " " << covar(i,i) << endl;
      covar(i,i)*=covar(i,i);
    } else {
      peak[i]=o2scl::vector_mean<mat_col_t>(t.get_nlines(),col);
      // Square standard deviation
      covar(i,i)=o2scl::vector_stddev<mat_col_t>
	(t.get_nlines(),col);
      cout << cols[i] << " " << peak[i] << " " << covar(i,i) << endl;
      covar(i,i)*=covar(i,i);
    }
  }
    
  // Setup off-diagonal covariance matrix
  for(size_t i=0;i<nparam;i++) {
    mat_col_t col_i(mvt,i);
    for(size_t j=i+1;j<nparam;j++) {
      mat_col_t col_j(mvt,j);
      double cov;
      if (weighted) {
	cov=o2scl::wvector_covariance(t.get_nlines(),col_i,col_j,
				      t["weight"]);
      } else {
	cov=o2scl::vector_covariance(t.get_nlines(),col_i,col_j);
      }
      covar(i,j)=cov;
      covar(j,i)=cov;
    }
  }

  // Create correlation matrix
  for(size_t i=0;i<nparam;i++) {
    for(size_t j=0;j<nparam;j++) {
      correl(i,j)=covar(i,j)/sqrt(covar(i,i))/sqrt(covar(j,j));
    }
  }

  hf.open_or_create(out_file);
  hf.setd_vec_copy("peak",peak);
  hf.setd_mat_copy("covar",covar);
  hf.setd_mat_copy("correl",correl);
  hf.close();

  return 0;
}

int mcmc_wrapper::make_gaussian2(std::vector<std::string> &sv, bool itive_com) {

  if (sv.size()<3) {
    cout << "Not enough arguments." << endl;
    exit(-1);
  }
  std::string in_file=sv[1];
  std::string out_file=sv[2];

  // Read previous results
  cout << "Reading file: " << in_file << endl;
  table_units<> t;
  hdf_file hf;
  hf.open(in_file);
  std::string name;
  hdf_input(hf,t,name);
  hf.close();

  // Remove rows with zero multiplier (empty)
  //t.delete_rows_func("abs(mult)<0.5");

  // Remove rows with small weight
  /*
    cout << t.get_nlines() << " rows." << endl;
    cout << "Deleting rows with small weight." << endl;
    t.delete_rows_func("log_wgt<(-100)");
    cout << t.get_nlines() << " rows remaining." << endl;
  */

  // Compute minimum and maximum vectors
  ubvector minv(nparam), maxv(nparam);
  for(size_t i=0;i<nparam;i++) {
    minv[i]=t.min(t.get_column_name(i+5));
    maxv[i]=t.max(t.get_column_name(i+5));
    double step=(maxv[i]-minv[i])/2.0;
    minv[i]-=step;
    maxv[i]+=step;
    if (minv[i]<low[i]) minv[i]=low[i];
    if (maxv[i]>high[i]) maxv[i]=high[i];
    cout << "low[" << i << "]=" << minv[i] << ";" << endl;
    cout << "high[" << i << "]=" << maxv[i] << ";" << endl;
  }
  exit(-1);

  cout << "Finding maximum weight." << endl;
  size_t max_row=t.lookup("log_wgt",t.max("log_wgt"));
    
  // Peak and covariance matrix
  ubvector peak(nparam);
  ubmatrix covar(nparam,nparam);
  ubmatrix correl(nparam,nparam);
  typedef matrix_column_gen<const_matrix_view_table<std::vector<double> > >
    mat_col_t;

  // Set peak with average and diagonal elements in covariance
  // matrix with variance
  for(size_t i=0;i<nparam;i++) {
    //peak[i]=(maxv[i]+minv[i])/2.0;
    peak[i]=t.get(i+5,max_row);
    double std=(maxv[i]-minv[i])/4.0;
    //covar(i,i)=(maxv[i]-minv[i])/20.0;
    cout << "Parameter " << i << " " << t.get_column_name(i+5)
	 << " min: " << minv[i] << " max: " << maxv[i]
	 << " peak: " << peak[i] 
	 << " std: " << std << endl;
    covar(i,i)=std*std;
  }
    
  // Setup off-diagonal covariance matrix
  for(size_t i=0;i<nparam;i++) {
    for(size_t j=i+1;j<nparam;j++) {
      covar(i,j)=0.0;
      covar(j,i)=0.0;
    }
  }

  // Create correlation matrix
  for(size_t i=0;i<nparam;i++) {
    for(size_t j=0;j<nparam;j++) {
      correl(i,j)=covar(i,j)/sqrt(covar(i,i))/sqrt(covar(j,j));
    }
  }

  hf.open_or_create(out_file);
  hf.setd_vec_copy("peak",peak);
  hf.setd_mat_copy("covar",covar);
  hf.setd_mat_copy("correl",correl);
  hf.close();

  return 0;
}

double mcmc_wrapper::gauss_min_func
(size_t np, const ubvector &p, 
 table_units<> &prev_res,
 o2scl::prob_dens_mdim_gaussian<> &mdim_gauss) {
  
  if (false) {
    for(size_t i=0;i<p.size();i++) {
      cout << p[i] << " ";
    }
    cout << endl;
  }
    
  bool bad=false;
  for(size_t i=nparam;i<2*nparam;i++) {
    if (p[i]<0.0) {
      //cout << "Bad: " << i << " " << p[i] << endl;
      //exit(-1);
      bad=true;
    }
  }
  if (bad==true) {
    return 1.0e14;
  }

  double ret=0.0;
  for(size_t row=0;row<prev_res.get_nlines();row++) {
      
    ubvector peak(nparam), xv(nparam);
    ubmatrix covar(nparam,nparam);
    double log_norm=p[2*nparam];
    for(size_t i=0;i<nparam;i++) {
      peak[i]=p[i];
      covar(i,i)=p[i+nparam];
      xv[i]=prev_res.get(i+5,row);
    }
    for(size_t i=0;i<nparam;i++) {
      for(size_t j=0;j<nparam;j++) {
	if (i!=j) covar(i,j)=0.0;
      }
    }
    mdim_gauss.set_covar(nparam,peak,covar);
    double model=mdim_gauss.log_pdf(xv)+log_norm;
    double exact=prev_res.get("log_wgt",row);
    if (row%40==0) {
      cout << row << " " << model << " " << exact << endl;
    }
    ret+=pow((model-exact)/fabs(exact)*100.0,2.0);
  }
  cout << ret << endl;
  return ret;
}
  
double mcmc_wrapper::gauss_min_func2
(size_t np, const ubvector &p, 
 table_units<> &prev_res,
 o2scl::prob_dens_mdim_gaussian<> &mdim_gauss) {

  size_t ix=0;
    
  ubvector peak(nparam);
  for(size_t i=0;i<nparam;i++) {
    peak[i]=p[ix];
    ix++;
  }
    
  ubmatrix covar(nparam,nparam);
  for(size_t i=0;i<nparam;i++) {
    if (p[ix]<=0.0) {
      return 1.0e20;
    }
    covar(i,i)=p[ix];
    ix++;
  }
  for(size_t i=0;i<nparam;i++) {
    for(size_t j=i+1;j<nparam;j++) {
      if (fabs(p[ix])>=1.0) {
	return 1.0e20;
      }
      covar(i,j)=p[ix]*sqrt(covar(i,i)*covar(j,j));
      covar(j,i)=covar(i,j);
      ix++;
    }
  }
    
  double log_norm=p[ix];
  mdim_gauss.set_covar(nparam,peak,covar);
  ubvector x(nparam);
  mdim_gauss(x);

  double ret=0.0;
  for(size_t row=0;row<prev_res.get_nlines();row++) {

    ubvector xv(nparam);
    for(size_t i=0;i<nparam;i++) {
      xv[i]=prev_res.get(i+5,row);
    }
    double model=mdim_gauss.log_pdf(xv)+log_norm;
    double exact=prev_res.get("log_wgt",row);
    if (row%40==0) {
      cout << row << " " << mdim_gauss.pdf(xv) << " "
	   << mdim_gauss.log_pdf(xv) << " "
	   << model << " " << exact << endl;
    }
    ret+=pow((model-exact)/fabs(exact)*100.0,2.0);
  }
  return ret;
}
  
int mcmc_wrapper::fit_gaussian(std::vector<std::string> &sv, bool itive_com) {

  // Read previous results
  table_units<> t;
  hdf_file hf;
  hf.open("/home/awsteiner/data/18/newmcmc_03_30_04_07_out");
  std::string name;
  hdf_input(hf,t,name);
  hf.close();
  hf.open("mg_3");
  ubvector peak(nparam);
  ubmatrix covar(nparam,nparam);
  hf.getd_vec_copy("peak",peak);
  hf.getd_mat_copy("covar",covar);
  hf.close();

  t.new_column("index");
  t.new_column("log_wgt_err");
  for(size_t i=0;i<t.get_nlines();i++) {
    t.set("index",i,i);
    t.set("log_wgt_err",i,fabs(t.get("log_wgt",i))/100.0);
  }

  o2scl::prob_dens_mdim_gaussian<> pdmg2;    
  multi_funct mf=
    std::bind(std::mem_fn<double(size_t, const ubvector &, 
				 table_units<> &,
				 o2scl::prob_dens_mdim_gaussian<> &)>
	      (&mcmc_wrapper::gauss_min_func),this,
	      std::placeholders::_1,std::placeholders::_2,
	      std::ref(t),std::ref(pdmg2));
		
  ubvector p(nparam*2+1), p2(nparam*2+1), p3(nparam*2+1);
  for(size_t i=0;i<nparam;i++) {
    p[i]=peak[i];
    p2[i]=peak[i]+sqrt(covar(i,i));
    p[i+nparam]=covar(i,i);
    p2[i+nparam]=covar(i,i)*1.2;
  }
  p[2*nparam]=-100.0;
  p2[2*nparam]=-110.0;

  /*
  // variance parameter for b3
  size_t ix=vi["xi"]+nparam;
  double val=mf(2*nparam+1,p);
  cout << "0 " << val << endl;
  for(size_t i=1;i<100;i++) {
  p[ix]*=1.2;
  val=mf(2*nparam+1,p);
  cout << i << " " << val << endl;
  }
  exit(-1);
  */
    
  /*
    for(size_t i=0;i<11;i++) {
    for(size_t k=0;k<2*nparam+1;k++) {
    p3[k]=p[k]-((double)i)/10.0*(p2[k]-p[k]);
    }
    double y=mf(nparam,p3);
    cout << i << " " << y << endl;
    }
    exit(-1);
  */
    
  double fmin;

  /*
    mmin_bfgs2<> mb;
    mb.err_nonconv=false;
    int ret=mb.mmin(2*nparam+1,p,fmin,mf);
  */

  anneal_gsl<> ga;
  ga.ntrial=400;
  ga.err_nonconv=false;
  for(size_t k=0;k<2*nparam+1;k++) {
    p3[k]=fabs(p2[k]-p[k])/100.0;
    cout << k << " " << p[k] << " " << p2[k] << " " << p3[k] << endl;
  }
  ga.set_step(2*nparam+1,p3);
  ga.min_step_ratio=0.0;
  ga.tol_abs/=1.0e2;
  ga.verbose=1;
  int ret=ga.mmin(2*nparam+1,p,fmin,mf);
    
  /*
    mmin_simp2<> ms;
    ms.verbose=1;
    ms.ntrial=1000;
    ms.err_nonconv=false;

    int ret=ms.mmin_twovec(2*nparam+1,p,p2,fmin,mf);
  */

  for(size_t i=0;i<nparam;i++) {
    cout << i << " " << peak[i] << " " << p[i] << " "
	 << covar(i,i) << " " << p[i+nparam] << endl;
    peak[i]=p[i];
    covar(i,i)=p[i+nparam];
  }

  hf.open_or_create("fg_0");
  hf.setd_vec_copy("peak",peak);
  hf.setd_mat_copy("covar",covar);
  hf.close();
    
  return 0;
}
  
int mcmc_wrapper::check_limits(std::vector<std::string> &sv, bool itive_com) {

  // Read previous results
  table_units<> t;
  vector<double> low2, high2;
  vector<string> param_list;
  size_t n_params;
  hdf_file hf;
  hf.open(sv[1]);
  std::string name;
  hdf_input(hf,t,name);
  hf.getd_vec("low",low2);
  hf.getd_vec("high",high2);
  hf.gets_vec_copy("param_list",param_list);
  hf.get_szt("n_params",n_params);
  hf.close();

  t.delete_rows_func("log_wgt<(-700) || Mns_max>3 || abs(mult)<0.5"); 
  
  for(size_t i=0;i<n_params;i++) {
    double mean=vector_mean(t.get_nlines(),t[t.get_column_name(i+5)]);
    // cant use sd b/c it's defined elsewhere
    double sd_loc=vector_stddev(t.get_nlines(),t[t.get_column_name(i+5)]);
    double min=vector_min_value<vector<double>,double>
      (t.get_nlines(),t[t.get_column_name(i+5)]);
    double max=vector_max_value<vector<double>,double>
      (t.get_nlines(),t[t.get_column_name(i+5)]);
    cout.width(3);
    cout << i << " ";
    cout.width(12);
    cout << param_list[i] << " ";
    cout.precision(4);
    cout.setf(ios::showpos);
    cout << low2[i] << " " << min << " "
         << mean-sd_loc << " " << mean << " " << mean+sd_loc << " "
         << max << " " << high2[i];
    if (mean+sd_loc>high2[i]) cout << " x";
    if (mean-sd_loc<low2[i]) cout << " y";
    cout << endl;
    cout.precision(6);
    cout.unsetf(ios::showpos);
  }
    
  return 0;
}
  
int mcmc_wrapper::fit_gaussian2(std::vector<std::string> &sv, bool itive_com) {

  // Read previous results
  table_units<> t;
  hdf_file hf;
  hf.open("/home/awsteiner/data/18/newmcmc_03_30_04_07_out");
  std::string name;
  hdf_input(hf,t,name);
  hf.close();
  hf.open("mg_3");
  ubvector peak(nparam);
  ubmatrix covar(nparam,nparam);
  hf.getd_vec_copy("peak",peak);
  hf.getd_mat_copy("covar",covar);
  hf.close();

  o2scl::prob_dens_mdim_gaussian<> pdmg2;    
  multi_funct mf=
    std::bind(std::mem_fn<double(size_t, const ubvector &, 
				 table_units<> &,
				 o2scl::prob_dens_mdim_gaussian<> &)>
	      (&mcmc_wrapper::gauss_min_func2),this,
	      std::placeholders::_1,std::placeholders::_2,
	      std::ref(t),std::ref(pdmg2));

  size_t nv=nparam*(nparam+1)/2+1+nparam;
  ubvector p(nv), p2(nv), p3(nv);
  size_t ix=0;
  for(size_t i=0;i<nparam;i++) {
    p[ix]=peak[i];
    ix++;
  }
  for(size_t i=0;i<nparam;i++) {
    p[ix]=covar(i,i);
    p2[ix]=p[ix]*1.2;
    p2[ix-nparam]=p[ix-nparam]+p[ix];
    ix++;
  }
  for(size_t i=0;i<nparam;i++) {
    for(size_t j=i+1;j<nparam;j++) {
      p[ix]=covar(i,j)/sqrt(covar(i,i)*covar(j,j));
      p[ix]/=100.0;
      if (p[ix]<0.0) {
	p2[ix]=-p[ix]*p[ix];
      } else {
	p2[ix]=p[ix]*p[ix];
      }
      if (fabs(p[ix])>=1.0) {
	cout << "Problem: " << i << " " << j << endl;
	exit(-1);
      }
      ix++;
    }
  }
  p[ix]=-100.0;
  p2[ix]=-90.0;
  ix++;
  if (ix!=nv) {
    cout << ix << " " << nv << endl;
    cout << "Problem2." << endl;
    exit(-1);
  }

  double fmin;

  anneal_gsl<> ga;
  ga.ntrial=400;
  ga.err_nonconv=false;
  for(size_t k=0;k<nv;k++) {
    p3[k]=fabs(p2[k]-p[k]);
  }
  ga.set_step(2*nparam+1,p3);
  ga.min_step_ratio=0.0;
  ga.tol_abs/=1.0e2;
  ga.verbose=1;
  int ret=ga.mmin(nv,p,fmin,mf);
    
  hf.open_or_create("fg_0");
  hf.setd_vec_copy("peak",peak);
  hf.setd_mat_copy("covar",covar);
  hf.close();
    
  return 0;
}
