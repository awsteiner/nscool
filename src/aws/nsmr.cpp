#include "nsmr.h"

using namespace std;
using namespace o2scl;

nsmr_data::nsmr_data() {
  
  // Read the M and R data files. These are read once and
  // shared across OpenMP cores.
  std::string data_files[18]={"../qlmxb/6304_H_nopl_syst_wilm.o2",
			      "../qlmxb/6304_He_nopl_syst_wilm.o2",
			      "../qlmxb/6397_H_syst_wilm.o2",
			      "../qlmxb/6397_He_syst_wilm3.o2",
			      "../qlmxb/M13_H_rs.o2",
			      "../qlmxb/M13_He_rs.o2",
			      "../qlmxb/M28_H_syst_wilm.o2",
			      "../qlmxb/M28_He_syst_wilm.o2",
			      //"../qlmxb/M30_H_syst_wilm.o2",
			      //"../qlmxb/M30_He_syst_wilm.o2",
			      "../qlmxb/M30_echi_atmos.o2",
			      "../qlmxb/M30_echi_x.o2",
			      //"../qlmxb/M30_Hatm.o2",
			      //"../qlmxb/M30_Heatm.o2",
			      "../qlmxb/wCen_H_syst_wilm.o2",
			      "../qlmxb/wCen_H_syst_wilm.o2",
			      "../qlmxb/X7_H_syst_wilm.o2",
			      "../qlmxb/X7_He_syst_wilm.o2",
			      "../nicer/0030_st_pst.o2",
			      "../nicer/0030_st_pst.o2",
            "../NICERxXMM/J0740_H_MR_t3d.o2",
            "../NICERxXMM/J0740_H_MR_t3d.o2"};
        
  int mpi_rank=0, mpi_size=1;
#ifndef NO_MPI
  // Get MPI rank, etc.
  MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
#endif

#ifndef NO_MPI
  // Ensure that multiple threads aren't writing to the
  // filesystem at the same time
  int tag=0, buffer=0;
  if (mpi_size>1 && mpi_rank>=1) {
    MPI_Recv(&buffer,1,MPI_INT,mpi_rank-1,
	     tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  }
#endif

  const size_t sz1= sizeof(data_files)/sizeof(data_files[0]);

  MR_data.resize(sz1);
  for(size_t j=0;j<sz1;j++) {
    o2scl_hdf::hdf_file hf;
    hf.open(data_files[j]);
    if (j==4 || j==5) {
      hdf_input(hf,MR_data[j],"rescaled_0");
    } else if (j<14) {
      hdf_input(hf,MR_data[j],"rescaled");
    } else {
      hdf_input(hf,MR_data[j],"table3d");
    }
    hf.close();
  }

#ifndef NO_MPI
  if (mpi_size>1 && mpi_rank<mpi_size-1) {
    MPI_Send(&buffer,1,MPI_INT,mpi_rank+1,
	     tag,MPI_COMM_WORLD);
  }
#endif

  // Read the name of the files and get the star names
  // into list
        
  for (size_t j=0;j<sz1;j+=2) {
    size_t pos1=data_files[j].find("/");
    std::string str1=data_files[j].substr(pos1+1);
    size_t pos2=str1.find("/");
    std::string str2=str1.substr(pos2+1);
    size_t pos = str2.find("_");
    std::string str=str2.substr(0,pos);
    list.push_back({str});
  }
}

void nsmr_data::init_data(vec_index &pvi,
			  std::vector<std::string> &param_units) {
  nsmr_data nd;
  for (size_t i=0;i<nd.list.size();i++) {
    nsmr &c=nd.list[i];
    pvi.append(((string)"mf_")+c.name);
    param_units.push_back("");
    pvi.append(((string)"eta_")+c.name);
    param_units.push_back("");
  }
  return;
}
