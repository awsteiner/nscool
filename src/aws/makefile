#----------------------------------------------------------------------
# Organize data files by date

MONTH = $(shell date +'%m')
DAY = $(shell date +'%d')
YEAR = $(shell date +'%y')

# ----------------------------------------------------------------------
# User-specific makefiles
# ----------------------------------------------------------------------

#include ../makefile.aws

# ----------------------------------------------------------------
# User-specific settings
# ----------------------------------------------------------------
# LIBS is the list of libraries
# LFC is the local fortran compiler
# LCXX is the local C++ compiler
# LCFLAGS are the local C++ compiler flags
# LFFLAGS are the local fortran compiler flags
# FLIBS are fortran system libraries

LIBS = -L/usr/local/lib -L/usr/lib/x86_64-linux-gnu/hdf5/serial \
	-lo2scl -lhdf5 -lgsl \
	-lreadline -lpython3.8
FLIBS = -lgfortran
# PLIBS = -L/usr/lib/x86_64-linux-gnu/ 
LCXX = g++
LFC = gfortran
LMPI_FC = mpif90
LMPI_CXX = mpic++
LCFLAGS = -O3 -I/usr/lib/x86_64-linux-gnu/hdf5/serial/include \
	-I/usr/local/include -DNO_MPI -DNO_OPENMP \
	-I/usr/include/python3.8 -DO2SCL_PYTHON \
	-Wall -Wno-unused -Wshadow -Wno-deprecated-declarations \
	-DO2SCL_NO_BOOST_MULTIPRECISION
LCFLAGS_OMP = -O3 -I/usr/lib/x86_64-linux-gnu/hdf5/serial/include \
	-I/usr/local/include -DNO_MPI \
	-I/usr/local/include -fopenmp \
	-I/usr/include/python3.8 -DO2SCL_PYTHON \
	-Wall -Wno-unused -Wshadow -Wno-deprecated-declarations \
	-DO2SCL_NO_BOOST_MULTIPRECISION
LFFLAGS = -O3
LMPI_CFLAGS = -DO2SCL_OPENMP -DO2SCL_MPI -O3 \
	-I/usr/lib/x86_64-linux-gnu/hdf5/serial/include \
	-I/usr/local/include -fopenmp \
	-I/usr/include/python3.8 -DO2SCL_PYTHON \
	-Wall -Wno-unused -Wshadow -Wno-deprecated-declarations \
	-DO2SCL_NO_BOOST_MULTIPRECISION
COMMENT = "default"
PYTHON_LDFLAGS = 

# ----------------------------------------------------------------------
# UTK generic makefile
# ----------------------------------------------------------------------

ifdef UTKNA_MAKEFILE

include $(UTKNA_MAKEFILE)

# UTK configuration
LIBS = $(UTKNA_O2SCL_LIBS) -lreadline
FLIBS = $(UTKNA_FLIBS)
LCXX = $(UTKNA_CXX) 
LFC = $(UTKNA_FC) 
LMPI_FC = $(UTKNA_MPI_FC) 
LMPI_CXX = $(UTKNA_MPI_CXX)
LCFLAGS = $(UTKNA_O2SCL_INCS) $(UTKNA_CFLAGS) -DNO_MPI -DNO_OPENMP \
	$(UTKNA_PYTHON_INCLUDES) -DO2SCL_NO_BOOST_MULTIPRECISION
LFFLAGS = $(UTKNA_FFLAGS)
LMPI_CFLAGS = $(UTKNA_O2SCL_INCS) $(UTKNA_MPI_CFLAGS) $(UTKNA_OPENMP_FLAGS) \
	$(UTKNA_PYTHON_INCLUDES) -DO2SCL_PYTHON -DO2SCL_NO_BOOST_MULTIPRECISION
PYTHON_INCLUDES = $(UTKNA_PYTHON_INCLUDES)
PYTHON_LDFLAGS = $(UTKNA_PYTHON_LDFLAGS)
COMMENT = $(UTKNA_COMMENT)

endif

# ----------------------------------------------------------------
# Main targets
# ----------------------------------------------------------------

OBJS  = precool.o conductivity.o conductivity_core.o \
	opacity.o neutrino.o neutrino_core.o neutrino_crust.o \
	data_eval.o spec_heat.o density.o tc.o tc_Ioffe.o \
	Tools.o conductivity_crust.o nscool_wrap.o \
	ins.o sxrt.o nsmr.o ins_like.o sxrt_like.o nsmr_like.o \
	rs.o 
OBJS_NOMPI_TEST = precool_nompi.o conductivity_nompi.o \
	conductivity_core_nompi.o \
	opacity_nompi.o neutrino_nompi.o neutrino_core_nompi.o \
	neutrino_crust_nompi.o \
	spec_heat_nompi.o density_nompi.o tc_nompi.o tc_Ioffe_nompi.o \
	Tools_nompi.o conductivity_crust_nompi.o 
OBJS_NOMPI = precool_nompi.o conductivity_nompi.o \
	conductivity_core_nompi.o \
	opacity_nompi.o neutrino_nompi.o neutrino_core_nompi.o \
	neutrino_crust_nompi.o data_eval_nompi.o \
	spec_heat_nompi.o density_nompi.o tc_nompi.o tc_Ioffe_nompi.o \
	Tools_nompi.o conductivity_crust_nompi.o nscool_wrap_nompi.o \
	ins_nompi.o sxrt_nompi.o nsmr_nompi.o ins_like_nompi.o \
	sxrt_like_nompi.o nsmr_like_nompi.o rs_nompi.o

OBJS_EMU = precool.o conductivity.o conductivity_core.o \
	opacity.o neutrino.o neutrino_core.o neutrino_crust.o \
	data_eval.o spec_heat.o density.o tc.o tc_Ioffe.o \
	Tools.o conductivity_crust.o nscool_wrap.o \
	ins.o sxrt.o nsmr.o ins_like.o sxrt_like.o nsmr_like.o \
	rs_emu.o 

OBJS_EMU_NOMPI  = precool_nompi.o conductivity_nompi.o \
	conductivity_core_nompi.o \
	opacity_nompi.o neutrino_nompi.o neutrino_core_nompi.o \
	neutrino_crust_nompi.o data_eval_nompi.o \
	spec_heat_nompi.o density_nompi.o tc_nompi.o tc_Ioffe_nompi.o \
	Tools_nompi.o conductivity_crust_nompi.o nscool_wrap_nompi.o \
	ins_nompi.o sxrt_nompi.o nsmr_nompi.o ins_like_nompi.o \
	sxrt_like_nompi.o nsmr_like_nompi.o rs_emu_nompi.o

OBJS_PG = precool_pg.o conductivity_pg.o conductivity_core_pg.o \
           opacity_pg.o neutrino_pg.o neutrino_core_pg.o neutrino_crust_pg.o \
           spec_heat_pg.o density_pg.o tc_pg.o tc_Ioffe_pg.o \
           Tools_pg.o conductivity_crust_pg.o 

help:
	@echo "test"
	@echo "settings"

settings:
	@echo $(COMMENT)
	@echo "libs:" $(LIBS)
	@echo "flibs:" $(FLIBS)
	@echo "lcxx:" $(LCXX)
	@echo "lfc:" $(LFC)
	@echo "lmpi_fc:" $(LMPI_FC)
	@echo "lmpi_cxx:" $(LMPI_CXX)
	@echo "lcflags:" $(LCFLAGS)
	@echo "lfflags:" $(LFFLAGS)
	@echo "lmpi_cflags:" $(LMPI_CFLAGS)

empty:

doc: empty
	cd sphinx/static; cat bib_header.txt > ../bib.rst
	cd sphinx/static; btmanip -parse nscool.bib -rst ../bib_temp.rst
	cd sphinx; cat bib_temp.rst >> bib.rst; rm -f bib_temp.rst
	cd doc; doxygen doxyfile
	cd sphinx; make html

test: test.o ../nscool_wrap.h
	$(LMPI_CXX) $(LMPI_CFLAGS) -o test test.o \
		nscool_wrap.o $(LIBS) $(FLIBS)

test.o: ../test.cpp
	$(LMPI_CXX) $(LMPI_CFLAGS) -o test.o -c ../test.cpp

test_nompi: $(OBJS_NOMPI_TEST) nscool_wrap_nompi.o NSCool_nompi.o \
	test_nompi.o ../nscool_wrap.h
	$(LCXX) $(LCFLAGS_OMP) -o test_nompi test_nompi.o \
		nscool_wrap_nompi.o \
		NSCool_nompi.o $(OBJS_NOMPI_TEST) $(LIBS) $(FLIBS)

test_nompi.o: ../test.cpp
	$(LCXX) $(LCFLAGS) -o test_nompi.o -c ../test.cpp 

slice: slice.o 
	$(LCXX) $(LCFLAGS) -o slice slice.o  \
	$(LIBS) $(FLIBS)

slice_nompi_omp: slice.o 
	$(LCXX) $(LCFLAGS_OMP) -o slice slice.o  \
	$(LIBS) $(FLIBS)

slice.o: slice.cpp 
	$(LCXX) $(LCFLAGS) -o slice.o -c slice.cpp 

nscool_wrap.o: ../nscool_wrap.cpp
	$(LMPI_CXX) $(LMPI_CFLAGS) -o nscool_wrap.o -c ../nscool_wrap.cpp 

nscool_wrap_nompi.o: ../nscool_wrap.cpp
	$(LCXX) $(LCFLAGS) -o nscool_wrap_nompi.o -c ../nscool_wrap.cpp  

ins.o: ins.cpp
	$(LMPI_CXX) $(LMPI_CFLAGS) -o ins.o -c ins.cpp 

ins_nompi.o: ins.cpp
	$(LCXX) $(LCFLAGS) -o ins_nompi.o -c ins.cpp 

sxrt.o: sxrt.cpp sxrt.h
	$(LMPI_CXX) $(LMPI_CFLAGS) -o sxrt.o -c sxrt.cpp 

sxrt_nompi.o: sxrt.cpp sxrt.h
	$(LCXX) $(LCFLAGS) -o sxrt_nompi.o -c sxrt.cpp

nsmr.o: nsmr.cpp
	$(LMPI_CXX) $(LMPI_CFLAGS) -o nsmr.o -c nsmr.cpp 

nsmr_nompi.o: nsmr.cpp
	$(LCXX) $(LCFLAGS) -o nsmr_nompi.o -c nsmr.cpp 

ins_like.o: ins_like.cpp  
	$(LMPI_CXX) $(LMPI_CFLAGS) -o ins_like.o -c ins_like.cpp 

ins_like_nompi.o: ins_like.cpp 
	$(LCXX) $(LCFLAGS) -o ins_like_nompi.o -c ins_like.cpp 

data_eval.o: data_eval.cpp  
	$(LMPI_CXX) $(LMPI_CFLAGS) -o data_eval.o -c data_eval.cpp 

data_eval_nompi.o: data_eval.cpp 
	$(LCXX) $(LCFLAGS) -o data_eval_nompi.o -c data_eval.cpp 

sxrt_like.o: sxrt_like.cpp  
	$(LMPI_CXX) $(LMPI_CFLAGS) -o sxrt_like.o -c sxrt_like.cpp 

sxrt_like_nompi.o: sxrt_like.cpp 
	$(LCXX) $(LCFLAGS) -o sxrt_like_nompi.o -c sxrt_like.cpp

nsmr_like.o: nsmr_like.cpp  
	$(LMPI_CXX) $(LMPI_CFLAGS) -o nsmr_like.o -c nsmr_like.cpp 

nsmr_like_nompi.o: nsmr_like.cpp 
	$(LCXX) $(LCFLAGS) -o nsmr_like_nompi.o -c nsmr_like.cpp 

rs.o: rs.cpp rs.h
	$(LMPI_CXX) $(LMPI_CFLAGS) -o rs.o -c rs.cpp 

rs_nompi.o: rs.cpp rs.h
	$(LCXX) $(LCFLAGS) -o rs_nompi.o -c rs.cpp

rs_emu.o: rs_emu.cpp rs.h
	$(LMPI_CXX) $(LMPI_CFLAGS) -o rs_emu.o -c rs_emu.cpp 

rs_emu_nompi.o: rs_emu.cpp rs.h
	$(LCXX) $(LCFLAGS) -o rs_emu_nompi.o -c rs_emu.cpp  

main.o: main.cpp 
	$(LMPI_CXX) $(LMPI_CFLAGS) -o main.o -c main.cpp 

main_nompi.o: main.cpp
	$(LCXX) $(LCFLAGS) -o main_nompi.o -c main.cpp 

rs: $(OBJS) NSCool.o main.o ../nscool_wrap.h 
	$(LMPI_CXX) $(LMPI_CFLAGS) -o rs main.o NSCool.o \
		$(OBJS) $(LIBS) $(FLIBS) 

rs_nompi: $(OBJS_NOMPI) NSCool_nompi.o main_nompi.o ../nscool_wrap.h
	$(LCXX) $(LCFLAGS) -o rs_nompi main_nompi.o NSCool_nompi.o \
		$(OBJS_NOMPI) $(LIBS) $(FLIBS)

rs_nompi_omp: $(OBJS_NOMPI) NSCool_nompi.o main_nompi.o ../nscool_wrap.h
	$(LCXX) $(LCFLAGS_OMP) -o rs_nompi_omp main_nompi.o NSCool_nompi.o \
		$(OBJS_NOMPI) $(LIBS) $(FLIBS)

rs_emu: $(OBJS_EMU) NSCool.o main.o ../nscool_wrap.h 
	$(LMPI_CXX) $(LMPI_CFLAGS) -o rs_emu main.o NSCool.o \
		$(OBJS_EMU) $(LIBS) $(FLIBS) $(PYTHON_LDFLAGS)

rs_emu_nompi: $(OBJS_EMU_NOMPI) NSCool_nompi.o main_nompi.o ../nscool_wrap.h
	$(LCXX) $(LCFLAGS) -o rs_emu_nompi main_nompi.o NSCool_nompi.o \
		$(OBJS_EMU_NOMPI) $(LIBS) $(FLIBS) $(PYTHON_LDFLAGS)

rs_emu_nompi_omp: $(OBJS_EMU_NOMPI) NSCool_nompi.o main_nompi.o ../nscool_wrap.h
	$(LCXX) $(LCFLAGS_OMP) -o rs_emu_nompi_omp main_nompi.o NSCool_nompi.o \
		$(OBJS_EMU_NOMPI) $(LIBS) $(FLIBS) $(PYTHON_LDFLAGS)

clean:
	-rm -f *.o rs rs_nompi rs_nompi_omp \
		rs_emu rs_emu_nompi rs_emu_nompi_omp \
		test test_nompi slice

# ----------------------------------------------------------------
# Main object files
# ----------------------------------------------------------------

NSCool.o: ../NSCool.f ../nscool_wrap.h
	$(LMPI_FC) $(LFFLAGS) -c ../NSCool.f -o NSCool.o
precool.o: ../precool.f
	$(LMPI_FC) $(LFFLAGS) -c ../precool.f -o precool.o
conductivity.o: ../conductivity.f
	$(LMPI_FC) $(LFFLAGS) -c ../conductivity.f -o conductivity.o
conductivity_core.o: ../conductivity_core.f
	$(LMPI_FC) $(LFFLAGS) -c ../conductivity_core.f -o conductivity_core.o
conductivity_crust.o: ../conductivity_crust.f
	$(LMPI_FC) $(LFFLAGS) -c ../conductivity_crust.f -o conductivity_crust.o
opacity.o: ../opacity.f
	$(LMPI_FC) $(LFFLAGS) -c ../opacity.f -o opacity.o
neutrino.o: ../neutrino.f
	$(LMPI_FC) $(LFFLAGS) -c ../neutrino.f -o neutrino.o
neutrino_core.o: ../neutrino_core.f
	$(LMPI_FC) $(LFFLAGS) -c ../neutrino_core.f -o neutrino_core.o
neutrino_crust.o: ../neutrino_crust.f
	$(LMPI_FC) $(LFFLAGS) -c ../neutrino_crust.f -o neutrino_crust.o
spec_heat.o: ../spec_heat.f
	$(LMPI_FC) $(LFFLAGS) -c ../spec_heat.f -o spec_heat.o
density.o: ../density.f
	$(LMPI_FC) $(LFFLAGS) -c ../density.f -o density.o
tc.o: ../tc.f
	$(LMPI_FC) $(LFFLAGS) -c ../tc.f -o tc.o
tc_Ioffe.o: ../tc_Ioffe.f
	$(LMPI_FC) $(LFFLAGS) -c ../tc_Ioffe.f -o tc_Ioffe.o
Tools.o: ../Tools.f
	$(LMPI_FC) $(LFFLAGS) -c ../Tools.f -o Tools.o

# ----------------------------------------------------------------
# Main object files (no MPI version)
# ----------------------------------------------------------------

NSCool_nompi.o: ../NSCool.f ../nscool_wrap.h
	$(LFC) $(LFFLAGS) -c ../NSCool.f -o NSCool_nompi.o
precool_nompi.o: ../precool.f
	$(LFC) $(LFFLAGS) -c ../precool.f -o precool_nompi.o
conductivity_nompi.o: ../conductivity.f
	$(LFC) $(LFFLAGS) -c ../conductivity.f -o conductivity_nompi.o
conductivity_core_nompi.o: ../conductivity_core.f
	$(LFC) $(LFFLAGS) -c ../conductivity_core.f -o conductivity_core_nompi.o
conductivity_crust_nompi.o: ../conductivity_crust.f
	$(LFC) $(LFFLAGS) -c ../conductivity_crust.f -o conductivity_crust_nompi.o
opacity_nompi.o: ../opacity.f
	$(LFC) $(LFFLAGS) -c ../opacity.f -o opacity_nompi.o
neutrino_nompi.o: ../neutrino.f
	$(LFC) $(LFFLAGS) -c ../neutrino.f -o neutrino_nompi.o
neutrino_core_nompi.o: ../neutrino_core.f
	$(LFC) $(LFFLAGS) -c ../neutrino_core.f -o neutrino_core_nompi.o
neutrino_crust_nompi.o: ../neutrino_crust.f
	$(LFC) $(LFFLAGS) -c ../neutrino_crust.f -o neutrino_crust_nompi.o
spec_heat_nompi.o: ../spec_heat.f
	$(LFC) $(LFFLAGS) -c ../spec_heat.f -o spec_heat_nompi.o
density_nompi.o: ../density.f
	$(LFC) $(LFFLAGS) -c ../density.f -o density_nompi.o
tc_nompi.o: ../tc.f
	$(LFC) $(LFFLAGS) -c ../tc.f -o tc_nompi.o
tc_Ioffe_nompi.o: ../tc_Ioffe.f
	$(LFC) $(LFFLAGS) -c ../tc_Ioffe.f -o tc_Ioffe_nompi.o
Tools_nompi.o: ../Tools.f
	$(LFC) $(LFFLAGS) -c ../Tools.f -o Tools_nompi.o

# ----------------------------------------------------------------
# Main object files (gprof version)
# ----------------------------------------------------------------

NSCool_pg.o: NSCool.f ../nscool_wrap.h
	$(LFC) $(LFFLAGS) -g -pg -c NSCool.f -o NSCool_pg.o
precool_pg.o: precool.f
	$(LFC) $(LFFLAGS) -g -pg -c precool.f -o precool_pg.o
conductivity_pg.o: conductivity.f
	$(LFC) $(LFFLAGS) -g -pg -c conductivity.f -o conductivity_pg.o
conductivity_core_pg.o: conductivity_core.f
	$(LFC) $(LFFLAGS) -g -pg -c conductivity_core.f \
		-o conductivity_core_pg.o
conductivity_crust_pg.o: conductivity_crust.f
	$(LFC) $(LFFLAGS) -g -pg -c conductivity_crust.f \
		-o conductivity_crust_pg.o
opacity_pg.o: opacity.f
	$(LFC) $(LFFLAGS) -g -pg -c opacity.f -o opacity_pg.o
neutrino_pg.o: neutrino.f
	$(LFC) $(LFFLAGS) -g -pg -c neutrino.f -o neutrino_pg.o
neutrino_core_pg.o: neutrino_core.f
	$(LFC) $(LFFLAGS) -g -pg -c neutrino_core.f -o neutrino_core_pg.o
neutrino_crust_pg.o: neutrino_crust.f
	$(LFC) $(LFFLAGS) -g -pg -c neutrino_crust.f -o neutrino_crust_pg.o
spec_heat_pg.o: spec_heat.f
	$(LFC) $(LFFLAGS) -g -pg -c spec_heat.f -o spec_heat_pg.o
density_pg.o: density.f
	$(LFC) $(LFFLAGS) -g -pg -c density.f -o density_pg.o
tc_pg.o: tc.f
	$(LFC) $(LFFLAGS) -g -pg -c tc.f -o tc_pg.o
tc_Ioffe_pg.o: tc_Ioffe.f
	$(LFC) $(LFFLAGS) -g -pg -c tc_Ioffe.f -o tc_Ioffe_pg.o
Tools_pg.o: Tools.f
	$(LFC) $(LFFLAGS) -g -pg -c Tools.f -o Tools_pg.o

test-steady-state:
	./rs -steady-state ss.o2

orca_test:
	./rs_nompi -threads 1 -set ptype gauss -set file_update_time 1 \
	       -set mcmc_verbose 1 -set verbose 2 -set max_iters 10 \
	       -initial-point-last guess_sept -set covar_dec_factor 100 \
	       -set prefix ../data/sep_gauss -mcmc > ot_gauss.scr 2>&1 &
	sleep 5
	./rs_nompi -threads 1 -set ptype step -set file_update_time 1 \
	       -set mcmc_verbose 1 -set verbose 2 -set max_iters 10 \
	       -initial-point-last guess_sept -set covar_dec_factor 100 \
	       -set prefix ../data/sep_step -mcmc > ot_step.scr 2>&1 &

ubuntu_test:
	./rs -threads 1 -set ptype step -set file_update_time 1 \
	       -set mcmc_verbose 2 -set verbose 3 -set max_iters 1 \
	       -initial-point-last guess_sept \
	       -set prefix ../data/temp -mcmc > ubuntu_test.scr 2>&1 

dec2:
	-mpirun --allow-run-as-root -np 2 ./rs_emu -threads 1 \
		-set w_tl_prof 0 -set file_update_time 3600 \
		-set aff_inv 1 -set n_walk 400 \
		-set mcmc_verbose 1 -set verbose 1 -set max_time 432000 \
		-set step_fac 2 -initial-point-last \
		"data/dec2_in_<rank>_out" \
		-set prefix data/dec2 -mcmc > data/dec2.out 2>&1 &
	-mpirun --allow-run-as-root -np 2 ./rs_emu -threads 1 \
		-set w_tl_prof 0 -set file_update_time 3600 \
		-set aff_inv 1 -set n_walk 400 -set R14_max 12.75 \
		-set mcmc_verbose 1 -set verbose 1 -set max_time 432000 \
		-set step_fac 2 -initial-point-last \
		"data/dec2_smallr_in_<rank>_out" \
		-set prefix data/dec2_smallr -mcmc \
		> data/dec2_smallr.out 2>&1 &

jan_emu:
	mpirun --allow-run-as-root -np 4 ./rs_emu -threads 1 \
		-set w_tl_prof 0 -set file_update_time 3600 \
		-set aff_inv 1 -set n_walk 400 \
		-set mcmc_verbose 2 -set verbose 2 -set max_time 432000 \
		-set step_fac 2 -initial-point-last \
		"data/jan_0_out" \
		-emu-aws "data/combined_emu" \
		-set prefix data/jan_emu -mcmc > data/je.out 2>&1 &

jan:
	-mpirun --allow-run-as-root -np 4 ./rs_emu -threads 1 \
		-set w_tl_prof 0 -set file_update_time 3600 \
		-set aff_inv 1 -set n_walk 400 \
		-set mcmc_verbose 1 -set verbose 1 -set max_time 432000 \
		-set step_fac 2 -initial-point-best \
		"data/mix_gaps_<rank>" \
		-set prefix data/jan -mcmc > data/jan.out 2>&1 &

jan2:
	-mpirun --allow-run-as-root -np 4 ./rs_emu -threads 1 \
		-set w_tl_prof 0 -set file_update_time 3600 \
		-set aff_inv 1 -set n_walk 400 \
		-set mcmc_verbose 1 -set verbose 1 -set max_time 432000 \
		-set step_fac 2 -initial-point-last \
		"data/combined_z_<rank>" \
		-set prefix data/jan2 -mcmc > data/jan2.out 2>&1 &

alt3:
	-mpirun --allow-run-as-root -np 2 ./rs_emu -threads 1 \
		-set w_tl_prof 0 -set file_update_time 3600 \
		-set aff_inv 1 -set n_walk 400 \
		-set mcmc_verbose 1 -set verbose 1 -set max_time 432000 \
		-set step_fac 2 -initial-point-last \
		"data/mix" \
		-set prefix data/alt3 -mcmc \
		> data/alt3.out 2>&1 &

dec2e_sub:
	-mpirun --allow-run-as-root -np 2 ./rs_emu -threads 1 \
		-set w_tl_prof 0 -set file_update_time 3600 \
		-set aff_inv 1 -set n_walk 400 \
		-set mcmc_verbose 1 -set verbose 1 -set max_time 172800 \
		-set step_fac 2 -initial-point-best \
		"data/dec2e_new_<rank>_out" \
		-set prefix data/dec2e -mcmc > data/dec2e.out 2>&1 &
#		-emu-aws "data/dec2e_in_<rank>_out" \
#mkdir -p data/dec_$(DAY)
#cp -i data/dec2e.out data/dec_$(DAY)
#cp -i data/dec2e_?_??? data/dec_$(DAY)
#cp data/dec2e_0_out data/dec2e_in_0_out
#cp data/dec2e_1_out data/dec2e_in_1_out

dec2e_smallr_sub:
	-mpirun --allow-run-as-root -np 2 ./rs_emu -threads 1 \
		-emu-aws "data/dec2e_smallr_in_<rank>_out" \
		-set w_tl_prof 0 -set file_update_time 3600 \
		-set aff_inv 1 -set n_walk 400 -set R14_max 12.75 \
		-set mcmc_verbose 1 -set verbose 1 -set max_time 172800 \
		-set step_fac 2 -initial-point-last \
		"data/dec2e_smallr_in_<rank>_out" \
		-set prefix data/dec2e_smallr -mcmc \
		> data/dec2e_smallr.out 2>&1
	mkdir -p data/dec_$(DAY)
	cp -i data/dec2e_smallr.out data/dec_$(DAY)
	cp -i data/dec2e_smallr_?_??? data/dec_$(DAY)
	cp data/dec2e_smallr_0_out data/dec2e_smallr_in_0_out
	cp data/dec2e_smallr_1_out data/dec2e_smallr_in_1_out

alt2e_sub: empty
	-mpirun --allow-run-as-root -np 2 ./rs_emu -threads 1 \
		-set w_tl_prof 0 -set file_update_time 3600 \
		-set aff_inv 1 -set n_walk 400 \
		-set mcmc_verbose 1 -set verbose 1 -set max_time 172800 \
		-set step_fac 2 -initial-point-best \
		"data/alt2e_new_<rank>_out" \
		-set prefix data/alt2e -mcmc > data/alt2e.out 2>&1 &
#	mkdir -p data/dec_$(DAY)
#	cp -i data/alt2e.out data/dec_$(DAY)
#	cp -i data/alt2e_?_??? data/dec_$(DAY)
#	cp data/alt2e_0_out data/alt2e_in_0_out
#	cp data/alt2e_1_out data/alt2e_in_1_out

alt2e_temp: empty
	-mpirun --allow-run-as-root -np 2 ./rs_emu -threads 1 \
		-emu-aws "data/alt2e_<rank>_out" \
		-set w_tl_prof 0 -set file_update_time 3600 \
		-set aff_inv 1 -set n_walk 400 \
		-set mcmc_verbose 1 -set verbose 1 -set max_time 86400 \
		-set step_fac 2 -initial-point-last \
		"data/alt2e_<rank>_out" \
		-set prefix data/alt2e -mcmc

alt2e:
	$(MAKE) alt2e_sub > alt2e_all.scr 2>&1
	$(MAKE) alt2e_sub >> alt2e_all.scr 2>&1
	$(MAKE) alt2e_sub >> alt2e_all.scr 2>&1

nove2a:
	-mpirun --allow-run-as-root -np 2 ./rs_emu -threads 1 \
		-emu-aws "data/guess_<rank>" \
		-set w_tl_prof 0 -set file_update_time 3600 \
		-set aff_inv 1 -set n_walk 400 \
		-set mcmc_verbose 1 -set verbose 1 -set max_time 432000 \
		-set step_fac 2 -initial-point-best \
		"data/guess_<rank>" \
		-set prefix data/nove2 -mcmc > data/nove2.out 2>&1 &

smtest:
	rs_emu_nompi -threads 1 \
		-set w_tl_prof 0 -set file_update_time 3600 \
		-set aff_inv 0 -set R14_max 12.75 \
		-set mcmc_verbose 3 -set verbose 3 -set max_iters 2 \
		-set step_fac 2 -initial-point-best \
		"data/sm_max" \
		-set prefix data/sm -mcmc

newdat:
	./rs_emu -threads 1 \
		-set w_tl_prof 0 -set file_update_time 3600 \
		-set aff_inv 1 -set n_walk 400 \
		-set mcmc_verbose 3 -set verbose 3 -set max_iters 1 \
		-set step_fac 2 -initial-point-last \
		"thin" \
		-set prefix nd1 -mcmc > nd1.out 2>&1 &

newdat_smallr:
	./rs_emu -threads 1 \
		-set w_tl_prof 0 -set file_update_time 3600 \
		-set aff_inv 1 -set n_walk 400 \
		-set mcmc_verbose 3 -set verbose 3 -set max_iters 1 \
		-set step_fac 2 -initial-point-last \
		"thin_smallr" \
		-set prefix nd1_sr -mcmc > nd1_sr.out 2>&1 &

em2t:
	rs_emu_nompi -threads 1 \
		-emu-aws "dec2e_0_out" \
		-set w_tl_prof 0 -set file_update_time 3600 \
		-set aff_inv 1 -set n_walk 400 \
		-set mcmc_verbose 1 -set verbose 1 -set max_time 432000 \
		-set step_fac 2 -initial-point-last \
		"dec2e_0_out" \
		-set prefix em2t -mcmc
#> data/nove2.out 2>&1 &

gmm:
	rs_emu_nompi -threads 1 \
		-set w_tl_prof 0 -set file_update_time 3600 \
		-set aff_inv 0 -set n_walk 1 -set ptype gmm \
		-set mcmc_verbose 1 -set verbose 1 -set max_time 432000 \
		-set step_fac 2 -initial-point-best \
		"data/jan_0_out" -set prefix gmm -mcmc

gmm2:
	mpirun --allow-run-as-root -np 4 rs_emu -threads 1 \
		-set w_tl_prof 0 -set file_update_time 3600 \
		-set aff_inv 0 -set n_walk 1 -set ptype gmm \
		-set mcmc_verbose 1 -set verbose 1 -set max_time 432000 \
		-set step_fac 2 -initial-point-best \
		"data/jan_temp" -set prefix data/gmm -mcmc > gmm.out 2>&1 &

mut:
	rs_emu_nompi -threads 1 \
		-set w_tl_prof 0 -set file_update_time 3600 \
		-set aff_inv 0 -set n_walk 1 -set max_iters 1 \
		-set mcmc_verbose 3 -set verbose 3 -set max_time 432000 \
		-set step_fac 2 -initial-point-best \
		"data/mu_test" 1.0e-6 6 \
		-set prefix data/mut -mcmc

kde1:
	rs_emu_nompi -threads 1 \
		-set w_tl_prof 0 -set file_update_time 3600 \
		-set aff_inv 0 -set n_walk 1 -set ptype kde \
		-set mcmc_verbose 1 -set verbose 1 -set max_time 432000 \
		-set step_fac 2 -initial-point-best \
		"data/combined_emu" 1.0e-6 6 -set prefix data/kde -mcmc

mmaxt:
	mpirun --allow-run-as-root -np 4 rs_emu -threads 1 \
		-set w_tl_prof 0 -set file_update_time 3600 \
		-set aff_inv 1 -set n_walk 400 \
		-set mcmc_verbose 1 -set verbose 1 -set max_time 432000 \
		-set step_fac 2 -initial-point-best \
		"combined_emu_mmax" \
		-set prefix mmaxt -mcmc

fdiff:
	cd .. && svn cat -r5 nscool_wrap.h > temp && diff temp nscool_wrap.h
	cd .. && svn cat -r5 nscool_wrap.cpp > temp && diff temp nscool_wrap.cpp

itest:
	interact -t 00:15:00 -A phy230028p -p RM-shared

itest2:
	mpirun -n 1 ./rs_emu -threads 1 \
		-set w_tl_prof 0 -set file_update_time 3600 \
		-set aff_inv 0 -set n_walk 1 -set ptype kde \
		-set mcmc_verbose 1 -set verbose 1 -set max_time 600 \
		-set step_fac 2 -initial-point-best \
		"combined_emu" -set prefix data/kde -mcmc
