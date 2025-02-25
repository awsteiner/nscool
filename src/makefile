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

LIBS = -L/usr/local/hdf5/lib \
	-lo2scl -lhdf5 -lgsl \
	-lreadline -lpython3.13
FLIBS = -lgfortran
# PLIBS = -L/usr/lib/x86_64-linux-gnu/ 
LCXX = g++
LFC = gfortran
LMPI_FC = mpif90
LMPI_CXX = ccache mpic++
LCFLAGS = -I/usr/local/hdf5/include \
	-I/usr/lib/python3.13/site-packages/numpy/_core/include \
	-DNO_MPI -DNO_OPENMP -DO2SCL_PYTHON \
	-I/usr/include/python3.13 \
	-Wall -Wno-unused -Wshadow -Wno-deprecated-declarations \
	-DO2SCL_NO_BOOST_MULTIPRECISION
LCFLAGS_OMP = -I/usr/local/hdf5/include \
	-DNO_MPI -DO2SCL_PYTHON \
	-fopenmp \
	-I/usr/include/python3.13 \
	-Wall -Wno-unused -Wshadow -Wno-deprecated-declarations \
	-DO2SCL_NO_BOOST_MULTIPRECISION
LFFLAGS = -O3
LMPI_CFLAGS = -I/usr/local/hdf5/include \
	-I/usr/lib/python3.13/site-packages/numpy/_core/include \
	-DO2SCL_OPENMP -DO2SCL_MPI -O3 \
	-DO2SCL_PLAIN_HDF5_HEADER -DO2SCL_HDF5_COMP \
	-I/usr/local/include -fopenmp \
	-I/usr/include/python3.13 -DO2SCL_PYTHON \
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
	$(UTKNA_PYTHON_INCLUDES)
LFFLAGS = $(UTKNA_FFLAGS)
LMPI_CFLAGS = $(UTKNA_O2SCL_INCS) $(UTKNA_MPI_CFLAGS) $(UTKNA_OPENMP_FLAGS) \
	$(UTKNA_PYTHON_INCLUDES)
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
	rs_emu.o 

OBJS_NOMPI  = precool_nompi.o conductivity_nompi.o \
	conductivity_core_nompi.o \
	opacity_nompi.o neutrino_nompi.o neutrino_core_nompi.o \
	neutrino_crust_nompi.o data_eval_nompi.o \
	spec_heat_nompi.o density_nompi.o tc_nompi.o tc_Ioffe_nompi.o \
	Tools_nompi.o conductivity_crust_nompi.o nscool_wrap_nompi.o \
	ins_nompi.o sxrt_nompi.o nsmr_nompi.o ins_like_nompi.o \
	sxrt_like_nompi.o nsmr_like_nompi.o rs_emu_nompi.o

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

OBJS_PREX  = precool.o conductivity.o conductivity_core.o \
	opacity.o neutrino.o neutrino_core.o neutrino_crust.o \
	spec_heat.o density.o tc.o tc_Ioffe.o \
	Tools.o conductivity_crust.o nscool_wrap.o \


OBJS_PREX_NOMPI  = precool_nompi.o conductivity_nompi.o \
	conductivity_core_nompi.o \
	opacity_nompi.o neutrino_nompi.o neutrino_core_nompi.o \
	neutrino_crust_nompi.o \
	spec_heat_nompi.o density_nompi.o tc_nompi.o tc_Ioffe_nompi.o \
	Tools_nompi.o conductivity_crust_nompi.o nscool_wrap_nompi.o \


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

test: precool.o conductivity.o conductivity_core.o \
	opacity.o neutrino.o neutrino_core.o neutrino_crust.o \
	spec_heat.o density.o tc.o tc_Ioffe.o \
	Tools.o conductivity_crust.o nscool_wrap.o NSCool.o test.o nscool_wrap.h
	$(LMPI_CXX) $(LMPI_CFLAGS) -o test test.o NSCool.o \
		precool.o conductivity.o conductivity_core.o \
	opacity.o neutrino.o neutrino_core.o neutrino_crust.o \
	spec_heat.o density.o tc.o tc_Ioffe.o \
	Tools.o conductivity_crust.o nscool_wrap.o $(LIBS) $(FLIBS)

test.o: test.cpp
	$(LMPI_CXX) $(LMPI_CFLAGS) -o test.o -c test.cpp

test_nompi_omp: $(OBJS_NOMPI) NSCool_nompi.o test_nompi.o nscool_wrap.h
	$(LCXX) $(LCFLAGS_OMP) -o test_nompi test_nompi.o NSCool_nompi.o \
		$(OBJS_NOMPI) $(LIBS) $(FLIBS)

test_nompi.o: test.cpp
	$(LCXX) $(LCFLAGS) -o test_nompi.o -c test.cpp 

slice: slice.o 
	$(LMPI_CXX) $(LMPI_CFLAGS) -o slice slice.o -lo2scl \
	$(LIBS)

slice_nompi_omp: slice.o 
	$(LCXX) $(LCFLAGS_OMP) -o slice slice.o  \
	$(LIBS) $(FLIBS)

slice.o: slice.cpp 
	$(LMPI_CXX) $(LMPI_CFLAGS) -o slice.o -c slice.cpp 

nscool_wrap.o: nscool_wrap.cpp 
	$(LMPI_CXX) $(LMPI_CFLAGS) -o nscool_wrap.o -c nscool_wrap.cpp 

nscool_wrap_nompi.o: nscool_wrap.cpp 
	$(LCXX) $(LCFLAGS) -o nscool_wrap_nompi.o -c nscool_wrap.cpp  

ins.o: ins.cpp ins.h
	$(LMPI_CXX) $(LMPI_CFLAGS) -o ins.o -c ins.cpp 

ins_nompi.o: ins.cpp ins.h
	$(LCXX) $(LCFLAGS) -o ins_nompi.o -c ins.cpp 

sxrt.o: sxrt.cpp sxrt.h
	$(LMPI_CXX) $(LMPI_CFLAGS) -o sxrt.o -c sxrt.cpp 

sxrt_nompi.o: sxrt.cpp sxrt.h
	$(LCXX) $(LCFLAGS) -o sxrt_nompi.o -c sxrt.cpp

nsmr.o: nsmr.cpp nsmr.h
	$(LMPI_CXX) $(LMPI_CFLAGS) -o nsmr.o -c nsmr.cpp 

nsmr_nompi.o: nsmr.cpp nsmr.h
	$(LCXX) $(LCFLAGS) -o nsmr_nompi.o -c nsmr.cpp 

ins_like.o: ins_like.cpp data_eval.h 
	$(LMPI_CXX) $(LMPI_CFLAGS) -o ins_like.o -c ins_like.cpp 

ins_like_nompi.o: ins_like.cpp data_eval.h
	$(LCXX) $(LCFLAGS) -o ins_like_nompi.o -c ins_like.cpp 

data_eval.o: data_eval.cpp  data_eval.h 
	$(LMPI_CXX) $(LMPI_CFLAGS) -o data_eval.o -c data_eval.cpp 

data_eval_nompi.o: data_eval.cpp data_eval.h 
	$(LCXX) $(LCFLAGS) -o data_eval_nompi.o -c data_eval.cpp 

sxrt_like.o: sxrt_like.cpp data_eval.h 
	$(LMPI_CXX) $(LMPI_CFLAGS) -o sxrt_like.o -c sxrt_like.cpp 

sxrt_like_nompi.o: sxrt_like.cpp data_eval.h
	$(LCXX) $(LCFLAGS) -o sxrt_like_nompi.o -c sxrt_like.cpp

nsmr_like.o: nsmr_like.cpp data_eval.h 
	$(LMPI_CXX) $(LMPI_CFLAGS) -o nsmr_like.o -c nsmr_like.cpp 

nsmr_like_nompi.o: nsmr_like.cpp data_eval.h
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

rs: $(OBJS) NSCool.o main.o nscool_wrap.h 
	$(LMPI_CXX) $(LMPI_CFLAGS) -o rs main.o NSCool.o \
		$(OBJS) $(LIBS) $(FLIBS) 

rs_nompi: $(OBJS_NOMPI) NSCool_nompi.o main_nompi.o nscool_wrap.h
	$(LCXX) $(LCFLAGS) -o rs_nompi main_nompi.o NSCool_nompi.o \
		$(OBJS_NOMPI) $(LIBS) $(FLIBS)

rs_nompi_omp: $(OBJS_NOMPI) NSCool_nompi.o main_nompi.o nscool_wrap.h
	$(LCXX) $(LCFLAGS_OMP) -o rs_nompi_omp main_nompi.o NSCool_nompi.o \
		$(OBJS_NOMPI) $(LIBS) $(FLIBS)

rs_emu: $(OBJS_EMU) NSCool.o main.o nscool_wrap.h 
	$(LMPI_CXX) $(LMPI_CFLAGS) -o rs_emu main.o NSCool.o \
		$(OBJS_EMU) $(LIBS) $(FLIBS) $(PYTHON_LDFLAGS)

rs_emu_nompi: $(OBJS_EMU_NOMPI) NSCool_nompi.o main_nompi.o nscool_wrap.h 
	$(LCXX) $(LCFLAGS) -o rs_emu_nompi main_nompi.o NSCool_nompi.o \
		$(OBJS_EMU_NOMPI) $(LIBS) $(FLIBS) $(PYTHON_LDFLAGS)

rs_emu_nompi_omp: $(OBJS_EMU_NOMPI) NSCool_nompi.o main_nompi.o nscool_wrap.h
	$(LCXX) $(LCFLAGS_OMP) -o rs_emu_nompi_omp main_nompi.o NSCool_nompi.o \
		$(OBJS_EMU_NOMPI) $(LIBS) $(FLIBS) $(PYTHON_LDFLAGS)

prex.o: prex.cpp
	$(LMPI_CXX) $(LMPI_CFLAGS) -o prex.o -c prex.cpp 

prex_nompi.o: prex.cpp
	$(LCXX) $(LCFLAGS) -o prex_nompi.o -c prex.cpp 

prex_nompi_omp.o: prex.cpp
	$(LCXX) $(LCFLAGS) -o prex_nompi_omp.o -c prex.cpp 

prex: $(OBJS_PREX) NSCool.o prex.o nscool_wrap.h
	$(LMPI_CXX) $(LMPI_CFLAGS) -o prex prex.o NSCool.o \
		$(OBJS_PREX) $(LIBS) $(FLIBS)

prex_nompi: $(OBJS_PREX_NOMPI) NSCool_nompi.o prex_nompi.o nscool_wrap.h
	$(LCXX) $(LCFLAGS) -o prex_nompi prex_nompi.o NSCool_nompi.o \
		$(OBJS_PREX_NOMPI) $(LIBS) $(FLIBS)

prex_nompi_omp: $(OBJS_PREX_NOMPI) NSCool_nompi.o prex_nompi_omp.o nscool_wrap.h
	$(LCXX) $(LCFLAGS_OMP) -o prex_nompi_omp prex_nompi_omp.o NSCool_nompi.o \
		$(OBJS_PREX_NOMPI) $(LIBS) $(FLIBS)

correl_skip.o: correl_skip.cpp
	$(LMPI_CXX) $(LMPI_CFLAGS) -o correl_skip.o -c correl_skip.cpp 

correl_skip: correl_skip.o
	$(LMPI_CXX) $(LMPI_CFLAGS) -o correl_skip correl_skip.o \
	$(LIBS) $(FLIBS) $(PYTHON_LDFLAGS)

emu_test: emu_test.o 
	$(LMPI_CXX) $(LMPI_CFLAGS) -o emu_test emu_test.o  \
	$(LIBS) $(FLIBS)

emu_test.o: emu_test.cpp 
	$(LMPI_CXX) $(LMPI_CFLAGS) -o emu_test.o -c emu_test.cpp 



clean:
	-rm -f *.o rs rs_nompi rs_nompi_omp \
		rs_emu rs_emu_nompi rs_emu_nompi_omp \
		prex prex_nompi \
		test test_nompi slice

# ----------------------------------------------------------------
# Main object files
# ----------------------------------------------------------------

NSCool.o:  NSCool.f nscool_wrap.h
	$(LMPI_FC) $(LFFLAGS) -c  NSCool.f -o NSCool.o
precool.o:  precool.f
	$(LMPI_FC) $(LFFLAGS) -c  precool.f -o precool.o
conductivity.o:  conductivity.f
	$(LMPI_FC) $(LFFLAGS) -c  conductivity.f -o conductivity.o
conductivity_core.o:  conductivity_core.f
	$(LMPI_FC) $(LFFLAGS) -c  conductivity_core.f -o conductivity_core.o
conductivity_crust.o:  conductivity_crust.f
	$(LMPI_FC) $(LFFLAGS) -c  conductivity_crust.f -o conductivity_crust.o
opacity.o:  opacity.f
	$(LMPI_FC) $(LFFLAGS) -c  opacity.f -o opacity.o
neutrino.o:  neutrino.f
	$(LMPI_FC) $(LFFLAGS) -c  neutrino.f -o neutrino.o
neutrino_core.o:  neutrino_core.f
	$(LMPI_FC) $(LFFLAGS) -c  neutrino_core.f -o neutrino_core.o
neutrino_crust.o:  neutrino_crust.f
	$(LMPI_FC) $(LFFLAGS) -c  neutrino_crust.f -o neutrino_crust.o
spec_heat.o:  spec_heat.f
	$(LMPI_FC) $(LFFLAGS) -c  spec_heat.f -o spec_heat.o
density.o:  density.f
	$(LMPI_FC) $(LFFLAGS) -c  density.f -o density.o
tc.o:  tc.f
	$(LMPI_FC) $(LFFLAGS) -c  tc.f -o tc.o
tc_Ioffe.o:  tc_Ioffe.f
	$(LMPI_FC) $(LFFLAGS) -c  tc_Ioffe.f -o tc_Ioffe.o
Tools.o:  Tools.f
	$(LMPI_FC) $(LFFLAGS) -c  Tools.f -o Tools.o

# ----------------------------------------------------------------
# Main object files (no MPI version)
# ----------------------------------------------------------------

NSCool_nompi.o:  NSCool.f nscool_wrap.h
	$(LFC) $(LFFLAGS) -c  NSCool.f -o NSCool_nompi.o
precool_nompi.o:  precool.f
	$(LFC) $(LFFLAGS) -c  precool.f -o precool_nompi.o
conductivity_nompi.o:  conductivity.f
	$(LFC) $(LFFLAGS) -c  conductivity.f -o conductivity_nompi.o
conductivity_core_nompi.o:  conductivity_core.f
	$(LFC) $(LFFLAGS) -c  conductivity_core.f -o conductivity_core_nompi.o
conductivity_crust_nompi.o:  conductivity_crust.f
	$(LFC) $(LFFLAGS) -c  conductivity_crust.f -o conductivity_crust_nompi.o
opacity_nompi.o:  opacity.f
	$(LFC) $(LFFLAGS) -c  opacity.f -o opacity_nompi.o
neutrino_nompi.o:  neutrino.f
	$(LFC) $(LFFLAGS) -c  neutrino.f -o neutrino_nompi.o
neutrino_core_nompi.o:  neutrino_core.f
	$(LFC) $(LFFLAGS) -c  neutrino_core.f -o neutrino_core_nompi.o
neutrino_crust_nompi.o:  neutrino_crust.f
	$(LFC) $(LFFLAGS) -c  neutrino_crust.f -o neutrino_crust_nompi.o
spec_heat_nompi.o:  spec_heat.f
	$(LFC) $(LFFLAGS) -c  spec_heat.f -o spec_heat_nompi.o
density_nompi.o:  density.f
	$(LFC) $(LFFLAGS) -c  density.f -o density_nompi.o
tc_nompi.o:  tc.f
	$(LFC) $(LFFLAGS) -c  tc.f -o tc_nompi.o
tc_Ioffe_nompi.o:  tc_Ioffe.f
	$(LFC) $(LFFLAGS) -c  tc_Ioffe.f -o tc_Ioffe_nompi.o
Tools_nompi.o:  Tools.f
	$(LFC) $(LFFLAGS) -c  Tools.f -o Tools_nompi.o

# ----------------------------------------------------------------
# Main object files (gprof version)
# ----------------------------------------------------------------

NSCool_pg.o: NSCool.f nscool_wrap.h
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

ubuntu_test:
	./rs -threads 1 -set ptype step -set file_update_time 1 \
	       -set mcmc_verbose 2 -set verbose 3 -set max_iters 1 \
	       -initial-point-last guess_sept \
	       -set prefix ../data/temp -mcmc > ubuntu_test.scr 2>&1 

iso_run:
	./rs -set ptype step -set file_update_time 1800 \
	       -set mcmc_verbose 1 -set verbose 2 -set max_iters 5 \
	       -initial-point-last guess_sept \
	       -set prefix ../data/isospin -mcmc 

neutrino_run:
	mpirun -np 3 ./rs_emu -threads 1 -set apply_emu 0 \
		-set w_tl_prof 0 -set file_update_time 1800 \
		-set mcmc_verbose 1 -set verbose 1 -set max_time 360000 \
		-set ptype gauss \
		-initial-point-first ../data/guess_jul_out \
	    -set prefix ../data/jul_iufsu_gauss -mcmc > neutrino.scr 2>&1 &

neutrino_ai:
	mpirun -np 6 ./rs -threads 1 \
		-set file_update_time 1800 \
		-set mcmc_verbose 1 -set verbose 1 -set max_time 864000 \
		-set aff_inv true -set step_fac 2.0 -set n_walk 640 \
		-initial-point-best "../data/neutrino_<rank>_out" \
		-set prefix ../data/neutrino_ai -mcmc > neutrino_ai.scr 2>&1 &

smallr:
	mpirun --allow-run-as-root \
		-np 1 ./rs_emu -threads 2 \
		-set w_tl_prof 0 -set file_update_time 3600 \
		-set mcmc_verbose 1 -set verbose 1 -set max_time 432000 \
		-set aff_inv 1 -set n_walk 200 \
		-set step_fac 2.0 -set R14_max 13.75 \
		-initial-point-best ../data/dec11_smallr_0_out \
	    -set prefix ../data/dec29_smallr -mcmc > data/syncsr.out 2>&1 &

larger:
	mpirun --allow-run-as-root \
		-np 1 ./rs_emu -threads 2 \
		-set w_tl_prof 0 -set file_update_time 3600 \
		-set mcmc_verbose 1 -set verbose 1 -set  max_time 432000 \
		-set aff_inv 1 -set n_walk 200 \
		-set step_fac 2.0 \
		-initial-point-best ../data/dec11_0_out \
	    -set prefix ../data/dec29 -mcmc > data/sync.out 2>&1 &

midr:
	mpirun --allow-run-as-root \
		-np 1 ./rs_emu -threads 2 \
		-set w_tl_prof 0 -set file_update_time 3600 \
		-set mcmc_verbose 1 -set verbose 1 -set max_time 432000 \
		-set aff_inv 1 -set n_walk 200 \
		-set step_fac 2.0 -set R14_max 13.7 \
		-initial-point-best ../plot/midinit \
	    -set prefix ../data/jan07_smallr -mcmc > data/syncsr.out 2>&1 &

covar_test1:
	mpirun --allow-run-as-root \
		-np 1 ./rs_emu -threads 1 \
		-set w_tl_prof 0 -set file_update_time 3600 \
		-set verbose 2 \
		-make_gaussian gprtest_thinned data/gprtestcorr

covar_test2:
	mpirun --allow-run-as-root \
		-np 1 ./rs_emu -threads 1 \
		-set w_tl_prof 0 -set file_update_time 3600 \
		-set verbose 2 \
		-make_gaussian ../data/emulated/jan_emu_uncorr_3_out ../data/uncorr_jan_3

emu_aff:
	-mpirun --allow-run-as-root \
		-np 1 ./rs_emu -threads 1 \
		-emu-aws "../data/aff_init3" \
		-set w_tl_prof 1 -set file_update_time 600 \
		-set aff_inv 1 -set n_walk 1289 -set step_fac 2.0 \
		-set mcmc_verbose 1 -set verbose 1 -set max_iters 1 \
		-initial-point-last ../data/aff_test_init \
		-set prefix ../data/emu_aff_run_2 -mcmc > emu.out 2>&1 &

emu_verify:
	-mpirun --allow-run-as-root \
		-np 1 ./rs_emu -threads 1 \
		-emu-aws "../data/emu_july" -set trmod emulator_gpr \
		-set w_tl_prof 1 -set file_update_time 600 \
		-set aff_inv 1 -set n_walk 35 -set step_fac 2.0 \
		-set mcmc_verbose 1 -set verbose 1 -set max_iters 1 \
		-initial-point-last ../data/emu_kde_aff \
		-set prefix ../data/gpr_verify -mcmc 

kde2:
	-mpirun --allow-run-as-root \
		-np 1 ./rs_emu -threads 1 \
		-set w_tl_prof 1 -set file_update_time 1 \
		-set aff_inv 0 -set n_walk 1 -set ptype kde \
		-set mcmc_verbose 2 -set verbose 1 -set max_time 600 \
		-initial-point-best ../data/kde_jun 1.0e-6 5 \
		-set prefix ../data/kde_jun_test -mcmc

kde1:
	-mpirun --allow-run-as-root \
		-np 4 ./rs_emu -threads 1 \
		-set w_tl_prof 1 -set file_update_time 1800 \
		-set aff_inv 0 -set n_walk 1 -set ptype kde \
		-set mcmc_verbose 2 -set verbose 1 -set max_time 432000 \
		-initial-point-last "../data/kde_aug_<rank>_out" \
		-set prefix ../data/kde_aug_2 -mcmc > kde2.out 2>&1 &

aff_run:
	-mpirun --allow-run-as-root \
		-np 4 ./rs_emu -threads 1 \
		-emu-aws "../data/aff_init3" \
		-set w_tl_prof 1 -set file_update_time 1800 \
		-set aff_inv 1 -set n_walk 50 -set step_fac 2.0 \
		-set mcmc_verbose 2 -set verbose 1 -set max_time 86400 \
		-initial-point-last "../data/jul30_<rank>_out" \
		-set prefix ../data/sep_12_dtr -mcmc > aff.out 2>&1 &


cv_run:
	mpirun --allow-run-as-root \
		-np 1 ./rs_emu -threads 1 \
		-set w_tl_prof 1 -set file_update_time 600 \
		-set aff_inv 1 -set n_walk 86 \
		-set mcmc_verbose 2 -set verbose 1 -set max_iters 1 \
		-initial-point-last "../data/andrew/kde_andrew3" \
	    -set prefix ../data/andrew/kde_andrew_cv -mcmc > data/cv.out 2>&1 &

ligo_fix:
	mpirun --allow-run-as-root \
		-np 1 ./rs_emu -threads 1 \
		-set w_tl_prof 1 -set file_update_time 600 \
		-set aff_inv 1 -set n_walk 87 \
		-set mcmc_verbose 2 -set verbose 1 -set max_iters 1 \
		-initial-point-last "../data/andrew_init1" \
	    -set prefix ../data/kde_andrew_lf -mcmc > cv.out 2>&1 &

sync:
	mpirun -np 1 \
		./rs_emu -threads 1 \
		-emu-aws "../data/aff-run-24/aff_run_0_out" \
		-set w_tl_prof 1 -set file_update_time 1 \
		-set mcmc_verbose 0 -set verbose 1 -set max_iters 1 \
		-set aff_inv 1 -set n_walk 1284 \
		-initial-point-last  ../data/emu_init \
	  -set prefix ../data/gpr_nov -mcmc 

bmark:
	mpirun -np 1 ./rs_emu -threads 1 \
		-set file_update_time 1800 \
		-set mcmc_verbose 2 -set verbose 1 -set max_iters 1 \
		-initial-point-best ../data/emu_init 1.0e-6 5 \
	  -set prefix ../data/bmark -mcmc

test_threads:
	mpirun -np 1 ./test -threads 2 

cv_test:
	mpirun --allow-run-as-root \
		-np 1 ./rs_emu -threads 1 \
		-set w_tl_prof 1 -set file_update_time 1800 \
		-set mcmc_verbose 2 -set verbose 2 -set max_iters 1 \
		-set ptype -step -set step_fac 2.0 \
		-initial-point-best ../data/kde_run/all/cv_all \
	    -set prefix ../data/cv_1e4 -mcmc 

cv_1:
	rm -f CVINS_0_0, CVINS_0_1, CVINS_0_2, CVINS_0_3, CVINS_0_4, CVINS_0_5, \
		CVINS_1_0, CVINS_1_1, CVINS_1_2, CVINS_1_3, CVINS_1_4, CVINS_1_5, \
		CVINS_2_0, CVINS_2_1, CVINS_2_2, CVINS_2_3, CVINS_2_4, CVINS_2_5, \
	./slice 

prex_run:
	./prex_nompi_omp -threads 1 -set ptype step -set step_fac 2.0 \
	-set file_update_time 1800 \
	-set mcmc_verbose 1 -set verbose 2 -set max_time  500 \
	-set prefix ../data/prex -mcmc > prex.scr 2>&1 &

#KDE CV runs
kde_cv:
	-mpirun --allow-run-as-root \
		-np 1 ./rs_emu -threads 1 \
		-set w_tl_prof 1 -set file_update_time 1800 \
		-set aff_inv 1 -set n_walk 247 \
		-set mcmc_verbose 1 -set verbose 1 -set max_iters 1 \
		-initial-point-last \
		"../data/kde_run/all/cv_all_init" \
		-set prefix ../data/kde_run/all/cv_1e4 -mcmc > data/kde.out 2>&1 &

nsatmos_MR.o2: empty
	rm -f ../qlmxb/M30_echi_atmos.o2
	o2graph -read ../aws/echi/nsatmos.o2 \
		-index \
		-delete-rows "N<40000" \
		-select-rows "gm>1" \
		-to-hist-2d r gm 18 18 \
		-to-table3d r gm like \
		-set interp_type 1 -refine 4 \
		-set obj_name rescaled \
		-internal ../qlmxb/M30_echi_atmos.o2 \
		-to-hist-2d like \
		-contours frac 0.68 ../qlmxb/M30_echi_atmos.o2 c68 \
		-contours frac 0.90 ../qlmxb/M30_echi_atmos.o2 c90 \
		-contours frac 0.99 ../qlmxb/M30_echi_atmos.o2 c99 
	o2graph -read ../qlmxb/M30_echi_atmos.o2 rescaled -den-plot like \
		-read ../qlmxb/M30_echi_atmos.o2 c68 -plot lw=2,color=white \
		-read ../qlmxb/M30_echi_atmos.o2 c90 -plot lw=2,color=white,ls=-- \
		-read ../qlmxb/M30_echi_atmos.o2 c99 -plot lw=2,color=white,ls=: \
		-show

nsx_MR.o2: empty
	rm -f ../qlmxb/M30_echi_x.o2
	o2graph -read ../aws/echi/nsx.o2 \
		-index \
		-delete-rows "N<40000" \
		-select-rows "gm>1" \
		-to-hist-2d r gm 18 18 \
		-to-table3d r gm like \
		-set interp_type 1 -refine 4 \
		-set obj_name rescaled \
		-internal ../qlmxb/M30_echi_x.o2 \
		-to-hist-2d like \
		-contours frac 0.68 ../qlmxb/M30_echi_x.o2 c68 \
		-contours frac 0.90 ../qlmxb/M30_echi_x.o2 c90 \
		-contours frac 0.99 ../qlmxb/M30_echi_x.o2 c99 
	o2graph -read ../qlmxb/M30_echi_x.o2 rescaled -den-plot like \
		-read ../qlmxb/M30_echi_x.o2 c68 -plot lw=2,color=white \
		-read ../qlmxb/M30_echi_x.o2 c90 -plot lw=2,color=white,ls=-- \
		-read ../qlmxb/M30_echi_x.o2 c99 -plot lw=2,color=white,ls=: \
		-show

j0740.o2: 
	o2graph -read ../NICERxXMM/J0740_H_MR_t3d.o2 table3d \
		-contours frac 0.68 ../NICERxXMM/J0740_H_MR_t3d.o2 c68 \
		-contours frac 0.90 ../NICERxXMM/J0740_H_MR_t3d.o2 c90 \
		-contours frac 0.99 ../NICERxXMM/J0740_H_MR_t3d.o2 c99 
	o2graph -read ../NICERxXMM/J0740_H_MR_t3d.o2 table3d -den-plot prob \
		-read ../NICERxXMM/J0740_H_MR_t3d.o2 c68 -plot lw=2,color=white \
		-read ../NICERxXMM/J0740_H_MR_t3d.o2 c90 -plot lw=2,color=white,ls=-- \
		-read ../NICERxXMM/J0740_H_MR_t3d.o2 c99 -plot lw=2,color=white,ls=: \
		-show

M30_He:
	o2graph -set colbar 1 -set verbose 1 -create table3d \
		R "grid:5,15.83,10.83/99" M "grid:0.5,2.5,2/99" \
		Rinfty "if(2*1.48*M/R>1,1e3,R/sqrt(1-2*1.48*M/R))" \
		-function "1.48*M/R" beta -function \
		"exp(-(Rinfty-11.5)^2/2/2.5^2)" temp1 \
		-function "1/(1+exp((beta-0.263)*30))" temp2 \
		-function "temp1*temp2" prob \
		-den-plot prob -contours 0.5 prob -plot "lw=2,c=w" -show 

# 7.5 to 8.9
# 1.2 at 8.2 is 0.217
M30_H:
	o2graph -set colbar 1 -set verbose 1 -create table3d \
		R "grid:5,15.83,10.83/99" M "grid:0.5,2.5,2/99" \
		Rinfty "if(2*1.48*M/R>1,1e3,R/sqrt(1-2*1.48*M/R))" \
		-function "1.48*M/R" beta -function \
		"exp(-(Rinfty-8.2)^2/2/0.7^2)" temp1 \
		-function "1/(1+exp((beta-0.217)*30))" temp2 \
		-function "temp1*temp2" prob \
		-den-plot prob -show

check_corr:
	acol -read ../data/kde_faja \
		-select log10_Tcn k_maxn del_kn log10_Tcp k_maxp del_kp \
		b c c_r m_s zeta c_s c_w b1 xi a1 a2 a3 a4 a5 a6 \
		b2 b3 alpha Q M_chirp q mf_J0822 eta_J0822 \
		mf_J085201 eta_J085201 mf_J104608 eta_J104608 \
		mf_1E1207 eta_1E1207 mf_J141256 eta_J141256 \
		t_J141256 mf_J160103 eta_J160103 mf_J1713 \
		eta_J1713 mf_J172054 eta_J172054 mf_J173203 \
		eta_J173203 mf_J181852 eta_J181852 mf_J185238 \
		eta_J185238 mf_J232327 eta_J232327 mf_J0205 \
		eta_J0205 mf_J0357 eta_J0357 mf_J0538 eta_J0538 \
		mf_J0554 eta_J0554 t_J0554 mf_J061705 eta_J061705 \
		mf_J0633_06 eta_J0633_06 t_J0633_06 mf_J0633_17 \
		t_J0633_17 mf_B0656 mf_B0833 eta_B0833 mf_B0950 \
		mf_B1055 t_B1055 mf_J1357 eta_J1357 t_J1357 \
		mf_B1706 eta_B1706 t_B1706 mf_J1740 t_J1740 \
		mf_J1741 t_J1741 mf_B1822 t_B1822 mf_B1823 \
		eta_B1823 t_B1823 mf_B1951 mf_J1957 eta_J1957 \
		t_J1957 mf_J2021 eta_J2021 t_J2021 mf_B2334 \
		eta_B2334 mf_J0726 t_J0726 mf_J1119 eta_J1119 \
		mf_B1509 eta_B1509 t_B1509 mf_J1718 eta_J1718 \
		t_J1718 mf_J0720 mf_J1308 eta_J1308 mf_J1605 eta_J1605 \
		mf_J1856 eta_J1856 mf_J0007 eta_J0007 mf_B1727 \
		eta_B1727 mf_B0114 t_B0114 mf_6304 eta_6304 \
		mf_6397 eta_6397 mf_M13 eta_M13 mf_M28 eta_M28 \
		mf_M30 eta_M30 mf_wCen eta_wCen mf_X7 eta_X7 \
		mf_0030 eta_0030 mf_J0740 eta_J0740 mf_1608 eta_1608 \
		mf_1659 eta_1659 mf_6440 eta_6440 mf_1709 eta_1709 \
		mf_00291 eta_00291 mf_1808 eta_1808 mf_1905 \
		eta_1905 log10_mdot_1905 \
		-internal ../data/kde_input_all

