#----------------------------------------------------------------------
# Organize data files by date

MONTH = $(shell date +'%m')
DAY = $(shell date +'%d')
YEAR = $(shell date +'%y')

# ----------------------------------------------------------------------
# User-specific makefiles
# ----------------------------------------------------------------------

#include makefile.aws

# ----------------------------------------------------------------
# User-specific settings
# ----------------------------------------------------------------
# LIBS is the list of libraries
# LFC is the local fortran compiler
# LCXX is the local C++ compiler
# LCFLAGS are the local C++ compiler flags
# LFFLAGS are the local fortran compiler flags
# FLIBS are fortran system libraries

LIBS = -lo2scl -lhdf5 -lgsl \
	-lreadline -lpython3.13
FLIBS = -lgfortran
# PLIBS = -L/usr/lib/x86_64-linux-gnu/ 
LCXX = g++
LFC = gfortran
LMPI_FC = mpif90
LMPI_CXX = ccache mpic++
LCFLAGS = -I/usr/lib/python3.13/site-packages/numpy/_core/include \
	-DNO_MPI -DNO_OPENMP -DO2SCL_PYTHON \
	-I/usr/include/python3.13 \
	-Wall -Wno-unused -Wshadow -Wno-deprecated-declarations \
	-DO2SCL_NO_BOOST_MULTIPRECISION
LCFLAGS_OMP = -DNO_MPI -DO2SCL_PYTHON \
	-fopenmp \
	-I/usr/include/python3.13 \
	-Wall -Wno-unused -Wshadow -Wno-deprecated-declarations \
	-DO2SCL_NO_BOOST_MULTIPRECISION
LFFLAGS = -O3
LMPI_CFLAGS = -I/usr/lib/python3.13/site-packages/numpy/_core/include \
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

OBJS  = precool.o conductivity.o conductivity_core.o \
	opacity.o neutrino.o neutrino_core.o neutrino_crust.o \
	spec_heat.o density.o tc.o tc_Ioffe.o \
	Tools.o conductivity_crust.o NSCool.o

OBJS_NOMPI  = precool_nompi.o conductivity_nompi.o \
	conductivity_core_nompi.o \
	opacity_nompi.o neutrino_nompi.o neutrino_core_nompi.o \
	neutrino_crust_nompi.o \
	spec_heat_nompi.o density_nompi.o tc_nompi.o tc_Ioffe_nompi.o \
	Tools_nompi.o conductivity_crust_nompi.o  NSCool_nompi.o

nscool.o: nscool.cpp  
	$(LMPI_CXX) $(LMPI_CFLAGS) -o nscool.o -c nscool.cpp 

nscool_nompi.o: nscool.cpp 
	$(LCXX) $(LCFLAGS) -o nscool_nompi.o -c nscool.cpp 

nscool_wrap.o: nscool_wrap.cpp  
	$(LMPI_CXX) $(LMPI_CFLAGS) -o nscool_wrap.o -c nscool_wrap.cpp 

nscool_wrap_nompi.o: nscool_wrap.cpp 
	$(LCXX) $(LCFLAGS) -o nscool_wrap_nompi.o -c nscool_wrap.cpp 

fnscool.o: f_Nscool.cpp 
	$(LMPI_CXX) $(LMPI_CFLAGS) -o fnscool.o -c f_Nscool.cpp  

fnscool: nscool.o fnscool.o
	$(LMPI_CXX) $(LMPI_CFLAGS) -o fnscool nscool.o \
	fnscool.o $(LIBS) $(FLIBS)

#f_density.o: f_density.cpp
#	$(LMPI_CXX) $(LMPI_CFLAGS) -o f_density.o -c f_density.cpp
#
#density: density.o f_density.o
#	$(LMPI_CXX) $(LMPI_CFLAGS) -o density density.o f_density.o \
#	$(OBJS) $(LIBS) $(FLIBS)
#
#test_conduct.o: test_conduct.cpp
#	$(LCXX) $(LCFLAGS) -o test_conduct.o -c test_conduct.cpp
#
#test_conduct: conductivity_core.o conductivity_crust.o conductivity.o test_conduct.o 
#	$(LMPI_CXX) $(LMPI_CFLAGS) -o test_conduct test_conduct.o \
#	conductivity_core.o conductivity_crust.o conductivity.o $(LIBS) $(FLIBS)
#
#test_neutrino.o: test_neutrino.cpp
#	$(LCXX) $(LCFLAGS) -o test_neutrino.o -c test_neutrino.cpp
#
#test_neutrino: neutrino_core.o neutrino_crust.o neutrino.o Tools.o test_neutrino.o 
#	$(LMPI_CXX) $(LMPI_CFLAGS) -o test_neutrino test_neutrino.o \
#	neutrino_core.o neutrino_crust.o neutrino.o Tools.o $(LIBS) $(FLIBS)
#
#test_density.o: f_density.cpp
#	$(LMPI_CXX) $(LMPI_CFLAGS) -o test_density.o -c f_density.cpp
#
#test_density: density.o test_density.o
#	$(LMPI_CXX) $(LMPI_CFLAGS) -o test_density test_density.o \
#	density.o $(LIBS) $(FLIBS)
#
#test_tc.o: test_tc.cpp
#	$(LCXX) $(LCFLAGS) -o test_tc.o -c test_tc.cpp
#
#test_tc: tc.o test_tc.o 
#	$(LMPI_CXX) $(LMPI_CFLAGS) -o test_tc test_tc.o \
#	tc.o $(LIBS) $(FLIBS)
#
#test_precool.o: test_precool.cpp
#	$(LMPI_CXX) $(LMPI_CFLAGS) -o test_precool.o -c test_precool.cpp
#
#test_precool: $(OBJS) nscool_wrap.o test_precool.o 
#	$(LMPI_CXX) $(LMPI_CFLAGS) -o test_precool test_precool.o \
#	$(OBJS) nscool_wrap.o $(LIBS) $(FLIBS)

test.o: test.cpp
	$(LMPI_CXX) $(LMPI_CFLAGS) -o test.o -c test.cpp

test: $(OBJS) nscool_wrap.o test.o
	$(LMPI_CXX) $(LMPI_CFLAGS) -o test test.o \
	$(OBJS) nscool_wrap.o $(LIBS) $(FLIBS)

clean:
	-rm -f *.o test fnscool

# ----------------------------------------------------------------
# Main object files
# ----------------------------------------------------------------

NSCool.o: NSCool.f nscool_wrap.h
	$(LMPI_FC) $(LFFLAGS) -c NSCool.f -o NSCool.o
precool.o: precool.f
	$(LMPI_FC) $(LFFLAGS) -c precool.f -o precool.o
conductivity.o: conductivity.f
	$(LMPI_FC) $(LFFLAGS) -c conductivity.f -o conductivity.o
conductivity_core.o: conductivity_core.f
	$(LMPI_FC) $(LFFLAGS) -c conductivity_core.f -o conductivity_core.o
conductivity_crust.o: conductivity_crust.f
	$(LMPI_FC) $(LFFLAGS) -c conductivity_crust.f -o conductivity_crust.o
opacity.o: opacity.f
	$(LMPI_FC) $(LFFLAGS) -c opacity.f -o opacity.o
neutrino.o: neutrino.f
	$(LMPI_FC) $(LFFLAGS) -c neutrino.f -o neutrino.o
neutrino_core.o: neutrino_core.f
	$(LMPI_FC) $(LFFLAGS) -c neutrino_core.f -o neutrino_core.o
neutrino_crust.o: neutrino_crust.f
	$(LMPI_FC) $(LFFLAGS) -c neutrino_crust.f -o neutrino_crust.o
spec_heat.o: spec_heat.f
	$(LMPI_FC) $(LFFLAGS) -c spec_heat.f -o spec_heat.o
density.o: density.f
	$(LMPI_FC) $(LFFLAGS) -c density.f -o density.o
tc.o: tc.f
	$(LMPI_FC) $(LFFLAGS) -c tc.f -o tc.o
tc_Ioffe.o: tc_Ioffe.f
	$(LMPI_FC) $(LFFLAGS) -c tc_Ioffe.f -o tc_Ioffe.o
Tools.o: Tools.f
	$(LMPI_FC) $(LFFLAGS) -c Tools.f -o Tools.o

# ----------------------------------------------------------------
# Main object files (no MPI version)
# ----------------------------------------------------------------

NSCool_nompi.o: NSCool.f nscool_wrap.h
	$(LFC) $(LFFLAGS) -c NSCool.f -o NSCool_nompi.o
precool_nompi.o: precool.f
	$(LFC) $(LFFLAGS) -c precool.f -o precool_nompi.o
conductivity_nompi.o: conductivity.f
	$(LFC) $(LFFLAGS) -c conductivity.f -o conductivity_nompi.o
conductivity_core_nompi.o: conductivity_core.f
	$(LFC) $(LFFLAGS) -c conductivity_core.f -o conductivity_core_nompi.o
conductivity_crust_nompi.o: conductivity_crust.f
	$(LFC) $(LFFLAGS) -c conductivity_crust.f -o conductivity_crust_nompi.o
opacity_nompi.o: opacity.f
	$(LFC) $(LFFLAGS) -c opacity.f -o opacity_nompi.o
neutrino_nompi.o: neutrino.f
	$(LFC) $(LFFLAGS) -c neutrino.f -o neutrino_nompi.o
neutrino_core_nompi.o: neutrino_core.f
	$(LFC) $(LFFLAGS) -c neutrino_core.f -o neutrino_core_nompi.o
neutrino_crust_nompi.o: neutrino_crust.f
	$(LFC) $(LFFLAGS) -c neutrino_crust.f -o neutrino_crust_nompi.o
spec_heat_nompi.o: spec_heat.f
	$(LFC) $(LFFLAGS) -c spec_heat.f -o spec_heat_nompi.o
density_nompi.o: density.f
	$(LFC) $(LFFLAGS) -c density.f -o density_nompi.o
tc_nompi.o: tc.f
	$(LFC) $(LFFLAGS) -c tc.f -o tc_nompi.o
tc_Ioffe_nompi.o: tc_Ioffe.f
	$(LFC) $(LFFLAGS) -c tc_Ioffe.f -o tc_Ioffe_nompi.o
Tools_nompi.o: Tools.f
	$(LFC) $(LFFLAGS) -c Tools.f -o Tools_nompi.o
