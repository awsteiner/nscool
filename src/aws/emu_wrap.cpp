#define PY_SSIZE_T_CLEAN
#include <python3.8/Python.h>
#include <o2scl/table.h>
#include <o2scl/misc.h>
#include <o2scl/tensor_grid.h>
#include <o2scl/hdf_eos_io.h>
#include <boost/numeric/ublas/vector.hpp>

using namespace std;
using namespace o2scl;
using namespace o2scl_hdf;
using namespace o2scl_const;

// Ublas vector typedef
typedef boost::numeric::ublas::vector<double> ubvector;

PyObject *train_modFile, *train_Xnames, *train_Ynames, 
         *train_xdata, *train_ydata, *tr_x, *tr_y,
         *train_trainClass, *train_pArgs, *train_trainMthd,
         *train_instance;

int train(std::string file_name) {

  // Load up the training data file
	hdf_file hf;
	hf.open(file_name);
	table_units<> tt;
  std::string namex;
  hdf_input(hf,tt,namex);
  hf.close();

  // Remove all the redundent points
  cout << "Before: " << tt.get_nlines() << endl;
  tt.delete_rows_func("log_wgt< -300");
  cout << "After: " << tt.get_nlines() << endl;

	// Load all the X and Y into a vector to be used later
	vector<string> pnames,pnames_X,pnames_Y;

	for(size_t i=5;i<tt.get_ncolumns();i++) {
      pnames.push_back(tt.get_column_name(i));
	}

  // Fill up the X and Y vectors names
  for(size_t i=0;i<pnames.size();i++){
    if(i<152){
      pnames_X.push_back(pnames[i]);
    } else {
      pnames_Y.push_back(pnames[i]);
    }
  }

  // Set up the training data sets
  train_Xnames = PyList_New(pnames_X.size());
  train_Ynames = PyList_New(pnames_Y.size());
  tr_x = PyList_New(tt.get_nlines());
  tr_y = PyList_New(tt.get_nlines());
  train_xdata = PyList_New(pnames_X.size());
  train_ydata = PyList_New(pnames_Y.size());

  // Fill up the python X train set with names and data
  for(size_t i=0;i<pnames_X.size();i++){
    PyList_SetItem(train_Xnames, i,
        PyUnicode_FromString(pnames_X[i].c_str()));
    for(size_t j=0;j<tt.get_nlines();j++){
      PyList_SetItem(tr_x, j, PyFloat_FromDouble(tt[i+5][j]));
    }
    //PyObject_Print(tr_x, stdout, Py_PRINT_RAW);
    //cout << endl;
    PyList_SetItem(train_xdata, i, tr_x);
  } 
  

  // Fill up the python Y train set with names and data
  for(size_t i=0;i<pnames_Y.size();i++){
    PyList_SetItem(train_Ynames, i,
        PyUnicode_FromString(pnames_Y[i].c_str()));
    for(size_t j=0;j<tt.get_nlines();j++){
      PyList_SetItem(tr_y, j, PyFloat_FromDouble(tt[i+5+pnames_X.size()][j]));
    }
    PyList_SetItem(train_ydata, i, tr_y);
  }

	string train_file = file_name;
	string test_file = "test_data";

  train_modFile = PyImport_ImportModule("emu");
	if (train_modFile == 0) {
    PyErr_Print();
    std::exit(1);
  	}
	cout << "Module imported" << endl;

	// Python class object
	train_trainClass = PyObject_GetAttrString(train_modFile, "modGpr");
  	assert(train_trainClass != 0);
	cout << "train_trainClass done" << endl;

	// Create an instance of the modGpr class
  	if(PyCallable_Check(train_trainClass)) {
    	train_instance = PyObject_CallObject(train_trainClass, 0);
  	}
  	assert(train_instance != 0);
	cout << "train_instance" << endl;
	
	// Python arguments for the modGpr::modTrain() function
  	train_pArgs = PyTuple_Pack(4, 
            PyUnicode_FromString(train_file.c_str()),
			     train_Xnames, train_Ynames, 1.0);
	
	cout << "train_pArgs done" << endl;

  	train_trainMthd = PyObject_GetAttrString(train_instance, "modTrain");
	cout << "train_trainMthd done" << endl;
  
  	// Call Python training function
  	if (PyCallable_Check(train_trainMthd)) {
    PyObject_CallObject(train_trainMthd, train_pArgs);
	cout << "pythoncall done" << endl;
  	}
	
	return 0;
}

int emu_points(std::vector<std::string> &sv){

  if(sv.size()<2){
    cout << "Need an emulated output filename." << endl;
  }
  if(sv.size()<3){
    cout << "Need an posterior output filename." << endl;
  }
  string emu_file = sv[1];
  string post_out = sv[2];
  
  // Initial row number
  size_t init_row = 0;
  if(sv.size()<4){
    cout << "Computing postesrior from the first row." << endl;
  }else{
  	init_row = o2scl::stoszt(sv[3]);
  	cout << "Computing postesrior from row number " << init_row << endl;
  }

  // Read emulated file to table
  o2scl::table_units<> emu_init_table;
  o2scl::table_units<> out_table;      
  hdf_file hf_emu;
  hf_emu.open(emu_file);
  size_t n_params=149;
  //hf_emu.get_szt("nparam",this->n_params);
  hdf_input(hf_emu,emu_init_table,"markov_chain_0");
  hf_emu.close();
  cout << "Emulated file copied to table" << endl;

  double log_wgt;
  ubvector emu_pars(n_params);

  // Open or create output file
  hdf_file hf_out;

  size_t nrows = emu_init_table.get_nlines();

  int pthread=0;
  bool set_col =false;
  std::clock_t start_time = std::clock();

  
  // Check start time, which can be used to update file after some interval
  // cout << "Start time : " << start_time << endl;
  
  for(size_t i=init_row; i<nrows; i++){

    //cout << "working on row : " << i  << endl;

    // copy parameter values to use in bamr_class::compute_point()
    for(size_t j=5;j<5+n_params;j++) {
      emu_pars(j-5) = emu_init_table.get(emu_init_table.get_column_name(j), i);
    }

    // Compute point success status
    //size_t iret = bc.compute_point(emu_pars, scr_out, log_wgt, test_point);

    //if(iret==0){
	//cout << "emulation failed" << endl;
      //}else{
      //continue;
    //}
  }
  
  hf_out.open_or_create(post_out);
  hdf_output(hf_out, out_table, "emulated");
  hf_out.close();
  
  return 0;
}

int main(int argc, char *argv[]){ 
  
  // Initialize Python
  Py_Initialize(); 
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append('./')");
	train("../data/temp_0_out1");
	//emu_points(pnames);
	Py_Finalize();

	return 0;
}