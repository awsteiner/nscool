#ifndef EMU__H
#define EMU__H

#include <iostream>
#include <string>
#include <cmath>

#include <boost/numeric/ublas/matrix.hpp>

#include <gsl/gsl_combination.h>

#include <o2scl/err_hnd.h>
#include <o2scl/vector.h>
#include <o2scl/vec_stats.h>
#include <o2scl/linear_solver.h>
#include <o2scl/columnify.h>
#include <o2scl/table.h>

#include <Python.h>

using namespace std;

namespace o2scl{
  template<class vec_t=ubvector, class vec2_t=ubvector>
  class emulator_py {

    protected:
      PyObject *emu_python; PyObject *mod_python; 
      PyObject *i_python; PyObject *eval_python;
    public:
      
      size_t n_out;

      emulator_py(){
        emu_python=NULL;
        mod_python=NULL;
        i_python=NULL;
        eval_python=NULL;
        }

      void set(string cls_name, size_t n_in,
        vector<string> list){

          if (!Py_IsInitialized()) {
          PyRun_SimpleString("print 'inital error!' ");
          return;
        }

        emu_python = PyImport_ImportModule("emu");
        mod_python = PyObject_GetAttrString(emu_python, cls_name.c_str());
        i_python = PyObject_CallObject(mod_python, 0);

        PyObject *py_nd_in = PyLong_FromSize_t(n_in);

        PyObject *py_param_list = PyList_New(list.size());
        for (size_t i=0;i<list.size();i++) {
          PyList_SetItem(py_param_list, i, PyUnicode_FromString(list[i].c_str()));
        }

        PyObject *set_python = PyObject_GetAttrString(i_python, "set");
        PyObject *info_python = PyTuple_Pack(2,py_nd_in, py_param_list);

        PyObject_CallObject(set_python, info_python);
        n_out = list.size()-n_in;
        eval_python = PyObject_GetAttrString(i_python, "eval"); 
        PyErr_Print();  
        return;
      }

      int eval_unc(const vec_t &p, double &log_wgt,
                 double &log_wgt_unc, std::vector<double> &dat, std::vector<double> &dat_unc){

        PyObject *py_p = PyList_New(p.size());
        for (size_t i=0;i<p.size();i++) {  
          PyList_SetItem(py_p, i, PyFloat_FromDouble(p[i]));
        }
        PyObject *result =  PyObject_CallObject(eval_python, PyTuple_Pack(1, py_p));

        if (result == NULL) {
            std::cout << "no result" << std::endl;
            for (size_t i=0; i < n_out; i++) {
              dat[i]=0;
              dat_unc[i]=0;
            }
        } else {
          // result and std_dev are stored in alternate order in the pyobject result_python
          for (size_t i=0; i<n_out; i++) {
            dat[i] = PyFloat_AsDouble(PyList_GetItem(result, static_cast<int>(2*i)));
            dat_unc[i]= PyFloat_AsDouble(PyList_GetItem(result, static_cast<int>(2*i+1)));
            //std::cout << dat[i] << ", " << dat_unc[i] << std::endl;
          }
        }
        log_wgt=dat[0];
        log_wgt_unc=dat_unc[0];
        Py_DECREF(py_p);Py_DECREF(result);
        PyErr_Print();
        //Py_Finalize();
        return 0;
      }  
    };
}
#endif