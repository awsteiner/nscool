#ifndef _SXRT_H_
#define _SXRT_H_

#include <vector>
#include <string>

#include <o2scl/misc.h>

typedef struct {
  /// Name
  std::string name;
  /// Accretion rate (\f$ \mathrm{M}_{\odot}/\mathrm{yr} \f$)
  double mdot;
  /// Luminosity (\f$ \mathrm{erg}/\mathrm{s} \f$)
  double L;
  /// True if the accretion rate is an upper limit
  bool mdot_ul;
  /// True if the luminosity is an upper limit
  bool L_ul;
} sxrt;

class sxrt_data {

public:

  /// List of SXRTs
  std::vector<sxrt> list;
  
  sxrt_data() {
    list.push_back({"1608",3.6e-10,5.3e33,false,false});
    list.push_back({"1659",1.7e-10,2.0e32,false,false});
    list.push_back({"6440",1.8e-10,3.4e32,false,false});
    list.push_back({"1709",1.8e-10,2.2e33,false,false});
    list.push_back({"00291",2.5e-12,1.9e32,false,false});
    list.push_back({"1808",9.0e-12,4.9e30,false,true});
    list.push_back({"1905",1.1e-10,1.0e31,true,true});
  };

  /** \brief Add the necessary parameters for the INS list 
      to the \c pvi object

      This function adds <tt>Mns_name</tt> for the mass for each star,
      <tt>eta_name</tt> for the envelope composition for each star,
      <tt>mdot_name</tt> for each star which has only an accretion
      rate upper limit, and <tt>L_name</tt> for each star which only
      has a luminosity upper limit.
   */
  void init_data(o2scl::vec_index &pvi,
		 std::vector<std::string> &param_units);
  
  /// Copy constructor 
  sxrt_data(const sxrt_data &sd) {
    list = sd.list;
  }

  /// Copy constructor with <tt>operator=()</tt>
  const sxrt_data &operator=(const sxrt_data &sd) {
    if (this != &sd) {
      list = sd.list;
    }
    return *this;
  }
  
};

#endif
