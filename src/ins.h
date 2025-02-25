#ifndef _INS_H_
#define _INS_H_

#include <vector>
#include <string>
#include <o2scl/misc.h>

typedef struct {
  /// Name 
  std::string name;
  /// Index of the star from Potekhin et al. (2020)
  int num;      
  /// Spin down age of the star (yr)
  double tc;    
  /// Lower limit for tk (yr)
  double tk_lo; 
  /// Kinetic age of the Star (yr)
  double tk;    
  /// Upper limit for tk (yr)
  double tk_hi; 
  /// Lower limit for L (10^33 ergs/s)
  double L_lo;  
  /// Luminosity (*10^33 ergs/s)
  double L;     
  /// Upper limit for L (10^33 ergs/s)
  double L_hi;  
  /// Lower limit for k_B*T (in MeV)
  double kT_lo; 
  /// k_B*T of the star (in MeV)
  double kT;    
  /// Upper limit for k_B*T (in MeV)
  double kT_hi; 
  /// Atmosphere parameter, false for eta=0 and true to allow eta to vary
  bool eta;
  /// Rescaled age 
  double t_hat;
  /// Rescaled age uncertainty
  double dt_hat;
  /// Rescaled luminosity
  double L_hat;
  /// Rescaled luminosity uncertainty
  double dL_hat;
} cooling_ns;

class ins_data {

public:
  
  ins_data();

  /// List of INSs
  std::vector<cooling_ns> list;
  
  /** \brief Add the necessary parameters for the INS list 
      to the \c pvi object

      This function adds <tt>Mns_name</tt> for the mass for each star,
      <tt>eta_name</tt> for each star which has eta=true
      <tt>t_name</tt> for each star which has only an age upper limit,
      and <tt>L_name</tt> for each star which only has a luminosity
      upper limit.
   */
  void init_data(o2scl::vec_index &pvi,
		 std::vector<std::string> &param_units);

  // Copy constructor ins_data id1(id2);
  ins_data(const ins_data &id) {
    list = id.list;
  }

  // Copy constructor ins_data id1=id2;
  const ins_data &operator=(const ins_data &id) {
    if (this != &id) {
      list = id.list;
    }
    return *this;
  }
};

#endif
