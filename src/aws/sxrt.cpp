#include "sxrt.h"

using namespace std;
using namespace o2scl;

void sxrt_data::init_data(vec_index &pvi,
			  std::vector<std::string> &param_units) {
  
  for (size_t i=0;i<list.size();i++){
    sxrt &c=list[i];
    pvi.append(((string) "mf_") + c.name);
    param_units.push_back("");
    pvi.append(((string) "eta_") + c.name);
    param_units.push_back("");
    if (c.mdot_ul) {
      pvi.append(((string) "log10_mdot_") + c.name);
      param_units.push_back("Msun/yr");
    }
    /*if (c.L_ul) {
      pvi.append(((string) "L_") + c.name);
      param_units.push_back("ergs/s");
    }*/
  }
  return;
}
