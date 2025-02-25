#include "ins.h"
#include <fstream>

using namespace std;
using namespace o2scl;

ins_data::ins_data() {
  // name, num, tc, tk_lo, tk, tk_hi
  // L_lo, L_, L_hi, kT_lo, kT, kT_hi, eta

  // NS 1E0102 was taken out on 2/12/21 since Yakovlev's website says
  // it's not a NS
  // list.push_back({"1E0102",1, 0.0, 1500, 2100, 2700, 6, 11,
  // 27, 224, 259, 302, true});
  
  // Updated last on 11/28/22
  //  J1740 changed from 0.09,0.19,0.5, 56,67,78
  //  J1856 tc changed from 3.76e6 to 3.72e6
  //  J173203 changed from 13.5, 17.4,19.3, 151, 153, 157
  //  J232327 is changed from 320.4, 329.4, 338.4,
  //                6.1, 0.0,9.4, 123, 0.0, 185
 
  // Weakly-magnetized thermal emitters
  list.push_back({"J0822",2,2.54e8,3700,4450,5200,
      3.9,5,6.1,240,0.0,475,true});
  list.push_back({"J085201",3,0.0,2400,0.0,5100,
      1,2,3,80,90,100,true});
  list.push_back({"J104608",4,0.0,1.1e4,0.0,3e4,
      0.077,0.0,0.62,40,0.0,70,true});
  list.push_back({"1E1207",5,3.01e8,2333,7000,21000,
      1.15,1.31,1.8,90,0.0,250,true});
  list.push_back({"J141256",6,2.85e5,0.0,0.0,0.0,
      1.5, 0.0,9,101,0.0,110,true});
  list.push_back({"J160103",7,0.0,600,800,1000,
      4.4,4.9,6.3,125,133,147,true});
  list.push_back({"J1713",8,0.0,1608,0.0,1609,
      1.7,2.0,2.3,128,136,145,true});
  list.push_back({"J172054",9,0.0,600,0.0,700,
      10.8,11.8,12.8,156,162,168,true});
  list.push_back({"J173203",10,0.0,2000,0.0,6000,
      7.8,10.8,14.5,150,156,162,true});
  list.push_back({"J181852",11,0.0,2700,3400,6000,
      4.2,8.4,15.2,110,130,150,true});
  list.push_back({"J185238",12,1.92e8,3200,6000,7800,
      8.4,10.4,12.8,132,133,134,true});
  list.push_back({"J232327",13,0.0,319.5,326.2,338.7,
      5.6,0.0,9.3,120,0.0,128,true});
  // Ordinary pulsars
  list.push_back({"J0205",14,5.37e3,819.4,819.5,819.6,
      0.08,0.19,0.34,43,49,54,true});
  list.push_back({"J0357",15,5.41e5,2.e5,0.0,1.3e6,
      0.004,0.015,0.04,30,36,45,true});
  list.push_back({"J0538",16,6.20e5,4e4,0.0,1.1e5,
      0.63,1.09,1.36,86,91,96,true});
  list.push_back({"J0554",61,5.17e4,0.0,0.0,0.0,
      0.14,0.19,0.27,45,48,51,true});
  list.push_back({"J061705",17,0.0,2.5e4,3e4,3.5e4,
      0.25,0.26,0.27,58,58.4,59,true});
  list.push_back({"J0633_06",18,5.92e4,0.0,0.0,0.0,
      0.08,0.2,0.8,49,53,57,true});
  list.push_back({"J0633_17",19,3.43e5,0.0,0.0,0.0,
      0.049,0.088,0.109,40,42,44,false});
  list.push_back({"B0656",20,1.11e5,3.1e5,3.5e5,4.0e5,
      0.32,0.35,0.38,67,68,69,false});
  list.push_back({"B0833",21,1.13e4,1.7e4,0.0,2.7e4,
      0.412,0.424,0.436,56,57,60,true});
  list.push_back({"B0950",60,1.75e7,6.0e5,1.9e6,7.6e6,
      2.0e-5,8.0e-5,2.3e-4,5,7,11,false});
  list.push_back({"B1055",22,5.35e5,0.0,0.0,0.0,
      0.03,0.1,0.2,65,68,71,false});
  list.push_back({"J1357",23,7.31e3,0.0,0.0,0.0,
      0.29,0.36,0.43,60,64,68,true});
  list.push_back({"B1706",24,1.75e4,0.0,0.0,0.0,
      0.06,0.71,0.87,40,70,230,true});
  list.push_back({"J1740",25,1.14e5,0.0,0.0,0.0,
      0.09,0.11,0.16,70,0.0,140,false});
  list.push_back({"J1741",26,3.86e5,0.0,0.0,0.0,
      0.21,0.31,0.45,58,60,62,false});
  list.push_back({"B1822",27,2.33e5,0.0,0.0,0.0,
      0.017,0.026,0.038,79,83,87,false});
  list.push_back({"B1823",28,2.14e4,0.0,0.0,0.0,
      0.36,0.45,0.54,92,97,101,true});
  list.push_back({"B1951",30,1.07e5,4.6e4,6.4e4,8.2e4,
      0.07,0.18,0.48,110,130,150,false});
  list.push_back({"J1957",31,8.4e5,0.0,0.0,0.0,
      0.005,0.0,0.014,20.1,0.0,25.3,true});
  list.push_back({"J2021",32,1.72e4,0.0,0.0,0.0,
      0.35,0.5,0.73,58,63,69,true});
  list.push_back({"B2334",33,4.06e4,7700,0.0,2.0e4,
      0.012,0.047,0.082,30,46,52,true});
  
  // High-B pulsars
  list.push_back({"J0726",34,1.86e5,0.0,0.0,0.0,
      0.3,0.4,0.84,63,74,80,false});
  list.push_back({"J1119",35,1.6e3,4.2e3,0.0,7.1e3,
      1.1,1.9,3.8,80,0.0,210,true});
  list.push_back({"B1509",36,1.56e3,0.0,0.0,0.0,
      7,9,11,133,142,149,true});
  list.push_back({"J1718",37,3.32e4,0.0,0.0,0.0,
      0.2,0.4,0.9,57,0.0,200,true});
  
  // Magnificent Seven (we only include 4)
  list.push_back({"J0720",40,1.9e6,7e5,8.5e5,1e6,0.11,0.19,
      0.32,91.4,92.4,93.3,false});
  list.push_back({"J1308",42,1.46e6,3e5,5.5e5,8e5,0.26,0.33,
      0.38,50,0.0,90,true});
  list.push_back({"J1605",43,0.0,3.8e5,4.4e5,5.1e5,0.007,0.0,
      0.5,35,0.0,120,true});
  list.push_back({"J1856",44,3.72e6,3.4e5,4.2e5,5e5,0.05,0.0,
      0.08,36,0.0,63,true});
  
  // Neutron stars with upper limits on thermal emission
  list.push_back({"J0007",46,1.39e4,9100,9200,9300,
      0.0,0.0,0.03,0.0,0.0,200,true});
  list.push_back({"B1727",48,8.05e4,4.0e4,5.0e4,6.0e4,
      0.0,0.0,0.035,0.0,0.0,33,true});
  
  // Middle-aged pulsars with measured thermal emission of hot spots
  list.push_back({"B0114",51,2.75e5,0.0,0.0,0.0,
      0.0041,0.0044,0.0047,150,170,190,false});
};

void ins_data::init_data(vec_index &pvi,
			 std::vector<std::string> &param_units) {

#ifdef PLOT_SCRIPTS  
  ofstream fout;
  
  fout.open("o2g.scr");

  fout << "o2graph -set logx 1 -set logy 1 \\" << endl;
  fout << "-error-point 4500 1e+29 800 1000 3e28 3.55e28 "
       << "elinewidth=1.3,ecolor=orange,marker='^',"
       << "mfc=orange,mec=orange -text 12000 1e+29 TINS "
       << "color=orange,fontsize=30 \\" << endl;
  fout << "-error-point 4500 3.3e+28 800 1000 1e28 1.2e28 "
       << "elinewidth=1.3,ecolor=black,marker='o',mfc=black,mec=black "
       << "-text 12000 3.3e+28 PSR color=black,fontsize=30 \\" << endl;
  fout << "-error-point 4500 1e+28 800 1000 3e27 4.55e27 "
       << "elinewidth=1.3,ecolor=red,marker='v',mfc=red,mec=red "
       << "-text 12000 1e+28 XINS color=red,fontsize=30 \\" << endl;
#endif
  
  for (size_t i = 0; i < list.size(); i++) {

    cooling_ns &c=list[i];

    pvi.append(((string)"mf_")+c.name);
    param_units.push_back("");

    if (c.eta) {
      pvi.append(((string) "eta_") + c.name);
      param_units.push_back("");
    }

    //Ignoring T for now
    /*if (c.kT != 0.0)
      {	
      //check if KT avarage data is available.
      dl_T.push_back(0.5 * (log10(c.kT / 8.617e-5) - 5.0)); 
      //Adding logT data
      T1 = log10((c.kT_hi) / 8.617e-5);
      T2 = log10((c.kT_lo) / 8.617e-5);
      dl_dT.push_back(0.5 * ((T1 - T2) / 2.0)); //Adding dT data
      }
      else if (c.kT_lo != 0.0)
      { //if only upper and lower limits are available
      dl_T.push_back(0.5 * (log10((c.kT_hi + c.kT_lo) / 
      (2 * 8.617e-5)) - 5.0));
      T1 = log10((c.kT_hi) / 8.617e-5);
      T2 = log10((c.kT_lo) / 8.617e-5);
      dl_dT.push_back(0.5 * ((T1 - T2) / 2.0));
      }
      else
      {
      dl_T.push_back(0.5 * (log10(c.kT_hi / 8.617e-5) - 5.0));
      ;
      dl_dT.push_back(0.0);
      }
    */
    
    if (c.tk==0.0 && c.tk_lo==0.0) {
      pvi.append(((string) "t_")+c.name);
      param_units.push_back("K");
    }

    //if (c.L==0.0 && c.L_lo==0.0) {
    //  pvi.append(((string) "L_")+c.name);
    //  param_units.push_back("ergs/s");
    //}

    if (c.tk!=0.0) {
      // If the central value for the kinetic age is known, then use
      // it and the specified range to determine t_hat and dt_hat
      c.t_hat = 0.2*(log10(c.tk)-2.0);
      double t1 = log10(c.tk_hi);
      double t2 = log10(c.tk_lo);
      c.dt_hat = 0.2 * ((t1 - t2)/2.0);
    } else if (c.tk_lo != 0.0) {
      // Otherwise, if only a range is specified, then
      // use the range to determine t_hat and dt_hat
      double t1 = log10(c.tk_hi);
      double t2 = log10(c.tk_lo);
      c.t_hat = 0.2*(((t1 + t2)/2.0)-2.0);
      c.dt_hat = 0.2*((t1 - t2)/2.0);
    } else {
      // Otherwise, only the spin-down age is known and we will treat
      // it as an upper limit. Set dt_hat to zero and t_hat will be
      // set by a parameter in ins_like.cpp
      c.dt_hat=0.0;
    }

    if (c.L != 0.0) {
      // If a central value for the luminosity is given, then use it
      // and the specified range to determine L_hat and dL_hat
      c.L_hat = 0.25*(log10(1.0e33*c.L)-30.0); 
      double L1 = log10(1.0e33*(c.L_hi));
      double L2 = log10(1.0e33*(c.L_lo));
      c.dL_hat = 0.25*(L1 - L2)/2.0; 
    } else if (c.L_lo != 0.0) {
      // Otherwise, if only a range is specified, then use the
      // range to determine L_hat and dL_hat
      double L1 = log10(1.0e33*(c.L_hi));
      double L2 = log10(1.0e33*(c.L_lo));
      c.L_hat = 0.25*(((L1 + L2)/2.0)-30);
      c.dL_hat = 0.25*(L1 - L2)/2.0;
    } else {
      // Otherwise, only an L upper limit is known and L_hat will
      // be set by a parameter in ins_like.cpp
      c.L_hat=0.0;
      c.dL_hat=0.0;
    }
    
    //cout << "INS: " << this << " " << i << " " << c.tk << " " 
    //<< c.t_hat << endl;
    
#ifdef PLOT_SCRIPTS
    
    double xerrlo=0.0, xerrhi=0.0, yerrlo=0.0, yerrhi=0.0;
    double te_x,te_y;
    string ecolor, marker;
    bool xuplim=false, yuplim=false;

    if(c.num<14) {
      ecolor="orange";
      marker="^";
    } else if(c.num>13 && c.num<39) {
      ecolor="black";
      marker="o";
    } else {
      ecolor="red";
      marker="v";
    }

    fout << "-error-point ";
      
    if (c.tk!=0.0) {
      fout << c.tk << " ";
      xerrlo=c.tk-c.tk_lo;
      xerrhi=c.tk_hi-c.tk;
      te_x=1.3*c.tk;
    } else if (c.tk_lo != 0.0) {
      fout << (c.tk_lo + c.tk_hi) / 2.0 << " ";
      xerrlo=(c.tk_hi-c.tk_lo) / 2.0 ;
      xerrhi=(c.tk_hi-c.tk_lo) / 2.0 ;
      te_x=1.3*((c.tk_lo + c.tk_hi) / 2.0);
    } else {
      fout << c.tc << " ";
      te_x=1.3*c.tc;
      xuplim=true;
      xerrlo=c.tc*0.2;
      xerrhi=c.tc*0.2;
    }
    if (c.L != 0.0) {
      fout << c.L*1.0e33 << " ";
      te_y=1.3*(c.L)*1.0e33;
      yerrlo=c.L-c.L_lo;
      yerrhi=c.L_hi-c.L;
    } else if (c.L_lo != 0.0) {
      fout << (c.L_hi+c.L_lo)/2.0*1.0e33 << " ";
      te_y=1.3*(((c.L_hi+c.L_lo)/2.0))*1.0e33;
      yerrlo=(c.L_hi-c.L_lo) / 2.0 ;
      yerrhi=(c.L_hi-c.L_lo) / 2.0 ;
    } else {
      fout << c.L_hi*1.0e33 << " ";
      te_y=1.3*(c.L_hi)*1.0e33;
      yuplim=true;
      yerrlo=c.L_hi*0.35;
      yerrhi=c.L_hi*0.35;
    }

    fout << xerrlo << " " << xerrhi << " "
	 << yerrlo*1.0e33 << " " << yerrhi*1.0e33 << " ";
    if (xuplim) {
      if (yuplim) {
	fout << "xuplims=True,uplims=True,";
      } else {
	fout << "xuplims=True,";
      }
    } else {
      if (yuplim) {
	fout << "uplims=True,";
      } 
    }
    fout << "elinewidth=1.3,ecolor=" << ecolor << ",marker='"
	 << marker << "',mfc=" << ecolor << ",mec=" << ecolor
	 << ",capsize=4 -text " << te_x << " " << te_y << " "
	 << c.num << " color=" << ecolor << ",fontsize=30 \\" << endl;
#endif
  }

#ifdef PLOT_SCRIPTS
  fout << "-modax labelsize=30 \\" << endl;
  fout << "-ylimits 1.0e27 1.0e35 -xlimits 1.584893e+02 '2e7' \\" << endl;
  fout << "-set font 30 -xtitle '$t	[yr]$' -ytitle "
       << "'$L_{\\gamma}^{\\infty} [ergs/s]$' \\" << endl;
  fout << "-show" << endl;
    
  fout.close();
#endif
  
  return;
}
