#include <iostream>
#include <string>
#include <cmath>
#include <vector>

using namespace std;

typedef struct{
    std::string name;
}nsmr;

int main(){
vector<nsmr> list;

std::string data_files[16]={"../qlmxb/6304_H_nopl_syst_wilm.o2",
				"../qlmxb/6304_He_nopl_syst_wilm.o2",
				"../qlmxb/6397_H_syst_wilm.o2",
				"../qlmxb/6397_He_syst_wilm3.o2",
				"../qlmxb/M13_H_rs.o2",
				"../qlmxb/M13_He_rs.o2",
				"../qlmxb/M28_H_syst_wilm.o2",
				"../qlmxb/M28_He_syst_wilm.o2",
				"../qlmxb/M30_H_syst_wilm.o2",
				"../qlmxb/M30_He_syst_wilm.o2",
				"../qlmxb/wCen_H_syst_wilm.o2",
				"../qlmxb/wCen_H_syst_wilm.o2",
				"../qlmxb/X7_H_syst_wilm.o2",
				"../qlmxb/X7_He_syst_wilm.o2",
				"../nicer/0030_st_pst.o2",
				"../nicer/0030_st_pst.o2"};

//std::size_t pos = data_files[4].find("_");
//std::string str2 = data_files[4].substr (6,pos-6);
 const size_t sz= sizeof(data_files)/sizeof(data_files[0]);

for (size_t j=0;j<sz;j+=2){ 
    size_t pos = data_files[j].find("_");
    string str= data_files[j].substr(9,pos-9);
    cout << j << " " << str <<  endl;
}


return 0;
}