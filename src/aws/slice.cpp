#include <vector>
#include <o2scl/table.h>
#include <o2scl/table3d.h>
#include <o2scl/hdf_file.h>
#include <o2scl/hdf_io.h>
#include "data_eval.h"


using namespace std;
using namespace o2scl;
using namespace o2scl_hdf;

// Place to try out new functions, maybe add to nscool_wrap later

// Adding separate functions as loops inside this function, maybe 
// make them as separate functions later.
void slice(string fname) {
    string name;
    table3d tl_prof;

    // Currently set only one bool to be true when running as the code 
    // writes into a single file, will change later when breaking into
    // separate functions
    bool rplot=true;
    bool yplot=false;
    bool lum_plot=false;
    
    bool ins_plot=true; // if not true, plot sxrt
    bool yinterp_plot=true;

    // Read the tl_prof data from file
    hdf_file hf;
    hf.open(fname);
    hdf_input(hf,tl_prof,name);
    hf.close();
    hf.open_or_create("new_file");
    ubvector xgrid=tl_prof.get_x_data();
    ubvector ygrid=tl_prof.get_y_data();

//----------------------------------------------------------------------------
    // This loop below computes all the neutrino luminosities at the surface 
    // of the star and puts them in a table lum_tl along with T_inf and M_dot 
    if (lum_plot){
        table_units<> lum_tl;
        lum_tl.line_of_names(((std::string)"Mdot Tinf Tinf_0 lmurca_nucl ")+
                             "lbrem_nucl lplasma lnpb lpbf_n1s0 lpbf_n3p2 "+
                             "lpbf_p1s0 lnu lphot ltot");

        for (size_t j=0;j<ygrid.size();j++){
        double lmurca_nucl=0;
        double lbrem_nucl=0;
        double lplasma=0;
        double lnpb=0;
        double lpbf_n1S0=0;
        double lpbf_n3P2=0;
        double lpbf_p1S0=0;
        double lnu=0;
        for (size_t i=1;i<=xgrid.size();i+=2){
            // Get the volume and redshift information from file
            double ephi=tl_prof.get(i,j,"ephi");
            double vol=tl_prof.get(i,j,"vol");

            // Get the emmissivities from the file
            double qmurca_nucl=tl_prof.get(i,j,"qmurca_nucl");
            double qbrem_nucl =tl_prof.get(i,j,"qbrem_nucl");
            double qplasma    =tl_prof.get(i,j,"qplasma");
            double qnpb       =tl_prof.get(i,j,"qnpb");
            double qpbf_n1S0  =tl_prof.get(i,j,"qpbf_n1s0");
            double qpbf_n3P2  =tl_prof.get(i,j,"qpbf_n3p2");
            double qpbf_p1S0  =tl_prof.get(i,j,"qpbf_p1s0");
            double qnu        =tl_prof.get(i,j,"qnu");

            // Compute the luminosities
            double e2p=ephi*ephi;
            lmurca_nucl=lmurca_nucl+qmurca_nucl*vol*e2p;
            lbrem_nucl =lbrem_nucl +qbrem_nucl *vol*e2p;
            lplasma    =lplasma    +qplasma    *vol*e2p;
            lnpb       =lnpb       +qnpb       *vol*e2p;
            lpbf_n1S0  =lpbf_n1S0  +qpbf_n1S0  *vol*e2p;
            lpbf_n3P2  =lpbf_n3P2  +qpbf_n3P2  *vol*e2p;
            lpbf_p1S0  =lpbf_p1S0  +qpbf_p1S0  *vol*e2p;
            lnu        =lnu        +qnu        *vol*e2p;
        }
        double lphot=tl_prof.get(xgrid.size()-1,j,"Linf")*3.826e33;
        //add ephi surface and remove redshift from temp
        double ep_surf=tl_prof.get(xgrid.size()-1,j,"ephi");
        double Tinf=tl_prof.get(xgrid.size()-1,j,"Tinf");
        double Tinf_0=Tinf*ep_surf;
        double lnu_tot=lmurca_nucl+lbrem_nucl+lplasma+lnpb+lpbf_n1S0
                        +lpbf_n3P2+lpbf_p1S0;
        double ltot=lnu_tot+lphot;
        double line[13]={ygrid[j],Tinf,Tinf_0,lmurca_nucl,lbrem_nucl,lplasma,
                        lnpb,lpbf_n1S0,lpbf_n3P2,lpbf_p1S0,lnu,lphot,ltot};
        lum_tl.line_of_data(13,line);
            }
        hdf_output(hf,lum_tl,"lum_tl");
        hf.close();
    }
//-------------------------------------------------------------------------------------
    // This loop below collects the emissivities and computes luminosities of 
    // different processes vs the radius and stores them in the lum_tl table 
    // for a  particular mass eta and Mdot for an SXRT.

    if(rplot) {
        table_units<> lum_tl;
        lum_tl.line_of_names(((std::string)"rad Tinf Tinf_0 rho ")+
                             "qmurca_nucl qbrem_nucl qplasma qnpb "+
                             "qpbf_n1s0 qpbf_n3p2 qpbf_p1s0 "+
                             "qdurca_np qdurca_lap qdurca_smn "+
                             "qdurca_smla qdurca_sms0 qdurca qnu "+
                             "lmurca_nucl lbrem_nucl lplasma lnpb "+
                             "lpbf_n1s0 lpbf_n3p2 lpbf_p1s0 "+
                             "ldurca_np ldurca_lap ldurca_smn "+
                             "ldurca_smla ldurca_sms0 ldurca lnu "+
                             "lphot ltot");

        // Choose here the value of Mdot
        size_t j=15;

        double lmurca_nucl=0;
        double lbrem_nucl=0;
        double lplasma=0;
        double lnpb=0;
        double lpbf_n1S0=0;
        double lpbf_n3P2=0;
        double lpbf_p1S0=0;

        double ldurca_np=0;
        double ldurca_lap=0;
        double ldurca_smn=0;
        double ldurca_smla=0;
        double ldurca_sms0=0;

        double ldurca=0;
        double lnu=0;

        for (size_t i=1;i<=xgrid.size();i+=2){
            // Get the volume and redshift information from file
            double ephi=tl_prof.get(i,j,"ephi");
            double vol=tl_prof.get(i,j,"vol");

            //get tc and rho information
            //double tcn=tl_prof.get(i,j,"tcn");
            //double tcp=tl_prof.get(i,j,"tcp");
            double rho=tl_prof.get(i,j,"rho");

            // Get the emmissivities from the file
            double qmurca_nucl=tl_prof.get(i,j,"qmurca_nucl");
            double qbrem_nucl =tl_prof.get(i,j,"qbrem_nucl");
            double qplasma    =tl_prof.get(i,j,"qplasma");
            double qnpb       =tl_prof.get(i,j,"qnpb");
            double qpbf_n1S0  =tl_prof.get(i,j,"qpbf_n1s0");
            double qpbf_n3P2  =tl_prof.get(i,j,"qpbf_n3p2");
            double qpbf_p1S0  =tl_prof.get(i,j,"qpbf_p1s0");
            double qnu        =tl_prof.get(i,j,"qnu");

            // Get the durca emmissivities
            double qdurca_np  =tl_prof.get(i,j,"qdurca_np");
            double qdurca_lap =tl_prof.get(i,j,"qdurca_lap");
            double qdurca_smn =tl_prof.get(i,j,"qdurca_smn");
            double qdurca_smla=tl_prof.get(i,j,"qdurca_smla");
            double qdurca_sms0=tl_prof.get(i,j,"qdurca_sms0");

            // Total durca emmissivity
            double qdurca=qdurca_np+qdurca_lap+qdurca_smn
                            +qdurca_smla+qdurca_sms0;

            // Compute the luminosities
            double e2p=ephi*ephi;
            lmurca_nucl=lmurca_nucl+qmurca_nucl*vol*e2p;
            lbrem_nucl =lbrem_nucl +qbrem_nucl *vol*e2p;
            lplasma    =lplasma    +qplasma    *vol*e2p;
            lnpb       =lnpb       +qnpb       *vol*e2p;
            lpbf_n1S0  =lpbf_n1S0  +qpbf_n1S0  *vol*e2p;
            lpbf_n3P2  =lpbf_n3P2  +qpbf_n3P2  *vol*e2p;
            lpbf_p1S0  =lpbf_p1S0  +qpbf_p1S0  *vol*e2p;

            // Compute the durca luminosities
            ldurca_np  =ldurca_np  +qdurca_np  *vol*e2p;
            ldurca_lap =ldurca_lap +qdurca_lap *vol*e2p;
            ldurca_smn =ldurca_smn +qdurca_smn *vol*e2p;
            ldurca_smla=ldurca_smla+qdurca_smla*vol*e2p;
            ldurca_sms0=ldurca_sms0+qdurca_sms0*vol*e2p;

            // Total durca luminosity
            ldurca=ldurca+ldurca_np+ldurca_lap+ldurca_smn
                    +ldurca_smla+ldurca_sms0;
                    
            // Compute the total neutrino luminosity
            lnu        =lnu        +qnu        *vol*e2p;

            double lphot=tl_prof.get(i,j,"Linf")*3.826e33;
            double Tinf=tl_prof.get(i,j,"Tinf");
            // Add ephi and remove redshift from temp
            double Tinf_0=Tinf*ephi;
            // Total neutrino luminosity
            double lnu_tot=lmurca_nucl+lbrem_nucl+lplasma+lnpb+lpbf_n1S0
                            +lpbf_n3P2+lpbf_p1S0+ldurca;
            // Total luminosity
            double ltot=lnu_tot+lphot;
            double line[34]={xgrid[i],Tinf,Tinf_0,log10(rho),
                        qmurca_nucl,qbrem_nucl,qplasma,qnpb,
                        qpbf_n1S0,qpbf_n3P2,qpbf_p1S0,
                        qdurca_np,qdurca_lap,qdurca_smn,
                        qdurca_smla,qdurca_sms0,qdurca,qnu,
                        lmurca_nucl,lbrem_nucl,lplasma,lnpb,
                        lpbf_n1S0,lpbf_n3P2,lpbf_p1S0,
                        ldurca_np,ldurca_lap,ldurca_smn,
                        ldurca_smla,ldurca_sms0,ldurca,lnu,
                        lphot,ltot};
            lum_tl.line_of_data(34,line);
        }
        hdf_output(hf,lum_tl,"lum_tl");
        hf.close();
    }
//--------------------------------------------------------------------------------
// This loop interpolates from the data in tl_prof to have a higher 
// resolution in the y-axis for the density plots 
if (yinterp_plot) {
    o2scl::table3d tl_prof3;
    int ysize=ygrid.size();
    // Set the new ygrid size
    int ysize_new=ysize*2;
    ubvector ygrid_new(ysize_new);
    //cout << "ygrid_new size "<< ysize_new << endl;
    if(ins_plot){
        o2scl::uniform_grid<double>
        tygrid=o2scl::uniform_grid_log_end<double>(log10(ygrid[0]),log10(ygrid[ysize-1]),
                ysize_new-1);
        for(size_t k=0;k<((size_t)ysize_new);k++){
            ygrid_new.insert_element(k,pow(10,tygrid[k]));
            //cout << ygrid_new[k] << endl;
        }

    }else{
    o2scl::uniform_grid<double>
        tygrid=o2scl::uniform_grid_end<double>(ygrid[0],ygrid[ysize-1],
                ysize_new-1);
        for(size_t k=0;k<((size_t)ysize_new);k++){
            ygrid_new.insert_element(k,tygrid[k]);
            //cout << ygrid_new[k] << endl;
        }
    }
    if(ins_plot){
        tl_prof3.set_xy("r",xgrid.size(),xgrid,"t",ysize_new,ygrid_new);
    }else{
        tl_prof3.set_xy("r",xgrid.size(),xgrid,"Mdot",ysize_new,ygrid_new);
    }
    tl_prof3.set_interp_type(o2scl::itp_nearest_neigh);
    //tl_prof3.set_interp_type(o2scl::itp_linear);
    for(size_t k=0;k<tl_prof.get_nslices();k++) {
      std::string sl_name=tl_prof.get_slice_name(k);
      if (sl_name!="qmax") {
	tl_prof3.add_slice_from_table(tl_prof,sl_name,sl_name);
      }
    }
    hdf_output(hf,((const o2scl::table3d &)(tl_prof3)),"tl_prof");
    hf.close();
}
//--------------------------------------------------------------------------------
    return;
}

void qdpto2(string name, string fname, bool H){
    //string name;

    ifstream myfile;
    myfile.open(name);

    hdf_file hf;
    hf.open_or_create(fname);
    table3d MR_data;
    o2scl::uniform_grid<double> rgrid_temp;
    o2scl::uniform_grid<double> mgrid_temp;
    if (H){
        rgrid_temp=o2scl::uniform_grid_end<double>(5.5,16,20);
        mgrid_temp=o2scl::uniform_grid_end<double>(0.5,2.2,20);      
    }else{
        rgrid_temp=o2scl::uniform_grid_end<double>(5.5,12,20);
        mgrid_temp=o2scl::uniform_grid_end<double>(0.5,1.8,20);
    }
    MR_data.set_xy("M",21,mgrid_temp,
                    "R",21,rgrid_temp);
    MR_data.line_of_names("chi2");
    int k=0;
    string line;
    double chi2_min=200;

    // Get the data from cory's table and put that into a 
    // table3d object

    while (std::getline(myfile, line)) {
        k+=1;
        if(k>=5){
        for(size_t i=0;i<21;i++) {
            //std::cout << l << "\n"; 
            //for(size_t j=0;j<21;j++) {
                size_t pos = line.find(" ");
                string str=line.substr(0,pos);
                line=line.substr(pos+1);
                double l=o2scl::stod(str);
                //cout << k << " " << i << " " << l << endl; 
                MR_data.set(k-5,i,"chi2",l);
                if(l<chi2_min){
                chi2_min=l; 
                }
            //}
        }          
      }
    }
    MR_data.add_constant("chi2_min",chi2_min);

    // Swap the M and R axis to use in our code

    table3d swap;
    swap.set_xy("R",21,rgrid_temp,
                        "M",21,mgrid_temp);
    swap.line_of_names("chi2 like");
    double c2m=MR_data.get_constant("chi2_min");
    cout << "Chi2_min: " << c2m << endl;
    for(size_t i=0;i<21;i++){
        for(size_t j=0;j<21;j++){
            double chi2=MR_data.get(j,i,"chi2");
            swap.set(i,j,"chi2",chi2);
            double like=exp(-chi2/2+c2m/2);
            //cout << "like: " << like << endl;
            swap.set(i,j,"like",like);
        }
    }

    // Interpolate to a higher number of grids

    table3d rescaled;
    size_t bin_sz=21;
    o2scl::uniform_grid<double> rgrid, mgrid;
    if (H){
        rgrid=o2scl::uniform_grid_end<double>(5.5,16,bin_sz);
        mgrid=o2scl::uniform_grid_end<double>(0.5,2.2,bin_sz);      
    }else{
        rgrid=o2scl::uniform_grid_end<double>(5.5,12,bin_sz);
        mgrid=o2scl::uniform_grid_end<double>(0.5,1.8,bin_sz);
    }

    rescaled.set_xy("R",bin_sz+1,rgrid,
                        "M",bin_sz+1,mgrid);
    rescaled.line_of_names("chi2 like");
    for(size_t i=0;i<bin_sz+1;i++){
        for(size_t j=0;j<bin_sz+1;j++){
            double chi2=swap.interp(rgrid[i],mgrid[j],"chi2");
            rescaled.set(i,j,"chi2",chi2);
            double like=swap.interp(rgrid[i],mgrid[j],"like");
            rescaled.set(i,j,"like",like);
        }
    }

    hdf_output(hf,((const o2scl::table3d &)(rescaled)),"rescaled");
    hf.close();
    return;
}

void J0740_data(string fname, bool H) {
    o2scl_hdf::hdf_file hf;
    std::string name;
    o2scl::table<> tab;
    hf.open(fname);
    hdf_input(hf,tab,name);
    hf.close();
    o2scl::table3d t3d;
    o2scl::hist_2d MR_hist;

    MR_hist.from_table(tab,"radius","mass",30,30);
    MR_hist.copy_to_table3d(t3d,"radius","mass","prob");
    
    // Interpolate to a higher number of grids

    const ubvector rgrid_temp=t3d.get_x_data();
    const ubvector mgrid_temp=t3d.get_y_data();

    size_t bin_sz=30;
    o2scl::uniform_grid<double> rgrid, mgrid;
    
    rgrid=o2scl::uniform_grid_end<double>(rgrid_temp[0],
                                        rgrid_temp[29],bin_sz);
    mgrid=o2scl::uniform_grid_end<double>(mgrid_temp[0],
                                        mgrid_temp[29],bin_sz); 
    
    o2scl::table3d table3d;
    table3d.set_xy("R",bin_sz+1,rgrid,
                        "M",bin_sz+1,mgrid);
    table3d.line_of_names("prob"); 

    for(size_t i=0;i<bin_sz+1;i++){
        for(size_t j=0;j<bin_sz+1;j++){
            double prob=t3d.interp(rgrid[i],mgrid[j],"prob");
            table3d.set(i,j,"prob",prob);
        }
    } 

    if(H){    
        hf.open_or_create("../NICERxXMM/J0740_H_MR_t3d.o2");
        hdf_output(hf,((const o2scl::table3d &)(table3d)),"table3d");
        hf.close();
    } else {
        hf.open_or_create("../NICERxXMM/J0740_He_MR_t3d.o2");
        hdf_output(hf,((const o2scl::table3d &)(table3d)),"table3d");
        hf.close();
    }
}

void int_cv(std::string fname, bool ins){

    string name;
    table3d tl_prof;

    // Read the tl_prof data from file
    hdf_file hf;
    hf.open(fname);
    hdf_input(hf,tl_prof,name);
    hf.close();
    hf.open_or_create("int_cv");
    ubvector xgrid=tl_prof.get_x_data();
    ubvector ygrid=tl_prof.get_y_data();

    cout << xgrid.size() << " " << ygrid.size() << endl;

    table_units<> cv_table;
    cv_table.line_of_names(((std::string)"time cv cv_n cv_p cv_e ")+
                            "cv_m cv_la cv_sm cv_s0 sv_sp cv_q");
    
    for (size_t j=0;j<ygrid.size();j++){

        double int_cv=0;

        double int_cv_n=0;
        double int_cv_p=0;
        double int_cv_e=0;
        double int_cv_m=0;

        double int_cv_la=0;
        double int_cv_sm=0;
        double int_cv_s0=0;
        double int_cv_q=0;

        for (size_t i=1;i<=xgrid.size();i+=2){
            // Get the volume and redshift information 
            // from file
            double ephi=tl_prof.get(i,j,"ephi");
            double vol=tl_prof.get(i,j,"vol");

            // Get the specific heats from the file
            double cv=tl_prof.get(i,j,"cv");
            double cv_n=tl_prof.get(i,j,"cv_n");
            double cv_p=tl_prof.get(i,j,"cv_p");
            double cv_e=tl_prof.get(i,j,"cv_e");
            double cv_m=tl_prof.get(i,j,"cv_m");

            double cv_la=tl_prof.get(i,j,"cv_la");
            double cv_sm=tl_prof.get(i,j,"cv_sm");
            double cv_s0=tl_prof.get(i,j,"cv_s0");
            double cv_q=tl_prof.get(i,j,"cv_q");

            // Compute the integrated luminosities
            int_cv=int_cv+cv*vol;
            int_cv_n=int_cv_n+cv_n*vol;
            int_cv_p=int_cv_p+cv_p*vol;
            int_cv_m=int_cv_m+cv_m*vol;
            int_cv_e=int_cv_e+cv_e*vol;

            int_cv_la=int_cv_la+cv_la*vol;
            int_cv_sm=int_cv_sm+cv_sm*vol;
            int_cv_s0=int_cv_s0+cv_s0*vol;
            int_cv_q=int_cv_q+cv_q*vol;
        }
        double ephi_surf=tl_prof.get(xgrid.size()-1,j,"ephi");
        double T_surf=ephi_surf*tl_prof.get(xgrid.size()-1,j,"Tinf");

        int_cv=int_cv/T_surf;
        int_cv_n=int_cv_n/T_surf;
        int_cv_p=int_cv_p/T_surf;
        int_cv_m=int_cv_m/T_surf;
        int_cv_e=int_cv_e/T_surf;

        int_cv_la=int_cv_la/T_surf;
        int_cv_sm=int_cv_sm/T_surf;
        int_cv_s0=int_cv_s0/T_surf;
        int_cv_q=int_cv_q/T_surf;

        double line[10]={ygrid[j],int_cv,int_cv_n,int_cv_p,int_cv_m,
                        int_cv_e,int_cv_la,int_cv_sm,int_cv_s0,int_cv_q};
        cv_table.line_of_data(10,line);
        //cout << ygrid[j] << " " << int_cv << endl;    
    }
    hdf_output(hf,cv_table,"cv_table");
    hf.close();
}


int main(int argc, char *argv[]){
    //slice("INS_0_2");
    qdpto2("../qlmxb/M_R_Hatm_20210114.qdp","../qlmxb/M30_Hatm.o2",1);
    qdpto2("../qlmxb/M_R_Heatm_20210114.qdp","../qlmxb/M30_Heatm.o2",0);
    J0740_data("J0740/STU/NICERxXMM/FI_H/run10/J0740_H_table.o2",1);
    //J0740_data("J0740/STU/NICERxXMM/FI_He/run10/J0740_He_table.o2",0);
    //int_cv("INS_0_2",1);
    return 0;
}
