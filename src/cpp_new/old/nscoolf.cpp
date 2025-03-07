#include <cmath>
#include <vector>
#include <iostream>
#include "precool.cpp"
#include "density.cpp"
#include "conductivity.cpp"
#include "opacity.cpp"
#include "neutrino.cpp"
#include "spec_heat.cpp"

class nscool {

  public:
    bool sxrt_mode;
    double eta,eta0,kfmax_n3p2,delkf_n3p2,tcmax_n3p2,tsurface,tdrip,tcore,naa;
    double sig,nu_e_s,nu_e_l,c_nu_str,p_nu_str;
    size_t mdot_index;
    int imax, idrip, ienv, isf, istop;
    std::vector<double> tprint,time_print;
    std::vector<double> rad,rrho,pres;
    std::vector<double> debar,dvol,emas,phi,rhod;

    std::vector<double> bar;
    std::vector<double> yneutr,yprot,yelect,ymuon;
    std::vector<double> ylambda,ysminus,yszero,ysplus;
    std::vector<double> yquarku,yquarkd,yquarks;
    std::vector<double> fhad,theta_k,theta_p;
    std::vector<double> a_cell,a_ion,z_ion,v_ion;

    std::vector<double> fp,fq,fr,fp1,fq1,fr1;
    std::vector<double> dfp,dfq,dfr,fa,fb,fc,ff,fj,fk;
    
    std::vector<double> f_gr_field,g_gr_field,h_gr_field;
    
    std::vector<double> cve,cvm,cvn,cvp,
                        cvla,cvsm,cvs0,cvsp,
                        cvqu,cvqd,cvqs;

    std::vector<double> mstp,mstn,mstla,mstsm,
                        msts0,mstsp,kfe,kfm,kfp,
                        kfn,kfla,kfsm,kfs0,kfsp,
                        kfqu,kfqd,kfqs;

    std::vector<int> idurca_np,idurca_lap,idurca_smn,
                    idurca_smla,idurca_sms0,idurca_quqd,
                    idurca_quqs;
    std::vector<double>  durca_ctrl_e,durca_ctrl_m,
                        durca_henon_e,durca_henon_m;

    std::vector<double> tcn,tcp,tcla,tcsm,
                      tcs0,tcsp,tcuu,tcdd,tcss,
                      tcud,tcus,tcds,tcu,tcd,
                      tcs,tcu1,tcu2,tcu3,tcd1,
                      tcd2,tcd3,tcs1,tcs2,tcs3;

    std::vector<double> orad,bar1,obar,rrho1,orrho;  
    std::vector<double> ephi,e2phi,a2ephin,dephi;  
    std::vector<double> temp,otemp,ntemp,ntemp1,delt,dtemp;
    std::vector<double> lum,olum,nlum,dell, dlum;
    
    std::vector<double> lambda,lambda1;
    std::vector<double> kappa,kappa1;
    std::vector<double> qnu,qnu1,qqq,qqq1;
    
    std::vector<double> qeebrem,qnpb,qplasma,qsynch,qbubble,qpair,
                      qphoto,qbrem_nn,qmurca_nucl,qbrem_nucl,
                      qmurca_hyp,qbrem_hyp,qdurca_np,qdurca_lap,
                      qdurca_smn,qdurca_smla,qdurca_sms0,qfast,
                      qdurca_q,qmurca_q,qpbf_n1s0,qpbf_n3p2,
                      qpbf_p1s0,qpbf_q;
    
    std::vector<double> heat,heat1;   
    std::vector<double> cv,cv1;
    std::vector<double> cv_n,cv_p,cv_e,cv_m,cv_l,cv_sm,
                      cv_s0,cv_sp,cv_q,cv_ion;
    std::vector<double> gamma,cryst;  
    std::vector<double> nbfield2;
    const double hbar = 1.054e-27, e = 4.803e-10, kb = 1.38e-16;
    const double g = 6.67e-8, c = 2.99792e10;
    const double msol = 1.989e33, lsol = 3.826e33;
    const double year = 3.1557600e7;
    const double pi = 3.1415926535;
    const int isize = 10000;
    int istrange,idt,istepmax,itrial_max,itrial_opt,istart;
    double htot,contraction,gammacryst,gammaliq;
    std::string version;
    double time0,timemax;
    double tcut,dtime,odtime,dtlimit,scale_dt0,scale_dt1,repeat;
    double mratt,mratl,mrats,tvar,svar,tcon,sxrt_Tinit;
    int pscreen,istep_debug,idump1,idump2,idump3,icvel_nodeg;
    double debug,pteff,ptemp_arg,pstar,tempmin,tempini,emnco,emncr,emp,p0,ptemp;
    double T_fact_drip, T_fact_surf;
    o2scl::table3d tl_prof;
    std::vector<double> temp_ygrid;
    int ifteff,sfn1s0,sfn3p2,sfp1s0,sfl1s0,fn1s0,fn3p2,fp1s0,fl1s0,sfquark;
    double eta_arg,mag_coeff,tb_acc0,eta_0,murca_increase;
    int inu_durca,inu_eion,inu_plasma,inu_synch,inu_n1s0_pbf,inu_n3p2_pbf,
        inu_p_pbf,inu_bubble,inu_photo,inu_pair,inu_nuts1,inu_nuts2,
        inu_nuts3,inu_nuts4,inu_nuts5;
    double rhoexo,cexo;
    int pexo,pexosn,pexosp,nonothing1,nonothing2,nonothing3,nonothing4;
    int iopacity,icon_crust,icon_core,iconnothing2,iconnothing3,iconnothing4,iconnothing5;
    double connothing1,connothing2,qimp,connothing4,connothing5;

    int i_heat_deep_crust,i_heat_deposit,i_dep,i_heat_convert,i_heat_vortex_creep;
    double t_dep,del_t_dep,total_heat,MeV_neutron;
    int i_heat_joule,i_heat_field_decay,i_heat_cold1,i_heat_cold2,i_heat_cold3,i_heat_cold4;
    double heat_cold1,heat_cold2,heat_cold3,heat_cold4;
    int ifield,i_gr_field,i0,i1;
    double bfield,j_44,nothing,start_b_defusion;
    int i_joule_heat,i_conb,i_eleconb,i_acc,icore,idec,itpmax;
    double m_dot0,t_acc0,t_acc1,t_acc2,alpha_acc,time_step_min,eta_Edd,X_Edd;
    double rhocore,rhodrip,rhoenv,rhosurf;
    nscool(){
      istrange=0;
      idt=1;
      htot=0.0;
      contraction=0.0;
      gammacryst=210.0;
      gammaliq=180.0;
      version = "NEW";
      // Read Numerical Parameters: nscool_num_param---------------------------
      time0=0.0;
      timemax=2.0e10;
      istepmax=1000000;
      itrial_max=20;
      itrial_opt=12;
      tcut=2.0;
      dtime=1.0e-12;
      dtlimit=3.15e15;
      scale_dt0=1.2;
      scale_dt1=1.5;
      repeat=0.2;
      istart=2;
      mratt=1.0e-12;
      mratl=1.0e-10;
      mrats=1.0e-10;
      tvar=1.20;
      svar=1.05;
      tcon=1.0e12;
      // ---------------------------------------------------------------------
      dtime=dtime*year;
      time0=time0*year;
      double odtime=dtime;
      double scale_dt=scale_dt1;
      // Initialize the cooling calculation: nscool_cool_param----------------
      pscreen=1;
      debug=1.0;
      istep_debug=0;
      pteff=0.0;
      ptemp_arg=1.0;
      pstar=0.0;
      idump1=1;
      idump2=111;
      idump3=421;
      tempmin=1.0e4;
      tempini=1.0e10;
      // If this is true, then the function 'cvelec()' is used
      // to handle the electron specific heat rather than
      // the simple expression for degenerate electrons
      icvel_nodeg=0;
      emnco=5.0;
      emncr=5.0;
      emp=3.0;
      p0=0.1;

      ptemp=0.0;
      // Setup time_print
      time_print.resize(25);
      time_print[0]=1.0e-10;
      time_print[1]=1.0e-4;
      time_print[2]=3.0e-4;
      time_print[3]=1.0e-3;
      time_print[4]=3.0e-3;
      time_print[5]=1.0e-2;
      time_print[6]=3.0e-2;
      time_print[7]=1.0e-1;
      time_print[8]=3.0e-1;
      time_print[9]=1.0e+0;
      time_print[10]=3.0e+0;
      time_print[11]=1.0e+1;
      time_print[12]=3.0e+1;
      time_print[13]=1.0e+2;
      time_print[14]=3.0e+2;
      time_print[15]=1.0e+3;
      time_print[16]=3.0e+3;
      time_print[17]=1.0e+4;
      time_print[18]=3.0e+4;
      time_print[19]=1.0e+5;
      time_print[20]=3.0e+5;
      time_print[21]=1.0e+6;
      time_print[22]=3.0e+6;
      time_print[23]=1.0e+7;
      time_print[24]=3.0e+7;
      T_fact_drip=0.8;
      T_fact_surf=0.5;
      eta=0.0;

      temp_ygrid.clear();
      for (size_t i=0;i<time_print.size();i++) {
        temp_ygrid.push_back(time_print[i]);
      }

      // The array 'tprint' the cooling code is actually zero-indexed,
      // but Dany's code ignores the first value.
      itpmax=time_print.size();
      for(int i=1;i<=itpmax;i++) {
        tprint[i]=time_print[i-1];
      }

      if (sxrt_mode) {
        tempini=sxrt_Tinit;
        timemax=1.5e-12;
      }
      //READ OUTER BOUNDARY PARAMETERS: Nscool_bound_param---------------------
      ifteff=3;
      eta_arg=eta;
      mag_coeff=3.0;
      tb_acc0=0.0;
      // ----------------------------------------------------------------------
      eta=eta/0.443;
      eta_0=eta;
      // READ PAIRING PARAMETERS: ---------------------------------------------
      sfn1s0=1;
      sfn3p2=101;
      sfp1s0=3;
      sfl1s0=0;
      fn1s0=1;
      fn3p2=1;
      fp1s0=1;
      fl1s0=1;
      sfquark=0;
      // READ NEUTRINO PARAMETERS: --------------------------------------------
      murca_increase=0.0;
      inu_durca=1;
      inu_eion=1;
      inu_plasma=1;
      inu_synch=0;
      inu_n1s0_pbf=1;
      inu_n3p2_pbf=1;
      inu_p_pbf=1;
      inu_bubble=0;
      inu_photo=0;
      inu_pair=0;
      inu_nuts1=0;
      inu_nuts2=0;
      inu_nuts3=0;
      inu_nuts4=0;
      inu_nuts5=0;
      rhoexo=1.2e25;
      cexo=1.0e25;
      pexo=0;
      pexosn=0;
      pexosp=0;
      nonothing1=0;
      nonothing2=0;
      nonothing3=0;
      nonothing4=0;
      // READ CONDUCTIVITY PARAMETERS: ----------------------------------------
      iopacity=0;
      icon_crust=3;
      icon_core=2;
      iconnothing2=0;
      iconnothing3=0;
      iconnothing4=0;
      iconnothing5=0;
      connothing1=0.0;
      connothing2=0.0;
      qimp=1.0e-10;
      connothing4=0.0;
      connothing5=0.0;
      // READ HEATING PARAMETERS: ---------------------------------------------
      i_heat_deep_crust=0;
      i_heat_deposit=0;
      t_dep=0.0;
      del_t_dep=0.0;
      total_heat=0.0;
      i_dep=0;
      i_heat_convert=0;
      MeV_neutron=0.0;
      i_heat_vortex_creep=0;
      j_44=0.0;
      i_heat_joule=0;
      i_heat_field_decay=0;
      i_heat_cold1=0;
      i_heat_cold2=0;
      i_heat_cold3=0;
      i_heat_cold4=0;
      heat_cold1=0.0;
      heat_cold2=0.0;
      heat_cold3=0.0;
      heat_cold4=0.0;
      // READ MAGNETIC FIELD PARAMETERS:--------------------------------------
      ifield = 0;
      i_gr_field = 1;
      bfield = 0.0e12;
      i0 = 113;
      i1 = 289;
      nothing = 0.1;
      start_b_defusion = 0.0;
      i_joule_heat = 0;
      i_conb = 0;
      i_eleconb = 0;
      // READ ACCRETION PARAMETERS:-------------------------------------------
      i_acc = 0;
      m_dot0 = 0.0;
      t_acc0 = 0.0;
      t_acc1 = 0.0;
      t_acc2 = 0.0;
      alpha_acc = 0.0;
      time_step_min = 0.0;
      eta_Edd = 0.0;
      X_Edd = 0.0;
      t_acc0 *= 3.15576e7;
      t_acc1 *= 3.15576e7;
      t_acc2 *= 3.15576e7;
      time_step_min *= 3.15576e7;
      eta = std::max(1.0e-50, eta);
      eta0 = std::max(1.0e-50, eta0);
      // READ STAR STRUCTURE LAYOUT: -----------------------------------------
      rhocore=1.6e14;
      rhodrip=4.0e11;
      rhoenv=1.0e8;
      rhosurf=1.0e10;
      icore=111;
      idec=60;
    }

    void NSCool(int irank,int iret,std::vector<double> neebrem_logt, 
            std::vector<double>neebrem_nalpha, std::vector<double> neebrem_n2,
            std::vector<double> sf_lgtau1,std::vector<double> sf_lgtau2,
            std::vector<std::vector<double>> sf_lgr, 
            std::vector<std::vector<double>> sf_lgr2){

      if (debug >= 1) 
        std::cout << "Initializing" << std::endl;
      // Get the time independent pieces of physics: ------------------------------
      grid(irank,idec,rhocore,rhodrip,rhoenv,rhosurf,
          imax,icore,idrip,ienv,rad,rrho,pres,dvol,emas,phi);

      get_core_chemistry(irank,version,imax,icore,rrho,
         bar,yneutr,yprot,yelect,ymuon,ylambda,ysminus,yszero,
         ysplus,yquarku,yquarkd,yquarks,theta_k,theta_p,fhad,
         mstn,mstp,mstla,mstsm,msts0,mstsp);

      if (icore==0) {
        iret=2;
        std::cout << 'Problem in get_core_chemistry().' << std::endl;
        return;
      }

      get_crust_chemistry(irank,debug,version,imax,icore,
          rrho,pres,debar,dvol,bar,a_cell,a_ion,z_ion,v_ion,
          yelect,yneutr);
      get_fermi_momenta(irank,imax,icore,rrho,bar,yneutr,
          yprot,yelect,ymuon,ylambda,ysminus,yszero,ysplus,yquarku,
          yquarkd,yquarks,fhad,theta_k,theta_p,
          kfn,kfp,kfe,kfm,kfla,kfsm,kfs0,kfsp,kfqu,kfqd,kfqs,
          idurca_np,idurca_lap,idurca_smn,idurca_smla,idurca_sms0,idurca_quqd,idurca_quqs,
          durca_ctrl_e,durca_ctrl_m,durca_henon_e,durca_henon_m);

      if (icore==0) {
        iret=4;
        std::cout << 'Problem in get_fermi_momenta().' << std::endl;
        return;
      }

      get_effective_masses(version,emnco,emncr,emp,
         kfn,kfp,mstn,mstp,mstla,mstsm,msts0,mstsp,idrip,icore);
      get_spec_heat_degenerate(cve,cvm,cvn,cvp,cvla,
         cvsm,cvs0,cvsp,cvqu,cvqd,cvqs,
         kfe,kfm,kfn,kfp,kfla,kfsm,kfs0,kfsp,kfqu,kfqd,kfqs,
         mstn,mstp,mstla,mstsm,msts0,mstsp,fhad,imax);
      get_Tc(irank,imax,icore,idrip,
          tcn,tcp,tcla,tcsm,tcs0,tcsp,
          tcuu,tcdd,tcss,tcud,tcus,tcds,
          tcu,tcd,tcs,
          sfn1s0,sfn3p2,sfp1s0,sfl1s0,
          fn1s0,fn3p2,fp1s0,fl1s0,
          kfmax_n3p2,delkf_n3p2,tcmax_n3p2,isf,
          kfn,kfp,kfla,kfqu,kfqd,kfqs,bar,fhad,yquarku,yquarkd,yquarks);
      get_degenerate_density(irank,rrho,pres,rhod,imax,ienv);
      // Calculate the T-independent coefficients: -------------------------------------
      if (debug >= 1) 
        std::cout << "Calculating T-independent coeff." << std::endl;

      for (int i=0;i<imax;++i) {
        ephi[i] = std::exp(phi[i]);
        e2phi[i] = ephi[i]*ephi[i];
        a2ephin[i] = std::pow(4.0*pi*rad[i]*rad[i],2)*ephi[i];
      }

      dephi[0]=0.0;
      for (int i=1;i<imax-1;++i) {
        dephi[i]=(ephi[i+1]-ephi[i-1])/(rad[i+1]-rad[i-1]);
      }
      dephi[imax-1]=dephi[imax-2];

      double radius=rad[imax-1];
      double root=std::sqrt(1.0-2.0*g*msol*emas[imax-1]/(radius*c*c));
      double factor=root/(4.0*pi*radius*radius)/(6.022e23)*1.0e39;
      double constant=4.0*pi*g*msol*emas[imax-1]*4.0/3.0*5.67e-5*e2phi[imax-1]/
                      pres[imax-1]/root;

      double gs=g*msol*emas[imax-1]/(radius*radius)/root;
      double gs14=gs/1.0e14;

      double compactness=2.0*g*msol*emas[imax-1]/(radius*c*c);
      // -----------------------------------------------------------------------------
      for (int i=imax;i>=1;i-=2) {
        nbfield2[i]=0.0;
      }
      // Calculate the initial Temp and Lum profiles ---------------------------------
      if (debug >= 1) 
        std::cout << "Calculating Initial T profile" << std::endl;
      // nscool_tptr_init ------------------------------------------------------------
      if (ifteff!=15) {
        if (tempini>0.0) {
          tsurface=T_fact_surf*ephi[imax]*tempini;
          tdrip=T_fact_drip*ephi[idrip]*tempini;
          tcore=1.0*ephi[0]*tempini;
        } else {
          tsurface=1.0e9;
          tdrip=2.0e10;
          tcore=1.0e11;
        }
      } else {
        tb_acc0*=ephi[imax];
        tsurface=tb_acc0;
        tdrip=tb_acc0;
        tcore=tb_acc0;
      }
      // -----------------------------------------------------------------------------
      double dec=1.1;
      for (int i=0;i<=icore;++i) {
        temp[i]=tcore;
      }
      for (int i=icore+1;i<=idrip;++i) {
        double w1=(std::log10(rad[idrip])-std::log10(rad[i]))/
              (std::log10(rad[idrip])-std::log10(rad[icore]));
        double w2=1.0-w1;
        double lt=w1*std::log10(tcore)+w2*std::log10(tdrip);
        temp[i]=std::pow(10.0,lt);
      }
      for (int i=idrip+1;i<=imax;++i) {
        double w1=(std::log10(rad[imax])-std::log10(rad[i]))/
              (std::log10(rad[imax])-std::log10(rad[idrip]));
        double w2=1.0-w1;
        double lt=w1*std::log10(tdrip)+w2*std::log10(tsurface);
        temp[i]=std::pow(10.0,lt);
      }
      dtemp[0]=0.0;
      for (int i=2;i<=imax-1;i+=2) {
        dtemp[i]=(temp[i+1]-temp[i-1])/(debar[i]+debar[i+1]);
      }
      // Calculate the inner envelope profile: ---------------------------------------
      if (debug >= 1) 
        std::cout << "Calculating envelope profile" << std::endl;

      for (int i=imax-1;i>=ienv+1;i-=2) {
        double x=(std::log(rrho[i+1])-std::log(rrho[i]))/
              (std::log(rrho[i+1])-std::log(rrho[i-1]));
        double y=(std::log(rrho[i])-std::log(rrho[i-1]))/
              (std::log(rrho[i+1])-std::log(rrho[i-1]));
        double ltemp=y*std::log(temp[i+1])+x*std::log(temp[i-1]);
        temp[i]=std::exp(ltemp);
      }

      for (int i=ienv;i<=imax;++i) {
        if (temp[i]<tcon) {
          density(temp[i]/ephi[i],pres[i],a_ion[i],z_ion[i],rrho[i]);
          rrho[i]=std::min(rrho[i],rhod[i]);
          bar[i]=6.022e-16*rrho[i];
          double dr=debar[i]/rrho[i]*factor; 
          rad[i+1]=rad[i]+dr;
        }
      }
      // ----------------------------------------------------------------------------
      if (debug>=1.0)
        std::cout << "Calculating initial L profile" << std::endl;


      for (int i=imax;i>=1;i-=2) {
        conduct(i,temp[i]/ephi[i],rrho[i],a_cell[i],a_ion[i],z_ion[i],qimp,
            nbfield2[i],sig,lambda[i],debug,nu_e_s,nu_e_l,icon_crust,icon_core,
            rhodrip,rhocore,kfe,kfm,kfn,kfp,kfla,kfsm,kfs0,kfsp,mstn,mstp,
            mstla,mstsm,msts0,mstsp,tcn,tcp,tcla,tcsm,tcs0,tcsp,isf,fhad,istrange);

        opacity(temp[i]/ephi[i],rrho[i],a_cell[i],z_ion[i],kappa[i],iopacity);

        double acd=7.56e-15*c/(3.0*kappa[i]*rrho[i]);
        fp[i]=(lambda[i]+4.0*acd*std::pow(temp[i]/ephi[i],3))*bar[i]/lsol;
      }

      lum[0]=0.0;
      for (int i=2;i<=imax-1;i+=2) {
        lum[i]=-(fp[i+1]+fp[i-1])/2.0*a2ephin[i]*dtemp[i];
        if (lum[i]==0.0) {
          lum[i]=1.0e-3;
        }
      }

      for (int i=1;i<=imax-2;i+=2) {
        dlum[i]=(nlum[i+1]-nlum[i-1])/(debar[i]+debar[i+1]);
      }
      // ---------------------------------------------------------------------------
      dlum[imax]=0.0;
      // ---------------------------------------------------------------------------
      for (int i=0;i<=imax;++i) {
        otemp[i]=temp[i];
        olum[i-1]=lum[i-1];
        orad[i]=rad[i];
        orrho[i]=rrho[i];
        rrho1[i]=rrho[i];
        obar[i]=bar[i];
        bar1[i]=bar[i];
      }
      // -----------------------------------------------------------------------
      // COOLING
      // -----------------------------------------------------------------------
      // starting from line 1063 again
      double time = time0;  // Initialize the time
      int icycle = 0;  // Initialize the counter for accretion cycles
      int itprint = 1;  // To print out only at the required times
      // THIS IS THE MAIN TIME LOOP:
      // -----------------------------------------------------------------------
      for (int istep=1;istep<=istepmax; istep++) {
        double debug=0.0;
        if (istep>=istep_debug) 
          debug = 1.0;
        if (debug>=1.0) 
          std::cout << "Going: istep=" << istep << std::endl;
        int itrial=0;
        double ratiot=1.e-2;
        double ratiol=1.e-2;
        if (dtime/year<=1.0e-20) {
        iret=1;
        std::cout << dtime/year << " NSCool failed. dtime too small." << std::endl;
        return;
      }

      if (debug>=1.0) {
        std::cout << "Guessing NLum & NTemp" << std::endl;
      }
      const double coeff_int = 0.8;
      for (int i=1;i<=imax;i+=2) {
        ntemp[i] = temp[i]+coeff_int*(temp[i]-otemp[i])*dtime/odtime;
      }

      dtemp[0]=0.0;
      for (int i=2;i<=imax-1;i+=2) {
        dtemp[i]=(ntemp[i+1]-ntemp[i-1])/(debar[i]+debar[i+1]);
      }

      nlum[0]=0.0;
      for (int i=2;i<=imax-1;i+=2) {
        nlum[i]=lum[i]+coeff_int*(lum[i]-olum[i])*dtime/odtime;
      }

      for (int i=1;i<=imax-2;i+=2) {
        dlum[i]=(nlum[i+1]-nlum[i-1])/(debar[i]+debar[i+1]);
      }

      dlum[imax] = 0.0;
      // Here needs to implement a goto loop
      itrial++;
      if (itrial==itrial_max+1) {
        double tcut = std::sqrt(scale_dt0);
        if (time<=1.0e5) {
          tcut=std::sqrt(scale_dt1);
        }
        dtime/=tcut;
        if (debug>0.4 && debug<0.6) {
          std::cout << "Exceeded iterations." << std::endl;
        }
        itrial=0;
      }

      for (int i=0;i<=imax;i++) {
        rad[i]=orad[i];
        rrho[i]=orrho[i];
        rrho1[i]=rrho[i];
      }

      double oteffective = teffective;
      // Calculate the new density in inner envelope at ntemp
      if (debug >= 1.0) 
        std::cout << "Calculating new density at NTemp" << std::endl;
      for (int i=imax-1;i>=ienv+1;i-=2) {
        double x=(std::log(rrho[i+1])-std::log(rrho[i]))/
                (std::log(rrho[i+1])-std::log(rrho[i-1]));
        double y=(std::log(rrho[i])-std::log(rrho[i-1])) /
                (std::log(rrho[i+1])-std::log(rrho[i-1]));
        double ltemp=y*std::log(ntemp[i+1])+x*std::log(ntemp[i-1]);
        ntemp[i]=std::exp(ltemp);
      }

      for (int i=ienv;i<=imax;i++) {
        if (ntemp[i]<tcon) {
          density(ntemp[i]/ephi[i],pres[i],a_cell[i],z_ion[i],rrho[i]);
          rrho[i]=std::min(rrho[i], rhod[i]);
          bar[i]=1.0e-39*6.022e23*rrho[i];
          double dr=debar[i]/rrho[i]*factor;
          rad[i+1]=rad[i]+dr;
          a2ephin[i]=std::pow(4.0*pi*rad[i]*rad[i],2)*ephi[i];
        }
      }
      // Calculate the physical parameters at ntemp
      if (debug >= 1.0) 
        std::cout << "Calculating physics at NTemp" << std::endl;
      for (int i=1;i<=imax;i+=2) {
        double t=ntemp[i]/ephi[i];
        double d=rrho[i];
        double a=a_cell[i];
        double a1=a_ion[i];
        double z=z_ion[i];
        neutrino(irank, i, t, d, a, z, qnu[i], qeebrem[i], qnpb[i], qplasma[i],
                 qsynch[i], qbubble[i], qpair[i], qphoto[i], qbrem_nn[i],
                 qmurca_nucl[i], qbrem_nucl[i], qmurca_hyp[i], qbrem_hyp[i],
                 qdurca_np[i], qdurca_lap[i], qdurca_smn[i], qdurca_smla[i],
                 qdurca_sms0[i], qfast[i], qdurca_q[i], qmurca_q[i],
                 qpbf_n1s0[i], qpbf_n3p2[i], qpbf_p1s0[i], qpbf_q[i],
                 debug, naa, nbfield2, rhodrip, rhocore, mstp, mstn, mstla,
                 mstsm, msts0, mstsp, kfe, kfm, kfp, kfn, kfqu, kfqd, kfqs,
                 bar, yelect, ymuon, fhad, theta_k, theta_p, v_ion,
                 rhoexo, cexo, pexo, c_nu_str, p_nu_str,
                 murca_increase, inu_durca, inu_eion, inu_plasma, inu_synch,
                 inu_n1s0_pbf, inu_n3p2_pbf, inu_p_pbf,
                 inu_bubble, inu_photo, inu_pair,
                 idurca_np, idurca_lap, durca_ctrl_e, durca_ctrl_m,
                 idurca_smn, idurca_smla, idurca_sms0,
                 idurca_quqd, idurca_quqs, tcn, tcp, tcla, tcu, tcd, tcs,
                 tcu1, tcu2, tcu3, tcd1, tcd2, tcd3, tcs1, tcs2, tcs3, isf,
                 neebrem_logt, neebrem_nalpha, neebrem_n2,
                 sf_lgtau1, sf_lgtau2, sf_lgr, sf_lgr2);

        qqq[i]=qnu[i]-heat[i];
        specheat(i,t,d,a,z,cv[i],
           cv_n[i],cv_p[i],cv_e[i],cv_m[i],
           cv_l[i],cv_sm[i],cv_s0[i],cv_sp[i],
           cv_q[i],cv_ion[i],cve,cvm,cvn,cvp,cvla,
          cvsm,cvs0,cvsp,cvqu,cvqd,cvqs,rhodrip,rhocore,fhad,istrange,
          tcn,tcp,tcla,tcsm,tcs0,tcsp,isf);
        if (debug >= 1.0) 
        std::cout << "Calling conduct: " << i << " " << d << " " << t << std::endl;
        conduct(i,t,d,a,a1,z,qimp,nbfield2[i],
           sig,lambda1[i],debug,
           nu_e_s,nu_e_l,icon_crust,icon_core,rhodrip,rhocore,
           kfe,kfm,kfn,kfp,kfla,kfsm,kfs0,kfsp,mstn,mstp,
           mstla,mstsm,msts0,mstsp,tcn,tcp,tcla,tcsm,tcs0,tcsp,isf,
           fhad,istrange);
        if (debug >= 1.0) 
        std::cout << "done" << std::endl;
        opacity(t,d,a,z,kappa[i],iopacity);
        double acd=7.56-15*c/(3*kappa[i]*d);
        fp[i]=(lambda[i]+4.*acd*std::pow(t,3))*bar[i]/lsol;
        fq[i]=bar1[i]/cv[i]*lsol;
        fr[i]=e2phi[i]*qqq[i]/cv[i]-
             ephi[i]*pres[i]/cv[i]*
             (std::log(rrho[i])-std::log(orrho[i]))/dtime*contraction;
      
      }
      // Calculate the physical parameters at (1-tinc)*ntemp
      if (debug >= 1.0) 
        std::cout << "Calculating physics at NTemp" << std::endl;
      for (int i=1;i<=imax;i+=2) {
        double t=ntemp1[i]/ephi[i];
        double d=rrho1[i];
        double a=a_cell[i];
        double a1=a_ion[i];
        double z=z_ion[i];
        double qn00,qn01,qn02,qn03,qn04,qn05,qn06,qn07,qn08,qn09,q10,qn11,
                qn12,qn13,qn14,qn15,qn16,qn17,qn18,qn19,q20,qn21,qn22,qn23,
                xx1,xx2,xx3,xx4,xx5,xx6,xx7,xx8,xx9,xx0;
        neutrino(irank, i, t, d, a, z, qnu1[i], qn00,qn01,qn02,qn03,qn04,qn05,
                 qn06,qn07,qn08,qn09,q10,qn11,qn12,qn13,qn14,qn15,qn16,qn17,
                 qn18,qn19,q20,qn21,qn22,qn23,
                 debug, naa, nbfield2, rhodrip, rhocore, mstp, mstn, mstla,
                 mstsm, msts0, mstsp, kfe, kfm, kfp, kfn, kfqu, kfqd, kfqs,
                 bar, yelect, ymuon, fhad, theta_k, theta_p, v_ion,
                 rhoexo, cexo, pexo, c_nu_str, p_nu_str,
                 murca_increase, inu_durca, inu_eion, inu_plasma, inu_synch,
                 inu_n1s0_pbf, inu_n3p2_pbf, inu_p_pbf,
                 inu_bubble, inu_photo, inu_pair,
                 idurca_np, idurca_lap, durca_ctrl_e, durca_ctrl_m,
                 idurca_smn, idurca_smla, idurca_sms0,
                 idurca_quqd, idurca_quqs, tcn, tcp, tcla, tcu, tcd, tcs,
                 tcu1, tcu2, tcu3, tcd1, tcd2, tcd3, tcs1, tcs2, tcs3, isf,
                 neebrem_logt, neebrem_nalpha, neebrem_n2,
                 sf_lgtau1, sf_lgtau2, sf_lgr, sf_lgr2);

        qqq1[i]=qnu1[i]-heat1[i];
        specheat(i,t,d,a,z,cv1[i],
           xx1,xx2,xx3,xx4,xx5,xx6,xx7,xx8,xx9,xx0,cve,cvm,cvn,cvp,cvla,
          cvsm,cvs0,cvsp,cvqu,cvqd,cvqs,rhodrip,rhocore,fhad,istrange,
          tcn,tcp,tcla,tcsm,tcs0,tcsp,isf);
        if (debug >= 1.0) 
        std::cout << "Calling conduct: " << i << " " << d << " " << t << std::endl;
        conduct(i,t,d,a,a1,z,qimp,nbfield2[i],
           sig,lambda1[i],debug,
           nu_e_s,nu_e_l,icon_crust,icon_core,rhodrip,rhocore,
           kfe,kfm,kfn,kfp,kfla,kfsm,kfs0,kfsp,mstn,mstp,
           mstla,mstsm,msts0,mstsp,tcn,tcp,tcla,tcsm,tcs0,tcsp,isf,
           fhad,istrange);
        if (debug >= 1.0) 
        std::cout << "done" << std::endl;
        opacity(t,d,a,z,kappa1[i],iopacity);
        double acd1=7.56-15*c/(3*kappa1[i]*d);
        fp1[i]=(lambda1[i]+4.*acd1*std::pow(t,3))*bar1[i]/lsol;
        fq1[i]=bar1[i]/cv[i]*lsol;
        fr1[i]=e2phi[i]*qqq1[i]/cv1[i]-
             ephi[i]*pres[i]/cv1[i]*
             (std::log(rrho1[i])-std::log(orrho[i]))/dtime*contraction;
      
      }
      // Calculate the derivatives of fp,fq & fr
      if (debug>=1.0) 
        std::cout << "Calculating derivatives of FP, FQ, FR" << std::endl;
      for (int i=1;i<=imax;i+=2) {
        double t=ntemp[i];
        double t1=ntemp1[i];
        dfp[i]=(fp[i]-fp1[i])/(t-t1);
        dfq[i]=(fq[i]-fq1[i])/(t-t1);
        dfr[i]=(fr[i]-fr1[i])/(t-t1);
      }
      if (debug>=1.0) 
        std::cout << "Calculating FF" << std::endl;
      ff[0]=0.0;
      for (int i=2;i<=imax-1;i+=2) {
        ff[i]=nlum[i]+0.5*(fp[i-1]+fp[i+1])*a2ephin[i]*dtemp[i];
        ff[i-1]=fr[i-1]+fq[i-1]*dlum[i-1]+
              (ntemp[i-1]-temp[i-1])/dtime;
      }
      ff[imax]=0.0;
      // Matrix inversion for Newton-Raphson method
      if (debug >= 1.0) 
        std::cout << "Newton-Raphson" << std::endl;
      for (int i=2;i<=imax-1;i+=2) {
        fa[i]=0.5*dfp[i+1]*a2ephin[i]*dtemp[i]+
            0.5*(fp[i+1]+fp[i-1])*a2ephin[i]/
                  (debar[i]+debar[i+1]);
        fb[i]=0.5*dfp[i-1]*a2ephin[i]*dtemp[i]-
            0.5*(fp[i+1]+fp[i-1])*a2ephin[i]/
                (debar[i]+debar[i+1]);
        fc[i]=1.0;
      }

      for (int i=1;i<=imax-2;i+=2) {
        fa[i]=fq[i]/(debar[i]+debar[i+1]);
        fb[i]=-fq[i]/(debar[i]+debar[i+1]);
        fc[i]=dfr[i]+dfq[i]*dlum[i]+1.0/dtime;
      }

      fk[1]=-ff[1]/fc[1];
      fj[1]=fa[1]/fc[1];
      for (int i=2;i<=imax-1;++i) {
        fk[i]=-(ff[i]+fb[i]*fk[i-1])/(fc[i]-fb[i]*fj[i-1]);
        fj[i]=fa[i]/(fc[i]-fb[i]*fj[i-1]);
      }

      if (debug >= 1.0) 
        std::cout << "Boundary Condition" << std::endl;
      double ntp;
      if (ifteff != 15) {
        double epsilon=1.0e-8;
        double precision=1.0e-12;
        double coeff=4.0*pi*radius*radius*5.67e-5*e2phi[imax]/lsol;
        double lhs=nlum[imax-1]+fk[imax-1]+fj[imax-1]*ntemp[imax];
        ntp=ntemp[imax];
        double tp0_keep=ntp;
        
        double tp0 = ntp;
        double teff0 = Teff(tp0/ephi[imax],ifteff,eta,
                          0.0, istep,time,z_ion[imax],
                          a_ion[imax],rrho[imax],debug,gs14,compactness);
        if (debug==-50.0) 
          std::cout << "Tb0, Te0 = " << tp0 << ", " << teff0 << std::endl;
        
        double tp1=(1.0+epsilon)*tp0;
        double teff1=Teff(tp1/ephi[imax],ifteff,eta,
                          0.0, istep,time,z_ion[imax],
                          a_ion[imax],rrho[imax],debug,gs14,compactness);
        if (debug==-50.0) 
          std::cout << "Tb1, Te1 = " << tp1 << ", " << teff1 << std::endl;
        
        double derivative=coeff*(std::pow(teff1,4)-std::pow(teff0,4))/(epsilon*tp0);
        derivative=-fj[imax-1]-derivative;
        if (debug==-50.0) 
          std::cout << "Derivative = " << derivative << std::endl;
        
        double function=lhs-fj[imax-1]*tp0-coeff*std::pow(teff0,4);
        if (debug==-50.0)
          std::cout << "Function = " << function << std::endl;
        
        ntp=tp0-function/derivative;
        if (debug==-50.0) {
          std::cout << "Del(Tp)/Tp = " << std::abs(tp0-ntp)/tp0 << std::endl;
          std::cout << "------> New Tb = " << ntp << std::endl;
        }
        
        if ((ntp<=0.0) || (ntp>1.0e12)) { // In case the method diverges
          tcut=std::sqrt(scale_dt0);
          if (time<=1.0e5) {
            tcut = std::sqrt(scale_dt1);
          }
          dtime = dtime/tcut;
          if (debug>0.4 && debug<0.6) 
            std::cout << "BC problem." << std::endl;
            itrial=0; // goto statement needs fix
        }
        
        if (std::abs(tp0-ntp)/tp0>precision) {
          tp0=ntp; // goto statement needs fix
        }
      } else {
        ntp=tb_acc0; // Fixed T_b for accretion
      }
      // Get ntemp & nlum
      if (debug >= 1.0) 
        std::cout << "Getting NTemp & NLum" << std::endl;
      delt[imax]=ntp-ntemp[imax];
      for (int i=imax-2;i>=1;i-=2) {
        dell[i+1]=fk[i+1]-fj[i+1]*delt[i+2];
        dell[i+1]=std::copysign(std::min(2.0e3*std::abs(nlum[i+1]),
                  std::abs(dell[i+1])),dell[i+1]);
        delt[i]=fk[i]-fj[i]*dell[i+1];
        delt[i]=std::copysign(std::min(0.5*ntemp[i],
              std::abs(delt[i])),delt[i]);
      }
      dell[0]=0.0; // This is the inner boundary condition !

      if (debug>0.4 && debug<0.6) {
        std::cout << "itrial,imax,delt,dell" << std::endl;
        std::cout << " " << itrial << " " << imax << " " 
        << delt[imax-2] << " " << dell[imax-1] << std::endl;
      }

      for (int i=0;i<imax; i+=2) {
        nlum[i]=nlum[i]+dell[i];
        ntemp[i+1]=ntemp[i+1]+delt[i+1];
      }

      dtemp[0]=0.0;
      for (int i=2;i<imax-1;i++) {
        dtemp[i]=(ntemp[i+1]-ntemp[i-1])/(debar[i]+debar[i+1]);
      }
      for (int i=1;i<imax-2;i+=2) {
        dlum[i]=(nlum[i+1]-nlum[i-1])/(debar[i]+debar[i+1]);
      }
      dlum[imax]=0.0; 
      //----------------------------------------------------------------
      // Analyze the results to see if it has converged
      // ---------------------------------------------------------------
      if (debug>=1.0) 
        std::cout << "Analyzing Results" << std::endl;

      double ratiot=0.0;
      double ratiol=0.0;
      int iratt=0;
      int iratl=0;

      for (int i=1;i<imax-2;i+=2) {
        double ratl=std::abs(dell[i+1])/(std::abs(nlum[i+1])+1.0e-12);
        if (ratl>ratiol) {
          ratiol=ratl;
          iratl=i+1;
        }
        double ratt=std::abs(delt[i])/(ntemp[i]+1.0e-30);
        if (ratt>ratiot) {
          ratiot=ratt;
          iratt=i;
        }
      }
      // This was used when doing dipolar magnetic field evolution.
      // Not used any more.
      double ratios=0.0;
      int irats=1;
      // ---------------------------------------------------------------
      if (pscreen==3) {
        if (ifield==2) {
          std::cout << "   dT/T=" << ratiot << "(" << iratt << ")"
                << "   dL/L=" << ratiol << "(" << iratl << ")"
                << "   dS/S=" << ratios << "(" << irats << ")" << std::endl;
        } else {
          std::cout << "   dT/T=" << ratiot << "(" << iratt << ")"
                << "   dL/L=" << ratiol << "(" << iratl << ")" << std::endl;
        }
      }
      // ---------------------------------------------------------------
      if (debug>0.4 && debug<0.6) {
        std::cout << "ratio,irat,mrat   "
              << ratiot << " " << iratt << " " << mratt << "   "
              << ratiol << " " << iratl << " " << mratl << std::endl;
      }
      // ---------------------------------------------------------------
      // Decide if converged or not:
      if (ratiot<mratt && ratiol<mratl && ratios<mrats) {
        continue;// Converged! Continue to the next time step.
      } else {
      // Not converged! Go back for another iteration.
        goto 2000;
      }

      // ---------------------------------------------------------------

      double luminosity=nlum[imax-1]/(ephi[imax-1]*ephi[imax-1]);
      double sign_l=std::abs(nlum[imax-1])/nlum[imax-1];
      double teffective=sign_l*std::pow(std::abs(luminosity)/
            (4.0*pi*radius*radius*5.67e-5),0.25)
            *std::pow(lsol,0.25)*ephi[imax-1];

      // c PHILOSOPHY OF TIME STEP CONTROL:
      //
      // (Time of step just finished is "time+dtime", not just time !)
      // The new "dtime" will be "scale_dt*dtime" with "scale_dt" calculated below.
      // Allows for 2 different "scale_dt": at early time, while relaxing from initial
      // conditions, accuracy is not important and one can allow for larger timestep:

      if (debug >= 1) 
        std::cout << "Calculating New Time Step" << std::endl;

      double scale_dt = scale_dt0;
      if (time <= 1.0e5) {
        scale_dt = scale_dt1;
      }

      // TEMP variation: "max_dtemp" is the max. relative variation of T in the
      // star and "icht" the zone where "max_dtemp" is obtained

      int icht = 0;
      double max_dtemp = 0.0;
      for (int i = 1; i <= imax - 2; i += 2) {
        double mdt = std::abs(temp[i] - ntemp[i]) / ntemp[i];
        if (mdt > max_dtemp) {
          max_dtemp = mdt;
          icht = i;
        }
      }
      // ---------------------------------------------------------------
      // c Check if "max_dtemp" is not too large. If "1+max_dtemp" exceeds "tvar",
      // then "scale_dt" is reduced correspondingly:
      double chtemp = 0.0;
      if ((tvar - 1.0) < max_dtemp) {
        scale_dt = scale_dt * (tvar - 1.0) / max_dtemp;
        if (time < 1.0e5) {
          scale_dt = std::min(scale_dt, scale_dt1);
        } else {
          scale_dt = std::min(scale_dt, scale_dt0);
        }
        chtemp = 1.0;
      }
      // ----------------------------------------------------------------
      // c If "scale_dt" has been reduced too much, i.e. "max_dtemp" too large,
      // then the present time-step is recalculated with a smaller "dtime"
      // unless one is still at the first few time-steps:

      if ((scale_dt < repeat) && (istep > istart)) {
        dtime = repeat * dtime;
        // --------------------------------------------------------------
        if (pscreen >= 2) {
          std::cout << std::endl;
          std::cout << "dtime too large, do it again" << std::endl;
          std::cout << "time=" << (time + dtime) / year << std::endl;
          std::cout << "dtime=" << dtime / year << std::endl;
          std::cout << "dtime ratio=" << dtime / odtime << std::endl;
          std::cout << std::endl;
        }
        // ----------------------------------------------------------------
        if (debug > 0.4 && debug < 0.6) 
          std::cout << "Temperature changed too much." << std::endl;
          itrial-0; // another goto statement needs fix
      }

      // In case convergence is reached in too many trials, scale_dt is reduced
      double chtrial;
      if ((itrial > itrial_opt) && (istep > istart)) {
        chtrial = 1.0; // this means the time-step is controlled by needing too many iterations
        double olddt = scale_dt;
        if (time < 1.0e5) {
          scale_dt = scale_dt / std::pow(scale_dt1, static_cast<double>(itrial - itrial_opt) / 2.0);
          scale_dt = std::min(scale_dt, scale_dt1);
        } else {
          scale_dt = scale_dt / std::pow(scale_dt0, (1.0 + static_cast<double>(itrial - itrial_opt)) / 2.0);
          scale_dt = std::min(scale_dt, scale_dt0);
        }
      } else {
        chtrial = 0.0;
      }
      // Before setting the next time and time-step, stuff are printed out and updated:
      // -------------------------------------------------------------------
      // End of iterations
      // -------------------------------------------------------------------
      for (int i = 1; i <= imax; i += 2) {
        otemp[i] = temp[i];
        temp[i] = ntemp[i];
        orrho[i] = rrho[i];
        orad[i] = rad[i];
        obar[i] = bar[i];
      }

      for (int i = 2; i <= imax - 1; i += 2) {
        olum[i] = lum[i];
        lum[i] = nlum[i];
        orrho[i] = rrho[i];
        orad[i] = rad[i];
        obar[i] = bar[i];
      }
      // -------------------------------------------------------------------
      //period=nperiod;
      // -------------------------------------------------------------------
      // Crystalization density:
      // -------------------------------------------------------------------
      if (debug>=1) 
        std::cout << "Calculating Crystalization Densities" << std::endl;

      for (int i=imax;i>=icore+2;i-=2) {
        gamma[i]=2.273e5*pow(z_ion[i],2)*pow(rrho[i]/a_cell[i],1.0/3.0)/
               (temp[i]/ephi[i]);
        if (gamma[i]<gammaliq) {
          cryst[i]=-1.0;
        } else if ((gamma[i]>=gammaliq) && (gamma[i]<=gammacryst)) {
          cryst[i]=0.0;
        } else {
          cryst[i]=1.0;
        }
      }
      for (int i=icore;i>= 0;i-=2)
        gamma[i]=1.0;
    
      int iliq=0;
      int icryst=0;

      for (int i=imax;i>=icore+2;i-=2) {
        if (cryst[i]==0.0) {
          iliq=i;
          continue;
        }
      }
      for (int i=imax;i>=icore+2;i-=2) {
        if (cryst[i]==1.0) {
          icryst=i;
          continue;
        }
      }

      // ----------------------------------------------------------------
      // Calculate the neutrino luminosity and heating:
      // ----------------------------------------------------------------
      if (debug>=1) 
        std::cout << "Calculating Total Neutrino Luminosity and Heating" << std::endl;
      double lnu=0.0;
      double lnu0=0.0;
      double lh=0.0;
      double lh0=0.0;
      for (int i=1;i<=imax;i+=2) {
        lnu+=e2phi[i]*qnu[i]*(dvol[i]+dvol[i+1]);
        lnu0+=ephi[i]*qnu[i]*(dvol[i]+dvol[i+1]);   // without energy red-shift
        lh+=e2phi[i]*heat[i]*(dvol[i]+dvol[i+1]);
        lh0+=ephi[i]*heat[i]*(dvol[i]+dvol[i+1]);   // without energy red-shift
      }
      lnu/=lsol;
      lh/=lsol;
      htot+=lh*dtime;
      // ---------------------------------------------------------------
      // CALCULATE THE INTEGRATED NEUTRINO LUMINOSITIES:
      // ---------------------------------------------------------------
      // Note: lnu_tot, calculated from qnu(i), is the garanteed total
      // neutrino luminosity. The other ones are only informative.
      double lmurca_nucl=0.0;
      double lbrem_nucl=0.0;
      double lplasma=0.0;
      double lnpb=0.0;
      double lpbf_n1S0=0.0;
      double lpbf_n3P2=0.0;
      double lpbf_p1S0=0.0;
      for (int i=1;i<=imax;i+=2) {
        double e2p=e2phi[i];
        lmurca_nucl+=qmurca_nucl[i]*(dvol[i]+dvol[i+1])*e2p;
        lbrem_nucl+=qbrem_nucl[i]*(dvol[i]+dvol[i+1])*e2p;
        lplasma+=qplasma[i]*(dvol[i]+dvol[i+1])*e2p;
        lnpb+=qnpb[i]*(dvol[i]+dvol[i+1])*e2p;
        lpbf_n1S0+=qpbf_n1s0[i]*(dvol[i]+dvol[i+1])*e2p;
        lpbf_n3P2+=qpbf_n3p2[i]*(dvol[i]+dvol[i+1])*e2p;
        lpbf_p1S0+=qpbf_p1s0[i]*(dvol[i]+dvol[i+1])*e2p;
      }
      // --------------------------------------------------------------
      // CALCULATE THE INTEGRATED SPECIFIC HEATS:
      // --------------------------------------------------------------
      // cv_tot_all, calculated from cv(i), is the garanteed total
      // specific heat. The other ones are only informative.
      double cv_core=0.0;
      double cv_crust=0.0;
      double cv_tot_all=0.0;
      double cv_tot_ion=0.0;
      double cv_tot_neu=0.0;
      double cv_tot_pro=0.0;
      double cv_tot_ele=0.0;
      double cv_tot_muo=0.0;
      double cv_tot_lam=0.0;
      double cv_tot_sim=0.0;
      double cv_tot_si0=0.0;
      double cv_tot_sip=0.0;
      double cv_tot_qrk=0.0;
      double cv_phot=0.0;
      for (int i=1;i<=imax;i+=2) {
        cv_tot_all+=cv[i]*(dvol[i] + dvol[i+1]);
        cv_tot_ion+= cv_ion[i]*(dvol[i]+dvol[i+1]);
        cv_tot_neu+= cv_n[i]*(dvol[i]+dvol[i+1]);
        cv_tot_pro+= cv_p[i]*(dvol[i]+dvol[i+1]);
        cv_tot_ele+= cv_e[i]*(dvol[i]+dvol[i+1]);
        cv_tot_muo+= cv_m[i]*(dvol[i]+dvol[i+1]);
        cv_tot_lam+= cv_l[i]*(dvol[i]+dvol[i+1]);
        cv_tot_sim+= cv_sm[i]*(dvol[i]+dvol[i+1]);
        cv_tot_si0+= cv_s0[i]*(dvol[i]+dvol[i+1]);
        cv_tot_sip+= cv_sp[i]*(dvol[i]+dvol[i+1]);
        cv_tot_qrk+= cv_q[i]*(dvol[i]+dvol[i+1]);
        cv_phot+=4.0*7.56e-15*pow((ntemp[i]/ephi[i]),3)*(dvol[i]+dvol[i+1]);
      }
      for (int i=1;i<=icore+2;i++) {
        cv_core+=cv[i]*(dvol[i]+dvol[i+1]);
      }
      for (int i=icore+2;i<=imax;i++) {
        cv_crust+=cv[i]*(dvol[i]+dvol[i+1]);
      }
      // -------------------------------------------------------------------
      // TEMP
      // -------------------------------------------------------------------
      if (debug>0.4 && debug<0.6) 
        std::cout << "time " << time+dtime/year << std::endl;

      if (ptemp>=1.0) {
        // AWS: Determine if we should output temperature at this iteration
        if (((time+dtime)/year >= tprint[itprint-1]) && (itprint<=itpmax)) {
          itprint++;

          // AWS: Compute weights and effective temperature
          double w1=((time+dtime)-tprint[itprint-2]*year)/dtime;
          double w2=1.0-w1;
          double logtemp=w1*log(sign_l*oteffective)+
                         w2*log(sign_l*teffective);
          double t_effective=sign_l*exp(logtemp);

          print_temp(istep,itprint-2,
                          tprint[itprint-2],t_effective,imax,w1,w2,otemp,
                          temp, olum, lum, rad, rrho, ephi, dvol, e2phi, tcn, tcp,
                          qnu, qeebrem, qnpb, qplasma, qsynch, qbubble, qpair,
                          qphoto, qbrem_nn, qmurca_nucl, qbrem_nucl, qmurca_hyp,
                          qbrem_hyp, qdurca_np, qdurca_lap, qdurca_smn,
                          qdurca_smla, qdurca_sms0, qfast, qdurca_q, qmurca_q,
                          qpbf_n1s0, qpbf_p1s0, qpbf_n3p2, qpbf_q);

          print_cv(itprint-2, imax, cv, cv_n,
                        cv_p, cv_e, cv_m, cv_l, cv_sm, cv_s0, cv_sp, cv_q);

        }
      }
      // ------------------------------------------------------------------
      // TEFF
      // ------------------------------------------------------------------
      if (pteff >= 1.0) {
        // skipping because Its not used
        cv_core = 0.0;
        cv_crust = 0.0;
        for (int i=0;i<icore;i+=2) {
          cv_core+=(dvol[i]+dvol[i+1])*cv[i];
        }
        for (int i=icore+2;i<imax;i+=2) {
          cv_crust+=(dvol[i]+dvol[i+1])*cv[i];
        }
        // -----------------------------------------------------------------
        if (pteff == 1.0) {
          std::cout << std::fixed << std::setprecision(3)
                  << std::setw(8) << istep
                  << std::scientific << std::setprecision(3)
                  << std::setw(12) << (time + dtime) / year
                  << std::setw(12) << teffective
                  << std::setw(12) << lum[imax - 1] * lsol
                  << std::setw(12) << lnu * lsol
                  << std::setw(12) << lh * lsol
                  << std::endl;
        } else {
          std::cout << "WARNING: Not Teff print out defined !" << std::endl;
          exit(-1);
        }
      }
      // -------------------------------------------------------------------
      nscool_main_out(irank,(time+dtime)/year,teffective,nlum[imax-1]*lsol,lnu*lsol,dtime/year,istop);

      if (istop > 0) {
        iret = 3;
        std::cout << dtime << " NSCool failed. Stepsize vanishing." << std::endl;
        continue; 
      }
      // -------------------------------------------------------------------
      // CALCULATE THE NEW TIME STEP
      // -------------------------------------------------------------------
      time=time+dtime;
      odtime=dtime;
      dtime=std::min(scale_dt*dtime,dtlimit);
      // -------------------------------------------------------------------
      // For accretion scenarios: time step must moreover be shortened 
      // dramatically when a new outburst is approaching (to make sure it is  
      // much shorter than the outburst duration, or rise time, or any relevant 
      // time scale which has to be resolved by the code):
      // -------------------------------------------------------------------
      // Transient FRED ("Fast rise and exponential decay")
      if (i_acc == 1) {
        double time_step_cut = 100.0;

        if (((time+3.0*dtime)>=t_acc0) && (icycle == 0)) {
          double timeleft=t_acc0-time;
          dtime=std::max(timeleft/3.0,t_acc2/time_step_cut);
        }

        if (time>t_acc0) {
          if (delt_acc/t_acc2 <= 10.0) {
            scale_dt0 = 1.05;
          } else {
            scale_dt0 = 1.2;
          }
          double t_next = t_acc0+static_cast<double>(icycle + 1)*t_acc1;
          if ((time+3.0*dtime)>=t_next) {
            double timeleft=t_next-time;
            dtime = std::max(timeleft/3.0,t_acc2/time_step_cut);
          }
        }

        if (time>=t_acc0) 
          dtlimit=t_acc1/20.0;
    
      }
      // Transient STEP----------------------------------------------------
      if (i_acc==2) {
        double timeleft,t_next,t_end,day;
        if (((time+2.0*dtime)>=t_acc0) && (time<t_acc0)) {
          timeleft=t_acc0-time;
          dtime=std::max(timeleft/3.0,time_step_min);
          // Print messages if needed
        }

        if (time > t_acc0) {
          t_next=t_acc0+static_cast<double>(icycle-1)*t_acc1+t_acc1;
          t_end=t_acc0+static_cast<double>(icycle-1)*t_acc1+t_acc2;

          if ((time+dtime>t_end) && (time+2.0*dtime>=t_next)) {
            timeleft=t_next-time;
            dtime=std::max(timeleft/3.0,time_step_min);
          } else if ((time<=t_end) && (time+2.0*dtime>=t_end)) {
            timeleft=t_end-time;
            dtime=std::max(timeleft/3.0,time_step_min);
          }
        }

        if (time>=t_acc0) {
          dtlimit=t_acc1/20.0;
        }
      }
      // Heat deposition--------------------------------------------------
      if (i_heat_deposit==1) {
        double t_slow=std::max(dtime,1000.0*del_t_dep);

        if (std::abs(t_dep-(time+dtime))<t_slow) {
          double timeleft=std::abs(t_dep-(time+dtime));
          dtime = std::max(timeleft/30.0,del_t_dep/100.0);
        }
      }

      if (time/year>=timemax) {
        return;
      }

      if ((sign_l*teffective)<tempmin) {
        return;
      }
    }
  };
};


