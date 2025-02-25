c *********************************************************************
c 
c DEBUGGING STRATEGY:
c debug.eq.0.9d0 : prints during reading of input parameter file `I*.dat'
c debug.ge.1.    : prints program steps
c debug.ge.2.    : prints program steps and entering & exiting main subroutines
c 
c
c
c debug.eq.-50. : print boundary condition details
c
c *********************************************************************
c 
c                             WARNING:
c
c          Index "i" for radial zones: my convention is that
c          - Temp is defined at ODD zone numbers
c          - Lum  is defined at EVEN zone numbers
c
c          i runs from imin=0 to imax
c          - imax MUST BE AN ODD NUMBER:
c            Temp is defined there and is used for the outer boundary condition
c          - icore, idrip and ienv should also be odd
c          - imin=0:
c            Lum is defined in the center of the star and Lum(0)=0
c *********************************************************************

      subroutine NSCool(irank,iret,neebrem_logt,neebrem_nalpha,
     1     neebrem_n2,sf_lgtau1,sf_lgtau2,sf_lgr,sf_lgr2)
      
      implicit real*8 (a-h,k-z)
      
      parameter (hbar=1.054d-27,e=4.803d-10,kb=1.38d-16)
      parameter (g=6.67d-8,c=2.99792d10)
      parameter (msol=1.989d33,lsol=3.826d33)
      parameter (year=3.1557600d7)
      parameter (pi=3.1415926535d0)

      parameter (isize=10000)

      parameter (gammacryst=210.d0,gammaliq=180.d0)

      dimension rad(0:isize),rrho(0:isize),pres(0:isize),
     2          debar(0:isize),dvol(0:isize),
     3          emas(0:isize),phi(0:isize),rhod(0:isize)

      dimension bar(0:isize),
     1          yneutr(0:isize),yprot(0:isize),
     2          yelect(0:isize),ymuon(0:isize),
     3          ylambda(0:isize),
     4          ysminus(0:isize),yszero(0:isize),ysplus(0:isize),
     5          yquarku(0:isize),yquarkd(0:isize),yquarks(0:isize),
     6          fhad(0:isize),
     7          theta_k(0:isize),theta_p(0:isize),
     8          a_cell(0:isize),a_ion(0:isize),z_ion(0:isize),
     9     v_ion(0:isize)

      dimension neebrem_logt(56),neebrem_nalpha(56),neebrem_n2(56)
      dimension sf_lgtau1(35),sf_lgtau2(35)
      dimension sf_lgr(35,35),sf_lgr2(35,35)
 
c -----------------------------------------------------------------------
c theta_k and theta_p are the chiral angles for Kaon and Pion condensates
c
c z_ion  is the charge number of the nuclei
c a_ion  is the mass number of the nuclei
c a_cell is the number of nucleons per Wigner-Seitz cell 
c        (i.e., a_cell=a_ion + # dripped neutrons)
c -----------------------------------------------------------------------

      dimension mstp(0:isize),mstn(0:isize),mstla(0:isize),
     2          mstsm(0:isize),msts0(0:isize),mstsp(0:isize)
      dimension kfe(0:isize),kfm(0:isize),kfp(0:isize),kfn(0:isize),
     2          kfla(0:isize),kfsm(0:isize),kfs0(0:isize),kfsp(0:isize),
     3          kfqu(0:isize),kfqd(0:isize),kfqs(0:isize)

c-------------------------------------------------------------------
c The effective masses ms** are the ratio of the 
c Landau effective mass to the free mass.
c The Fermi momenta are in units of fm^-1
c-------------------------------------------------------------------

      dimension idurca_np(0:isize),idurca_lap(0:isize),
     2          idurca_smn(0:isize),idurca_smla(0:isize),
     3          idurca_sms0(0:isize),
     4          idurca_quqd(0:isize),idurca_quqs(0:isize),
     5          durca_ctrl_e(0:isize),durca_ctrl_m(0:isize),
     6          durca_henon_e(0:isize),durca_henon_m(0:isize)
      
      dimension tcn(0:isize),tcp(0:isize),tcla(0:isize),
     2          tcsm(0:isize),tcs0(0:isize),tcsp(0:isize),
     3          tcuu(0:isize),tcdd(0:isize),tcss(0:isize),
     4          tcud(0:isize),tcus(0:isize),tcds(0:isize),
     5          tcu(0:isize),tcd(0:isize),tcs(0:isize)

c April 2004: seperate Tc for flavor AND color:
      dimension tcu1(0:isize),tcu2(0:isize),tcu3(0:isize),
     2          tcd1(0:isize),tcd2(0:isize),tcd3(0:isize),
     3          tcs1(0:isize),tcs2(0:isize),tcs3(0:isize)

c**** Input/Output:

      character*90 filename
      character*79 model
      character*80  f_concryst
      dimension tprint(0:50)
      integer pscreen
      character*3 version
      character*150 f_i,f_Teff,f_Temp,f_Star

c***  AWS new integer to stop if iteration is failing      
      integer istop
      
c**** Heating:

c     Deep crustal heating:
      dimension q_deep_crust(0:isize)
c     Old stuff: MAY NOT WORK !
      real*8 j_44,j_heat
      dimension qdeposit(0:isize)
      dimension mag_field(0:isize),j_heat(0:isize)

c**** Boundary condition:

      character*100 f_TbTs

C***** Local variable used only in the main program

      dimension orad(0:isize),bar1(0:isize),obar(0:isize),
     2          rrho1(0:isize),orrho(0:isize)
      dimension ephi(0:isize),e2phi(0:isize),a2ephin(0:isize),
     2          dephi(0:isize)
      dimension temp(0:isize),otemp(0:isize),ntemp(0:isize)
      dimension ntemp1(0:isize),delt(0:isize),dtemp(0:isize)
      dimension lum(0:isize),olum(0:isize),nlum(0:isize)
      dimension dell(0:isize),dlum(0:isize)

      dimension lambda(0:isize),lambda1(0:isize)
      dimension kappa(0:isize),kappa1(0:isize)
      dimension qnu(0:isize),qnu1(0:isize)
      dimension qqq(0:isize),qqq1(0:isize)
      dimension qeebrem(0:isize),
     1          qnpb(0:isize),qplasma(0:isize),qsynch(0:isize),
     1          qbubble(0:isize),qpair(0:isize),qphoto(0:isize),
     2          qbrem_nn(0:isize),
     3          qmurca_nucl(0:isize),qbrem_nucl(0:isize),
     4          qmurca_hyp(0:isize),qbrem_hyp(0:isize),
     5          qdurca_np(0:isize),
     5          qdurca_lap(0:isize),qdurca_smn(0:isize),
     6          qdurca_smla(0:isize),qdurca_sms0(0:isize),
     7          qfast(0:isize),
     8          qdurca_q(0:isize),qmurca_q(0:isize),
     9          qpbf_n1s0(0:isize),qpbf_n3p2(0:isize),
     x          qpbf_p1s0(0:isize),qpbf_q(0:isize)
      dimension heat(0:isize),heat1(0:isize)
      dimension cv(0:isize),cv1(0:isize)
      dimension cv_n(0:isize),cv_p(0:isize),cv_e(0:isize),cv_m(0:isize),
     1      cv_l(0:isize),cv_sm(0:isize),cv_s0(0:isize),cv_sp(0:isize),
     2      cv_q(0:isize),cv_ion(0:isize)
      dimension gamma(0:isize),cryst(0:isize)

      dimension nbfield2(0:isize)
      
c     Accuracy checking:
c The equation nlum+fp*a2ephi*dtemp is very delicate when the star is
c isothermal: to get convinced that it is solved use:
c      real*16 nlum,ntemp,dtemp,ff
c If these are only real*8 then it may look like the equation is not
c solved properly. However it is, by the Henyey method as soon as
c dell & delt are small enough. That's why it is considered as solved
c either when it is numerically solved (maxff1<chff1) or ratiol<mratl

      dimension fp(0:isize),fq(0:isize),fr(0:isize)
      dimension fp1(0:isize),fq1(0:isize),fr1(0:isize)
      dimension dfp(0:isize),dfq(0:isize),dfr(0:isize)
      dimension fa(0:isize),fb(0:isize),fc(0:isize),ff(0:isize)
      dimension fj(0:isize),fk(0:isize)

      dimension f_gr_field(0:isize),g_gr_field(0:isize),
     1          h_gr_field(0:isize)
      
      dimension cve(0:isize),cvm(0:isize),cvn(0:isize),cvp(0:isize),
     1     cvla(0:isize),cvsm(0:isize),cvs0(0:isize),cvsp(0:isize),
     2     cvqu(0:isize),cvqd(0:isize),cvqs(0:isize)
      
c Auxiliary variables:
      character*5 what

c AWS: setup return value

      iret=0
      
c *********************************************************************
c **********************     LET'S GO !      **************************
c *********************************************************************

      i_model=0

c *********************************************************************
c ************   BEGINNING OF A NEW MODEL CALCULATION   ***************
c *********************************************************************

 1234 continue

      i_model=i_model+1

      idt=1
      htot=0.0
      contraction=0.d0

c *********************************************************************
c *****************     GET INPUT MODEL FILES    **********************
c *********************************************************************
      
caws      if (i_model.eq.1) then
c ***  Choose between two input: **************************************
c      Ask for the input file *****************************************
caws       write(6,*) 'Input master file ='
caws       read(5,*)filename
c ***  Can add here the directory where "Cool_*.in" is:
caws       filename='Model_1/'//filename
c ***  Or define it completely here: **********************************
c       filename='Model_1/Cool_Try.in'
c       write(6,*)'Using as input: ',filename
c**********************************************************************
caws       open(unit=15,file=filename,status='old')
caws      else
caws       read(15,*,end=9997,err=9997)
caws      end if
caws      read(15,*,end=9997,err=9997)version
caws      if (version.eq.'STR') then
caws       istrange=1
caws      else
caws       istrange=0
caws      end if
c     *** BASIC MODEL FILES: **********************************************
caws      read(15,*,end=9997,err=9997)
      if (i_model.eq.1) then
         version='NEW'
      else
         goto 9997
      end if
c *** OTHER MODEL FILES: **********************************************
caws      read(15,*,end=9997,err=9997)
c *** OUTPUT FILES: ***************************************************
caws      read(15,*)                    
caws      read(15,*,end=9997,err=9997)f_i
caws      read(15,*,end=9997,err=9997)f_Teff
caws      read(15,*,end=9997,err=9997)f_Temp
c     aws      read(15,*,end=9997,err=9997)f_Star
      f_i=' '
      f_Teff='Teff.dat'
      f_Temp='Temp.dat'
      f_Star='Star.dat'
c      f_Teff='Model_1/Teff_Try.dat'
c      f_Temp='Model_1/Temp_Try.dat'
c      f_Star='Model_1/Star_Try.dat'
c**********************************************************************
c *** PRINT ON THE SCREEN THE FILES:
c Notice: pscreen will be read from file "I.dat"
c     AWS. pscreen is now set in the c wrapper
c      pscreen=1
c$$$      if (pscreen.gt.0) then
c$$$       print *,'-------------------------------------------------'
c$$$       print *,'Here are the files: '
c$$$       write(6,*)version(1:istrlen(version))
c$$$       write(6,*)f_i(1:istrlen(f_i))
c$$$       write(6,*)f_Teff(1:istrlen(f_Teff))
c$$$       write(6,*)f_Temp(1:istrlen(f_Temp))
c$$$       write(6,*)f_Star(1:istrlen(f_Star))
c$$$       print *,'-------------------------------------------------'
c$$$       print *,'LET''S GO !'
c$$$       print *,'-------------------------------------------------'
c$$$       print '(1a28,1a50)','****************************',
c$$$     1  '**************************************************'
c$$$      end if

c *********************************************************************
c *****************     READ THE ABOVE FILES     **********************
c *********************************************************************

c     *********************************************************************
c     **************     Read Numerical Parameters    *********************
c     *********************************************************************

      call nscool_num_param(irank,time0,timemax,istepmax,itrial_max,
     1     itrial_opt,tcut,dtime,dtlimit,scale_dt0,scale_dt1,repeat,
     2     istart,mratt,mratl,mrats,tvar,svar,tcon)
      
      dtime=dtime*year
      time0=time0*year
      odtime=dtime
      scale_dt=scale_dt1

c     *********************************************************************
c     ********     Initialize the cooling calculation     *****************
c     *********************************************************************

      call nscool_cool_param(irank,pscreen,debug_keep,istep_debug,pteff,
     1     ptemp,pstar,idump1,idump2,idump3,tempmin,tempini,
     2     icvel_nodeg,emnco,emncr,emp,p0,itpmax,tprint)

      debug=debug_keep

c     READ OUTER BOUNDARY PARAMETERS: ************************************

      call nscool_bound_param(irank,ifteff,eta,mag_coeff,tb_acc0)
      
      eta=eta/0.443d0
      eta_0=eta

c     READ PAIRING PARAMETERS: ********************************************
c$$$  if (debug.ge.1.) print *,'Opening I_Pairing*.dat'
c$$$  open(unit=20,file=f_Pairing,status='old')
c$$$  read(20,*)
c$$$  read(20,*)sfn1s0
c$$$  read(20,*)sfn3p2
c$$$  read(20,*)sfp1s0
c$$$  read(20,*)sfl1s0
c$$$  read(20,*)fn1s0
c$$$  read(20,*)fn3p2
c$$$  read(20,*)fp1s0
c$$$  read(20,*)fl1s0
c$$$  read(20,*)sfquark
c$$$  if (sfquark.eq.1.) read(20,*)f_quark_tc       ! File for Quark pairing
c$$$  if (sfn3p2.eq.150.) then
c$$$  read(20,*)
c$$$  read(20,*)kfmax_n3p2
c$$$  read(20,*)delkf_n3p2
c$$$  read(20,*)tcmax_n3p2
c$$$  write (*,*) 'tcmax: ',tcmax_n3p2
c$$$  end if
c$$$  close(unit=20,status='keep')
      sfn1s0=1
      sfn3p2=101
      sfp1s0=3
      sfl1s0=0
      fn1s0=1
      fn3p2=1
      fp1s0=1
      fl1s0=1
      sfquark=0

c     READ NEUTRINO PARAMETERS: ********************************************
c$$$  if (debug.ge.1.) print *,'Opening I_Neutrino*.dat'
c$$$  open(unit=20,file=f_Neutrino,status='old')
c$$$  read(20,*)
c$$$  read(20,*)murca_increase
c$$$  read(20,*)inu_durca
c$$$  read(20,*)inu_eion
c$$$  read(20,*)inu_plasma
c$$$  read(20,*)inu_synch
c$$$  read(20,*)inu_n1s0_pbf
c$$$  read(20,*)inu_n3p2_pbf
c$$$  read(20,*)inu_p_pbf
c$$$  read(20,*)inu_bubble
c$$$  read(20,*)inu_photo
c$$$  read(20,*)inu_pair
c$$$  read(20,*)inu_nuts1       ! Not used so far
c$$$  read(20,*)inu_nuts2       ! Not used so far
c$$$  read(20,*)inu_nuts3       ! Not used so far
c$$$  read(20,*)inu_nuts4       ! Not used so far
c$$$  read(20,*)inu_nuts5       ! Not used so far
c$$$  read(20,*)
c$$$  read(20,*)rhoexo
c$$$  read(20,*)cexo
c$$$  read(20,*)pexo
c$$$  read(20,*)pexosn
c$$$  read(20,*)pexosp
c$$$  read(20,*)nonothing1      ! Not used so far
c$$$  read(20,*)nonothing2      ! Not used so far
c$$$  read(20,*)nonothing3      ! Not used so far
c$$$  read(20,*)nonothing4      ! Not used so far
c$$$  close(unit=20,status='keep')
      murca_increase=0.0
      inu_durca=1
      inu_eion=1
      inu_plasma=1
      inu_synch=0
      inu_n1s0_pbf=1
      inu_n3p2_pbf=1
      inu_p_pbf=1
      inu_bubble=0
      inu_photo=0
      inu_pair=0
      inu_nuts1=0
      inu_nuts2=0
      inu_nuts3=0
      inu_nuts4=0
      inu_nuts5=0
      rhoexo=1.2d25
      cexo=1.0d25
      pexo=0
      pexosn=0
      pexosp=0
      nonothing1=0
      nonothing2=0
      nonothing3=0
      nonothing4=0
c     READ CONDUCTIVITY PARAMETERS: ********************************************
c$$$  if (debug.ge.1.) print *,'Opening I_Conduct*.dat'
c$$$  open(unit=20,file=f_Conduct,status='old')
c$$$  read(20,*)
c$$$  read(20,*)iopacity
c$$$  read(20,*)icon_crust
c$$$  read(20,*)icon_core
c$$$  read(20,*)iconnothing2      ! Not used so far
c$$$  read(20,*)iconnothing3      ! Not used so far
c$$$  read(20,*)iconnothing4      ! Not used so far
c$$$  read(20,*)iconnothing5      ! Not used so far
c$$$  read(20,*)connothing1       ! Not used so far
c$$$  read(20,*)connothing2       ! Not used so far
c$$$  read(20,*)qimp
c$$$  read(20,*)connothing4       ! Not used so far
c$$$  read(20,*)connothing5       ! Not used so far
c$$$  close(unit=20,status='keep')
      iopacity=0
      icon_crust=3
      icon_core=2
      iconnothing2=0
      iconnothing3=0
      iconnothing4=0
      iconnothing5=0
      connothing1=0.0
      connothing2=0.0
      qimp=1.0d-10
      connothing4=0.0
      connothing5=0.0

c     READ HEATING PARAMETERS: ******************************************
c$$$  if (debug.ge.1.) print *,'Opening I_Heat*.dat'
c$$$  open(unit=20,file=f_Heat,status='old')
c$$$  read(20,*)
c$$$  c      Deep crustal heating:
c$$$  read(20,*)i_heat_deep_crust
c$$$  c      Sudden heating:
c$$$  read(20,*)i_heat_deposit
c$$$  read(20,*)t_dep
c$$$  read(20,*)del_t_dep
c$$$  read(20,*)total_heat
c$$$  read(20,*)i_dep
c$$$  c      Baryon --> SQM conersion heating:
c$$$  read(20,*)i_heat_convert           ! these values are overwritten by the ones
c$$$  read(20,*)MeV_neutron              ! read from the file I_Strange
c$$$  c      Vortex creep heating:
c$$$  read(20,*)i_heat_vortex_creep
c$$$  read(20,*)j_44
c$$$  c      Joule heating:
c$$$  read(20,*)i_heat_joule
c$$$  c      Field decay heating:
c$$$  read(20,*)i_heat_field_decay
c$$$  c      Heating parameters for future use:
c$$$  read(20,*)i_heat_cold1
c$$$  read(20,*)i_heat_cold2
c$$$  read(20,*)i_heat_cold3
c$$$  read(20,*)i_heat_cold4
c$$$  read(20,*)heat_cold1
c$$$  read(20,*)heat_cold2
c$$$  read(20,*)heat_cold3
c$$$  read(20,*)heat_cold4
c$$$  close(unit=20,status='keep')
      i_heat_deep_crust=0
      i_heat_deposit=0
      t_dep=0.d0
      del_t_dep=0.d0
      total_heat=0.d0
      i_dep=0
      i_heat_convert=0
      MeV_neutron=0
      i_heat_vortex_creep=0
      j_44=0.d0
      i_heat_joule=0
      i_heat_field_decay=0
      i_heat_cold1=0
      i_heat_cold2=0
      i_heat_cold3=0
      i_heat_cold4=0
      heat_cold1=0
      heat_cold2=0
      heat_cold3=0
      heat_cold4=0

c     READ MAGNETIC FIELD PARAMETERS: **********************************
c$$$  if (debug.ge.1.) print *,'Opening I_Bfield*.dat'
c$$$  open(unit=20,file=f_Bfield,status='old')
c$$$  read(20,*)
c$$$  read(20,*)ifield
c$$$  read(20,*)i_gr_field
c$$$  read(20,*)bfield0
c$$$  read(20,*)i0
c$$$  read(20,*)i1
c$$$  read(20,*)nothing
c$$$  read(20,*)start_b_diffusion
c$$$  read(20,*)i_joule_heat
c$$$  read(20,*)i_conb
c$$$  read(20,*)i_eleconb
c$$$  close(unit=20,status='keep')
      ifield=0
      i_gr_field=1
      bfield=0.0d12
      i0=113
      i1=289
      nothing=0.1
      start_b_defusion=0.0
      i_joule_heat=0
      i_conb=0
      i_eleconb=0

c     READ ACCRETION PARAMETERS: **************************************
c$$$  if (debug.ge.1.) print *,'Opening I_Accretion*.dat'
c$$$  open(unit=20,file=f_Accretion,status='old')
c$$$  read(20,*)        
c$$$  read(20,*)i_acc       
c$$$  read(20,*)m_dot0
c$$$  read(20,*)t_acc0
c$$$  t_acc0=t_acc0*3.15576d7  ! convert year=365.25 days to seconds
c$$$  read(20,*)t_acc1
c$$$  t_acc1=t_acc1*3.15576d7 
c$$$  read(20,*)t_acc2
c$$$  t_acc2=t_acc2*3.15576d7 
c$$$  read(20,*)alpha_acc
c$$$  read(20,*)time_step_min
c$$$  time_step_min=time_step_min*3.15576d7 
c$$$  read(20,*)eta_Edd
c$$$  read(20,*)X_Edd
c$$$  close(unit=20,status='keep')
      i_acc=0
      m_dot0=0.0
      t_acc0=0.0
      t_acc1=0.0
      t_acc2=0.0
      alpha_acc=0.0
      t_step_min=0.0
      eta_Edd=0.0
      X_Edd=0.0

      t_acc0=t_acc0*3.15576d7
      t_acc1=t_acc1*3.15576d7
      t_acc2=t_acc2*3.15576d7
      time_step_min=time_step_min*3.15576d7
      eta=max(1.d-50,eta)
      eta0=max(1.d-50,eta0)
c     READ STRANGE QUARK MATTER PARAMETERS: ***************************
c$$$  if (istrange.eq.1) then
c$$$  if (debug.ge.1.) print *,'Opening I_Stange*.dat'
c$$$  open(unit=20,file=f_Strange,status='old')
c$$$  read(20,*)c_nu_str          ! SQM neutrino emission
c$$$  read(20,*)p_nu_str          ! SQM neutrino emission
c$$$  read(20,*)c_con_str         ! SQM thermal conductivity
c$$$  read(20,*)p_con_str         ! SQM thermal conductivity
c$$$  read(20,*)c_cv_str          ! SQM specific heat
c$$$  read(20,*)i_heat_convert    ! Heating from baryon -> SQM conversion
c$$$  read(20,*)MeV_neutron       ! Heating from baryon -> SQM conversion
c$$$  close(unit=20,status='keep')
c$$$  else if (istrange.eq.0) then
c$$$  c_nu_str      =0.d0
c$$$  p_nu_str      =0.d0
c$$$  c_con_str     =0.d0
c$$$  p_con_str     =0.d0
c$$$  c_cv_str      =0.d0
c$$$  i_heat_convert=0.d0
c$$$  MeV_neutron   =0.d0
c$$$  else
c$$$  print *,'NSCool_READ: istrange not defined !'
c$$$  stop
c$$$  end if

c     READ STAR STRUCTURE LAYOUT FILE: ********************************
c$$$  if (debug.ge.1.) print *,'Opening I_Structure*.dat'
c$$$  read(20,*)rhocore,rhodrip,rhoenv,rhosurf
c$$$  read(20,*)icore,idec
c$$$  close(unit=20,status='keep')

c     rhocore is the lowest density in the core, i.e.
c         if rho=rhocore then it has to be considered as in the core
c

      rhocore=1.6e14
      rhodrip=4.0d11
      rhoenv=1.0d8
      rhosurf=1.0d10
      icore=111
      idec=60

c *********************************************************************
c *****************        INITIALIZATION        **********************
c *********************************************************************
      if (debug.ge.1.) print *,'Initializing'

c *********************************************************************
c *** Get the time independent pieces of physics: *********************
c *********************************************************************
c     get_core_chemistry MUST be called BEFORE get_crust_chemistry

      call grid(irank,idec,rhocore,rhodrip,rhoenv,rhosurf,
     1     imax,icore,idrip,ienv,rad,rrho,pres,dvol,emas,phi)
      call get_core_chemistry(irank,version,imax,icore,rrho,
     1     bar,yneutr,yprot,yelect,ymuon,ylambda,ysminus,yszero,
     2     ysplus,yquarku,yquarkd,yquarks,theta_k,theta_p,fhad,
     3     mstn,mstp,mstla,mstsm,msts0,mstsp)
      
      if (icore.eq.0) then
         iret=2
         print *,'Problem in get_core_chemistry().'
         goto 9997
      end if
      
      call get_crust_chemistry(irank,debug,version,imax,icore,
     1     rrho,pres,debar,dvol,bar,a_cell,a_ion,z_ion,v_ion,
     2     yelect,yneutr)
      call get_fermi_momenta(irank,imax,icore,rrho,bar,yneutr,
     1     yprot,yelect,ymuon,ylambda,ysminus,yszero,ysplus,yquarku,
     2     yquarkd,yquarks,fhad,theta_k,theta_p,
     3     kfn,kfp,kfe,kfm,kfla,kfsm,kfs0,kfsp,kfqu,kfqd,kfqs,
     4     idurca_np,idurca_lap,durca_ctrl_e,durca_ctrl_m,
     5     idurca_smn,idurca_smla,idurca_sms0,idurca_quqd,idurca_quqs,
     6     durca_henon_e,durca_henon_m)

      if (icore.eq.0) then
         iret=4
         print *,'Problem in get_fermi_momenta().'
         goto 9997
      end if
      
      call get_effective_masses(version,emnco,emncr,emp,
     1     kfn,kfp,mstn,mstp,mstla,mstsm,msts0,mstsp,idrip,icore)
      call get_spec_heat_degenerate(cve,cvm,cvn,cvp,cvla,
     1     cvsm,cvs0,cvsp,cvqu,dvqd,cvqs,
     2     kfe,kfm,kfn,kfp,kfla,kfsm,kfs0,kfsp,kfqu,kfqd,kfqs,
     3     mstn,mstp,mstla,mstsm,msts0,mstsp,fhad,imax)
      call get_Tc(irank,imax,icore,idrip,
     1     tcn,tcp,tcla,tcsm,tcs0,tcsp,
     2     tcuu,tcdd,tcss,tcud,tcus,tcds,
     3     tcu,tcd,tcs,
     4     sfn1s0,sfn3p2,sfp1s0,sfl1s0,
     5     fn1s0,fn3p2,fp1s0,fl1s0,
     6     kfmax_n3p2,delkf_n3p2,tcmax_n3p2,isf,
     7     kfn,kfp,kfla,kfqu,kfqd,kfqs,bar,fhad,yquarku,yquarkd,yquarks)
      call get_degenerate_density(irank,rrho,pres,rhod,imax,ienv)
      
c *********************************************************************
c ***** Calculate the T-independent coefficients **********************
c *********************************************************************
      if (debug.ge.1.) print *,'Calculating T-independent coeff.'

      do i=0,imax
       ephi(i)=dexp(phi(i))
       e2phi(i)=ephi(i)**2
       a2ephin(i)=(4.d0*pi*rad(i)**2)**2*ephi(i)
      end do

      dephi(0)=0.d0
      do i=1,imax-1
       dephi(i)=(ephi(i+1)-ephi(i-1))/(rad(i+1)-rad(i-1))
      end do
      dephi(imax)=dephi(imax-1)

      radius=rad(imax)
      root=dsqrt(1.-2.d0*g*msol*emas(imax)/rad(imax)/c**2)
      factor=root/(4.d0*pi*rad(imax)**2)/6.022d23*1.d39
      constant=4.d0*pi*g*msol*emas(imax)*4.d0/3.d0*5.67d-5*
     1         e2phi(imax)/pres(imax)/root

      gs=g*msol*emas(imax)/rad(imax)**2/root
      gs14=gs/1.d14

      compactness=2.d0*g*msol*emas(imax)/rad(imax)/c**2

c *********************************************************************
c *** Initialize some more stuff: *************************************
c *********************************************************************

c$$$      if (i_heat_deep_crust.eq.1) then
c$$$       call initialize_heating_deep_crust
c$$$      end if
c$$$      if (i_heat_deposit.eq.1) then
c$$$       call initialize_heating_deposit
c$$$      end if
c$$$      if (i_heat_convert.eq.1) then
c$$$       call initialize_heating_convert(MeV_neutron)
c$$$      end if
c$$$      call get_beta(emas(imax),rad(imax),beta_rot,m_i)
c$$$c      if (ifield.ne.0) call initialize_dipole(i0,i1)
c$$$c      call initialize_accretion_rate
c$$$      if (i_heat_vortex_creep.eq.1) then
c$$$       call initialize_heating_vortex_creep(imax,rad,rrho,tcn,dvol,j_44)
c$$$      end if
c BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
c This is to initialize the Bfield safely 
c when Bfield will be eliminated:
      do i=imax,1,-2
       nbfield2(i)=0.d0
      end do

c *********************************************************************
c ********* Calculate the initial Temp and Lum profiles ***************
c *********************************************************************

c **** calculate the t & l profile ************************************
      
      if (debug.ge.1.) print *,'Calculating Initial T profile'
c$$$       if (ifteff.ne.15) then
c$$$       tsurface=ephi(imax) *1.d11
c$$$       tdrip   =ephi(idrip)*3.d11
c$$$       tcore   =ephi(0)    *1.d12
c$$$       if (tempini.ne.0.) then
c$$$        tsurface=0.5 * ephi(imax) *tempini
c$$$        tdrip   =0.8 * ephi(idrip)*tempini
c$$$        tcore   =1.0 * ephi(0)    *tempini
c$$$       else
c$$$        tsurface=1.e9
c$$$        tdrip   =2.e10
c$$$        tcore   =1.e11
c$$$       end if
c$$$      else                     ! this is for fixed T_b = tb_acc0
c$$$       tb_acc0=tb_acc0*ephi(imax)
c$$$       tsurface=tb_acc0
c$$$       tdrip   =tb_acc0
c$$$       tcore   =tb_acc0
c$$$      end if
      call nscool_tptr_init(irank,ifteff,tempini,ephi(imax),
     1     ephi(idrip),ephi(0),tsurface,tdrip,tcore,tb_acc0)
      dec=1.1

      do i=0,icore
       temp(i)=tcore
      end do
      do i=icore+1,idrip
       w1=(dlog10(rad(idrip))-dlog10(rad(  i  )))/
     1    (dlog10(rad(idrip))-dlog10(rad(icore)))
       w2=1.d0-w1
       lt=w1*dlog10(tcore)+w2*dlog10(tdrip)
       temp(i)=10.d0**lt
      end do

      do i=idrip+1,imax
       w1=(dlog10(rad(imax))-dlog10(rad(  i  )))/
     1    (dlog10(rad(imax))-dlog10(rad(idrip)))
       w2=1.d0-w1
       lt=w1*dlog10(tdrip)+w2*dlog10(tsurface)
       temp(i)=10.d0**lt
      end do

      dtemp(0)=0.
      do i=2,imax-1,2
       dtemp(i)=(temp(i+1)-temp(i-1))/(debar(i)+debar(i+1))
      end do

c *********************************************************************
c ***** calculate the inner envelope profile **************************

      if (debug.ge.1.) print *,'Calculating envelope profile'
      do i=imax-1,ienv+1,-2
       x=(dlog(rrho(i+1))-dlog(rrho(i)))/
     1   (dlog(rrho(i+1))-dlog(rrho(i-1)))
       y=(dlog(rrho(i))-dlog(rrho(i-1)))/    
     1   (dlog(rrho(i+1))-dlog(rrho(i-1)))
       ltemp=y*dlog(temp(i+1))+x*dlog(temp(i-1))
c       ltemp=y*log(temp(i+1))+x*log(temp(i-1))
       temp(i)=dexp(ltemp)
      end do

      do i=ienv,imax
       if (temp(i).lt.tcon) then      
        call density(temp(i)/ephi(i),pres(i),a_ion(i),z_ion(i),rrho(i))
        rrho(i)=min(rrho(i),rhod(i))
        bar(i)=6.022d-16*rrho(i)
        dr=debar(i)/rrho(i)*factor
        rad(i+1)=rad(i)+dr
       end if
      end do

c *********************************************************************

      if (debug.ge.1.) print *,'Calculating initial L profile'
      do i=imax,1,-2
         call conduct(i,temp(i)/ephi(i),rrho(i),
     1        a_cell(i),a_ion(i),z_ion(i),qimp,
     2        nbfield2(i),
     3        sig,lambda(i),debug,
     4        nu_e_s,nu_e_l,icon_crust,icon_core,rhodrip,rhocore,
     5        kfe,kfm,kfn,kfp,kfkfla,kfsm,kfs0,kfsp,mstn,mstp,
     6        mstla,mstsm,msts0,mstsp,tcn,tcp,tcla,tcsm,tcs0,tcsp,isf,
     7        fhad,istrange)
         call opacity(temp(i)/ephi(i),rrho(i),a_cell(i),z_ion(i),
     1        kappa(i),iopacity)
         acd=7.56d-15*c/(3.d0*kappa(i)*rrho(i))
         fp(i)=(lambda(i)+4.d0*acd*(temp(i)/ephi(i))**3)*bar(i)/lsol
      end do

      lum(0)=0.
      do i=2,imax-1,2
       lum(i)=-(fp(i+1)+fp(i-1))/2.d0*a2ephin(i)*dtemp(i)
       if (lum(i).eq.0.0) lum(i)=1.d-3
      end do

      do i=1,imax-2,2
       dlum(i)=(nlum(i+1)-nlum(i-1))/(debar(i)+debar(i+1))
      end do
c look +++++++++++++++++++++++++++++++++++++++++++++
      dlum(imax)=0.
c      dlum(imax)=dlum(imax-2)
c ++++++++++++++++++++++++++++++++++++++++++++++++++
     
      do i=1,imax
       otemp(i)=temp(i)
       olum(i-1)=lum(i-1)
      end do

c *********************************************************************

      do i=0,imax
       orad(i)=rad(i)
       orrho(i)=rrho(i)
       rrho1(i)=rrho(i)
       obar(i)=bar(i)
       bar1(i)=bar(i)
      end do

c *********************************************************************


c *********************************************************************
c ****************** Open print out files *****************************
c *********************************************************************

c *********************************************************************
c PRINT OUT STAR PROPERTIES *******************************************
c *********************************************************************
      if (pstar.eq.1.) then
c       open(unit=49,file=f_Star,status='new')
        itext=5
        write(49,'(4i10)')itext,imax,icore,idrip
        write(49,*)
        write(49,'(a79)')model
        write(49,*)
        write(49,
     1   '(a5,2a15,3a12,3x,
     2     8a12,3x,
     3     3a12,3x,
     4     a5,
     5     a5,3a7)')
     1     'i  ','rad  ','emas  ','rho  ','pres ','nb ',
     2     ' kf(e)','kf(mu)',' kf(p)',' kf(n)',
     2        'kf(la)','kf(S-)','kf(S0)','kf(S+)',
     3     'Tc(n) ','Tc(p) ','Tc(la) ',
     4     'Durca',
     5     'nSF',' Acel',' Aion',' Zion'
        write(49,*)
c       Print out core:
        do i=0,icore
         if (i.gt.isf)       what='  1s0'
         if (i.le.isf)       what='  3p2'
         if (tcn(i).eq.1.d0) what='  no '
         write(49,
     1    '(i5,1p2e15.6,1p3e12.3,3x,
     2      1p8e12.3,3x,
     3      1p3e12.3,3x,
     4      5i1,
     5      a5,3a7)')
     1      i,rad(i)/100.,emas(i),rrho(i),pres(i),bar(i),
     2      kfe(i),kfm(i),kfp(i),kfn(i),kfla(i),kfsm(i),kfs0(i),kfsp(i),
     3      tcn(i),tcp(i),tcla(i),
     4      idurca_np(i),
     4       idurca_lap(i),idurca_smn(i),idurca_smla(i),idurca_sms0(i),
     5      what,' N/A ',' N/A ',' N/A '
        end do
c       Print out inner crust:
        do i=icore+1,idrip
         if (i.gt.isf)       what='  1s0'
         if (i.le.isf)       what='  3p2'
         if (tcn(i).eq.1.d0) what='  no '
         write(49,
     1     '(i5,1p2e15.6,1p3e12.3,3x,
     2       1p1e12.3,24x,1p1e12.3,48x,3x,
     3       1p1e12.3,24x,3x,
     4       5x,       
     5       a5,0p3f7.1)')       
     1       i,rad(i)/100.,emas(i),rrho(i),pres(i),bar(i),
     2       kfe(i),kfn(i),
     3       tcn(i),
     5       what,a_cell(i),a_ion(i),z_ion(i)
        end do
c       Print out outer crust:
        do i=idrip+1,imax
         what=' '
         write(49,
     1     '(i5,1p2e15.6,1p3e12.3,3x,
     2       1p1e12.3,84x,3x,
     3       36x,3x,
     4       5x,
     5       a5,0p3f7.1)')
     1       i,rad(i)/100.,emas(i),rrho(i),pres(i),bar(i),
     2       kfe(i),
     5       what,a_cell(i),a_ion(i),z_ion(i)
        end do
       close(unit=49,status='keep')
      end if
c ***********************************************************************
c **** OPEN TEFF file: **************************************************
c ***********************************************************************
      if (pteff.ge.1.) then
      if (debug.ge.1.) print *,'Opening Teff file'
       rstar=rad(imax)
       mstar=emas(imax)
       rdurca=-1.d0
       mdurca=-1.d0
       rhodurca=-1.d0
       do i=0,icore
        if(idurca_np(i).eq.1)then
         rdurca=rad(i)
         mdurca=emas(i)
         rhodurca=rrho(i)
        end if
       end do
       if(jexo.gt.-1) then
        rexo=rad(jexo)
        mexo=emas(jexo)
       end if
       rcore=rad(icore)
       mcore=emas(icore)
       rincrust=rad(idrip)-rcore
       mincrust=emas(idrip)-mcore
       routcrust=rad(imax)-rad(idrip)
       moutcrust=pres(idrip)*4.d0*pi*(rad(idrip)/msol**.25)**4*
     1           root/g/emas(idrip)/msol

c$$$       call nscool_teff_header(imax,rad,emas,pres,idurca_np,jexo,
c$$$     1      icore,idrip,msol,root,g)
       
c       open(unit=19,file=f_Teff,status='new')

       write(19,*)
       write(19,*)model
       write(19,*)
       write(19,1230)'Mstar     =',    mstar,'Rstar     =',    rstar
       if(rdurca.gt.0.)then
        write(19,1230)'Mdurca    =',   mdurca,'Rdurca    =',   rdurca
       end if
       if(jexo.gt.-1)then
        write(19,1230)'Mexo      =',     mexo,'Rexo      =',     rexo
       end if
       write(19,1230)'Mcore     =',    mcore,'Rcore     =',    rcore
       write(19,1230)'Mincrust  =', mincrust,'Rincrust  =', rincrust
       write(19,1230)'Moutcrust =',moutcrust,'Routcrust =',routcrust
       write(19,*)
1230   format(10x,2(1a15,1p1e10.3))

       write(19,9750)
     1   'icore=',icore,'sfn1s0=',sfn1s0,               'emnco=',emnco,
     2     'rhoexo=',rhoexo
       write(19,9751)
     1   'idrip=',idrip,'sfn3p2=',sfn3p2,'fn3p2=',fn3p2,'emncr=',emncr,
     2     '  cexo=',  cexo
       write(19,9751)
     1   ' imax=', imax,'sfp1s0=',sfp1s0,'fp1s0=',fp1s0,'  emp=',  emp,
     2     '  pexo=',  pexo
       write(19,*)
 9750  format(a8,i4,a10,0p1f3.0,8x,  5x    ,a10,0p1f3.0,a10,1p1e12.3)
 9751  format(a8,i4,a10,0p1f3.0,a8,0p1f05.2,a10,0p1f3.0,a10,1p1e12.3)
 9752  format(a8,i4,a10,0p1f3.0,8x,  5x    ,a10,0p1f3.0,a10,0p1f05.0)

       if (rdurca.le.0.)then
          write(19,9791)'DUrca not possible'
       else
          if (inu_durca.eq.0) then
             write(19,9791)'DUrca possible but TURNED OFF'
          else 
             write(19,9792)'DUrca active above Rho =',rhodurca,
     2                     'gm/cm^3'
          end if
       end if
       write(19,9790)'e-ion bremstrahlung:',inu_eion,
     2                  'bubble neutrinos:',inu_bubble
       write(19,9790)'plasma neutrinos:',inu_plasma,
     2               ' neutron 1S0 pbf:',inu_n1s0_pbf
       write(19,9790)'  pair neutrinos:',inu_pair,
     2               ' neutron 3P2 pbf:',inu_n3p2_pbf
       write(19,9790)' photo neutrinos:',inu_photo,
     2               '  proton 1S0 pbf:',inu_p_pbf
       write(19,*)
 9790  format (2(1a30,1i3))
 9791  format(a30)
 9792  format(a30,1p1e12.3,a8)

       if (ifteff.eq.0) then
        write(19,'(a30,a40)')'Envelope from file: ',f_TbTs
       else if (ifteff.eq.1) then
        write(19,'(a40)')' Iron envelope from Gundmundsson et al'
       else if (ifteff.eq.2) then
        write(19,'(a40)')' Iron envelope from Nomoto & Tsuruta'
       else if (ifteff.eq.3) then
        etap=eta
        if (eta.le.1.d-40) etap=0.0
        write(19,'(a58,1p1e8.2)')
     2    'Accreted envelope from Potekhin et al. with Eta = ',etap
       else if (ifteff.eq.10) then
        write(19,'(a40,1p1e10.1,a3)')
     2    ' Magnetized Iron envelope with',0.d0,' G'
       else if (ifteff.eq.11) then
        write(19,'(a40,1p1e10.1,a3)')
     2    ' Magnetized Accreted envelope with',0.d0,' G'
       else if (ifteff.eq.15) then
        write(19,'(a40,1p1e10.1)')
     2    'Fixed outer boundary temperature T_b =',tb_acc0     
       end if
       write(19,*)
c ***********************************************************************
       write(19,*)
       if (pteff.eq.1.0) then
        write(19,1231)
     1   'Step',' Time  ',' Teff   ',
     2   ' L_phot  ',' L_nu   ',' L_heat '
        write(19,1231)
     1    '   ','[years]',' at inf [K] ',
     2    '[erg/sec]','[erg/sec]','[erg/sec]'
        write(19,*)
 1231   format(a8,a12,5x,4a12)
       else
        write(19,1235)
     1  'step','Time      ','Teff   ',
     2  'B field ','Period ',
     3  'dt  ','dt/odt','dtemp','itrial',
     4  'Temp(i=',idump1,')','Temp(i=',idump2,')','Temp(i=',idump3,')',
     5  'Cycle','M_dot  ',
     6  'Heat  ','L_H   ','L_nu(core)','L_nu(crust',
     7  'Cv_core ','Cv_crust '
1235    format(
     1   1a8 , 1a22 , 1a16 ,
     2   1a12 , 1a12 ,
     3   1a10 , 2a8 , 1a8 ,
     4   3(a12,i3,a1) ,
     5   1a8 , 1a14 ,
     6   4a12,
     7   2a12    )
        write(19,1236)
     1   ' ','[years]     ','at inf [K] ',
     2   '[G]   ','[sec.] ',
     3   ' ',' ',' ',' ',   
     4   rrho(idump1),rrho(idump2),rrho(idump3),
     5   ' ','[Msun/yr]',
     6   '[erg] ','[erg/sec]','[erg/sec]','[erg/sec]'
 1236   format(
     1   1a8 , 1a22 , 1a16 ,
     2   1a12 , 1a12 ,
     3   1a10 , 2a8 , 1a8 ,
     4   1p3e16.3 ,
     5   1a8 , 1a14 ,
     6   4a12 )
        write(19,*)
       end if
      end if
c ***********************************************************************
c ***********************************************************************
c ***********************************************************************


c *********************************************************************
c ************************     COOLING     ****************************
c *********************************************************************

c *********************************************************************
      time=time0       ! Initialize the time
      icycle=0         ! Initialize the counter for accretion cycles
c *********************************************************************

c *********************************************************************
c      itprint=0       ! To print out the initial T and L profiles
      itprint=1        ! To print out only at the required times
c *********************************************************************

c *********************************************************************
c     THIS IS THE MAIN TIME LOOP:
      do 9999 istep=1,istepmax
c *********************************************************************
         
        debug=0.
        if (istep.ge.istep_debug) debug=debug_keep
        if (debug.ge.1.) print *,'Going: istep=',istep

c *********************************************************************
2345  itrial=0         ! Branch back here in case:
                       !  - Too many iteration in Newton-Raphson
                       !  - Envelope boundary condition cannot be solved
                       !  - Temp has changed too much
c *********************************************************************

      ratiot=1.d-2
      ratiol=1.d-2

c     AWS: New test if dtime vanishes
c     AWS, 3/4/18: changed from 1.0d-20 to 1.0d-18 to help solve
c     hangups with MCMC     
      if (dtime/year.le.1.0d-20) then
         iret=1
         print *,dtime/year,"NSCool failed. dtime too small."
         goto 9997
      end if
      
c *********************************************************************
      
c     ---------------------------------------------------------------------
      if (pscreen.ge.2) then
       print '(1a28,1a50)','****************************',
     1  '**************************************************'
       print '(2a10,1i5,1a53)','**********','step#=',istep,
     1  '**************************************************'
       print '(1a28,1a50)','****************************',
     1  '**************************************************'
       read(5,*)        ! Wait for <ENTER> befor printing out
       print '(2(a10,1p1e10.3),a30,0p1f6.3)',
     1                'time =',(time+dtime)/year,
     2               'dtime =',dtime/year,
     3        'dtime/odtime =',dtime/odtime
       print *
       if (chtemp.eq.1.) then
        print '(a42,0p1f5.2,a9,1p1e9.2,a3,1p1e9.2)',
     1        'dtime limited by TEMP change, max_dtemp =',max_dtemp,
     2        'at rho=',rrho(icht),'T=',temp(icht)
       end if
c       if (chstoke.eq.1.) then
c        print '(a42,0p1f5.2,a9,1p1e9.2,a3,1p1e9.2)',
c     1        'dtime limited by STOKE change, mdstoke =',mdstoke,
c     2        'at rho=',rrho(ichs),'S=',stoke(ichs)
c       end if
       if (chtrial.eq.1) then
        print '(a40)',
     1        '   dtime limited by ITRIAL'
       end if
      end if
c ---------------------------------------------------------------------

c *********************************************************************
c ***** Calculate ntemp & nlum for first guess ************************
c *********************************************************************

      if (debug.ge.1.) print *,'Guessing NLum & NTemp'
      coeff_int=0.8d0
      do i=1,imax,2
       ntemp(i)=temp(i)+coeff_int*(temp(i)-otemp(i))*dtime/odtime
      end do
      dtemp(0)=0.d0
      do i=2,imax-1,2
       dtemp(i)=(ntemp(i+1)-ntemp(i-1))/(debar(i)+debar(i+1))
      end do

      nlum(0)=0.
      do i=2,imax-1,2
       nlum(i)=lum(i)+coeff_int*(lum(i)-olum(i))*dtime/odtime
      end do
      do i=1,imax-2,2
       dlum(i)=(nlum(i+1)-nlum(i-1))/(debar(i)+debar(i+1))
      end do
c look +++++++++++++++++++++++++++++++++++++++++++++
      dlum(imax)=0.d0
c      dlum(imax)=dlum(imax-2)
c ++++++++++++++++++++++++++++++++++++++++++++++++++

c ---------------------------------------------------------------------
      if (pscreen.eq.3)then
       print '(1i3,1a5)', 0,     '-----'
      end if
c ---------------------------------------------------------------------

c *********************************************************************
c ***** Branch here if new trial **************************************
c *********************************************************************

c *********************************************************************
2000  itrial=itrial+1                 ! This is the Newton-Raphson loop
c *********************************************************************
       if (itrial.eq.itrial_max+1)then
        tcut=dsqrt(scale_dt0)
        if (time.le.1.e5) tcut=dsqrt(scale_dt1)
        dtime=dtime/tcut
        if (debug.gt.0.4 .and. debug.lt.0.6) then
           print *,'Exceeded iterations.'
        end if 
        goto 2345
       end if

      do i=0,imax
       rad(i)=orad(i)
       rrho(i)=orrho(i)
       rrho1(i)=rrho(i)
      end do   

      oteffective=teffective
c ---------------------------------------------------------------------
       if (pscreen.eq.3)then
        print '(1i3,1a5)', itrial,'-----'
       end if
c ---------------------------------------------------------------------

c *********************************************************************
c ***** Calculate the new density in inner envelope at ntemp **********
c *********************************************************************

      if (debug.ge.1.) print *,'Calculating new density at NTemp'
      do i=imax-1,ienv+1,-2
       x=(dlog(rrho(i+1))-dlog(rrho(i)))/
     1   (dlog(rrho(i+1))-dlog(rrho(i-1)))
       y=(dlog(rrho(i))-dlog(rrho(i-1)))/    
     1   (dlog(rrho(i+1))-dlog(rrho(i-1)))
       ltemp=y*log(ntemp(i+1))+x*log(ntemp(i-1))
       ntemp(i)=dexp(ltemp)
      end do

      do i=ienv,imax
       if (ntemp(i).lt.tcon) then      
        call density(ntemp(i)/ephi(i),
     1               pres(i),a_cell(i),z_ion(i),rrho(i))
        rrho(i)=min(rrho(i),rhod(i))
        bar(i)=1.d-39 * 6.022d23*rrho(i)
        dr=debar(i)/rrho(i)*factor
        rad(i+1)=rad(i)+dr
        a2ephin(i)=(4.d0*pi*rad(i)**2)**2*ephi(i)
       end if
      end do

c *********************************************************************
c ***** Calculate the physical parameters at ntemp ********************
c *********************************************************************

      if (debug.ge.1.) print *,'Calculating physics at NTemp'
      do i=1,imax,2
       t=ntemp(i)/ephi(i)
       d=rrho(i)
       a=a_cell(i)
       a1=a_ion(i)
       z=z_ion(i)
       call neutrino(irank,i,t,d,a,z,qnu(i),
     1      qeebrem(i),qnpb(i),qplasma(i),qsynch(i),qbubble(i),
     1      qpair(i),qphoto(i),qbrem_nn(i),
     2      qmurca_nucl(i),qbrem_nucl(i),qmurca_hyp(i),qbrem_hyp(i),
     3      qdurca_np(i),qdurca_lap(i),
     3      qdurca_smn(i),qdurca_smla(i),qdurca_sms0(i),
     4      qfast(i),
     5      qdurca_q(i),qmurca_q(i),
     6      qpbf_n1s0(i),qpbf_n3p2(i),qpbf_p1s0(i),qpbf_q(i),
     7      debug,naa,nbfield2,rhodrip,rhocore,
     8      mstp,mstn,mstla,mstsm,msts0,mstsp,kfe,kfm,kfp,kfn,
     9      kfqu,kfqd,kfqs,bar,yelect,ymuon,fhad,theta_k,theta_p,v_ion,
     a      rhoexo,cexo,pexo,c_nu_str,p_nu_str,
     b      murca_increase,inu_durca,inu_eion,inu_plasma,inu_synch,
     c      inu_n1s0_pbf,inu_n3p2_pbf,inu_p_pbf,
     d      inu_bubble,inu_photo,inu_pair,
     e      idurca_np,idurca_lap,durca_ctrl_e,durca_ctrl_m,
     f      idurca_smn,idurca_smla,idurca_sms0,
     g      idurca_quqd,idurca_quqs,tcn,tcp,tcla,tcu,tcd,tcs,
     h      tcu1,tcu2,tcu3,tcd1,tcd2,tcd3,tcs1,tcs2,tcs3,isf,
     i      neebrem_logt,neebrem_nalpha,neebrem_n2,
     j      sf_lgtau1,sf_lgtau2,sf_lgr,sf_lgr2)
c       call heating(i,time,dtime,t,d,a,a1,z,tcp(i),
c     1              m_dot/ephi(i),i_eleconb,
c     2              ephi(i),dephi(i),heat(i))
       qqq(i)=qnu(i)-heat(i)
       call specheat(i,t,d,a,z,cv(i),
     1      cv_n(i),cv_p(i),cv_e(i),cv_m(i),
     2      cv_l(i),cv_sm(i),cv_s0(i),cv_sp(i),
     3      cv_q(i),cv_ion(i),cve,cvm,cvn,cvp,cvla,
     4     cvsm,cvs0,cvsp,cvqu,dvqd,cvqs,rhodrip,rhocore,fhad,istrange,
     5     tcn,tcp,tcla,tcsm,tcs0,tcsp,isf)
       call conduct(i,t,d,a,a1,z,qimp,nbfield2(i),
     1              sig,lambda(i),debug,
     2              nu_e_s,nu_e_l,icon_crust,icon_core,rhodrip,rhocore,
     3     kfe,kfm,kfn,kfp,kfkfla,kfsm,kfs0,kfsp,mstn,mstp,
     4      mstla,mstsm,msts0,mstsp,tcn,tcp,tcla,tcsm,tcs0,tcsp,isf,
     5      fhad,istrange)
       call opacity(t,d,a,z,kappa(i),iopacity)
       acd=7.56d-15*c/(3.d0*kappa(i)*d)
       fp(i)=(lambda(i)+4.d0*acd*t**3)*bar(i)/lsol
       fq(i)=bar(i)/cv(i)*lsol
       fr(i)=e2phi(i)*qqq(i)/cv(i)
     1       -ephi(i)*pres(i)/cv(i)
     2       *(dlog(rrho(i))-dlog(orrho(i)))/dtime*contraction
      end do

c *********************************************************************
c ***** Calculate the new density at (1-tinc)*ntemp *******************
c *********************************************************************

      if (debug.ge.1.) print *,'Calculating density at NTemp'''
      tinc=max(1.d-12,ratiot/1.d1)
      do i=1,imax,2
       ntemp1(i)=ntemp(i)*(1.d0-tinc)
      end do

      do i=imax-1,ienv+1,-2
       x=(dlog(rrho(i+1))-dlog(rrho(i)))/
     1   (dlog(rrho(i+1))-dlog(rrho(i-1)))
       y=(dlog(rrho(i))-dlog(rrho(i-1)))/    
     1   (dlog(rrho(i+1))-dlog(rrho(i-1)))
       ltemp=y*log(ntemp1(i+1))+x*log(ntemp1(i-1))
       ntemp1(i)=dexp(ltemp)
      end do

      do i=ienv,imax
       if (ntemp(i).lt.tcon) then      
        call density(ntemp1(i)/ephi(i),pres(i),a_ion(i),z_ion(i),
     1               rrho1(i))
        rrho1(i)=min(rrho1(i),rhod(i))
        bar1(i)=1.d-39 * 6.022d23*rrho1(i)
       end if
      end do

c *********************************************************************
c ***** Calculate the physical parameters at (1-tinc)*ntemp ***********
c *********************************************************************

      if (debug.ge.1.) print *,'Calculating physics at NTemp'''
      do i=1,imax,2
       t=ntemp1(i)/ephi(i)
       d=rrho1(i)
       a=a_cell(i)
       a1=a_ion(i)
       z=z_ion(i)
       call neutrino(irank,i,t,d,a,z,qnu1(i),
     1      qn00,qn01,qn02,qn03,qn04,qn05,qn06,qn07,qn08,qn09,q10,
     2      qn11,qn12,qn13,qn14,qn15,qn16,qn17,qn18,qn19,q20,
     3      qn21,qn22,qn23,
     4      debug,naa,nbfield2,rhodrip,rhocore,
     8      mstp,mstn,mstla,mstsm,msts0,mstsp,kfe,kfm,kfp,kfn,
     9      kfqu,kfqd,kfqs,bar,yelect,ymuon,fhad,theta_k,theta_p,v_ion,
     a      rhoexo,cexo,pexo,c_nu_str,p_nu_str,
     b      murca_increase,inu_durca,inu_eion,inu_plasma,inu_synch,
     c      inu_n1s0_pbf,inu_n3p2_pbf,inu_p_pbf,
     d      inu_bubble,inu_photo,inu_pair,
     e      idurca_np,idurca_lap,durca_ctrl_e,durca_ctrl_m,
     f      idurca_smn,idurca_smla,idurca_sms0,
     g      idurca_quqd,idurca_quqs,tcn,tcp,tcla,tcu,tcd,tcs,
     h      tcu1,tcu2,tcu3,tcd1,tcd2,tcd3,tcs1,tcs2,tcs3,isf,
     i      neebrem_logt,neebrem_nalpha,neebrem_n2,
     j      sf_lgtau1,sf_lgtau2,sf_lgr,sf_lgr2)

c     call heating(i,time,dtime,t,d,a,a1,z,tcp(i),
c     1              m_dot/ephi(i),i_eleconb,
c     2              ephi(i),dephi(i),heat1(i))
       qqq1(i)=qnu1(i)-heat1(i)
       call specheat(i,t,d,a,z,cv1(i),
     1      xx1,xx2,xx3,xx4,xx5,xx6,xx7,xx8,xx9,xx0,
     2      cve,cvm,cvn,cvp,cvla,
     3      cvsm,cvs0,cvsp,cvqu,dvqd,cvqs,rhodrip,rhocore,fhad,istrange,
     4      tcn,tcp,tcla,tcsm,tcs0,tcsp,isf)
       if (debug.ge.1.) print '(a20,i5,1p2e12.3)',
     1      'Calling conduct',i,d,t
       call conduct(i,t,d,a,a1,z,qimp,nbfield2(i),
     1      sig,lambda1(i),debug,
     2      nu_e_s,nu_e_l,icon_crust,icon_core,rhodrip,rhocore,
     3      kfe,kfm,kfn,kfp,kfkfla,kfsm,kfs0,kfsp,mstn,mstp,
     4      mstla,mstsm,msts0,mstsp,tcn,tcp,tcla,tcsm,tcs0,tcsp,isf,
     5      fhad,istrange)
       if (debug.ge.1.) print *,'Done'
       call opacity(t,d,a,z,kappa1(i),iopacity)
       acd1=7.56d-15*c/(3.d0*kappa1(i)*d)

       fp1(i)=(lambda1(i)+4.*acd1*t**3)*bar1(i)/lsol
       fq1(i)=bar1(i)/cv1(i)*lsol
       fr1(i)=e2phi(i)*qqq1(i)/cv1(i)
     1        -ephi(i)*pres(i)/cv1(i)
     2        *(dlog(rrho1(i))-dlog(orrho(i)))/dtime*contraction
      end do

c *********************************************************************
c ***** Calculate the derivatives of fp,fq & fr ***********************
c *********************************************************************

      if (debug.ge.1.) print *,'Calculating derivatives of FP, FQ, FR'
       do i=1,imax,2
        t =ntemp(i)
        t1=ntemp1(i)
        dfp(i)=(fp(i)-fp1(i))/(t-t1)
        dfq(i)=(fq(i)-fq1(i))/(t-t1)
        dfr(i)=(fr(i)-fr1(i))/(t-t1)
       end do

c *********************************************************************
c ***** Calculate ff **************************************************
c *********************************************************************

      if (debug.ge.1.) print *,'Calculating FF'
      ff(0)=0.d0
      do i=2,imax-1,2
       ff(i)=nlum(i)+.5d0*(fp(i-1)+fp(i+1))*a2ephin(i)*dtemp(i)
       ff(i-1)=fr(i-1)+fq(i-1)*dlum(i-1)+(ntemp(i-1)-temp(i-1))/dtime
      end do
c look +++++++++++++++++++++++++++++++++++++++++++++
      ff(imax)=0.d0
c      ff(imax)=fr(i-max)+fq(imax)*dlum(imax)+
c     1         (ntemp(imax)-temp(imax))/dtime
c ++++++++++++++++++++++++++++++++++++++++++++++++++

c *********************************************************************
c ***** Matrix inversion for Newton-Raphson method ********************
c *********************************************************************

      if (debug.ge.1.) print *,'Newton-Raphson'
      do i=2,imax-1,2
       fa(i)=.5d0*dfp(i+1)*a2ephin(i)*dtemp(i)+
     1       .5d0*(fp(i+1)+fp(i-1))*a2ephin(i)/(debar(i)+debar(i+1))
       fb(i)=.5d0*dfp(i-1)*a2ephin(i)*dtemp(i)-
     1       .5d0*(fp(i+1)+fp(i-1))*a2ephin(i)/(debar(i)+debar(i+1))
       fc(i)=1.d0
      end do

      do i=1,imax-2,2
       fa(i)=fq(i)/(debar(i)+debar(i+1))
       fb(i)=-fq(i)/(debar(i)+debar(i+1))
       fc(i)=dfr(i)+dfq(i)*dlum(i)+1.d0/dtime
      end do

      fk(1)=-ff(1)/fc(1)
      fj(1)=+fa(1)/fc(1)
      do i=2,imax-1
       fk(i)=-(ff(i)+fb(i)*fk(i-1))/(fc(i)-fb(i)*fj(i-1))
       fj(i)=fa(i)/(fc(i)-fb(i)*fj(i-1))
      end do

c *********************************************************************
c ***************** Boundary condition ********************************
c *********************************************************************

      if (debug.ge.1.) print *,'Boundary Condition'
      if (ifteff.ne.15) then
       epsilon=1.d-8
       precision=1.d-12
       coeff=4.d0*pi*radius**2*5.67d-5*e2phi(imax)/lsol
       lhs=nlum(imax-1)+fk(imax-1)+fj(imax-1)*ntemp(imax)
       ntp=ntemp(imax)
       tp0_keep=ntp
 7654  tp0=ntp
       teff0=nscool_teff(irank,tp0/ephi(imax),ifteff,eta,
     1      0.d0,istep,time,ts1,ts2,z_ion(imax),
     2      a_ion(imax),rrho(imax),debug,gs14,compactness)
       if(debug.eq.-50.) print *,'Tb0, Te0 =',tp0,teff0
       tp1=(1.d0+epsilon)*tp0
       teff1=nscool_teff(irank,tp1/ephi(imax),ifteff,eta,
     1      0.d0,istep,time,ts1,ts2,z_ion(imax),
     2      a_ion(imax),rrho(imax),debug,gs14,compactness)
       if(debug.eq.-50.) print *,'Tb1, Te1 =',tp1,teff1
       derivative=coeff*(teff1**4-teff0**4)/(epsilon*tp0)
       derivative=-fj(imax-1)-derivative
       if(debug.eq.-50.) print *,'Derivative =',derivative
       function=lhs-fj(imax-1)*tp0-coeff*teff0**4
       if(debug.eq.-50.) print *,'Function =',function
       ntp=tp0-function/derivative
       if(debug.eq.-50.) print *,'Del(Tp)/Tp =',abs(tp0-ntp)/tp0
       if(debug.eq.-50.) print *,'------> New Tb =',ntp
       if ((ntp.le.0.).or.(ntp.gt.1.e12)) then ! In case the method diverges
                                 ! restart iterations with shorter time step
        tcut=dsqrt(scale_dt0)
        if (time.le.1.e5) tcut=dsqrt(scale_dt1)
        dtime=dtime/tcut
        if (debug.gt.0.4 .and. debug.lt.0.6) then
           print *,'BC problem.'
        end if 
        goto 2345
      end if
      if(abs(tp0-ntp)/tp0.gt.precision) goto 7654
      else
       ntp=tb_acc0            ! Fixed T_b for accretion
      end if

c *********************************************************************
c ****** Get ntemp & nlum *********************************************
c *********************************************************************

      if (debug.ge.1.) print *,'Getting NTemp & NLum'
      delt(imax)=ntp-ntemp(imax)
      do i=imax-2,1,-2
         dell(i+1)=fk(i+1)-fj(i+1)*delt(i+2)
         dell(i+1)=sign(min(2.d3*abs(nlum(i+1)),abs(dell(i+1))),
     1        dell(i+1))
         delt(i)=fk(i)-fj(i)*dell(i+1)
         delt(i) = sign(min(.5d0*ntemp(i),abs(delt(i))),delt(i))
      end do
      dell(0)=0.d0     ! This is the inner boundary condition !

      if (debug.gt.0.4 .and. debug.lt.0.6) then
         print '(2x,a,i4,1x,i4,1x,E10.3,1x,E10.3)',
     2        'itrial,imax,delt,dell',
     1        itrial,imax,delt(imax-2),dell(imax-1)
      end if
      
c Check for matrix inversion: *****************************************
c      if (debug.ge.1.) print *,'Checking Matrix Inversion'
c      max_equ_t=0.d0
c      max_equ_l=0.d0
c      do i=imax-2,1,-2
c       equ_t=ff( i )+
c     1       fa( i )*dell(i+1)+fb( i )*dell(i-1)+fc( i )*delt( i )
c       equ_t=abs(equ_t)/(abs(ntemp(i)+delt(i))+1.d-8)*dtime
c       equ_l=ff(i+1)+
c     1       fa(i+1)*delt(i+2)+fb(i+1)*delt( i )+fc(i+1)*dell(i+1)
c       equ_l=abs(equ_l)/(abs(nlum(i+1)+dell(i+1)+1.d-8))
c       if (equ_t.gt.max_equ_t) max_equ_t=equ_t
c       if (equ_l.gt.max_equ_l) max_equ_l=equ_l
c      end do
c *********************************************************************

      do i=0,imax-1,2
       nlum(i)=nlum(i)+dell(i)
       ntemp(i+1)=ntemp(i+1)+delt(i+1)
      end do

      dtemp(0)=0.d0
      do i=2,imax-1
       dtemp(i)=(ntemp(i+1)-ntemp(i-1))/(debar(i)+debar(i+1))
      end do
      do i=1,imax-2,2
       dlum(i)=(nlum(i+1)-nlum(i-1))/(debar(i)+debar(i+1))
      end do
c look +++++++++++++++++++++++++++++++++++++++++++++
      dlum(imax)=0.d0
c      dlum(imax)=dlum(imax-2)
c ++++++++++++++++++++++++++++++++++++++++++++++++++

c *********************************************************************
c ***** Analyze the results to see if it has converged ****************
c *********************************************************************

      if (debug.ge.1.) print *,'Analyzing Results'
      ratiot=0.d0
      ratiol=0.d0
      do i=1,imax-2,2
       ratl=abs(dell(i+1))/(abs(nlum(i+1))+1.d-12)
       if(ratl.gt.ratiol)then
        ratiol=ratl
        iratl=i+1
       end if
       ratt=abs(delt(i)/(ntemp(i)+1.d-30))
       if (ratt.gt.ratiot)then
        ratiot=ratt
        iratt=i
       end if
      end do

c BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
c This was used when doing dipolar magnetic field evolution.
c Not used any more.
      ratios=0.d0
      irats=1
c BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB

c ---------------------------------------------------------------------
      if (pscreen.eq.3) then
       if (ifield.eq.2) then
        print '(3x,3(a10,1p1e10.3,a3,i4,a1))',
     1                                     'dT/T=',ratiot,'(',iratt,')',
     2                                     'dL/L=',ratiol,'(',iratl,')',
     3                                     'dS/S=',ratios,'(',irats,')'
       else
        print '(3x,2(a10,1p1e10.3,a3,i4,a1))',
     1                                     'dT/T=',ratiot,'(',iratt,')',
     2                                     'dL/L=',ratiol,'(',iratl,')'
       end if
      end if
c ---------------------------------------------------------------------

      if (debug.gt.0.4 .and. debug.lt.0.6) then
         print '(2x,a,2(E10.3,1x,i4,1x,E10.3))',
     1        'ratio,irat,mrat',
     1        ratiot,iratt,mratt,ratiol,iratl,mratl
      end if

c *********************************************************************
c     Decide if converged or not:
      if ((ratiot.lt.mratt).and.(ratiol.lt.mratl).and.(ratios.lt.mrats))
     x then
        continue        ! Converged ! continue to next time step
       else 
        goto 2000       ! Not converged ! Go back for another iteration
       end if
c *********************************************************************

      luminosity=nlum(imax-1)/ephi(imax-1)**2
      sign_l=abs(nlum(imax-1))/nlum(imax-1)
      teffective=sign_l*
     2   (abs(luminosity)/(4.d0*pi*radius**2*5.67d-5))**.25d0*
     3   lsol**.25d0*ephi(imax-1)

c     In case of accretion with fixed outer Tb, lum at the surface can
c     become negative: the surface is injecting heat into the star (from
c     the surface nuclear burning).

c *********************************************************************
c ******* PREPARATION TO CALCULATE THE NEW TIME STEP  *****************
c *********************************************************************
c
c This is a delicate part, based on experience and many trials and errors.
c It works pretty well, so avoid changing it !
c
c PHILOSOPHY OF TIME STEP CONTROL:
c
c (Time of step just finished is "time+dtime", not just time !)
c The new "dtime" will be "scale_dt*dtime" with "scale_dt" calculated below.
c Allows for 2 different "scale_dt": at early time, while relaxing from initial
c conditions, accuracy is not important and one can allow for larger timestep:
c "scale_dt0" and "scale_dt1" are read from the file 
c NUM_PARAM.dat in NSCool_READ.inc.f
c and are the maximum allowed relative increase in "dtime"
c *********************************************************************

      if (debug.ge.1.) print *,'Calculating New Time Step'

      scale_dt=scale_dt0
      if (time.le.1.e5) scale_dt=scale_dt1

c *********************************************************************
c TEMP variation: "max_dtemp" is the max. relative variation of T in the
c                 star and "icht" the zone where "max_dtemp" is obtained
      icht=0
      max_dtemp=0.d0
      do i=1,imax-2,2
       mdt=abs(temp(i)-ntemp(i))/ntemp(i)
       if (mdt.gt.max_dtemp)then
        max_dtemp=mdt
        icht=i
       end if
      end do
c *********************************************************************
c Check if "max_dtemp" is not too large. If "1+max_dtemp" exceeds "tvar",
c then "scale_dt" is reduced correspondingly:
c ("tvar" read from the file NUM_PARAM.dat in NSCool_READ.inc.f)
      chtemp=0.d0
      if((tvar-1.d0).lt.max_dtemp)then
       scale_dt=scale_dt*(tvar-1.)/max_dtemp
       if (time.lt.1.d5) then
        scale_dt=min(scale_dt,scale_dt1)
       else
        scale_dt=min(scale_dt,scale_dt0)
       end if
       chtemp=1.d0     ! this means the time-step is controlled b
      end if           ! a too large max_dtemp
c BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
c       chstoke=0.d0
c BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
c *********************************************************************
c If "scale_dt" has been reduced too much, i.e. "max_dtemp" too large,
c then the present time-step is recalculated with a smaller "dtime"
c unless one is still at the first few time-steps:
c ("repeat" and "istart" are read from the file
c   NUM_PARAM.dat in NSCool_READ.inc.f)
      if ( (scale_dt.lt.repeat).and.(istep.gt.istart) ) then
       dtime=repeat*dtime 
c ---------------------------------------------------------------------
       if (pscreen.ge.2) then
        write(6,*)
        print '(1a40)','dtime too large, do it again'
        print '(1a13,1p1e12.3)','time=',(time+dtime)/year
        print '(1a13,1p1e12.3)','dtime=',dtime/year
        print '(1a13,1p1e12.3)','dtime ratio=',dtime/odtime
        write(6,*)
       end if
c     ---------------------------------------------------------------------
       if (debug.gt.0.4 .and. debug.lt.0.6) then
          print *,'Temperature changed too much.'
       end if 
       goto 2345
      end if
c In case convergence is reached in too many trials, scale_dt is reduced
      if ((itrial.gt.itrial_opt).and.(istep.gt.istart)) then
       chtrial=1.d0     ! this means the time-step is controlled by needing too many iterations
       olddt=scale_dt
       if (time.lt.1.d5) then
        scale_dt=scale_dt/
     x           scale_dt1**(float(itrial-itrial_opt)/2.d0)
        scale_dt=min(scale_dt,scale_dt1)
       else
        scale_dt=scale_dt/
     x           scale_dt0**((1.d0+float(itrial-itrial_opt))/2.d0)
        scale_dt=min(scale_dt,scale_dt0)
       end if
      else
       chtrial=0.d0
      end if
c Before setting the next time and time-step, stuff are printed out and updated:

c *********************************************************************
c ***** End of iterations
c *********************************************************************

      do 171 i=1,imax,2
       otemp(i)=temp(i)
       temp(i)=ntemp(i)
       orrho(i)=rrho(i)
       orad(i)=rad(i)       
       obar(i)=bar(i)
171   continue
       do 172 i=2,imax-1,2
       olum(i)=lum(i)
       lum(i)=nlum(i)
       orrho(i)=rrho(i)
       orad(i)=rad(i)       
       obar(i)=bar(i)
172   continue

c BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
c BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB

c *********************************************************************
c Spin down: **********************************************************
c *********************************************************************
c      call spin_down(period,dtime,beta_rot,bfield2(imax),nperiod)
c      dp_dt=(nperiod-period)/dtime
c      p_av=(nperiod+period)/2.d0            ! average P
c      omega=2.d0*pi/p_av**2
c      domega_dt=-2.d0*pi*dp_dt/p_av**2
c      energy_loss=m_i*omega*domega_dt       ! Spin-down power
      period=nperiod
c +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
c
c Stuff below, till the next +++++ line is only informative and not
c used in the calculations.
c
c *********************************************************************
c Crystalization density: *********************************************
c *********************************************************************
      if (debug.ge.1.) print *,'Calculating Crystalization Densities'
      do i=imax,icore+2,-2
       gamma(i)=2.273d5*z_ion(i)**2*(rrho(i)/a_cell(i))**(1.d0/3.d0)/
     1          (temp(i)/ephi(i))
       if (gamma(i).lt.gammaliq) then
        cryst(i)=-1.
       else if ( (gamma(i).ge.gammaliq  ) .and.
     1           (gamma(i).le.gammacryst)       ) then
        cryst(i)=0.d0
       else
        cryst(i)=1.d0
       end if
      end do
      do i=icore,0,-2
       gamma(i)=1.d0
      end do
      iliq=0
      icryst=0
      do i=imax,icore+2,-2
       if (cryst(i).eq.0.d0) then
        iliq=i
        goto 600
       end if
      end do
600   continue
      do i=imax,icore+2,-2
       if (cryst(i).eq.1.d0) then
        icryst=i
        goto 700
       end if
      end do
700   continue

c *********************************************************************
c ***** Calculate the neutrino luminosity and heating: ****************
c *********************************************************************
       if (debug.ge.1.) print *,
     1     'Calculating Total Neutrino Luminosity and Heating'
       lnu =0.d0
       lnu0=0.d0
       lh  =0.d0
       lh0 =0.d0
       do i=1,imax,2
        lnu =lnu +e2phi(i)* qnu(i)*(dvol(i)+dvol(i+1))
        lnu0=lnu0+ephi(i) * qnu(i)*(dvol(i)+dvol(i+1))   ! without energy red-shift
        lh  =lh  +e2phi(i)*heat(i)*(dvol(i)+dvol(i+1))
        lh0 =lh0 +ephi(i) *heat(i)*(dvol(i)+dvol(i+1))   ! without energy red-shift
       end do
       lnu=lnu/lsol
       lh =lh/lsol
       htot=htot+lh*dtime

c ***** CALCULATE THE INTEGRATED NEUTRINO LUMINOSITIES: ***************
c     Note: lnu_tot, calculated from qnu(i), is the garanteed total
c     neutrino luminosity. The other ones are only informative.
      lmurca_nucl=0.0d0
      lbrem_nucl =0.0d0
      lplasma    =0.0d0
      lnpb       =0.0d0
      lpbf_n1S0  =0.0d0
      lpbf_n3P2  =0.0d0
      lpbf_p1S0  =0.0d0
      do i=1,imax,2
       e2p=e2phi(i)
       lmurca_nucl=lmurca_nucl+qmurca_nucl(i)*(dvol(i)+dvol(i+1))*e2p
       lbrem_nucl =lbrem_nucl +qbrem_nucl(i) *(dvol(i)+dvol(i+1))*e2p
       lplasma    =lplasma    +qplasma(i)    *(dvol(i)+dvol(i+1))*e2p
       lnpb       =lnpb       +qnpb(i)       *(dvol(i)+dvol(i+1))*e2p
       lpbf_n1S0  =lpbf_n1S0  +qpbf_n1S0(i)  *(dvol(i)+dvol(i+1))*e2p
       lpbf_n3P2  =lpbf_n3P2  +qpbf_n3P2(i)  *(dvol(i)+dvol(i+1))*e2p
       lpbf_p1S0  =lpbf_p1S0  +qpbf_p1S0(i)  *(dvol(i)+dvol(i+1))*e2p
      end do

c ***** CALCULATE THE INTEGRATED SPECIFIC HEATS: **********************
c     cv_tot_all, calculated from cv(i), is the garanteed total
c     specific heat. The other ones are only informative.
      cv_core=0.d0
      cv_crust=0.d0
      cv_tot_all=0.d0
      cv_tot_ion=0.d0
      cv_tot_neu=0.d0
      cv_tot_pro=0.d0
      cv_tot_ele=0.d0
      cv_tot_muo=0.d0
      cv_tot_lam=0.d0
      cv_tot_sim=0.d0
      cv_tot_si0=0.d0
      cv_tot_sip=0.d0
      cv_tot_qrk=0.d0
      cv_phot=0.d0
      do i=1,imax,2
       cv_tot_all=cv_tot_all+cv(i)   *(dvol(i)+dvol(i+1))
       cv_tot_ion=cv_tot_ion+cv_ion(i)*(dvol(i)+dvol(i+1))
       cv_tot_neu=cv_tot_neu+cv_n(i)  *(dvol(i)+dvol(i+1))
       cv_tot_pro=cv_tot_pro+cv_p(i)  *(dvol(i)+dvol(i+1))
       cv_tot_ele=cv_tot_ele+cv_e(i)  *(dvol(i)+dvol(i+1))
       cv_tot_muo=cv_tot_muo+cv_m(i)  *(dvol(i)+dvol(i+1))
       cv_tot_lam=cv_tot_lam+cv_l(i)  *(dvol(i)+dvol(i+1))
       cv_tot_sim=cv_tot_sim+cv_sm(i) *(dvol(i)+dvol(i+1))
       cv_tot_si0=cv_tot_si0+cv_s0(i) *(dvol(i)+dvol(i+1))
       cv_tot_sip=cv_tot_sip+cv_sp(i) *(dvol(i)+dvol(i+1))
       cv_tot_qrk=cv_tot_qrk+cv_q(i)  *(dvol(i)+dvol(i+1))
       cv_phot= 4.*7.56e-15*(ntemp(i)/ephi(i))**3  *(dvol(i)+dvol(i+1))
      end do
      do i=1,icore+2
       cv_core=cv_core+cv(i)*(dvol(i)+dvol(i+1))
      end do
      do i=icore+2,imax
       cv_crust=cv_crust+cv(i)*(dvol(i)+dvol(i+1))
      end do

c *********************************************************************
c ******************** print out results ******************************
c *********************************************************************

c *********************************************************************
c                                  TEMP
c *********************************************************************
      if (debug.gt.0.4 .and. debug.lt.0.6) then
         print *,'time',time+dtime/year
      end if
      
      if (ptemp.ge.1.) then

c AWS: Determine if we should output temperature at this iteration
         if ( ( ((time+dtime)/year) .ge. tprint(itprint)) .and.
     1        (itprint.le.itpmax)) then
            
            itprint=itprint+1

c AWS: Compute weights and effective temperature            
            w1=((time+dtime)-tprint(itprint-1)*year)/dtime
            w2=1.d0-w1
            logtemp=w1*dlog(sign_l*oteffective)+
     1           w2*dlog(sign_l* teffective)
            t_effective=sign_l*dexp(logtemp)
            
            call nscool_print_temp(irank,istep,itprint-2,
     1           tprint(itprint-1),t_effective,imax,w1,w2,otemp,
     2           temp,olum,lum,rad,rrho,ephi,dvol,e2phi,tcn,tcp,
     3           qnu,qeebrem,qnpb,qplasma,qsynch,qbubble,qpair,
     4           qphoto,qbrem_nn,qmurca_nucl,qbrem_nucl,qmurca_hyp,
     5           qbrem_hyp,qdurca_np,qdurca_lap,qdurca_smn,
     6           qdurca_smla,qdurca_sms0,qfast,qdurca_q,qmurca_q,
     7           qpbf_n1s0,qpbf_p1s0,qpbf_n3p2,qpbf_q)

            call nscool_print_cv(irank,itprint-2,imax,cv,cv_n,
     1           cv_p,cv_e,cv_m,cv_l,cv_sm,cv_s0,cv_sp,cv_q)

         end if
      end if
c *********************************************************************
c                               TEFF
c *********************************************************************

      if (pteff.ge.1.0) then
       if ((float(idump1)/2.).ne.float(idump1/2)) then
c        temp1=temp(idump1)/ephi(idump1)
        temp1=temp(idump1)
       else
c        temp1=(temp(idump1-1)+temp(idump1+1))/2./ephi(idump1)
        temp1=(temp(idump1-1)+temp(idump1+1))/2.
       end if
       if ((float(idump2)/2.).ne.float(idump2/2)) then
c        temp2=temp(idump2)/ephi(idump2)
        temp2=temp(idump2)
       else
c        temp2=(temp(idump2-1)+temp(idump2+1))/2./ephi(idump2)
        temp2=(temp(idump2-1)+temp(idump2+1))/2.
       end if
       if ((float(idump3)/2.).ne.float(idump3/2)) then
c        temp3=temp(idump3)/ephi(idump3)
        temp3=temp(idump3)
       else
c        temp3=(temp(idump3-1)+temp(idump3+1))/2./ephi(idump3)
        temp3=(temp(idump3-1)+temp(idump3+1))/2.
       end if
       cv_core =0.d0
       cv_crust=0.d0
       do i=1,icore,2
        cv_core=cv_core+(dvol(i)+dvol(i+1))*cv(i)
       end do
       do i=icore+2,imax,2
        cv_crust=cv_crust+(dvol(i)+dvol(i+1))*cv(i)
       end do
c*********************************************************************
       if (pteff.eq.1.0) then
        write(19,601) istep,(time+dtime)/year,teffective,
     1               lum(imax-1)*lsol,lnu*lsol,lh*lsol
 601    format(i8,1p1e12.3,5x,1p4e12.3)
c*********************************************************************
       else
          print *,'WARNING: Not Teff print out defined !'
          stop
       end if
c*********************************************************************
c       write(19,615) istep,(time+dtime)/year,teffective,
c     1               0.d0,period,
c     2               dtime/year,dtime/odtime,max_dtemp,itrial,
c     3               omegab_tau1,omegab_tau2,omegab_tau3,
c     4               icycle,m_dot*3.15e7/2.e33,
c     5               htot*lsol,lh*lsol,
c     6               lcore+lexo,lcrust
c*********************************************************************
c       write(19,615) istep,(time+dtime)/year,teffective,
c     1               0.d0,period,
c     2               dtime/year,dtime/odtime,max_dtemp,itrial,
c     3               temp1,temp2,temp3,
c     4               icycle,m_dot*3.15e7/2.e33,
c     5               htot*lsol,lh*lsol,(lcore+lexo)*lsol,lcrust*lsol,
c     6               cv_core,cv_crust
c615    format(i8 , 1p1e22.15 , 1p1e16.8 ,
c     1        1p1e12.3 , 1p1e12.3 ,        
c     2        1p1e10.2 , 0p2f8.3, i8 ,
c     3        1p3e16.8 ,
c     4        i8 , 1p1e14.5,
c     5        1p4e12.3,
c     6        1p2e12.3)       
c*********************************************************************
c       write(19,615) istep,(time+dtime)/year,teffective,
c     1               0.d0,
c     2               lum(imax-1)*lsol,lnu_tot,
c     3               lmurca_nucl,lbrem_nucl,lplasma,
c     4               lpbf_n1S0,lpbf_n3P2,lpbf_p1S0
c615    format(i8 , 1p1e12.3 , 1p1e12.3 , 3x ,
c     1        1p1e12.3, 5x ,
c     2        1p1e12.3 , 2x , 1p1e12.3 , 2x ,
c     3        1p3e12.3 ,
c     4        1p3e12.3)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c       menv= (emas(imax)/gs14**2) * eta
c       write(19,615) istep,(time+dtime)/year,teffective,ts1,ts2,
c     1               temp1,temp2,temp3,
c     2               0.d0,
c     3               lum(imax-1)*lsol,lnu_tot,
c     4               lmurca_nucl,lbrem_nucl,lplasma,lnpb,
c     5               lpbf_n1S0,lpbf_n3P2,lpbf_p1S0
c615    format(i8 , 1p1e12.3 , 1p3e12.3 , 3x ,
c     1        1p3e12.3,        
c     2        1p1e12.3, 5x ,
c     3        1p1e12.3 , 2x , 1p1e12.3 , 2x ,
c     4        1p4e12.3 ,
c     5        1p3e12.3)
c*********************************************************************
      end if



c ----------------------------------------------------------------------
      if (pscreen.ge.2) then
       print *
       print '(1a21,1i3,1a9)','Iteration finished:',
     1                    itrial,'trials.'
       print *

       print '(a8,1p1e12.3,a30,1p1e12.3,a10,i5)','Teff =',teffective,
     1                   'max delt/ntemp=',max_dtemp,'at i=',icht
       print *
       if ((iliq.eq.0).and.(icryst.eq.0)) then
        print *,'   Whole crust liquid'
       else if (icryst.eq.imax) then
        print *,'   Whole crust solid'
       else if ((iliq.eq.0).and.(icryst.ne.0)) then
        print '(22x,a10,1p1e12.3,a3,i3,a1)',
     1    'Rho_s=',rrho(icryst),'(',icryst,')'
       else if ((iliq.ne.0).and.(icryst.eq.0)) then
        print '(a10,1p1e12.3,a3,i3,a1)',
     1    'Rho_l=',rrho(iliq),'(',iliq,')'
       else
        print '(2(a10,1p1e12.3,a3,i3,a1))',
     1    'Rho_l=',rrho(iliq),'(',iliq,')',
     2    'Rho_s=',rrho(icryst),'(',icryst,')'
       end if
       print *
       print '(5x,3(a10,1p1e10.3))',
     1   'L(ph)=',luminosity*lsol*e2phi(imax-1),
     2   'L(nu) =',lnu*lsol,
     3   'L(H)  =',lh*lsol
       lh_red=lh
       print *
       if ((i_acc.eq.1).or.(i_acc.eq.2)) then
        print '(a12,1p1e10.2)','M_dot =',m_dot*3.15576d7/2.d33
        print '(a12,1i10,a12,0p1f8.2)',
     1       'Cycle number',icycle,'Phase =',t_burst/t_acc1
        print *
       end if
      end if
c     ---------------------------------------------------------------
      call nscool_main_out(irank,(time+dtime)/year,teffective,
     1     nlum(imax-1)*lsol,lnu*lsol,dtime/year,istop)
      if (istop.gt.0) then
         iret=3
         print *,dtime,"NSCool failed. Stepsize vanishing."
         goto 9997
      end if
c     ---------------------------------------------------------------

      if (pscreen.eq.1) then
         if (i_acc.eq.0) then
c$$$            print '(i5,1p1e12.3,3x,1p1e12.3,3x,1p3e12.3)',
c$$$     1           istep,(time+dtime)/year,teffective,
c$$$     2           nlum(imax-1)*lsol,lnu*lsol,lh*lsol
         else
            if (icycle.ge.1) then
               if (t_burst.le.t_acc2) then
                  t_check=t_burst/t_acc2
               else
                  t_check=0.d0
               end if
            end if
            print 
     x           '(i7,1p1e22.15,1p1e14.6,1p3e12.3,
     x1p1e15.3,i10,0p1f10.3,1p1e12.3,
     x5x,1p2e12.3)',
     1           istep,(time+dtime)/year,teffective,
     2           nlum(imax-1)*lsol,lnu*lsol,lh*lsol,
     3           odtime,icycle,t_check,m_dot*3.15576d7/2.d33,
     3           ntemp(idrip)/ephi(idrip),ntemp(1)/ephi(1)
         end if
      end if
c ----------------------------------------------------------------------------

c *********************************************************************
c ****************  CALCULATE THE NEW TIME STEP  **********************
c *********************************************************************
      time=time+dtime
      odtime=dtime
      dtime=min(scale_dt*dtime,dtlimit)

c *********************************************************************
c For accretion scenarios: time step must moreover be shortened 
c dramatically when a new outburst is approaching (to make sure it is  
c much shorter than the outburst duration, or rise time, or any relevant 
c time scale which has to be resolved by the code):

c Transient FRED ("Fast rise and exponential decay") *******************
      if (i_acc.eq.1) then
       time_step_cut=100.d0
       if (((time+3.*dtime).ge.t_acc0).and.(icycle.eq.0)) then
        timeleft=t_acc0-time
        dtime=max(timeleft/3.d0,t_acc2/time_step_cut)
       end if
       if (time.gt.t_acc0) then
        if (delt_acc/t_acc2.le.10.d0) then
         scale_dt0=1.05d0
        else
         scale_dt0=1.2d0
        end if
        t_next=t_acc0+float(icycle+1)*t_acc1
        if ((time+3.*dtime).ge.t_next) then
         timeleft=t_next-time
         dtime=max(timeleft/3.d0,t_acc2/time_step_cut)
        end if
       end if
       if (time.ge.t_acc0) dtlimit=t_acc1/20.d0
      end if

c Transient STEP ******************************************************
      if (i_acc.eq.2) then
       if (((time+2.*dtime).ge.t_acc0).and.(time.lt.t_acc0)) then
        timeleft=t_acc0-time
        dtime=max(timeleft/3.d0,time_step_min)
        day=86400.d0
c        print '(a30)','Approaching first burst:'
c        print '(2(a20,0p1f18.8))',
c     1        'Time left =',timeleft/day,'DTime =',dtime/day
       end if
       if (time.gt.t_acc0) then
        t_next=t_acc0 + float(icycle-1)*t_acc1 + t_acc1
        t_end =t_acc0 + float(icycle-1)*t_acc1 + t_acc2
c       Slow down if approaching next burst
        if ((time+dtime.gt.t_end).and.
     1      (time+2.*dtime.ge.t_next)) then
         timeleft=t_next-time
         dtime=max(timeleft/3.d0,time_step_min)
c        print '(a30)','Approaching next burst:'
c        print '(2(a20,0p1f18.8))',
c     1        'Time left =',timeleft/day,'DTime =',dtime/day
c       Slow down if approaching end of burst
c        else if ((time+dtime.le.t_end).and.
c     1           (time+2.*dtime.ge.t_end)) then
        else if ((time.le.t_end).and.
     1           (time+2.*dtime.ge.t_end)) then
c        else if (time+2.*dtime.ge.t_end) then
         timeleft=t_end-time
         dtime=max(timeleft/3.d0,time_step_min)
c        print '(a30)','Approaching end of burst:'
c        print '(2(a20,0p1f18.8))',
c     1        'Time left =',timeleft/day,'DTime =',dtime/day
        end if
       end if
       if (time.ge.t_acc0) dtlimit=t_acc1/20.d0
      end if

c Heat Deposition *****************************************************
      if (i_heat_deposit.eq.1) then
       t_slow=max(dtime,1000.*del_t_dep)
       if (abs(t_dep-(time+dtime)).lt.t_slow) then
        timeleft=abs(t_dep-(time+dtime))
        dtime=max(timeleft/30.d0,del_t_dep/100.d0)
       end if
      end if
c *********************************************************************

      if (time/year.ge.timemax) goto 9998
      if ((sign_l*teffective).lt.tempmin) goto 9998

c *********************************************************************
9999  continue     ! This is the `end do' for the main time integration
c *********************************************************************

c *********************************************************************
9998  continue     ! Jump here if time > timemax
c *********************************************************************

c Close the output file for the present model: ************************
c      close(unit=10,status='keep')
c      close(unit=19,status='keep')

c Go back to the beginning: do the next model listed in the "Input file"
      goto 1234

c *********************************************************************
 9997 continue    ! This is the real end of it !
c *********************************************************************

c      close(unit=15,status='keep')

c      print *,'Done !'

      end

c *********************************************************************
