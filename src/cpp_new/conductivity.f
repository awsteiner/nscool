c     ***********************************************************************
c     ***********************************************************************
      subroutine conduct(i,T,rho,A,A1,Z,Q,magfield,
     1     sigma,lambda,debug,
     2     nu_e_s,nu_e_l,icon_crust,icon_core,rhodrip,rhocore,
     3     kfe,kfm,kfn,kfp,kfkfla,kfsm,kfs0,kfsp,mstn,mstp,
     4     mstla,mstsm,msts0,mstsp,tcn,tcp,tcla,tcsm,tcs0,tcsp,isf,
     5     fhad,istrange)
c     ***** CHECKED ON : ?
      implicit real*8 (a-h,k-z)
      parameter (isize=10000)

      dimension tcn(0:isize),tcp(0:isize),tcla(0:isize),
     2          tcsm(0:isize),tcs0(0:isize),tcsp(0:isize)

      dimension mstp(0:isize),mstn(0:isize),mstla(0:isize),
     2          mstsm(0:isize),msts0(0:isize),mstsp(0:isize)
      dimension kfe(0:isize),kfm(0:isize),kfp(0:isize),kfn(0:isize),
     2          kfla(0:isize),kfsm(0:isize),kfs0(0:isize),kfsp(0:isize)

      dimension fhad(0:isize)

      istrange=0
      
c     ***********************************************************************
      if (debug.ge.2.) print *,'Entering conduct:',
     1     ' T, rho, A, A1, Z, Qimp =',T,rho,A,A1,Z,Q
c     ***********************************************************************
      if  (rho.ge.rhocore) then
         if (istrange.eq.0) then
            if (i.le.isf) then
               isfn=3
            else
               isfn=1
            end if
            call con_core(icon_core,debug,
     1           T,kfe(i),kfm(i),
     2           kfp(i) ,mstp(i) ,Tcp(i) ,
     3           kfn(i) ,mstn(i) ,Tcn(i) ,isfn,
     4           kfla(i),mstla(i),Tcla(i),
     5           kfsm(i),mstsm(i),Tcsm(i),
     6           kfs0(i),msts0(i),Tcs0(i),
     7           kfsp(i),mstsp(i),Tcsp(i),
     8           fhad(i),
     9           sigma,lambda,
     x           nu_e_s,nu_e_l)
         else if (istrange.eq.1) then
            lambda=c_con_str/(T/1.d9)**p_con_str
         else
            print *,'conduct: istrange not defined !'
            stop
         end if
      else
         call con_crust(icon_crust,debug,
     1        T,rho,kfe(i),A,A1,Z,Q,sigma,lambda,
     2        nu_e_s,nu_e_l,rhodrip)
      end if
c     ***********************************************************************
      if (debug.ge.2.) print *,'Exiting conduct:',
     1     ' sigma, lambda=',sigma,lambda
c     ***********************************************************************
      return
      end
c     ***********************************************************************
c     ***********************************************************************


