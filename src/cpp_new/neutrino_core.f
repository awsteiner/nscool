c     *********************************************************************

      subroutine numurca_nucl(i,t,qmurca_nucl,tcn,tcp,isf,
     1     mstn,mstp,kfe,kfm,kfn,kfp)

c     DP: checked partially on March 30, 1993

c**********************************************************************
c     Rates and suppression factors from:                              
c     Yakovlev & Levenfish, A&A 297 (1995): p. 717.                    
c**********************************************************************
      implicit real*8 (a-h,k-z)
      parameter (isize=10000)
      parameter (pi = 3.14159265d0)

      dimension tcn(0:isize),tcp(0:isize)
      dimension mstp(0:isize),mstn(0:isize)
      dimension kfe(0:isize),kfm(0:isize),kfp(0:isize),kfn(0:isize)

c     ***************************
      fexp(x)=dexp(max(x,-7.d2))
c     ***************************
c     SINGLET PAIRING:
      u_1s0(t)=dsqrt(1.d0-t)*(1.456d0-0.157d0/dsqrt(t)+1.764d0/t)
c     Murca_n:  n+n -> n+p+e+nu
      rmurca_n_p1s0(u)=
     1     fexp(3.4370d0-dsqrt((3.4370d0)**2+(1.d0*u)**2))*0.5d0*
     2     ((0.1477d0+dsqrt((0.8523d0)**2+(0.1175d0*u)**2))**7.5d0+
     3     (0.1477d0+dsqrt((0.8523d0)**2+(0.1297d0*u)**2))**5.5d0)
c     rmurca_n_n1s0(u)=rmurca_p_p1s0(u)
      rmurca_n_n1s0(u)=
     1     fexp(5.3390d0-dsqrt((5.3390d0)**2+(2.d0*u)**2))*
     2     (0.2414d0+dsqrt((0.7586d0)**2+(0.1318d0*u)**2))**7.0d0
c     Murca_p:  n+p -> p+p+e+nu
      rmurca_p_p1s0(u)=
     1     fexp(5.3390d0-dsqrt((5.3390d0)**2+(2.d0*u)**2))*
     2     (0.2414d0+dsqrt((0.7586d0)**2+(0.1318d0*u)**2))**7.0d0
c     rmurca_p_n1s0(u)=rmurca_n_p1s0(u)
      rmurca_p_n1s0(u)=
     1     fexp(3.4370d0-dsqrt((3.4370d0)**2+(1.d0*u)**2))*0.5d0*
     2     ((0.1477d0+dsqrt((0.8523d0)**2+(0.1175d0*u)**2))**7.5d0+
     3     (0.1477d0+dsqrt((0.8523d0)**2+(0.1297d0*u)**2))**5.5d0)
c     ***************************
c     TRIPLET PAIRING:
      u_3p2B(t)=dsqrt(1.d0-t)*(0.7893d0+1.188d0/t)
c     Murca_p:  n+p -> p+p+e+nu
      rmurca_p_n3p2B(u)=
     1     fexp(2.3980d0-dsqrt((2.3980d0)**2+(1.d0*u)**2))*0.5d0*
     2     ((0.1612d0+dsqrt((0.8388d0)**2+(0.1117d0*u)**2))**7+
     3     (0.1612d0+dsqrt((0.8388d0)**2+(0.1274d0*u)**2))**5)
c     Murca_n:  n+n -> n+p+e+nu
      rmurca_n_n3p2B(u,t)=39.1d0*t*fexp(-1.188d0/t)*rmurca_p_n3p2B(u)
c     This `rmurca_p_n3p2B' is exact in the limit t<<1 and is only approximate
c     when t~<1 since Yakovlev and Levenfish only calculated the t<<1 limit.

c     u_3p2C(t)=dsqrt(1.d0-t**4)/t*
c     1          (5.875d0-1.419d0*t**4+0.500d0*t**8)
c     r_3p2C(u)=1.d0/0.d0
c     ******************************
c     ******************************
      if (kfn(i)*kfp(i).eq.0.) then
         qmurca_nucl=0.d0
         return
      end if
c     ******************************

      rmn=mstn(i)
      rmp=mstp(i)

c     Murca_n:  n+n -> n+p+e+nu:
      alpha_n =1.76d0-0.63d0*(1.68d0/kfn(i))**2
      beta_n  =0.68d0
      qmurca_n=8.55d21 *rmn**3*rmp   * 
     1     (kfe(i)/1.68d0 + 
     1     kfm(i)/1.68d0) *
     2     alpha_n * beta_n * (t/1.d9)**8 
c     Murca_p:  n+p -> p+p+e+nu:
      alpha_p =alpha_n
      beta_p  =beta_n
c     qmurca_p=8.55d21 *rmn   *rmp**3* 
c     1         (kfe(i)/1.68d0 * (1.d0-kfe(i)/4.d0/kfp(i))+
c     1          kfm(i)/1.68d0 * (1.d0-kfm(i)/4.d0/kfp(i)) ) *
c     2         alpha_p * beta_p * (t/1.d9)**8
      qmurca_p=8.55d21 *rmn   *rmp**3* 
     1     (kfe(i)/1.68d0) * 
     1     (kfe(i)+3.d0*kfp(i)-kfn(i))**2/(8.d0*kfe(i)*kfp(i)) * ! From Oleg
     2     alpha_p * beta_p * (t/1.d9)**8

c     **** effect of superfluidity :

      if(t .lt. tcn(i))then
         if (i.ge.isf) then
            tt=t/tcn(i)
            u=u_1s0(tt)
            rmurca_n_n=rmurca_n_n1s0(u)
            rmurca_p_n=rmurca_p_n1s0(u)
         else
            tt=t/tcn(i)
            u=u_3p2B(tt)
            rmurca_n_n=rmurca_n_n3p2B(u,tt)
            rmurca_p_n=rmurca_p_n3p2B(u)
         end if
      else
         rmurca_n_n=1.0d0
         rmurca_p_n=1.0d0
      end if

      if(t .lt. tcp(i))then
         tt=t/tcp(i)
         u=u_1s0(tt)
         rmurca_n_p=rmurca_n_p1s0(u)
         rmurca_p_p=rmurca_p_p1s0(u)
      else
         rmurca_n_p=1.0d0
         rmurca_p_p=1.0d0
      end if

      rmurca_n=min(rmurca_n_p,rmurca_n_n)
      rmurca_p=min(rmurca_p_p,rmurca_p_n)

      qmurca_n=rmurca_n*qmurca_n
      qmurca_p=rmurca_p*qmurca_p
      qmurca_nucl=qmurca_n+qmurca_p

      return
      end

c**********************************************************************
c**********************************************************************

      subroutine nubrem_crust_nn(i,t,vion,qbrem_nn,tcn,isf,kfn,mstn)

c     **** checked partially on March 30, 1993

c**********************************************************************
c     Includes the Levenfish & Yakovlev suppression factors for DURCA,
c     and only Boltzmann factors for MURCA.
c**********************************************************************

      implicit real*8 (a-h,k-z)
      parameter (isize=10000)
      parameter (pi = 3.14159265d0)

      dimension tcn(0:isize),kfn(0:isize),mstn(0:isize)
      
c     ***************************
      fexp(x)=dexp(max(x,-7.d2))
c     ***************************
c     SINGLET PAIRING:
      u_1s0(t)=dsqrt(1.d0-t)*(1.456d0-0.157d0/dsqrt(t)+1.764d0/t)
c     Brem_nn:  n+n -> n+n+2nu
      rbrem_nn_n1s0(u)=
     1     (0.1747d0+dsqrt((0.8253d0)**2+(.07933d0*u)**2))**2*
     2     fexp(4.228d0-dsqrt((4.228d0)**2+(4.d0*u)**2))/2.0d0 +
     3     (0.7333d0+dsqrt((0.2667d0)**2+(0.1678d0*u)**2))**7.5d0*
     4     fexp(7.762d0-dsqrt((7.762d0)**2+(9.d0*u)**2))/2.0d0
c     ***************************
c     TRIPLET PAIRING:
      u_3p2B(t)=dsqrt(1.d0-t)*(0.7893d0+1.188d0/t)
c     Brem_nn:  n+n -> n+n+2nu
      rbrem_nn_n3p2B(u)=rbrem_nn_n1s0(u)
c     **********************************************
      n_nu=3.d0                 ! Number of neutrino famillies !
c     Brem_nn:  n+n -> n+n+2nu
      alpha_nn =0.59d0
      beta_nn  =0.56d0
      qbrem_nn=n_nu * 7.4d19 *       mstn(i)**4     * 
     1     (kfn(i)/1.68d0) * alpha_nn * beta_nn * (t/1.d9)**8
c     **** effect of superfluidity :
      if(t .lt. tcn(i))then
         if (i.ge.isf) then
            tt=t/tcn(i)
            u=u_1s0(tt)
            rbrem_nn=rbrem_nn_n1s0(u)
         else
            tt=t/tcn(i)
            u=u_3p2B(tt)
            rbrem_nn=rbrem_nn_n3p2B(u)
         end if
      else
         rbrem_nn=1.0d0
      end if
c     **** Reduction from ion's volume:
      
      qbrem_nn=rbrem_nn*qbrem_nn * (1.d0-vion)

      return
      end

c**********************************************************************

      subroutine nubrem_nucl(i,t,qbrem_nucl,tcn,tcp,isf,
     1     kfn,kfp,mstn,mstp)

c     **** checked partially on March 30, 1993

c**********************************************************************
c     Includes the Levenfish & Yakovlev suppression factors for DURCA,
c     and only Boltzmann factors for MURCA.
c**********************************************************************

      implicit real*8 (a-h,k-z)
      parameter (isize=10000)
      parameter (pi = 3.14159265d0)

      dimension tcn(0:isize),tcp(0:isize)

      dimension mstp(0:isize),mstn(0:isize)
      dimension kfp(0:isize),kfn(0:isize)
      
c     ***************************
      fexp(x)=dexp(max(x,-7.d2))
c     ***************************
c     SINGLET PAIRING:
      u_1s0(t)=dsqrt(1.d0-t)*(1.456d0-0.157d0/dsqrt(t)+1.764d0/t)
c     Brem_nn:  n+n -> n+n+2nu
      rbrem_nn_p1s0(u)=1.0d0    ! Not affected by proton pairing !
      rbrem_nn_n1s0(u)=
     1     (0.1747d0+dsqrt((0.8253d0)**2+(.07933d0*u)**2))**2*
     2     fexp(4.228d0-dsqrt((4.228d0)**2+(4.d0*u)**2))/2.0d0 +
     3     (0.7333d0+dsqrt((0.2667d0)**2+(0.1678d0*u)**2))**7.5d0*
     4     fexp(7.762d0-dsqrt((7.762d0)**2+(9.d0*u)**2))/2.0d0
c     Brem_np:  n+p -> n+p+2nu
      rbrem_np_p1s0(u)=
     1     (0.9982d0+dsqrt((0.0018d0)**2+(0.3815d0*u)**2))**1*
     2     fexp(1.306d0-dsqrt((1.306d0)**2+(1.d0*u)**2))/2.732d0 +
     3     (0.3949d0+dsqrt((0.6051d0)**2+(0.2666d0*u)**2))**7*
     4     fexp(3.303d0-dsqrt((3.303d0)**2+(4.d0*u)**2))/1.577d0
      rbrem_np_n1s0(u)=rbrem_np_p1s0(u)
c     Brem_pp:  p+p -> p+p+2nu
      rbrem_pp_p1s0(u)=rbrem_nn_n1s0(u)
      rbrem_pp_n1s0(u)=1.0d0    ! Not affected by neutron pairing !
c     ***************************
c     TRIPLET PAIRING:
      u_3p2B(t)=dsqrt(1.d0-t)*(0.7893d0+1.188d0/t)
c     Brem_nn:  n+n -> n+n+2nu
      rbrem_nn_n3p2B(u)=rbrem_nn_n1s0(u)
c     Brem_np:  n+p -> n+p+2nu
      rbrem_np_n3p2B(u)=rbrem_np_n1s0(u)
c     Brem_pp:  p+p -> p+p+2nu
      rbrem_pp_n3p2B(u)=1.0     ! Not affected by neutron pairing !
c     ***************************
c     ****** murca : **********************************************
      n_nu=3.d0                 ! Number of neutrino famillies !
c     Brem_nn:  n+n -> n+n+2nu
      alpha_nn =0.59d0
      beta_nn  =0.56d0
      qbrem_nn=n_nu * 7.4d19 *       mstn(i)**4     * 
     1     (kfn(i)/1.68d0) * alpha_nn * beta_nn * (t/1.d9)**8
c     Brem_np:  n+p -> n+p+2nu
      alpha_np =1.06d0
      beta_np  =0.66d0
      qbrem_np=n_nu * 1.5d20 * mstn(i)**2*mstp(i)**2 * 
     1     (kfp(i)/1.68d0) * alpha_np * beta_np * (t/1.d9)**8
c     Brem_pp:  p+p -> p+p+2nu
      alpha_pp=0.11d0
      beta_pp=0.7d0
      qbrem_pp=n_nu * 7.4d19 *       mstp(i)**4      * 
     1     (kfp(i)/1.68d0) * alpha_pp * beta_pp * (t/1.d9)**8
c     **** effect of superfluidity :

      if(t .lt. tcn(i))then
         if (i.ge.isf) then
            tt=t/tcn(i)
            u=u_1s0(tt)
            rbrem_nn_n=rbrem_nn_n1s0(u)
            rbrem_np_n=rbrem_np_n1s0(u)
            rbrem_pp_n=rbrem_pp_n1s0(u)
         else
            tt=t/tcn(i)
            u=u_3p2B(tt)
            rbrem_nn_n=rbrem_nn_n3p2B(u)
            rbrem_np_n=rbrem_np_n3p2B(u)
            rbrem_pp_n=rbrem_pp_n3p2B(u)
         end if
      else
         rbrem_nn_n=1.0d0
         rbrem_np_n=1.0d0
         rbrem_pp_n=1.0d0
      end if

      if(t .lt. tcp(i))then
         tt=t/tcp(i)
         u=u_1s0(tt)
         rbrem_nn_p=rbrem_nn_p1s0(u)
         rbrem_np_p=rbrem_np_p1s0(u)
         rbrem_pp_p=rbrem_pp_p1s0(u)
      else
         rbrem_nn_p=1.0d0
         rbrem_np_p=1.0d0
         rbrem_pp_p=1.0d0
      end if

      rbrem_nn=min(rbrem_nn_p,rbrem_nn_n)
      rbrem_np=min(rbrem_np_p,rbrem_np_n)
      rbrem_pp=min(rbrem_pp_p,rbrem_pp_n)
      qbrem_nn=rbrem_nn*qbrem_nn
      qbrem_np=rbrem_np*qbrem_np
      qbrem_pp=rbrem_pp*qbrem_pp

      qbrem_nucl=qbrem_nn+qbrem_np+qbrem_pp

      return

      end

c     *********************************************************************
c     H Y P E R O N S
c     *********************************************************************
      subroutine numurca_hyp(i,t,qmurca_hyp)
      implicit real*8(a-h,k-z)
      qmurca_hyp=0.d0
      return
      end

      subroutine nubrem_hyp(i,t,qbrem_hyp)
      implicit real*8(a-h,k-z)
      qbrem_hyp=0.0d0
      return
      end

      subroutine nudurca_h(irank,i,t,rho,qdurca_np,qdurca_lap,
     1     qdurca_smn,qdurca_smla,qdurca_sms0,tcn,tcp,tcla,isf,
     2     bar,yelect,ymuon,mstp,mstn,mstla,mstsm,msts0,mstsp,
     3     durca_ctrl_e,durca_ctrl_m,idurca_lap,idurca_smla,
     4     idurca_smn,idurca_sms0,idurca_np,
     5     sf_lgtau1,sf_lgtau2,sf_lgr,sf_lgr2)
      
c     **** checked partially on March 30, 1993

c**********************************************************************
c     From Levenfish & Yakovlev, Astron. Lett. 20 (1994), p. 43        *
c**********************************************************************

      implicit real*8 (a-h,k-z)
      parameter (pi = 3.14159265d0)
      parameter (isize=10000)

      dimension bar(0:isize),
     2     yelect(0:isize),ymuon(0:isize)

      dimension tcn(0:isize),tcp(0:isize),tcla(0:isize)
      
      dimension mstp(0:isize),mstn(0:isize),mstla(0:isize),
     2     mstsm(0:isize),msts0(0:isize),mstsp(0:isize)
      dimension idurca_np(0:isize),idurca_lap(0:isize),
     2     idurca_smn(0:isize),idurca_smla(0:isize),
     3     idurca_sms0(0:isize)

      dimension durca_ctrl_e(0:isize),durca_ctrl_m(0:isize)
      
      dimension sf_lgtau1(35),sf_lgtau2(35)
      dimension sf_lgr(35,35),sf_lgr2(35,35)
      
c     ***************************
      fexp(x)=dexp(max(x,-7.d2))

      u_1s0(t)=dsqrt(1.d0-t)*(1.456d0-0.157d0/dsqrt(t)+1.764d0/t)
      u_3p2B(t)=dsqrt(1.d0-t)*(0.7893d0+1.188d0/t)
c     u_3p2C(t)=dsqrt(1.d0-t**4)/t*
c     1          (5.875d0-1.419d0*t**4+0.500d0*t**8)

      r_1s0(u)=(0.2312d0+dsqrt(0.7688d0**2+(0.1438d0*u)**2))**5.5d0*
     1     fexp(3.427d0-dsqrt(3.427d0**2+u**2))
      r_3p2B(u)=(0.2546d0+dsqrt(0.7454d0**2+(0.1284d0*u)**2))**5*
     1     fexp(2.701d0-dsqrt(2.701d0**2+u**2))
      r_1s0_1s0(u1,u2)=func_r_1s0_1s0(u1,u2)
      r_1s0_3p2B(t1,t2)=func_r_1s0_3P2B(irank,t1,t2,
     1     sf_lgtau1,sf_lgtau2,sf_lgr,sf_lgr2)
!     WARNING: this uses t1 & t2
!     instead of u1 & u2
c     r_3p2C(u)=1.d0/0.d0

c****************************

      rmn = mstn(i)
      rmp = mstp(i)
      rmla=mstla(i)
      rmsm=mstsm(i)
      rms0=msts0(i)
      rmsp=mstsp(i)

      rate_np  =1.0000d0
      rate_lap =0.0394d0
      rate_smn =0.0125d0
      rate_smla=0.2055d0
      rate_sms0=0.6052d0

c     **** n-p:
      if (.FALSE.) then

c     ------------------------------------------------------
c     Dany's method for neutron-proton direct Urca
         
         if     (idurca_np(i).eq.1) then
            qdurca_np= rate_np * 4.24d27*rmn*rmp*(t/1.d9)**6*
     1           (abs(yelect(i))*bar(i)/0.16d0)**(1.d0/3.d0)
         else if(idurca_np(i).eq.2) then
            qdurca_np= rate_np * 4.24d27*rmn*rmp*(t/1.d9)**6*
     1           ((abs(yelect(i))*bar(i)/0.16d0)**(1.d0/3.d0)+
     2           (abs( ymuon(i))*bar(i)/0.16d0)**(1.d0/3.d0))
         else
            qdurca_np= 0.d0
         end if
         
      else
         
c     ------------------------------------------------------
c     New method for neutron-proton direct Urca
         
         qdurca_np=durca_ctrl_e(i)*rate_np*4.24d27*rmn*rmp*(t/1.d9)**6*
     1        (abs(yelect(i))*bar(i)/0.16d0)**(1.d0/3.d0)+
     2        durca_ctrl_m(i)*rate_np * 4.24d27*rmn*rmp*(t/1.d9)**6*
     3        (abs(ymuon(i))*bar(i)/0.16d0)**(1.d0/3.d0)
      end if
      
c     Pairing suppression:
      if ((t.gt.tcn(i)).and.(t.gt.tcp(i))) then
         r_np=1.d0
      else if ((t.gt.tcn(i)).and.(t.le.tcp(i))) then
         tt=t/tcp(i)
         u=u_1s0(tt)
         r_np=r_1s0(u)
      else if ((t.le.tcn(i)).and.(t.gt.tcp(i))) then
         if (i.ge.isf) then 
            tt=t/tcn(i)
            u=u_1s0(tt)
            r_np=r_1s0(u)
         else
            tt=t/tcn(i)
            u=u_3p2B(tt)
            r_np=r_3p2B(u)
         end if
      else
         tt1=t/tcp(i)
         u1=u_1s0(tt1)
         if (i.ge.isf) then 
            tt2=t/tcn(i)
            u2=u_1s0(tt2)
            r_np=r_1s0_1s0(u1,u2)
         else
            tt2=t/tcn(i)
            u2=u_3p2B(tt2)      ! Not needed for r_1s0_3p2B
            r_np=r_1s0_3p2B(tt1,tt2)
         end if
      end if
c     **** la-p:
      if     (idurca_lap(i).eq.1)then
         qdurca_lap= rate_lap * 4.24d27*rmla*rmp*(t/1.d9)**6*
     1        (abs(yelect(i))*bar(i)/0.16d0)**(1.d0/3.d0)
      else if(idurca_lap(i).eq.2)then
         qdurca_lap= rate_lap * 4.24d27*rmla*rmp*(t/1.d9)**6*
     1        ((abs(yelect(i))*bar(i)/0.16d0)**(1.d0/3.d0)+
     2        (abs( ymuon(i))*bar(i)/0.16d0)**(1.d0/3.d0))
      else
         qdurca_lap= 0.d0
      end if
c     **** sm-n:
      if     (idurca_smn(i).eq.1)then
         qdurca_smn= rate_smn * 4.24d27*rmn*rmsm*(t/1.d9)**6*
     1        (abs(yelect(i))*bar(i)/0.16d0)**(1.d0/3.d0)
      else if(idurca_smn(i).eq.2)then
         qdurca_smn= rate_smn * 4.24d27*rmn*rmsm*(t/1.d9)**6*
     1        ((abs(yelect(i))*bar(i)/0.16d0)**(1.d0/3.d0)+
     2        (abs( ymuon(i))*bar(i)/0.16d0)**(1.d0/3.d0))
      else
         qdurca_smn= 0.d0
      end if
c     **** sm-la:
      if     (idurca_smla(i).eq.1)then
         qdurca_smla= rate_smla * 4.24d27*rmsm*rmla*(t/1.d9)**6*
     1        (abs(yelect(i))*bar(i)/0.16d0)**(1.d0/3.d0)
      else if(idurca_smla(i).eq.2)then
         qdurca_smla= rate_smla * 4.24d27*rmsm*rmla*(t/1.d9)**6*
     1        ((abs(yelect(i))*bar(i)/0.16d0)**(1.d0/3.d0)+
     2        (abs( ymuon(i))*bar(i)/0.16d0)**(1.d0/3.d0))
      else
         qdurca_smla= 0.d0
      end if
c     **** sm-s0:
      if     (idurca_sms0(i).eq.1)then
         qdurca_sms0= rate_sms0 * 4.24d27*rmsm*rms0*(t/1.d9)**6*
     1        (abs(yelect(i))*bar(i)/0.16d0)**(1.d0/3.d0)
      else if(idurca_sms0(i).eq.2)then
         qdurca_sms0= rate_sms0 * 4.24d27*rmsm*rms0*(t/1.d9)**6*
     1        ((abs(yelect(i))*bar(i)/0.16d0)**(1.d0/3.d0)+
     2        (abs( ymuon(i))*bar(i)/0.16d0)**(1.d0/3.d0))
      else
         qdurca_sms0= 0.d0
      end if

c     **** effect of superfluidity :

c     (Note : isf is the zone where neutron pairing shifts from 3P2 to 1S0)

c     Neutron pairing suppression:
      if(t.lt.tcn(i)) then
         if (i.ge.isf) then 
            tt=t/tcn(i)
            u=u_1s0(tt)
            rn=r_1s0(u)
         else
            tt=t/tcn(i)
            u=u_3p2B(tt)
            rn=r_3p2B(u)
         end if
      else
         rn=1.0 d0
      end if
c     Proton pairing suppression:
      if(t .lt. tcp(i))then
         tt=t/tcp(i)
         u=u_1s0(tt)
         rp=r_1s0(u)
      else
         rp=1.0d0
      end if
c     Lambda pairing suppression:
      if(t .lt. tcla(i))then
         tt=t/tcla(i)
         u=u_1s0(tt)
         rla=r_1s0(u)
      else
         rla=1.0d0
      end if
c     Sigma- pairing suppression:
      rsm=1.0d0
c     Sigma0 pairing suppression:
      rs0=1.0d0
c     Sigma+ pairing suppression:
      rsp=1.0d0
c**** 
c     r_np  =min(rn,rp)
      r_lap =min(rla,rp)
      r_smn =min(rsm,rn)
      r_smla=min(rsm,rla)
      r_sms0=min(rsm,rs0)

      qdurca_np  =r_np  *qdurca_np
      qdurca_lap =r_lap *qdurca_lap
      qdurca_smn =r_smn *qdurca_smn
      qdurca_smla=r_smla*qdurca_smla
      qdurca_sms0=r_sms0*qdurca_sms0

c     qdurca =qdurca_np  + qdurca_lap +
c     2        qdurca_smn + qdurca_smla+ qdurca_sms0

      return

      end

c     *********************************************************************
c     *********************************************************************
      function func_r_1s0_1s0(v1,v2)
c     CHECKED ON MARCH 7, 2001 against Fig. 2 of L&Y Paper
      implicit real*8 (a-h,k-z)
      parameter (pi = 3.14159265d0,gamma=5040.d0/457.d0/pi**6)
      u=v1**2+v2**2
      w=v1**2-v2**2
      u1=1.8091d0+dsqrt(v1**2+2.2476d0**2)
      u2=1.8091d0+dsqrt(v2**2+2.2476d0**2)
      p=(u+12.421d0+dsqrt(w**2+16.350d0*u+45.171d0))/2.d0
      q=(u+12.421d0-dsqrt(w**2+16.350d0*u+45.171d0))/2.d0
      ps=(u+dsqrt(w**2+5524.8d0*u+6.7737d0))/2.d0
      pe=(u+0.43847d0+dsqrt(w**2+8.3680d0*u+491.32d0))/2.d0
      D=(u1*u2)**1.5/(2.d0*4.0567d0**5)*(u1**2+u2**2)*
     1     dexp(-u1-u2+8.1134d0)
      K0=dsqrt(p-q)/120.d0*(6.d0*p**2+83.d0*p*q+16.d0*q**2)-
     1     dsqrt(p)*q/8.d0*(4.d0*p+3.d0*q)*
     2     dlog((dsqrt(p)+dsqrt(p-q))/dsqrt(q))
      K1=pi**2*dsqrt(p-q)/6.d0*(p+2.d0*q)-
     1     pi**2/2.d0*q*dsqrt(p)*
     2     dlog((dsqrt(p)+dsqrt(p-q))/dsqrt(q))
      K2=7.d0*pi**4/60.d0*dsqrt(p-q)
      S=gamma*(K0+K1+0.42232d0*K2)*dsqrt(pi/2.d0)*ps**0.25*
     1     dexp(-dsqrt(pe))
      
      func_r_1s0_1s0=u/(u+0.9163d0)*S+D

      return
      end
c     *********************************************************************
c     *********************************************************************
c     CHECKED ON MARCH 8, 2001 against Fig. 3 of L&Y Paper
      
      function func_r_1s0_3p2B(irank,t1,t2,sf_lgtau1,sf_lgtau2,
     1     sf_lgr,sf_lgr2)
      
      implicit real*8 (a-h,k-z)
      dimension sf_lgtau1(35),sf_lgtau2(35)
      dimension sf_lgr(35,35),sf_lgr2(35,35)
      
      lt1=log10(t1)
      lt2=log10(t2)
      call splint2(sf_lgtau1,sf_lgtau2,sf_lgr,sf_lgr2,35,35,lt1,lt2,lr)
      func_r_1s0_3p2B=10.d0**lr
c     HERE DANY ccccccccccccccccccccccccccccccccccccccccccccccccccc
      lt=sqrt(lt1**2+lt2**2)
      lt_limit=3.
      if (lt.le.lt_limit) then
         call splint2(sf_lgtau1,sf_lgtau2,sf_lgr,sf_lgr2,35,35,
     1        lt1,lt2,lr)
         func_r_1s0_3p2B=10.d0**lr
      else
         lt1=lt1/lt
         lt2=lt2/lt
         call splint2(sf_lgtau1,sf_lgtau2,sf_lgr,sf_lgr2,35,35,
     1        lt1,lt2,lr)
         func_r_1s0_3p2B=10.d0**lr * exp(-lt/lt_limit)
      end if
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      return
      end

c     *********************************************************************
c     *********************************************************************

      subroutine nufast(i,t,rho,qfast,tcn,tcp,isf,bar,theta_k,theta_p,
     1     yelect,rhoexo,cexo,pexo,mstn,mstp,kfe)

c     **** checked partially on March 30, 1993

c**********************************************************************
c     Includes the Levenfish & Yakovlev suppression factors for DURCA,
c**********************************************************************

      implicit real*8 (a-h,k-z)
      parameter (isize=10000)

      parameter (pi = 3.14159265d0)

      dimension bar(0:isize),yelect(0:isize),
     1     theta_k(0:isize),theta_p(0:isize)
      dimension tcn(0:isize),tcp(0:isize)
      dimension mstp(0:isize),mstn(0:isize),kfe(0:isize)
      
c     ***************************
      fexp(x)=dexp(max(x,-7.d2))

      u_1s0(t)=dsqrt(1.d0-t)*(1.456d0-0.157d0/dsqrt(t)+1.764d0/t)
      r_1s0(u)=(0.2312d0+dsqrt(0.7688d0**2+(0.1438d0*u)**2))**5.5d0*
     1     fexp(3.427d0-dsqrt(3.427d0**2+u**2))

      u_3p2B(t)=dsqrt(1.-t)*(5.596d0+8.424d0/t)
      r_3p2B(u)=(0.2546d0+dsqrt(0.7454d0**2+(0.01811d0*u)**2))**5*
     1     fexp(2.701d0-dsqrt(2.701d0**2+u**2/(16.d0*pi)))

c     u_3p2C(t)=dsqrt(1.d0-t**4)/t*
c     1          (5.875d0-1.419d0*t**4+0.500d0*t**8)
c     r_3p2C(u)=d01./0.d0

c*******************************

      rmn=mstn(i)
      rmp=mstp(i)

      u=bar(i)/0.16d0
      ratio=.319d0/(abs(yelect(i))*u)**(1.d0/3.d0)
      zero=0.d0
      f=max(zero,1-ratio**2)
      f=dsqrt(f)

c     **** Kaon urca: 

      if (theta_k(i).ne.0.) then
         g_a=1.0d0
         mu_el=kfe(i)*197.d0
         qkaon= 5.d0/4.d0*dsin(theta_k(i))**2 * sin(0.223d0)**2 *
     1        2.21d26 * mstn(i) * mstp(i) * (mu_el/100.d0) *
     2        (1.d0+3.d0*g_a**2) * (t/1.d9)**6
c     From Thorsson et al, Phys. Rev. D52, p. 3739, 1995
      else
         qkaon=0.d0
      end if

c     **** eurca :

      if(rho.ge.rhoexo)then
         qexo=cexo*(rho/2.8d14)**(2.d0/3.d0)*(t/1.d9)**pexo
c     rhoexo2=1.4d25
c     if (rho.ge.rhoexo2) then
c     qexo=101.d0*qexo
c     end if
      else 
         qexo=0.d0
      end if

c     **** effect of superfluidity :

c     (Note : isf is the zone where neutron pairing shifts from 3P2 to 1S0)

      if( (t.lt.tcp(i)) .and. (t.lt.tcn(i)) )then
         if (i.ge.isf) then 
            un=u_1s0(t/tcn(i))
            rn=r_1s0(un)
         else
            un=u_3p2B(t/tcn(i))
            rn=r_3p2B(un)
         end if
         up=u_1s0(t/tcp(i))
         rp=r_1s0(up)
         r=min(rn,rp)
      else if(t .lt. tcn(i)) then
         if (i.ge.isf) then 
            u=u_1s0(t/tcn(i))
            r=r_1s0(u)
         else
            u=u_3p2B(t/tcn(i))
            r=r_3p2B(u)
         end if
      else if(t .lt. tcp(i))then
         u=u_1s0(t/tcp(i))
         r=r_1s0(u)
      else
         r=1.d0
      end if

      qkaon=qkaon*r

      r_exo=r
      if ((pexosn.eq.0.).and.(pexosp.eq.0.)) r_exo=1.
      qexo =qexo *r_exo

      qfast=qkaon+qexo
      return

      end

c     *********************************************************************

      subroutine nu_1s0_pbf(T,Tc,mst,kf,q_1s0_pbf)

c     This subroutine uses only the Axial part: good for n & p !
c     From : Neutrino emission due to proton pairing in neutron stars
c     Kaminker, A. D.; Haensel, P.; Yakovlev, D. G.
c     1999A&A...345L..14K
c     Vector part is put to zero according to:
c     "Vector current conservation and neutrino emission from 
c     singlet-paired baryons in neutron stars"
c     Leinson, L.B., & Perez, A.
c     2006PhLB..638..114L
      implicit real*8 (a-h,k-z)
      if (T.le.Tc) then
         pf=kf*197.d0
         vf=pf/mst/940.d0
         a_v=0.d0
         a_a=1.60d0*vf**2*(mst**2+11.d0/42.d0)
         a=a_v+a_a
c     The vector part in "a" has been put to zero !
         tau=T/Tc
         u=dsqrt(1.d0-tau)*(1.456d0-0.157d0/dsqrt(tau)+1.764d0/tau)
         q_1s0_pbf=1.170d21*mst**2*vf*(T/1.d9)**7*3.d0*a*
     1        control_pbf_1S0(u)
      else
         q_1s0_pbf=0.0d0
      end if
      return
      end
c     *********************************************************************
c     *********************************************************************
      subroutine nu_n3p2_B_pbf(T,Tc,mst,kf,q_n3p2_pbf)
c     This subroutine uses only the Axial part from
c     Neutrino emission due to Cooper pairing of nucleons in cooling neutron stars
c     Yakovlev, D. G.; Kaminker, A. D.; Levenfish, K. P.
c     1999A&A...343..650Y
c     Vector part is put to zero according to:
c     Neutrino emission due to Cooper pairing in neutron stars
c     Leinson, L.B., & Perez, A.
c     2006astro.ph..6653L
c     
c     For the j=2 m=0 gap
      implicit real*8 (a-h,k-z)
      parameter (pi = 3.14159265d0,rhonucl=2.8d14)
      if (T.le.Tc) then
         pf=kf*197.d0
         vf=pf/mst/940.d0
         g_A=1.26d0
         a_v=0.d0
c     AWS: Changed to satisfy Leinson's complaint about anomalous
c     axial contributions         
         a_a=0.5d0*g_A**2
         a=a_v+a_a
c     The vector part in "a" has been put to zero !
         tau=T/Tc
         u=dsqrt(1.d0-tau)*(0.7893d0+1.764d0/tau)
         q_n3p2_pbf=1.170d21*mst**2*vf*(T/1.d9)**7*3.d0*a*
     1        control_pbf_3P2_B(u)
      else
         q_n3p2_pbf=0.0d0
      end if
      return
      end
c     *********************************************************************
c     *********************************************************************
      function control_pbf_1S0(v)
      implicit real*8 (a-h,k-z)
      x=0.602d0*v**2+0.5942d0*v**4+0.288d0*v**6
      y=dsqrt(0.5547d0+dsqrt(0.4453d0**2+0.01130*v**2))
      z=dexp(-dsqrt(4.d0*v**2+2.245d0**2)+2.245d0)
      control_pbf_1S0=x*y*z
      return
      end
c     *********************************************************************
c     *********************************************************************
      function control_pbf_3P2_B(v)
      implicit real*8 (a-h,k-z)
      x=(1.204d0*v**2+3.733d0*v**4+0.3191*v**6)/(1.d0+0.3511*v**2)
      y=(0.7591+dsqrt(0.2409d0**2+0.3145d0*v**2))**2
      z=dexp(-dsqrt(4.d0*v**2+0.4616d0**2)+0.4616d0)
      control_pbf_3P2_B=x*y*z
      return
      end
c     *********************************************************************
c     *********************************************************************
      function control_pbf_3P2_C(v)
      implicit real*8 (a-h,k-z)
      x=0.4013d0*v**2-0.043d0*v**4+0.002172d0*v**6
      y=1.d0/
     1     (1.d0-2.018d-1*v**2+2.601d-2*v**4-1.477d-3*v**6+4.34d-5*v**8)
      control_pbf_3P2_C=x*y
      return
      end

c     *********************************************************************

c     *********************************************************************
c     
c     Q U A R K   P R O C E S S E S
c     
c     *********************************************************************
      subroutine nudurca_q(i,t,rho,qdurca_q,tcu1,tcu2,tcu3,
     1     tcd1,tcd2,tcd3,tcs1,tcs2,tcs3,kfe,kfm,kfqu,kfqd,kfqs,
     2     idurca_quqd,idurca_quqs)
c     *** Checked on April 28, 2004              <======  TO BE DONE !?!?!?
c**********************************************************************
c     Includes the Levenfish & Yakovlev 
c     suppression factors for nucleon DURCA
c**********************************************************************
      implicit real*8 (a-h,k-z)
      parameter (pi = 3.14159265d0)
      parameter (g_fermi=1.436d-49,theta_c=0.239,mev=1.6d-6)
      parameter (h_bar=1.054d-27,kb=1.38d-16,c_light=3.d10)
      parameter (isize=10000)

c     April 2004: seperate Tc for flavor AND color:
      dimension tcu1(0:isize),tcu2(0:isize),tcu3(0:isize),
     2     tcd1(0:isize),tcd2(0:isize),tcd3(0:isize),
     3     tcs1(0:isize),tcs2(0:isize),tcs3(0:isize)

      dimension kfe(0:isize),kfm(0:isize),
     1     kfqu(0:isize),kfqd(0:isize),kfqs(0:isize)
      dimension idurca_quqd(0:isize),idurca_quqs(0:isize)
      
c     ***************************
      fexp(x)=dexp(max(x,-7.d2))

      u_1s0(t)=dsqrt(1.d0-t)*(1.456d0-0.157d0/dsqrt(t)+1.764d0/t)
      r_1s0(u)=(0.2312d0+dsqrt(0.7688d0**2+(0.1438d0*u)**2))**5.5d0*
     1     fexp(3.427d0-dsqrt(3.427d0**2+u**2))

c     u_3p2b(t)=dsqrt(1.d0-t)*(5.596d0+8.424d0/t)
c     r_3p2b(u)=(0.2546d0+dsqrt(0.7454d0**2+(0.01811d0*u)**2))**5*
c     1           fexp(2.701d0-dsqrt(2.701d0**2+u**2/(16.d0*pi)))

      r_1s0_1s0(u1,u2)=func_r_1s0_1s0(u1,u2)
c     r_1s0_3p2B(t1,t2)=func_r_1s0_3P2B(t1,t2) ! WARNING: this uses t1 & t2
c     ! instead of u1 & u2
c     u_3p2c(t)=dsqrt(1.d0-t**4)/t*
c     1           (5.875d0-1.419d0*t**4+0.500d0*t**8)
c     r_3p2c(u)=1.d0/0.d0
c****************************

c     * AWS Needs to be fixed to restore quark functionality
      alpha_c=0.d0
      
c     ******************************************
c     **** u-d: for each color:
      coeff_ud= (1.d0/3.d0) * 914./315.*
     1     (g_fermi*dcos(theta_c)/(h_bar**5*c_light**3))**2*
     2     alpha_c
      if (idurca_quqd(i).eq.1)then
         qdurca_quqd=coeff_ud*1.d+39*
     1        kfqd(i)*kfqu(i)*kfe(i)*h_bar**3*(kb*t)**6
      else if(idurca_quqd(i).eq.2)then
         qdurca_quqd=coeff_ud*1.d+39*
     1        (kfqd(i)*kfqu(i)*kfe(i)*h_bar**3*(kb*t)**6
     2        +kfqd(i)*kfqu(i)*kfm(i)*h_bar**3*(kb*t)**6)
      else
         qdurca_quqd= 0.d0
      end if
c     Pairing suppression: COLOR 1
      if ((t.gt.tcu1(i)).and.(t.gt.tcd1(i))) then
         r_ud1=1.d0
      else if ((t.gt.tcu1(i)).and.(t.le.tcd1(i))) then
         tt=t/tcd1(i)
         u=u_1s0(tt)
         r_ud1=r_1s0(u)
      else if ((t.le.tcu1(i)).and.(t.gt.tcd1(i))) then
         tt=t/tcu1(i)
         u=u_1s0(tt)
         r_ud1=r_1s0(u)
      else
         tt1=t/tcu1(i)
         u1=u_1s0(tt1)
         tt2=t/tcd1(i)
         u2=u_1s0(tt2)
         r_ud1=r_1s0_1s0(u1,u2)
      end if
c     Pairing suppression: COLOR 2
      if ((t.gt.tcu2(i)).and.(t.gt.tcd2(i))) then
         r_ud2=1.d0
      else if ((t.gt.tcu2(i)).and.(t.le.tcd2(i))) then
         tt=t/tcd2(i)
         u=u_1s0(tt)
         r_ud2=r_1s0(u)
      else if ((t.le.tcu2(i)).and.(t.gt.tcd2(i))) then
         tt=t/tcu2(i)
         u=u_1s0(tt)
         r_ud2=r_1s0(u)
      else
         tt1=t/tcu2(i)
         u1=u_1s0(tt1)
         tt2=t/tcd2(i)
         u2=u_1s0(tt2)
         r_ud2=r_1s0_1s0(u1,u2)
      end if
c     Pairing suppression: COLOR 3
      if ((t.gt.tcu3(i)).and.(t.gt.tcd3(i))) then
         r_ud3=1.d0
      else if ((t.gt.tcu3(i)).and.(t.le.tcd3(i))) then
         tt=t/tcd3(i)
         u=u_1s0(tt)
         r_ud3=r_1s0(u)
      else if ((t.le.tcu3(i)).and.(t.gt.tcd3(i))) then
         tt=t/tcu3(i)
         u=u_1s0(tt)
         r_ud3=r_1s0(u)
      else
         tt1=t/tcu3(i)
         u1=u_1s0(tt1)
         tt2=t/tcd3(i)
         u2=u_1s0(tt2)
         r_ud3=r_1s0_1s0(u1,u2)
      end if
c     Putting everything together (3 colors + pairing suppression) :

      qdurca_quqd= (r_ud1+r_ud2+r_ud3) * qdurca_quqd

c     ******************************************
c     **** u-s: for each color
      theta_34=pi/4.            ! <----- A rough approximation !
      coeff_us=(1.d0/3.d0) * 457.*pi/840.*
     1     (g_fermi*dsin(theta_c)/(h_bar**5*c_light**3))**2*
     2     (1.d0-dcos(theta_34))
      if (idurca_quqs(i).eq.1)then
         mus=dsqrt(kfqs(i)**2*1.d+26*c_light**2*h_bar**2+
     1        (mev*strange_mass)**2)
         qdurca_quqs=coeff_us*1.e+26*
     1        mus/c_light*kfqu(i)*kfe(i)*h_bar**2*(kb*t)**6
      else if(idurca_quqs(i).eq.2)then
         mus=dsqrt(kfqs(i)**2*1.e+13*c_light**2*h_bar**2+
     1        (mev*strange_mass)**2)
         qdurca_quqs=coeff_us*1.e+26*
     1        (mus/c_light*kfqu(i)*kfe(i)*h_bar**2*(kb*t)**6
     2        +mus/c_light*kfqu(i)*kfm(i)*h_bar**2*(kb*t)**6)
      else
         qdurca_quqs= 0.d0
      end if
c     Pairing suppression: COLOR 1
      if ((t.gt.tcu1(i)).and.(t.gt.tcs1(i))) then
         r_us1=1.d0
      else if ((t.gt.tcu1(i)).and.(t.le.tcs1(i))) then
         tt=t/tcs1(i)
         u=u_1s0(tt)
         r_us1=r_1s0(u)
      else if ((t.le.tcu1(i)).and.(t.gt.tcs1(i))) then
         tt=t/tcu1(i)
         u=u_1s0(tt)
         r_us1=r_1s0(u)
      else
         tt1=t/tcu1(i)
         u1=u_1s0(tt1)
         tt2=t/tcs1(i)
         u2=u_1s0(tt2)
         r_us1=r_1s0_1s0(u1,u2)
      end if
c     Pairing suppression: COLOR 2
      if ((t.gt.tcu2(i)).and.(t.gt.tcs2(i))) then
         r_us2=1.d0
      else if ((t.gt.tcu2(i)).and.(t.le.tcs2(i))) then
         tt=t/tcs2(i)
         u=u_1s0(tt)
         r_us2=r_1s0(u)
      else if ((t.le.tcu2(i)).and.(t.gt.tcs2(i))) then
         tt=t/tcu2(i)
         u=u_1s0(tt)
         r_us2=r_1s0(u)
      else
         tt1=t/tcu2(i)
         u1=u_1s0(tt1)
         tt2=t/tcs2(i)
         u2=u_1s0(tt2)
         r_us2=r_1s0_1s0(u1,u2)
      end if
c     Pairing suppression: COLOR 3
      if ((t.gt.tcu3(i)).and.(t.gt.tcs3(i))) then
         r_us3=1.d0
      else if ((t.gt.tcu3(i)).and.(t.le.tcs3(i))) then
         tt=t/tcs3(i)
         u=u_1s0(tt)
         r_us3=r_1s0(u)
      else if ((t.le.tcu3(i)).and.(t.gt.tcs3(i))) then
         tt=t/tcu3(i)
         u=u_1s0(tt)
         r_us3=r_1s0(u)
      else
         tt1=t/tcu3(i)
         u1=u_1s0(tt1)
         tt2=t/tcs3(i)
         u2=u_1s0(tt2)
         r_us3=r_1s0_1s0(u1,u2)
      end if
c     Putting everything together (3 colors + pairing suppression) :

      qdurca_quqs= (r_us1+r_us2+r_us3) * qdurca_quqs

c     All Quark Durca processes:

      qdurca_q = qdurca_quqd  + qdurca_quqs

      return
      end
c     *********************************************************************
c     *********************************************************************

      subroutine numurca_q(i,t,rho,qmurca_q,kfqu,tcu,tcd)

c     **** checked
      
c**********************************************************************
c     Includes the Levenfish & Yakovlev suppression factors for MURCA,
c**********************************************************************

      implicit real*8 (a-h,k-z)
      parameter (pi = 3.14159265d0)
      parameter (g_fermi=1.436d-49,theta_c=0.239,mev=1.6d-6)
      parameter (h_bar=1.054d-27,kb=1.38d-16,c_light=3.d10)

      parameter (isize=10000)

      dimension kfqu(0:isize),tcu(0:isize),tcd(0:isize)
     
      num_coeff=1.e0            ! Remains to be calculated

      qmurca_q=num_coeff*
     1     (alpha_c*g_fermi*cos(theta_c)/
     2     h_bar**5/c_light**4)**2 *
     3     (1.e13*kfqu(i)*h_bar) * (kb*t)**8

      
      
c     Up quark pairing suppression:
      if (t.lt.tcu(i)) then
         r_u=dexp(-1.76*tcu(i)/t)
      else
         r_u=1.0
      end if
c     Down quark pairing suppression:
      if (t.lt.tcd(i)) then
         r_d=dexp(-1.76*tcd(i)/t)
      else
         r_d=1.0
      end if
c     c Strange quark pairing suppression:
c     if (t.lt.tcs(i)) then
c     r_s=dexp(-1.76*tcs(i)/t)
c     else
c     r_s=1.0
c     end if
c**** 
      r_ud=r_u*r_d
      r_us=r_u*r_s

      qmurca_q = qmurca_q * r_ud

      return

      end


c**********************************************************************
c**********************************************************************
      function function_pbf_1S0(v)
      implicit real*8 (a-h,k-z)
      parameter(pi=3.1415926535)
      x=0.602d0*v**2+0.5942d0*v**4+0.288d0*v**6
      y=dsqrt(0.5547d0+dsqrt(0.4453d0**2+0.01130*v**2))
      z=dexp(-dsqrt(4.d0*v**2+2.245d0**2)+2.245d0)
      function_pbf_1S0=x*y*z
      return
      end
c     *********************************************************************
c     *********************************************************************




