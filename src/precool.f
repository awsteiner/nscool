c     *********************************************************************
c     *********************************************************************
      subroutine grid(irank,idec,rhocore,rhodrip,rhoenv,rhosurf,
     1     imax,icore,idrip,ienv,rad,rrho,pres,dvol,emas,phi)
      implicit real*8 (a-h,k-z)
      parameter (isize=10000)
      character*150 f_i,f_Teff,f_Temp,f_Star
      character*3 version
      dimension rad(0:isize),rrho(0:isize),pres(0:isize),
     2     dvol(0:isize),
     3     emas(0:isize),phi(0:isize)
c*************************************************
c     dvol(i) is the volume of the half-shell between 
c     the spheres of radii rad(i) and rad(i+1)
c     (As calculated in subroutine grid)
c*************************************************
      dimension rad_t(10000),bar_t(10000),rho_t(10000),pres_t(10000),
     1     emas_t(10000),phi_t(10000)
      parameter (pi=3.1415926535d0)

c     DP: Define zone indices: icore, idrip & isurf
      
      icore=2*((icore-1)/2)+1   ! Makes sure icore is odd
      idel1=int(Log10(rhocore/rhodrip)*float(idec))
      idel1=2*(idel1/2)         ! Makes sure idel1 is even
      idrip=icore+idel1
      idel2=int(Log10(rhodrip/rhosurf)*float(idec))
      idel2=2*(idel2/2)         ! Makes sure idel2 is even
      isurf=idrip+idel2

c     AWS: Read TOV profile. Because of the 'dimension' line above,
c     arrays rad_t, bar_t, rho_t, pres_t, emas_t, and phi_t are unit
c     indexed.
      
      call nscool_star_struct(irank,icore,rhocore,rad_t,bar_t,rho_t,
     1     pres_t,emas_t,phi_t,rad,jmax,jcore,w1,w2)

c$$$  c     ****
c$$$  jdrip=0
c$$$  jcore=0
c$$$  do j=1,jmax
c$$$  c     AWS: convert m to cm
c$$$  rad_t(j)=rad_t(j)*100.d0
c$$$  if ((rho_t(j).lt.rhocore).and.(jcore.eq.0)) then
c$$$  jcore=j-1
c$$$  end if
c$$$  end do
c$$$  c     ****
      
c     DP: Get the core radius exactly:
      drho=rho_t(jcore)-rho_t(jcore+1)
      w1=(rhocore-rho_t(jcore+1))/drho
      w2=1.d0-w1
      rad_core=w1*rad_t(jcore)+w2*rad_t(jcore+1)
c     
c     Define Star grid: ****************************************
c     
c     dvol = physical volume between rad(i-1) and rad(i)
c     
c     CORE: zoning with (approximate) constant volume: *********
      do i=0,icore
         rad(i)=(float(i)/float(icore))**(1./3.) * rad_core
      end do

c     debar(0)=0.d0
c     bar(0)=bar_t(1)! bar is better defined in get_*_chemistry, from the EOS
      rrho(0)=rho_t(1)
      emas(0)=0.d0
      phi(0)=phi_t(1)
      pres(0)=pres_t(1)
      dvol(0)=0.d0
      j=0
      do i=1,icore
 500     j=j+1
         if (rad_t(j).lt.rad(i)) goto 500
         delrad=rad_t(j)-rad_t(j-1)
         w1=(rad_t(j)-rad(i))/delrad
         w2=1.d0-w1
c     bar(i) =w1*bar_t(j-1) +w2*bar_t(j)     ! bar is better defined in get_*_chemistry, from the EOS
         rrho(i)=w1*rho_t(j-1) +w2*rho_t(j)
         emas(i)=w1*emas_t(j-1)+w2*emas_t(j)
         phi(i) =w1*phi_t(j-1) +w2*phi_t(j)
         pres(i)=w1*pres_t(j-1)+w2*pres_t(j)
         if (i.eq.1) then
            dvol(i)=4./3.*pi*rad(i)**3
         else
            dvol(i)=4.*pi*((rad(i-1)+rad(i))/2.)**2*
     x           (rad(i)-rad(i-1)) /
     x           dsqrt(1.-2.92d5*(emas(i-1)+emas(i))/(rad(i-1)+rad(i)))
         end if
c     debar(i)=(bar(i-1)+bar(i))/2. * dvol(i)
         j=j-1
      end do
c     ***   Just to make sure (exact accuracy, not from above interpolations)
      rrho(icore)=rhocore
c     Crust: zoning with "idec" zone per decade in density: ****
      dlogrho=log10(rhocore/rhodrip)
      dlrho=dlogrho/float(idrip-icore)
      do i=icore+1,idrip
         lrho=log10(rhocore)-float(i-icore)*dlrho
         rrho(i)=10.d0**lrho
      end do
c     ***
      dlogrho=log10(rhodrip/rhosurf)
      dlrho=dlogrho/float(isurf-idrip)
      i=idrip
      do i=idrip+1,isurf
         lrho=log10(rhodrip)-float(i-idrip)*dlrho
         rrho(i)=10.d0**lrho
      end do
      j=0
      do i=icore+1,isurf
 600     j=j+1
         if (rho_t(j).gt.rrho(i)) goto 600
         dellrho=log10(rho_t(j-1))-log10(rho_t(j))
         w1=(log10(rrho(i))-log10(rho_t(j)))/dellrho
         w2=1.d0-w1
c     bar(i) =w1*bar_t(j-1) +w2*bar_t(j)   ! bar is better defined in get_*_chemistry, from the EOS
         rad(i) =w1*rad_t(j-1) +w2*rad_t(j)
         emas(i)=w1*emas_t(j-1)+w2*emas_t(j)
         phi(i) =w1*phi_t(j-1) +w2*phi_t(j)
         pres(i)=w1*pres_t(j-1)+w2*pres_t(j)
         dvol(i)=4.*pi*((rad(i-1)+rad(i))/2.)**2*
     x        (rad(i)-rad(i-1)) /
     x        dsqrt(1.-2.92d5*(emas(i-1)+emas(i))/(rad(i-1)+rad(i)))
c     debar(i)=(bar(i-1)+bar(i))/2. * dvol(i)
         j=j-1
      end do
      dvol(isurf+1)=dvol(isurf)
c     ***   Just to make sure (exact accuracy, not from above interpolations)
      rrho(idrip)=rhodrip
      rrho(isurf)=rhosurf
c     Find the envelope boundary: ******************************
      ienv=isurf+2
      do i=isurf,idrip,-2
         if (rrho(i).lt.rhoenv) ienv=i
      end do
c     ***
      imax=isurf
      return
      end

c     *********************************************************************
c     *********************************************************************
      subroutine get_core_chemistry(irank,version,imax,icore,rrho,
     1     bar,yneutr,yprot,yelect,ymuon,ylambda,ysminus,yszero,
     2     ysplus,yquarku,yquarkd,yquarks,theta_k,theta_p,fhad,
     3     mstn,mstp,mstla,mstsm,msts0,mstsp)
c     *********************************************************************
c     This subroutine calculates the concentrations Y's of 
c     all particles in the core using the EOS table.
c     *********************************************************************
      implicit real*8 (a-h,k-z)
      parameter (isize=10000)
      dimension rrho(0:isize)

      dimension bar(0:isize),
     1     yneutr(0:isize),yprot(0:isize),
     2     yelect(0:isize),ymuon(0:isize),
     3     ylambda(0:isize),
     4     ysminus(0:isize),yszero(0:isize),ysplus(0:isize),
     5     yquarku(0:isize),yquarkd(0:isize),yquarks(0:isize),
     6     fhad(0:isize),
     7     theta_k(0:isize),theta_p(0:isize)
     
      dimension mstp(0:isize),mstn(0:isize),mstla(0:isize),
     2     mstsm(0:isize),msts0(0:isize),mstsp(0:isize)
      dimension rho_t(500),nbar_t(500)
      dimension yneutr_t(500),yprot_t(500),
     2     yelect_t(500),ymuon_t(500),
     3     ylambda_t(500),
     4     ysminus_t(500),yszero_t(500),ysplus_t(500),
     5     yquarku_t(500),yquarkd_t(500),yquarks_t(500),
     6     fhad_t(500)
      dimension mstp_t(0:isize),mstn_t(0:isize),mstla_t(0:isize),
     2     mstsm_t(0:isize),msts0_t(0:isize),mstsp_t(0:isize)
      dimension theta_k_t(500),theta_p_t(500)
      character*3 version

      it=99999
      i0=99999
      
      call nscool_core_comp(irank,rho_t,nbar_t,yelect_t,ymuon_t,
     1     yneutr_t,yprot_t,ylambda_t,ysminus_t,yszero_t,ysplus_t,
     2     mstp_t,mstn_t,mstla_t,mstsm_t,msts0_t,mstsp_t,ix)

      if (ix.eq.0) then
         icore=0
         return
      end if
         
      do i=1,ix
         theta_k_t(i)=0.0d0
         theta_p_t(i)=0.0d0
         yquarku_t(i)=0.0d0
         yquarkd_t(i)=0.0d0
         yquarks_t(i)=0.0d0
         ylambda_t(i)=0.0d+00
         ysminus_t(i)=0.0d+00
         yszero_t(i)=0.0d+00
         ysplus_t(i)=0.0d+00
         mstla_t(i)=0.0d+00
         mstsm_t(i)=0.0d+00
         msts0_t(i)=0.0d+00
         mstsp_t(i)=0.0d+00
      end do
      xnut=9.2819d+32

c     ***** Interpolate the particle concentrations: ********
      i1=1
      do i0=0,icore
         if(rrho(i0).ge.rho_t(1))then
            i1=1
            i2=2
         else if(rrho(i0).le.rho_t(ix))then
            i1=ix-1
            i2=ix
         else
            i=i1-1
 5768       i=i+1
            if((rrho(i0).ge.rho_t(i+1)).and.(rrho(i0).le.rho_t(i)))then
               i1=i
               i2=i+1
            else
               goto 5768
            end if
         end if
         x1=(dlog(rho_t(i2))-dlog(rrho(i0)))/
     1        (dlog(rho_t(i2))-dlog(rho_t(i1)))
         x2=(dlog(rrho(i0))-dlog(rho_t(i1)))/
     1        (dlog(rho_t(i2))-dlog(rho_t(i1)))
c     print '(1p3e12.3,5x,0p2f10.5,2i5)',
c     1       rho_t(i1),rrho(i0),rho_t(i2),x1,x2,i1,i2
         bar(i0)    =x1*nbar_t(i1)   +x2*nbar_t(i2)
         yelect(i0) =x1*yelect_t(i1) +x2*yelect_t(i2)
         ymuon(i0)  =x1*ymuon_t(i1)  +x2*ymuon_t(i2)
         yneutr(i0) =x1*yneutr_t(i1) +x2*yneutr_t(i2)
         yprot(i0)  =x1*yprot_t(i1)  +x2*yprot_t(i2)
         ylambda(i0)=x1*ylambda_t(i1)+x2*ylambda_t(i2)
         ysminus(i0)=x1*ysminus_t(i1)+x2*ysminus_t(i2)
         yszero(i0) =x1*yszero_t(i1) +x2*yszero_t(i2)
         ysplus(i0) =x1*ysplus_t(i1) +x2*ysplus_t(i2)
         yquarku(i0)=x1*yquarku_t(i1)+x2*yquarku_t(i2)
         yquarkd(i0)=x1*yquarkd_t(i1)+x2*yquarkd_t(i2)
         yquarks(i0)=x1*yquarks_t(i1)+x2*yquarks_t(i2)
         theta_k(i0)=x1*theta_k_t(i1)+x2*theta_k_t(i2)
         theta_p(i0)=x1*theta_p_t(i1)+x2*theta_p_t(i2)
         if ((version.eq.'old').or.(version.eq.'new').or.
     1        (version.eq.'NEW')) then
            fhad(i0)=1.d0
         else if (version.eq.'QRK') then
            fhad(i0)=   x1*fhad_t(i1)   +x2*fhad_t(i2)
            if (fhad(i0).gt.1.0) fhad(i0)=1.d0
            if (fhad(i0).lt.0.0) fhad(i0)=0.d0
         end if
c     NOTE: the Y's are defined such that 
c     Y_i*bar=number of particle of type i per fm^3
c     but the density of baryons in the baryon phase is then
c     Y_i*bar/fhad 
c     since they occupy only a fraction fhad of the volume
c     and for quarks it is
c     Y_q*bar/(1-fhad)
c     *** Check for consistency:
         bnuc=yneutr(i0)+yprot(i0)
         bhyp=ylambda(i0)+ysminus(i0)+yszero(i0)+ysplus(i0) 
         bqua=1./3.*(yquarku(i0)+yquarkd(i0)+yquarks(i0))
         btot=bnuc+bhyp+bqua
         qlep=-yelect(i0)-ymuon(i0)
         qnuc=yprot(i0)
         qhyp=ysplus(i0)-ysminus(i0)
         qqua=1./3.*(2.*yquarku(i0)-yquarkd(i0)-yquarks(i0))
         qtot=qlep+qnuc+qhyp+qqua
         if (abs(btot-1.0).gt.1.e-2) then
            print '(a30,i5,1p1e12.3,5x,0p2f10.5)',
     1           'i, rho, Btot, Qtot = ',
     2           i0,rrho(i0),btot,qtot
            print *,'Btot not equal to 1 !',i0,yneutr(i0),yprot(i0)
            print *,ylambda(i0),ysminus(i0),yszero(i0),ysplus(i0)
            print *,bnuc,bhyp,bqua,btot
            icore=0
            return
         end if
         if (abs(qtot).gt.1.e-2) then
            print '(a30,i5,1p1e12.3,5x,0p2f10.5)',
     1           'i, rho, Btot, Qtot = ',
     2           i0,rrho(i0),btot,qtot
            print *,'Qtot not equal to 0 !',i0,qlep,qnuc,qhyp,qqua,qtot
            print *,yneutr(i0),yprot(i0),yelect(i0),ymuon(i0)
            icore=0
            return
         end if
c     Get the baryon effective masses if NEW:
         if (version.eq.'NEW') then
            mstp(i0) =x1*mstp_t(i1) +x2*mstp_t(i2)
            mstn(i0) =x1*mstn_t(i1) +x2*mstn_t(i2)
            mstla(i0)=x1*mstla_t(i1)+x2*mstla_t(i2)
            mstsm(i0)=x1*mstsm_t(i1)+x2*mstsm_t(i2)
            msts0(i0)=x1*msts0_t(i1)+x2*msts0_t(i2)
            mstsp(i0)=x1*mstsp_t(i1)+x2*mstsp_t(i2)
         end if
      end do
c     *** Clean up Y's in the crust, just in case:
c     (yelect & yneutr will be calculated in "get_crust_chemistry")
      do i0=icore+1,imax
         yelect(i0) =0.d0
         ymuon(i0)  =0.d0
         yneutr(i0) =0.d0
         yprot(i0)  =0.d0
         ylambda(i0)=0.d0
         ysminus(i0)=0.d0
         yszero(i0) =0.d0
         ysplus(i0) =0.d0
         yquarku(i0)=0.d0
         yquarkd(i0)=0.d0
         yquarks(i0)=0.d0
         theta_k(i0)=0.d0
         theta_p(i0)=0.d0
         fhad(i0)=1.d0          ! Also just in case
      end do
c     *****
      return
      end
c     *************************************************************************
c     *********************************************************************
      subroutine get_crust_chemistry(irank,debug,version,imax,icore,
     1     rrho,pres,debar,dvol,bar,a_cell,a_ion,z_ion,v_ion,
     2     yelect,yneutr)
c     *********************************************************************
c     This subroutine calculates the chemical composition, A, A' & Z
c     and the e & n Y's in the crust using the crust_cc table.
c     *********************************************************************
      implicit real*8 (a-h,k-z)
      parameter (isize=10000)

      dimension rrho(0:isize),pres(0:isize),
     1     debar(0:isize),dvol(0:isize)
      dimension bar(0:isize),
     1     a_cell(0:isize),a_ion(0:isize),z_ion(0:isize),
     2     v_ion(0:isize),yelect(0:isize),yneutr(0:isize)
      
      dimension rho_t(500),pres_t(500),bar_t(500),
     1     A_cell_t(500),A_ion_t(500),Z_ion_t(500)
      character*3 version

      if (debug.ge.1.) then
         print *,'Entering subroutine get_crust_chemistry'
      end if
      
      call nscool_crust_comp(irank,Z_ion_t,A_ion_t,A_cell_t,
     1     bar_t,pres_t,rho_t,jmax)
      
      jget_drip=0
      do j=jmax,1,-1
         if (jget_drip.eq.0) then
            if (A_cell_t(j).ne.A_ion_t(j)) then
               jdrip=j
               jget_drip=1
            end if
         end if
      end do

c     **************************************************************
c     Make sure that rho_t and bar_t at jmax are smaller than in core:
      jjmax=jmax
      do j=jmax,1,-1
         if (rho_t(j).ge.rrho(icore)) then
            jjmax=j-1
         end if
      end do
      jmax=jjmax
c     **************************************************************
c     This is for interpolation from the last crust_EOS line
c     up to the core, in a, hopefully, consistent way:
c     ***************
c     First version:
c     - Take rho & bar from core EOS:
      jmax=jmax+1
      rho_t(jmax) =rrho(icore)
      bar_t(jmax) =bar(icore)
      pres_t(jmax)=pres(icore)
c     - Keep Z constant:
      Z_ion_t(jmax)=Z_ion_t(jmax-1)
c     - Take A_ion constant: results in that the driped neutron
c     fraction grows and kfn only has a small discontinuity
c     (this is the same recipe as below for lower densities)
      A_ion_t(jmax) =A_ion_t(jmax-1)
c     - Take Z/A_cell consistent with Ye in core_EOS:
      A_cell_t(jmax)=Z_ion_t(jmax) / yelect(icore)
c     ***************
c     Second version:
c     
c     To be done !
c     
c     **************************************************************
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     do j=jmax,1,-1
c     print '(i5,1p3e12.3,5x,0p3f12.1)',
c     x          j,rho_t(j),pres_t(j),bar_t(j),
c     x          A_cell_t(j),A_ion_t(j),Z_ion_t(j)
c     end do
c     read(5,*)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      j=jmax
      do i=icore+1,imax
 100     j=j-1
         if (rrho(i).le.rho_t(j)) goto 100
         dd=(rho_t(j+1)-rho_t(j))
         w2=(rrho(i)-rho_t(j))/dd
         w1=1.d0-w2
         if (A_ion_t(j+1).eq.A_cell_t(j+1)) then
            A_ion(i) =A_ion_t(j+1)
            A_cell(i)=A_cell_t(j+1)
            Z_ion(i) =Z_ion_t(j+1)
         else
c     This is the old version:
c     A_cell(i)=w1*A_cell_t(j)+w2*A_cell_t(j+1)
c     A_ion(i) =w1*A_ion_t(j) +w2*A_ion_t(j+1)
c     Z_ion(i) =Z_ion_t(j+1)
c     New version: A & Z constant, only A_cell changes:
            A_cell(i)=w1*A_cell_t(j)+w2*A_cell_t(j+1)
            A_ion(i) =A_ion_t(j+1)
            Z_ion(i) =Z_ion_t(j+1)
         end if
c     ****  Get the baryon number density:
         bar(i)=w1*bar_t(j)+w2*bar_t(j+1)
c     ****  Calculate the fraction of volume occupied by ions:
         r1=1.1d0               ! Scale parameter, in fm
         vion=4.d0/3.d0*3.14159d0 * r1**3 * a_ion(i) ! vion in fm^3
         vion=1.d-39 * vion     ! in cm^3
         nion=rrho(i)/(1.66d-24*a_ion(i)) ! ion density per cm^3
         v_ion(i)=nion*vion
         v_ion(i)=min(1.d0,v_ion(i)) ! Make sure it's < 1
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     print *
c     print '(i5,1p1e12.3,0p1f17.12,0p3f10.1)',
c     1       j+1,rho_t(j+1),bar_t(j+1),
c     2       a_cell_t(j+1),a_ion_t(j+1),z_ion_t(j+1)
c     print '(i5,1p1e12.3,0p1f17.12,0p3f10.1)',
c     1       i ,rrho( i ),bar(  i ),
c     2        a_cell(  i ),a_ion(  i ),z_ion(  i )
c     print '(i5,1p1e12.3,0p1f17.12,0p3f10.1)',
c     1       j ,rho_t( j ),bar_t( j ),
c     2        a_cell_t( j ),a_ion_t( j ),z_ion_t( j )
c     read(5,*)
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
         j=j+1
      end do
c$$$  c Find the neutron drip point :
c$$$  c HERE DANY: idrip is now defined in subroutine grid
c$$$  do i=imax,icore+2,-2
c$$$  if (A_ion(i).eq.A_cell(i)) idrip=i-2
c$$$  end do
c$$$  c       Following is just in case strange stars are considered:
c$$$  c       and garantees that neutron drip is in the crust !
c$$$  if (idrip.eq.icore) idrip=icore+2
c$$$  rhodrip=rrho(idrip)
c     Calculate the Y's of the e & n:
      do i=icore+1,imax
         yelect(i) =Z_ion(i)/A_cell(i)
         yneutr(i) =(A_cell(i)-A_ion(i))/A_cell(i)
      end do
c     Clean up the core, just in case:
      do i=0,icore
         Z_ion(i) =0.d0
         A_ion(i) =0.d0
         A_cell(i)=0.d0
         v_ion(i) =0.d0
      end do
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     do i=icore+1,imax,2
c     if (i.eq.idrip) print*,'+++++++++++++'
c     print '(i5,1p3e12.3,3x,0p3f12.1,3x,1p3e12.3)',
c     x         i,rrho(i),pres(i),bar(i),
c     x         A_cell(i),A_ion(i),Z_ion(i),v_ion(i),yelect(i),yneutr(i)
c     if (i.eq.idrip) print*,'+++++++++++++'
c     end do
c     read(5,*)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

c     *********************************************************************
      debar(0)=0.d0
      do i=1,imax
         debar(i)=(bar(i-1)+bar(i))/2. * dvol(i)
      end do
c     *********************************************************************

c     *****
      if (debug.ge.1.) then
         print *,'Exiting subroutine get_crust_chemistry'
      end if
c     *****
      return
      end
c     *********************************************************************

c     *********************************************************************
      subroutine get_fermi_momenta(irank,imax,icore,rrho,bar,yneutr,
     1     yprot,yelect,ymuon,ylambda,ysminus,yszero,ysplus,yquarku,
     2     yquarkd,yquarks,fhad,theta_k,theta_p,
     3     kfn,kfp,kfe,kfm,kfla,kfsm,kfs0,kfsp,kfqu,kfqd,kfqs,
     4     idurca_np,idurca_lap,durca_ctrl_e,durca_ctrl_m,
     5     idurca_smn,idurca_smla,idurca_sms0,idurca_quqd,idurca_quqs,
     6     durca_henon_e,durca_henon_m)
      
c     *********************************************************************
c     This subroutine calculates the Fermi momentum of all fermions
c     *********************************************************************
      implicit real*8 (a-h,k-z)
      parameter (isize=10000)

      dimension rrho(0:isize)

      dimension bar(0:isize),
     1     yneutr(0:isize),yprot(0:isize),
     2     yelect(0:isize),ymuon(0:isize),
     3     ylambda(0:isize),
     4     ysminus(0:isize),yszero(0:isize),ysplus(0:isize),
     5     yquarku(0:isize),yquarkd(0:isize),yquarks(0:isize),
     6     fhad(0:isize),
     7     theta_k(0:isize),theta_p(0:isize)
      dimension kfe(0:isize),kfm(0:isize),kfp(0:isize),kfn(0:isize),
     2     kfla(0:isize),kfsm(0:isize),kfs0(0:isize),kfsp(0:isize),
     3     kfqu(0:isize),kfqd(0:isize),kfqs(0:isize)
      dimension idurca_np(0:isize),idurca_lap(0:isize),
     2     idurca_smn(0:isize),idurca_smla(0:isize),
     3     idurca_sms0(0:isize),
     4     idurca_quqd(0:isize),idurca_quqs(0:isize),
     5     durca_ctrl_e(0:isize),durca_ctrl_m(0:isize),
     6     durca_henon_e(0:isize),durca_henon_m(0:isize)
      
      
      parameter(pi=3.14159265d0)
c     ****  Calculate the fermi momenta in the core:
      do j=0,icore
c----------------------------------------------
         if (fhad(j).ne.0.d0) then
            nn =max(0.e0,yneutr(j) *bar(j)) / fhad(j)
            np =max(0.e0,yprot(j)  *bar(j)) / fhad(j)
            nla=max(0.e0,ylambda(j)*bar(j)) / fhad(j)
            nsm=max(0.e0,ysminus(j)*bar(j)) / fhad(j)
            ns0=max(0.e0,yszero(j) *bar(j)) / fhad(j)
            nsp=max(0.e0,ysplus(j) *bar(j)) / fhad(j)
         else
            nn =0.d0
            np =0.d0
            nla=0.d0
            nsm=0.d0
            ns0=0.d0
            nsp=0.d0
         end if
         if (fhad(j).ne.1.d0) then
            nqu=max(0.e0,yquarku(j)*bar(j)) / (1.d0-fhad(j))
            nqd=max(0.e0,yquarkd(j)*bar(j)) / (1.d0-fhad(j))
            nqs=max(0.e0,yquarks(j)*bar(j)) / (1.d0-fhad(j))
         else
            nqu=0.d0
            nqd=0.d0
            nqs=0.d0
         end if
         kfn(j) =(3.d0*pi**2*nn )**(1.d0/3.d0)
         kfp(j) =(3.d0*pi**2*np )**(1.d0/3.d0)
         kfla(j)=(3.d0*pi**2*nla)**(1.d0/3.d0)
         kfsm(j)=(3.d0*pi**2*nsm)**(1.d0/3.d0)
         kfs0(j)=(3.d0*pi**2*ns0)**(1.d0/3.d0)
         kfsp(j)=(3.d0*pi**2*nsp)**(1.d0/3.d0)
         kfqu(j)=(3.d0*pi**2*nqu)**(1.d0/3.d0)
         kfqd(j)=(3.d0*pi**2*nqd)**(1.d0/3.d0)
         kfqs(j)=(3.d0*pi**2*nqs)**(1.d0/3.d0)
c----------------------------------------------
         ne=abs(yelect(j)*bar(j))
         nm=abs( ymuon(j)*bar(j))
         kfe(j)=(3.d0*pi**2*ne)**(1.d0/3.d0)
         if (yelect(j).le.0.d0) kfe(j)=-kfe(j)
         kfm(j)=(3.d0*pi**2*nm)**(1.d0/3.d0)
         if ( ymuon(j).le.0.d0) kfm(j)=-kfm(j)
c----------------------------------------------
c     Check all this stuff:
         coeff=3.d0*pi**2
         nn = kfn(j)**3/coeff * fhad(j)
         np = kfp(j)**3/coeff * fhad(j)
         nla=kfla(j)**3/coeff * fhad(j)
         nsm=kfsm(j)**3/coeff * fhad(j)
         ns0=kfs0(j)**3/coeff * fhad(j)
         nsp=kfsp(j)**3/coeff * fhad(j)
         nqu=kfqu(j)**3/coeff * (1.d0-fhad(j))
         nqd=kfqd(j)**3/coeff * (1.d0-fhad(j))
         nqs=kfqs(j)**3/coeff * (1.d0-fhad(j))
         ne = kfe(j)**3/coeff * 1.d0
         nm = kfm(j)**3/coeff * 1.d0
         charge_l=-ne-nm
         charge_h=(np+nsp-nsm)
         charge_q=(2./3.*nqu-1./3.*nqd-1./3.*nqs)
         charge=(charge_l+charge_h+charge_q)
         baryon_h=(nn+np+nla+nsm+ns0+nsp)
         baryon_q=1./3.*(nqu+nqd+nqs)
         baryon=(baryon_h+baryon_q)
         if (abs(charge).ge.1.d-2) then
            print *,'Charge neutrality violated at:'
            print '(i5,a6,1p1e12.3,a18,0p1f12.5)',
     1           j,'Rho= ',rrho(j),': charge/fm3= ',charge
            icore=0
            return
         end if
         barrel=baryon/bar(j)
         if ((abs(barrel)-1.d0).ge.1.d-2) then
            print *,'Baryons do not sum up to baryon density at:'
            print '(i5,a6,1p1e12.3,a26,0p1f12.5)',
     1           j,'Rho= ',rrho(j),': sum(baryons)/baryon#= ',baryon
            icore=0
            return
         end if
      end do

c     ------------------------------------------------------
c     Calculate the fermi momenta in the crust:
      
      do j=icore+1,imax
         ne=yelect(j)*bar(j)
         nn=yneutr(j)*bar(j)
         kfe(j) =(3.d0*pi**2*ne)**(1.d0/3.d0)
         kfm(j) =0.d0
         kfn(j) =(3.d0*pi**2*nn)**(1.d0/3.d0)
         kfp(j) =0.d0
         kfla(j)=0.d0
         kfsm(j)=0.d0
         kfs0(j)=0.d0
         kfsp(j)=0.d0
         kfqu(j)=0.d0
         kfqd(j)=0.d0
         kfqs(j)=0.d0
      end do

c     ------------------------------------------------------
c     Check for direct Urca process

      if (.FALSE.) then

c     ------------------------------------------------------
c     Dany's method for neutron-proton direct Urca
         
         do j=0,icore
            if    ( (kfp(j) .lt.kfn(j) +kfe(j) ).and.
     1           (kfn(j) .lt.kfp(j) +kfe(j) ).and.
     2           (kfe(j) .lt.kfp(j) +kfn(j) )     ) then
               idurca_np(j)=1
               if    ( (kfp(j) .lt.kfn(j) +kfm(j) ).and.
     1              (kfn(j) .lt.kfp(j) +kfm(j) ).and.
     2              (kfm(j) .lt.kfp(j) +kfn(j) )     ) then
                  idurca_np(j)=2
               end if
            else
               idurca_np(j)=0
            end if
         end do
         
      else

c     ------------------------------------------------------
c     New method for neutron-proton direct Urca

         fix_durca=0.d0
         alpha_durca_frac=1.0d-8
         beta_durca_frac=1.d0
         
c     AWS: Get direct Urca settings
c     AWS: alpha is the broadening parameter and beta is the
c     fractional decrease of the direct Urca threshold
         call nscool_urca_settings(irank,fix_durca,alpha_durca_frac,
     1        beta_durca_frac)

c     AWS: Compute triangle squared areas
         do j=0,icore
            s=(kfp(j)+kfn(j)+kfe(j))/2.0
            durca_henon_e(j)=s*(s-kfp(j))*(s-kfe(j))*(s-kfn(j))
            s=(kfp(j)+kfn(j)+kfm(j))/2.0
            durca_henon_m(j)=s*(s-kfp(j))*(s-kfm(j))*(s-kfn(j))
         end do

c     AWS: Use linear interpolation to compute direct Urca
c     density thresholds
         if (fix_durca .eq. 0.d0) then
            nb_durca_e=0.0
            nb_durca_m=0.0
            do j=0,icore-1
               if (nb_durca_e .eq. 0.0) then
                  if (durca_henon_e(j)*durca_henon_e(j+1).lt.0.0) then
                     nb_low=bar(j)
                     nb_high=bar(j+1)
                     nb_durca_e=nb_low-(nb_high-nb_low)*
     1                    durca_henon_e(j)/(durca_henon_e(j+1)-
     2                    durca_henon_e(j))
                  end if
               end if
               if (nb_durca_m .eq. 0.0) then
                  if (durca_henon_m(j)*durca_henon_m(j+1).lt.0.0) then
                     nb_low=bar(j)
                     nb_high=bar(j+1)
                     nb_durca_m=nb_low-(nb_high-nb_low)*
     1                    durca_henon_m(j)/(durca_henon_m(j+1)-
     2                    durca_henon_m(j))
                  end if
               end if
            end do
         else
            nb_durca_e=fix_durca
            nb_durca_m=fix_durca
         endif

c     AWS: Now compute direct Urca control functions from threshold
c     densities
c     SH: Add two possible modifications
c     early onset (step function); broadening
         do j=0,icore
            idurca_np(j)=0
            durca_ctrl_e(j)=0.0
            durca_ctrl_m(j)=0.0

            if (nb_durca_e .gt. 0.0) then
               if (bar(j) .ge. (1.d0+alpha_durca_frac)*beta_durca_frac*
     1              nb_durca_e) then
                  idurca_np(j)=1
                  durca_ctrl_e(j)=1.0
               else if (bar(j) .ge. (1.d0-alpha_durca_frac)*
     1                 beta_durca_frac*nb_durca_e) then
                  idurca_np(j)=3
                  durca_ctrl_e(j)=1.d0/2.d0+1.d0/
     1                 (2.d0*alpha_durca_frac)*(bar(j)-nb_durca_e)/
     2                 nb_durca_e
                  if (nb_durca_m .gt. 0.0) then
                     if (bar(j) .ge. (1.d0+alpha_durca_frac)*
     1                    beta_durca_frac*nb_durca_m) then
                        idurca_np(j)=2
                        durca_ctrl_m(j)=1.0
                     else if (bar(j) .ge. (1.d0-alpha_durca_frac)*
     1                       beta_durca_frac*nb_durca_m) then
                        idurca_np(j)=4
                        durca_ctrl_m(j)=1.d0/2.d0+1.d0/
     1                       (2.d0*alpha_durca_frac)*
     2                       (bar(j)-nb_durca_m)/nb_durca_m
                     end if
                  end if
               end if
            end if
         end do

      endif
c     ------------------------------------------------------
c     Dany's method for hyperon and quark direct Urca
      
      do j=0,icore
c     la-p:
         if     ( (kfp(j) .lt.kfla(j)+kfe(j) ).and.
     1        (kfla(j).lt. kfp(j)+kfe(j) ).and.
     2        (kfe(j) .lt. kfp(j)+kfla(j))     ) then
            idurca_lap(j)=1
            if    ( (kfp(j) .lt.kfla(j)+kfm(j) ).and.
     1           (kfla(j).lt. kfp(j)+kfm(j) ).and.
     2           (kfm(j) .lt. kfp(j)+kfla(j))     ) then
               idurca_lap(j)=2
            end if
         else
            idurca_lap(j)=0
         end if
c     sm-n:
         if     ( (kfsm(j).lt.kfn(j) +kfe(j) ).and.
     1        (kfn(j) .lt.kfsm(j)+kfe(j) ).and.
     2        (kfe(j) .lt.kfsm(j)+kfn(j) )     ) then
            idurca_smn(j)=1
            if    ( (kfsm(j).lt.kfn(j) +kfm(j) ).and.
     1           (kfn(j) .lt.kfsm(j)+kfm(j) ).and.
     2           (kfm(j) .lt.kfsm(j)+kfn(j) )     ) then
               idurca_smn(j)=2
            end if
         else
            idurca_smn(j)=0
         end if
c     sm-la:
         if     ( (kfsm(j).lt.kfla(j)+kfe(j) ).and.
     1        (kfla(j).lt.kfsm(j)+kfe(j) ).and.
     2        (kfe(j) .lt.kfsm(j)+kfla(j))     ) then
            idurca_smla(j)=1
            if    ( (kfsm(j).lt.kfla(j)+kfm(j) ).and.
     1           (kfla(j).lt.kfsm(j)+kfm(j) ).and.
     2           (kfm(j) .lt.kfsm(j)+kfla(j))    ) then
               idurca_smla(j)=2
            end if
         else
            idurca_smla(j)=0
         end if
c     sm-s0:
         if     ( (kfsm(j).lt.kfs0(j)+kfe(j) ).and.
     1        (kfs0(j).lt.kfsm(j)+kfe(j) ).and.
     2        (kfe(j) .lt.kfsm(j)+kfs0(j))     ) then
            idurca_sms0(j)=1
            if    ( (kfsm(j).lt.kfs0(j)+kfm(j) ).and.
     1           (kfs0(j).lt.kfsm(j)+kfm(j) ).and.
     2           (kfm(j) .lt.kfsm(j)+kfs0(j))     ) then
               idurca_sms0(j)=2
            end if
         else
            idurca_sms0(j)=0
         end if
c     qu-qd:
         if     ( (kfqu(j).lt.kfqd(j)+kfe(j) ).and.
     1        (kfqd(j).lt.kfqu(j)+kfe(j) ).and.
     2        (kfe(j) .lt.kfqu(j)+kfqd(j))     ) then
            idurca_quqd(j)=1
            if    ( (kfqu(j).lt.kfqd(j)+kfm(j) ).and.
     1           (kfqd(j).lt.kfqu(j)+kfm(j) ).and.
     2           (kfm(j) .lt.kfqu(j)+kfqd(j))     ) then
               idurca_quqd(j)=2
            end if
         else
            idurca_quqd(j)=0
         end if
c     qu-qs:
         if     ( (kfqu(j).lt.kfqs(j)+kfe(j) ).and.
     1        (kfqs(j).lt.kfqu(j)+kfe(j) ).and.
     2        (kfe(j) .lt.kfqu(j)+kfqs(j))     ) then
            idurca_quqs(j)=1
            if    ( (kfqu(j).lt.kfqs(j)+kfm(j) ).and.
     1           (kfqs(j).lt.kfqu(j)+kfm(j) ).and.
     2           (kfm(j) .lt.kfqu(j)+kfqs(j))     ) then
               idurca_quqs(j)=2
            end if
         else
            idurca_quqs(j)=0
         end if
      end do     
      return
      end

c     *********************************************************************
c     *********************************************************************
      subroutine get_effective_masses(version,emnco,emncr,emp,
     1     kfn,kfp,mstn,mstp,mstla,mstsm,msts0,mstsp,idrip,icore)

c     *********************************************************************
c     This subroutine calculates the baryon effective masses
c     (in a lousy way) incase thery are not given by the EOS table !
c     *********************************************************************
      implicit real*8 (a-h,k-z)
      parameter (isize=10000)

      dimension mstp(0:isize),mstn(0:isize),mstla(0:isize),
     2     mstsm(0:isize),msts0(0:isize),mstsp(0:isize)
      dimension kfp(0:isize),kfn(0:isize)

      character*3 version

c     Crust neutrons:
      do i=icore+1,idrip
         mstn(i)=min(1.d0,1.09-0.11*kfn(i))
      end do
      
      return
      end
c     *********************************************************************
c     *********************************************************************
c     *********************************************************************
      subroutine get_spec_heat_degenerate(cve,cvm,cvn,cvp,cvla,
     1     cvsm,cvs0,cvsp,cvqu,dvqd,cvqs,
     2     kfe,kfm,kfn,kfp,kfla,kfsm,kfs0,kfsp,kfqu,kfqd,kfqs,
     3     mstn,mstp,mstla,mstsm,msts0,mstsp,fhad,imax)
c     *********************************************************************
c     This subroutine calculates Cv/T for degenerate particles
c     *********************************************************************
      implicit real*8 (a-h,k-z)
      parameter(pi=3.14159265)
      parameter (isize=10000)
      dimension fhad(0:isize)
      dimension kfe(0:isize),kfm(0:isize),kfp(0:isize),kfn(0:isize),
     2     kfla(0:isize),kfsm(0:isize),kfs0(0:isize),kfsp(0:isize),
     3     kfqu(0:isize),kfqd(0:isize),kfqs(0:isize)
      dimension mstp(0:isize),mstn(0:isize),mstla(0:isize),
     2     mstsm(0:isize),msts0(0:isize),mstsp(0:isize)
      dimension cve(0:isize),cvm(0:isize),cvn(0:isize),cvp(0:isize),
     1     cvla(0:isize),cvsm(0:isize),cvs0(0:isize),cvsp(0:isize),
     2     cvqu(0:isize),cvqd(0:isize),cvqs(0:isize)
      
      do j=0,imax
         pfe=kfe(j)*197.d0
         me =sqrt(0.511d0**2+pfe**2) ! electron effective mass
         pfm=kfm(j)*197.d0
         mm =sqrt(105.d0**2+pfm**2) ! muon effective mass
         pfn=kfn(j)*197.d0
         mn =939.56d0*mstn(j)
         pfp=kfp(j)*197.d0
         mp =938.27d0*mstp(j)
         pfla=kfla(j)*197.d0
         mla=1116.0d0*mstla(j)
         pfsm=kfsm(j)*197.d0
         msm=1193.0d0*mstsm(j)
         pfs0=kfs0(j)*197.d0
         ms0=1193.0d0*msts0(j)
         pfsp=kfsp(j)*197.d0
         msp=1193.0d0*mstsp(j)
         pfqu=kfqu(j)*197.d0
         mqu=sqrt(5.d0**2+pfqu**2) ! up effective mass
         pfqd=kfqd(j)*197.d0
         mqd=sqrt(8.d0**2+pfqd**2) ! down effective mass
         pfqs=kfqs(j)*197.d0
         mqs=sqrt(strange_mass**2+pfqs**2) ! strange effective mass
         cve(j) =cvt_deg(pfe,me)
         cvm(j) =cvt_deg(pfm,mm)
         cvn(j) =cvt_deg(pfn,mn)   * fhad(j)
         cvp(j) =cvt_deg(pfp,mp)   * fhad(j)
         cvla(j)=cvt_deg(pfla,mla) * fhad(j)
         cvsm(j)=cvt_deg(pfsm,msm) * fhad(j)
         cvs0(j)=cvt_deg(pfs0,ms0) * fhad(j)
         cvsp(j)=cvt_deg(pfsp,msp) * fhad(j)
         cvqu(j)=cvt_deg(pfqu,mqu) * (1.d0-fhad(j))
         cvqd(j)=cvt_deg(pfqd,mqd) * (1.d0-fhad(j))
         cvqs(j)=cvt_deg(pfqs,mqs) * (1.d0-fhad(j))
      end do   
      return
      end
c     *********************************************************************
c     *********************************************************************
c     *********************************************************************
      subroutine get_Tc(irank,imax,icore,idrip,
     1     tcn,tcp,tcla,tcsm,tcs0,tcsp,
     2     tcuu,tcdd,tcss,tcud,tcus,tcds,
     3     tcu,tcd,tcs,
     4     sfn1s0,sfn3p2,sfp1s0,sfl1s0,
     5     fn1s0,fn3p2,fp1s0,fl1s0,
     6     kfmax_n3p2,delkf_n3p2,tcmax_n3p2,isf,
     7     kfn,kfp,kfla,kfqu,kfqd,kfqs,bar,fhad,yquarku,yquarkd,yquarks)
c     *********************************************************************
c     This subroutine calculate Tc for all baryons and quarks
c     *********************************************************************
      implicit real*8 (a-h,k-z)
      
      parameter (isize=10000)

      dimension bar(0:isize),fhad(0:isize),yquarku(0:isize),
     2     yquarkd(0:isize),yquarks(0:isize)
      
      dimension kfp(0:isize),kfn(0:isize),
     2     kfla(0:isize),kfqu(0:isize),kfqd(0:isize),kfqs(0:isize)
      
      dimension tcn(0:isize),tcp(0:isize),tcla(0:isize),
     2     tcsm(0:isize),tcs0(0:isize),tcsp(0:isize),
     3     tcuu(0:isize),tcdd(0:isize),tcss(0:isize),
     4     tcud(0:isize),tcus(0:isize),tcds(0:isize),
     5     tcu(0:isize),tcd(0:isize),tcs(0:isize)

      dimension tcrit(10),rho_lo(10),rho_hi(10)

c     AWS: determine gap parameters
      call nscool_gaps(irank,sfn1s0,dinput_n1tc,dinput_n1kf,
     1     dinput_n1dk,sfn3p2,dinput_n3tc,dinput_n3kf,
     2     dinput_n3dk,sfp1s0,dinput_p1tc,dinput_p1kf,dinput_p1dk)
      
c     Just to be safe:
      do i=0,imax
         tcn(i) =1.0d0
         tcp(i) =1.0d0
         tcla(i)=1.0d0
         tcuu(i)=1.0d0
         tcdd(i)=1.0d0
         tcss(i)=1.0d0
         tcud(i)=1.0d0
         tcus(i)=1.0d0
         tcds(i)=1.0d0
         tcu(i) =1.0d0
         tcd(i) =1.0d0
         tcs(i) =1.0d0
      end do
c     ***** 1s0 neutron superfluidity *************************************
      if (sfn1s0.eq.1.) then
         do i=0,idrip
c            call cpp_tcn1_sfb(kfn(i),tcn(i))
c            tcn(i)=max(1.d0,tcn(i))*fn1s0
            
            tcn(i)=max(1.d0,tcn1_sfb(kfn(i)))*fn1s0
c            if (tcn1_sfb(kfn(i)).gt.0.d0) then
c               print *,tcn1_sfb(kfn(i))
c               call cpp_tcn1_sfb(kfn(i),tcn(i))
c               print *,tcn(i)
c               stop
c            end if

         end do
      else if (sfn1s0.eq.2.) then
         do i=0,idrip
            tcn(i)=max(1.d0,tcn1_ccdk(kfn(i)))*fn1s0
         end do
      else if (sfn1s0.eq.3.) then
         do i=0,idrip
            tcn(i)=max(1.d0,tcn1_wap(kfn(i)))*fn1s0
         end do
      else if (sfn1s0.eq.4.) then
         do i=0,idrip
            tcn(i)=max(1.d0,tcn1_gc(kfn(i)))*fn1s0
         end do
      else if (sfn1s0.eq.5.) then
         do i=0,idrip
            tcn(i)=max(1.d0,tcn1_gipsf(kfn(i)))*fn1s0
         end do
c     Ioffe gaps:
      else if (sfn1s0.eq.201.) then
         do i=0,idrip
            tcn(i)=max(1.d0,Tc_Ioffe_1ns(kfn(i)))*fn1s0
         end do
      else if (sfn1s0.eq.202.) then
         do i=0,idrip
            tcn(i)=max(1.d0,Tc_Ioffe_2ns(kfn(i)))*fn1s0
         end do
      else if (sfn1s0.eq.203.) then
         do i=0,idrip
            tcn(i)=max(1.d0,Tc_Ioffe_3ns(kfn(i)))*fn1s0
         end do
      else if (sfn1s0.eq.150.) then
         tcmax_n1s0=dinput_n1tc
         kfmax_n1s0=dinput_n1kf
         delkf_n1s0=dinput_n1dk
         do i=0,idrip
            temp=tcmax_n1s0*
     1           exp(-(kfn(i)-kfmax_n1s0)**2/delkf_n1s0**2)*fn1s0
            if (temp.ge.tcn(i)) then
               tcn(i)=temp
               if (isf.eq.i-1) isf=i
            end if
         end do
      end if
c     ***** 3p2 neutron superfluidity *************************************
      isf=-1
c     isf will be the largest zone number at which triplet pairing is present
      if(sfn3p2.eq.1.)then
         do i=0,idrip
            temp=tcn3_hgrr(kfn(i))*fn3p2
            if (temp.ge.tcn(i)) then
               tcn(i)=temp
               isf=i
            end if
         end do
      else if(sfn3p2.eq.2.)then
         do i=0,idrip
            temp=tcn3_ao(kfn(i))*fn3p2
            if (temp.ge.tcn(i)) then
               tcn(i)=temp
               isf=i
            end if
         end do
      else if(sfn3p2.eq.3.)then
         do i=0,idrip
            temp=tcn3_ao_m1(kfn(i))*fn3p2
            if (temp.ge.tcn(i)) then
               tcn(i)=temp
               isf=i
            end if
         end do
      else if(sfn3p2.eq.4.)then
         do i=0,idrip
            temp=tcn3_t72(kfn(i))*fn3p2
            if (temp.ge.tcn(i)) then
               tcn(i)=temp
               isf=i
            end if
         end do
      else if(sfn3p2.eq.5.)then
         temp=tcn3_t72_m1(kfn(i))*fn3p2
         do i=0,idrip
            if (temp.ge.tcn(i)) then
               tcn(i)=temp
               isf=i
            end if
         end do
      else if(sfn3p2.eq.6.)then
         do i=0,idrip
            temp=tcn3_bcll92(kfn(i))*fn3p2
            if (temp.ge.tcn(i)) then
               tcn(i)=temp
               isf=i
            end if
         end do
      else if(sfn3p2.eq.7.)then
         do i=0,idrip
            temp=tcn3_eehjo96_nr(kfn(i))*fn3p2
            if (temp.ge.tcn(i)) then
               tcn(i)=temp
               isf=i
            end if
         end do
      else if(sfn3p2.eq.8.)then
         do i=0,idrip
            temp=tcn3_eehjo96_r(kfn(i))*fn3p2
            if (temp.ge.tcn(i)) then
               tcn(i)=temp
               isf=i
            end if
         end do
c     Minimal Cooling paper gaps:
      else if((sfn3p2.ge.100.).and.(sfn3p2.lt.200.)) then
         if (sfn3p2.eq.101.) then ! Gap "a"
            kfmax_n3p2=1.8d0
            delkf_n3p2=0.5d0
            tcmax_n3p2=1.0d9
         else if(sfn3p2.eq.102.) then ! Gap "b"
            kfmax_n3p2=2.0d0
            delkf_n3p2=0.5d0
            tcmax_n3p2=3.0d9
         else if(sfn3p2.eq.103.) then ! Gap "c"
            kfmax_n3p2=2.5d0
            delkf_n3p2=0.7d0
            tcmax_n3p2=1.0d10
c     AWS: New gap parameters from NSCool parameters
         else if(sfn3p2.eq.150.) then
            tcmax_n3p2=dinput_n3tc
            kfmax_n3p2=dinput_n3kf
            delkf_n3p2=dinput_n3dk
         end if
         do i=0,idrip
            temp=tcmax_n3p2*
     1           exp(-(kfn(i)-kfmax_n3p2)**2/delkf_n3p2**2)*fn3p2
            if (temp.ge.tcn(i)) then
               tcn(i)=temp
               if (isf.eq.i-1) isf=i
            end if
         end do
c     Ioffe gaps:
      else if (sfn3p2.eq.201.) then
         do i=0,idrip
            temp=max(1.d0,Tc_Ioffe_1nt(kfn(i)))*fn3p2
            if (temp.ge.tcn(i)) then
               tcn(i)=temp
               isf=i
            end if
         end do
      else if (sfn3p2.eq.202.) then
         do i=0,idrip
            temp=max(1.d0,Tc_Ioffe_2nt(kfn(i)))*fn3p2
            if (temp.ge.tcn(i)) then
               tcn(i)=temp
               isf=i
            end if
         end do
      else if (sfn3p2.eq.203.) then
         do i=0,idrip
            temp=max(1.d0,Tc_Ioffe_3nt(kfn(i)))*fn3p2
            if (temp.ge.tcn(i)) then
               tcn(i)=temp
               isf=i
            end if
         end do
c     Uniform Tc gap:
      else if(sfn3p2.ge.1.e3)then
         do i=0,icore
            tcn(i)=sfn3p2
         end do
         isf=icore
      end if
c     ***** 1s0 proton superconductivity **********************************
      if(sfp1s0.eq.1.)then
         do i=0,icore
            tcp(i)=max(1.d0,tcp1_ccy_ms(kfp(i)))*fp1s0
         end do
      else if(sfp1s0.eq.2.)then
         do i=0,icore
            tcp(i)=max(1.d0,tcp1_ccy_ps(kfp(i)))*fp1s0
         end do
      else if(sfp1s0.eq.3.)then
         do i=0,icore
            tcp(i)=max(1.d0,tcp1_t73(kfp(i)))*fp1s0
         end do
      else if(sfp1s0.eq.4.)then
         do i=0,icore
            tcp(i)=max(1.d0,tcp1_ns(kfp(i)))*fp1s0
         end do
      else if(sfp1s0.eq.5.)then
         do i=0,icore
            tcp(i)=max(1.d0,tcp1_ao(kfp(i)))*fp1s0
         end do
      else if(sfp1s0.eq.6.)then  
         do i=0,icore
            tcp(i)=max(1.d0,tcp1_bcll92(kfp(i)))*fp1s0
         end do
      else if(sfp1s0.eq.7.)then
         do i=0,icore
            tcp(i)=max(1.d0,tcp1_ccdk(kfp(i)))*fp1s0
         end do
c     Using Neutron 1S0 gaps for Protons:
      else if(sfp1s0.eq.21.)then
         do i=0,icore
            tcp(i)=max(1.d0,tcn1_t72(kfp(i)))*fp1s0
         end do
      else if(sfp1s0.eq.22.)then
         do i=0,icore
            tcp(i)=max(1.d0,tcn1_awp_2(kfp(i)))*fp1s0
         end do
      else if(sfp1s0.eq.23.)then
         do i=0,icore
            tcp(i)=max(1.d0,tcn1_awp_3(kfp(i)))*fp1s0
         end do
c     Ioffe gaps:
      else if (sfp1s0.eq.201.) then
         do i=0,icore
            tcp(i)=max(1.d0,Tc_Ioffe_1p(kfp(i)))*fp1s0
         end do
      else if (sfp1s0.eq.202.) then
         do i=0,icore
            tcp(i)=max(1.d0,Tc_Ioffe_2p(kfp(i)))*fp1s0
         end do
      else if (sfp1s0.eq.203.) then
         do i=0,icore
            tcp(i)=max(1.d0,Tc_Ioffe_3p(kfp(i)))*fp1s0
         end do
c     Uniform Tc gap:
      else if(sfp1s0.ge.1.e3)then
         do i=0,icore
            tcp(i)=sfp1s0
         end do
c     AWS: Arbitrary proton gap:
      else if(sfp1s0.eq.150) then
         tcmax_p1s0=dinput_p1tc
         kfmax_p1s0=dinput_p1kf
         delkf_p1s0=dinput_p1dk
         do i=0,icore
            tcp(i)=dinput_p1tc*
     1           exp(-(kfp(i)-dinput_p1kf)**2/dinput_p1dk**2)*fp1s0
         end do
      end if
c     ***** 1s0 Lambda superfluidity **************************************
      if (sfl1s0.eq.1.) then
         do i=0,icore
            tcla(i)=max(1.d0,tcla1_bb(kfla(i),bar(i)))*fl1s0
         end do
      end if 
c     ***** Quark pairing *************************************************
c$$$  open(unit=32,file=f_quark_tc,status='old',err=6616)
c$$$  read(32,*)
c$$$  read(32,*)
c$$$  read(32,*)
c$$$  read(32,*)pf0_uu,dpf_uu,gap_uu
c$$$  read(32,*)pf0_dd,dpf_dd,gap_dd
c$$$  read(32,*)pf0_ss,dpf_ss,gap_ss
c$$$  read(32,*)pf0_ud,dpf_ud,gap_ud
c$$$  read(32,*)pf0_us,dpf_us,gap_us
c$$$  read(32,*)pf0_ds,dpf_ds,gap_ds
c$$$  close(unit=32,status='keep')
      do i=0,icore
c     **** uu:
         pf_uu=kfqu(i)*.197
         gap=dexp(-(pf_uu-pf0_uu)**2/dpf_uu**2)
         tcuu(i)=1.1604e13*gap_uu*gap
         if (yquarku(i).eq.0.) tcuu(i)=0.0
c     **** dd:
         pf_dd=kfqd(i)*.197
         gap=dexp(-(pf_dd-pf0_dd)**2/dpf_dd**2)
         tcdd(i)=1.1604e13*gap_dd*gap
         if (yquarkd(i).eq.0.) tcdd(i)=0.0
c     **** ss:
         pf_ss=kfqs(i)*.197
         gap=dexp(-(pf_ss-pf0_ss)**2/dpf_ss**2)
         tcss(i)=1.1604e13*gap_ss*gap
         if (yquarks(i).eq.0.) tcss(i)=0.0
c     **** ud:
         pf_ud=(kfqu(i)+kfqd(i))/2.*.197
         gap=dexp(-(pf_ud-pf0_ud)**2/dpf_ud**2)
         tcud(i)=1.1604e13*gap_ud*gap
         if ((yquarku(i).eq.0.).or.(yquarkd(i).eq.0.)) tcud(i)=0.0
c     *** us:
         pf_us=(kfqu(i)+kfqs(i))/2.*.197
         gap=dexp(-(pf_us-pf0_us)**2/dpf_us**2)
         tcus(i)=1.1604e13*gap_us*gap
         if ((yquarku(i).eq.0.).or.(yquarks(i).eq.0.)) tcus(i)=0.0
c     *** ds:
         pf_ds=(kfqd(i)+kfqs(i))/2.*.197
         gap=dexp(-(pf_ds-pf0_ds)**2/dpf_ds**2)
         tcds(i)=1.1604e13*gap_ds*gap
         if ((yquarkd(i).eq.0.).or.(yquarks(i).eq.0.)) tcds(i)=0.0
c     *** Take only the maximum Tc:
         tcu(i)=max(tcuu(i),tcud(i),tcus(i))
         tcd(i)=max(tcdd(i),tcud(i),tcds(i))
         tcs(i)=max(tcss(i),tcus(i),tcds(i))
         if (fhad(i).eq.1.0) then
            tcu(i)=1.d0
            tcd(i)=1.d0
            tcs(i)=1.d0
         end if
      end do
      goto 6617
c     Following is in case the file "f_quark_tc" is not found:
 6616 continue
c     print *,'+++++++++++++++++++++++++++++++++++++++++++++++++++++++'
c     print *,'WARNING:'
c     print *,'  File ',f_quark_tc(1:istrlen(f_quark_tc)),
c     x        'not found ==> Quark Tc=0 !'
c     print *,'+++++++++++++++++++++++++++++++++++++++++++++++++++++++'
      do i=0,icore
         tcu(i)=1.d0
         tcd(i)=1.d0
         tcs(i)=1.d0
      end do
 6617 continue
c     *********************************************
c     Just in case the above formulas give Tc<0 instead of Tc=0
c     at the edges where Tc is almost 0
      do i=0,idrip
         tcn(i) =abs( tcn(i))
         tcp(i) =abs( tcp(i))
         tcla(i)=abs(tcla(i))
         tcuu(i)=abs(tcuu(i))
         tcdd(i)=abs(tcdd(i))
         tcss(i)=abs(tcss(i))
         tcud(i)=abs(tcud(i))
         tcus(i)=abs(tcus(i))
         tcds(i)=abs(tcds(i))
         tcu(i) =abs( tcu(i))
         tcd(i) =abs( tcd(i))
         tcs(i) =abs( tcs(i))
      end do
c     *********************************************
c$$$      print *,irank,imax,idrip,icore,sfn1s0,sfn3p2,sfp1s0
c$$$      print *,fn1s0,fn3p2,fp1s0
c$$$      do i=70,75
c$$$         print *,i,tcn(i),tcp(i),kfp(i),tcuu(i),tcdd(i),tcss(i)
c$$$         print *,i,tcud(i),tcus(i),tcds(i),tcu(i),tcd(i),tcs(i)
c$$$      end do
c$$$      call nscool_tc_new(irank,imax,idrip,icore,sfn1s0,dinput_n1tc,
c$$$     1     dinput_n1kf,dinput_n1dk,sfp1s0,dinput_p1tc,dinput_p1kf,
c$$$     2     dinput_p1dk,sfn3p2,dinput_n3tc,dinput_n3kf,
c$$$     3     dinput_n3dk,fn1s0,fp1s0,fn3p2,kfn,kfp,kfla,kfsm,kfs0,kfsp,
c$$$     4     kfu,kfd,kfs,tcn,tcp,tcla,tcuu,tcdd,tcss,tcud,tcus,
c$$$     5     tcds,tcu,tcd,tcs,isf)      
c$$$      print *,irank,imax,idrip,icore,sfn1s0,sfn3p2,sfp1s0
c$$$      print *,fn1s0,fn3p2,fp1s0
c$$$      do i=70,75
c$$$         print *,i,tcn(i),tcp(i),kfp(i),tcuu(i),tcdd(i),tcss(i)
c$$$         print *,i,tcud(i),tcus(i),tcds(i),tcu(i),tcd(i),tcs(i)
c$$$      end do
c$$$      stop
      return
c     *********************************************************************
      end
c     *********************************************************************
c     *********************************************************************
c     *********************************************************************
      subroutine get_degenerate_density(irank,rrho,pres,rhod,imax,ienv)
      Implicit real*8 (a-h,k-z)
      parameter (isize=10000)

      dimension rrho(0:isize),pres(0:isize),rhod(0:isize)
      dimension rho2(0:isize),pres2(0:isize)
      
c     *** Read crust chemical composition:

      itext=99999

      call nscool_crust_eos(irank,rho2,pres2,idata)
      
c     ***
      do i=0,ienv-1
         rhod(i)=rrho(i)
      end do
      do i=imax,ienv,-1
         j=0
 765     j=j+1
         if ((pres(i).ge.pres2( j )).and.
     1        (pres(i).le.pres2(j+1))     ) then
            x=(dlog(pres(i))-dlog(pres2(j)))/
     1           (dlog(pres2(j+1))-dlog(pres2(j)))
            y=(dlog(pres2(j+1))-dlog(pres(i)))/
     1           (dlog(pres2(j+1))-dlog(pres2(j)))
            lrhod=y*dlog(rho2(j))+x*dlog(rho2(j+1))
            rhod(i)=max(dexp(lrhod),rrho(i))
            if ((i.lt.imax).and.(rrho(i+1).eq.rhod(i+1))) then
               rhod(i)=rrho(i)
            end if
         else
            goto 765
         end if
      end do
      return
      end
c     *********************************************************************
c***************************************************************************
      function cvt_deg(pf,m)
c**************************************************************c
c     Calculates the degenerate specific heat per cm^3 over T:    c
c     Cv/T                                   c
c     For spin 1/2 fermions !                                     c
c     pf and m must be in MeV, but cvt is returned in cgs units.  c
c     m must be the Landau effective mass, i.e., m* for baryons   c
c     and sqrt(m**2+pf**2) for leptons !                          c
c     CHECKED ON MARCH 913, 2001                      c
c**************************************************************c
      implicit real*8(a-h,k-z)
      parameter(pi=3.14159265d0)
      parameter (kb=1.38d-16, MeV=1.602d-6)
      if (pf.eq.0.0d0) then
         cvt_deg=0.0d0
      else
         N0 = 2.d0 * m*pf/2.d0/pi**2
         cvt_deg=pi**2/3.d0 * N0
      end if
      cvt_deg=cvt_deg * kb**2/MeV/197.d0**3*1.d39 ! converts to cgs units
      return
      end
c***************************************************************************
