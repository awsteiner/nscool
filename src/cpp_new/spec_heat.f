      subroutine specheat(i,t,rho,aion,zion,cv,
     1     cvneutron,cvproton,cvelectron,cvmuon,
     2     cvlambda,cvsminus,cvszero,cvsplus,
     3     cvquark,cvions,cve,cvm,cvn,cvp,cvla,
     4     cvsm,cvs0,cvsp,cvqu,dvqd,cvqs,rhodrip,rhocore,fhad,istrange,
     5     tcn,tcp,tcla,tcsm,tcs0,tcsp,isf)

c ****** checked on March 30, 1993

c***********************************************************************
c  Includes the Levenfish-Yakovlev coefficients for pairing suppression
c***********************************************************************

      implicit real*8(a-h,k-z)
      parameter (isize=10000)
      parameter (pi = 3.14159265)

      dimension tcn(0:isize),tcp(0:isize),tcla(0:isize),
     2          tcsm(0:isize),tcs0(0:isize),tcsp(0:isize)

      dimension cve(0:isize),cvm(0:isize),cvn(0:isize),cvp(0:isize),
     1     cvla(0:isize),cvsm(0:isize),cvs0(0:isize),cvsp(0:isize),
     2     cvqu(0:isize),cvqd(0:isize),cvqs(0:isize)

      dimension fhad(0:isize)
 
c -----------------------------------------------------------------------
c theta_k and theta_p are the chiral angles for Kaon and Pion condensates
c
c z_ion  is the charge number of the nuclei
c a_ion  is the mass number of the nuclei
c a_cell is the number of nucleons per Wigner-Seitz cell 
c        (i.e., a_cell=a_ion + # dripped neutrons)
c -----------------------------------------------------------------------
      
c ***************************

      fexp(x)=dexp(max(x,-7.d2))
      u_1s0(t)=dsqrt(1.-t)*(1.456-0.157/dsqrt(t)+1.764/t)
      r_1s0(u)=(0.4186+dsqrt(1.007**2+(0.5010*u)**2))**2.5*
     1         fexp(1.456-dsqrt(1.456**2+u**2))

      u_3p2(t)=dsqrt(1.-t)*(5.596+8.424/t)
      r_3p2(u)=(0.6893+dsqrt(0.790**2+(0.03983*u)**2))**2*
     1         fexp(1.934-dsqrt(1.934**2+u**2/(16.*pi)))

c ***************************

c ****** get Cv-ions :
     
      if (rho .lt. rhocore) then
       call cvion(t,rho,aion,zion,cv_ions)
c       call cvnrot(t,rho,aion,zion,cv_nrot)
       cv_nrot=0.d0
c       call cvnvib(t,i,rho,aion,zion,cv_nvib,theta_vib)
       cv_nvib=0.d0
       cvions=cv_ions+cv_nrot+cv_nvib
      else
       cvions=0.
      end if

c ****** get Cv-electrons :

      if ((rho .lt. rhodrip).and.(icvel_nodeg.eq.1)) then
       call cvelec(t,rho,aion,zion,i,cvelectron)
      else
       cvelectron=cve(i)*t
      end if

c ***** get Cv-muons :

      cvmuon=cvm(i)*t

c ****** get Cv-neutrons :

      if ((rho.ge.rhodrip).and.(istrange.eq.0)) then
c HERE DANY cccccccccccccccccccccccc
c       raise=1.1d0
       raise=1.0d0
cccccccccccccccccccccccccccccccccccc
       if (t .lt. raise*tcn(i))then
        t0=min(0.999999999999d0,t/tcn(i))
        if (i.le.isf) then
         u=u_3p2(t0)
         r=r_3p2(u)
        else
         u=u_1s0(t0)
         r=r_1s0(u)
        end if
        if (t .gt. tcn(i)) then
         w1=(raise*tcn(i)-t)/((raise-1.)*tcn(i))
         w2=1.-w1
         r=w1*r+w2*1.0
        end if
       else
        r=1.d0
       end if
       cvneutron =cvn(i)*t*r
      else
       cvneutron=0.d0
      end if
c here cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     Milano specific heat:
c      if ((rho.ge.rhodrip).and.(i.gt.isf)) then
c       call cvn_milano(t,rho,cvneutron)
c      end if
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

c ****** get Cv-protons :
  
      if ((rho.ge.rhocore).and.(istrange.eq.0)) then
       raise=1.1d0
       if(t .lt. raise*tcp(i))then
        t0=min(0.999999999999d0,t/tcp(i))
        u=u_1s0(t0)
        r=r_1s0(u)
        if (t .gt. tcp(i)) then
         w1=(raise*tcp(i)-t)/((raise-1.)*tcp(i))
         w2=1.-w1
         r=w1*r+w2*1.0
        end if
       else
        r=1.d0
       end if
       cvproton  =cvp(i)*t*r
      else
       cvproton=0.d0
      end if

c ****** get Cv-lambdas :
  
      if ((rho.ge.rhocore).and.(istrange.eq.0)) then
       raise=1.1
       if(t .lt. raise*tcla(i))then
        t0=min(0.999999999999d0,t/tcla(i))
        u=u_1s0(t0)
        r=r_1s0(u)
        if (t .gt. tcla(i)) then
         w1=(raise*tcla(i)-t)/((raise-1.)*tcla(i))
         w2=1.-w1
         r=w1*r+w2*1.0
        end if
       else
        r=1.d0
       end if
       cvlambda  =cvla(i)*t*r
      else
       cvlambda=0.d0
      end if

c ****** get Cv-Sigma- :
  
      if ((rho.ge.rhocore).and.(istrange.eq.0)) then
       raise=1.1
       if(t .lt. raise*tcsm(i))then
        t0=min(0.999999999999d0,t/tcsm(i))
        u=u_1s0(t0)
        r=r_1s0(u)
        if (t .gt. tcsm(i)) then
         w1=(raise*tcsm(i)-t)/((raise-1.)*tcsm(i))
         w2=1.-w1
         r=w1*r+w2*1.0
        end if
       else
        r=1.d0
       end if
       cvsminus  =cvsm(i)*t*r
      else
       cvsminus=0.d0
      end if

c ****** get Cv-Sigma0 :
  
      if ((rho.ge.rhocore).and.(istrange.eq.0)) then
       raise=1.1
       if(t .lt. raise*tcs0(i))then
        t0=min(0.999999999999d0,t/tcs0(i))
        u=u_1s0(t0)
        r=r_1s0(u)
        if (t .gt. tcs0(i)) then
         w1=(raise*tcs0(i)-t)/((raise-1.)*tcs0(i))
         w2=1.-w1
         r=w1*r+w2*1.0
        end if
       else
        r=1.d0
       end if
       cvszero  =cvs0(i)*t*r
      else
       cvszero=0.d0
      end if

c ****** get Cv-Sigma+ :
  
      if ((rho.ge.rhocore).and.(istrange.eq.0)) then
       raise=1.1
       if(t .lt. raise*tcsp(i))then
        t0=min(0.999999999999d0,t/tcsp(i))
        u=u_1s0(t0)
        r=r_1s0(u)
        if (t .gt. tcsp(i)) then
         w1=(raise*tcsp(i)-t)/((raise-1.)*tcsp(i))
         w2=1.-w1
         r=w1*r+w2*1.0
        end if
       else
        r=1.d0
       end if
       cvsplus  =cvsp(i)*t*r
      else
       cvsplus=0.d0
      end if

c ****** get Cv-photons :

c      cvphot= 4.*7.56e-15*t**3
      cvphot=0.d0

c ***** get Cv-quarks:

      if (rho.ge.rhocore) then
         cvquark=c_cv_str*(T/1.d9)
      else
         cvquark=0.d0
      end if

c ****** Total Cv :
      cvelectron=cvelectron*fhad(i)
      cvmuon    =cvmuon    *fhad(i)
      cvproton  =cvproton  *fhad(i)
      cvneutron =cvneutron *fhad(i)
      cvlambda  =cvlambda  *fhad(i)
      cvsminus  =cvsminus  *fhad(i)
      cvszero   =cvszero   *fhad(i)
      cvsplus   =cvsplus   *fhad(i)
      cvquark   =cvquark*(1.d0-fhad(i))

      cv=cvions+
     2   cvelectron+cvmuon+
     3   cvproton+cvneutron+
     4   cvlambda+cvsminus+cvszero+cvsplus+
     5   cvphot+cvquark

      return

      end

c *********************************************************************
c *********************************************************************

      subroutine cvion (t,rho,a,z,cv)
     
c *** checked on oct. 22 1990 *******

      implicit real*8(a-h,k-z)
      parameter (rhodrip=4.3e11)
     
      dimension cv0(0:14)
     
      data bcv,ccv,dcv,hcv,cte/.95043,.18956,-.81487,3225,1.417e2/
      data cv0/0.0,2.956,2.829,2.633,2.389,2.118,1.840,1.572,
     1             1.323,1.102,0.909,0.745,0.609,0.496,0.404/

     
      gamma=2.273e5*z**2*(rho/a)**(1./3.)/t
      if (rho .ge. rhodrip) then
       a1=3.*z
      else
       a1=a
      endif
      nionkb=1.38e-16*6.022e23*rho/a
      delta=1./t*z*dsqrt(rho/(a1*a))*6.022e23
      if (gamma .le. .1) then
       cv=1.5*nionkb
       return
      else if (gamma .le. .2) then
       cv1=1.5*nionkb
       cv2=nionkb*(.75*bcv*gamma**.25+1.25*ccv/gamma**.25+dcv+1.5)
       cv=(gamma-.1)/.1*cv2+(.2-gamma)/.1*cv1
       return
      else if (gamma .le. 178.) then
       cv=nionkb*(.75*bcv*gamma**.25+1.25*ccv/gamma**.25+dcv+1.5)
       return
      else if ((gamma.le.210.).and.(delta.ge.1.e19)) then
       cv1=nionkb*(1.5+3.*hcv/gamma**2+1.5)
       cv0(0)=1.5+3.*hcv/gamma**2+1.5
       i1=int(delta*2.e-20)
       cv2=nionkb*(cv0(i1)+(delta*2.e-20-i1)*(cv0(i1+1)-cv0(i1)))
       cv=(gamma-178.)/32.*cv2+(210.-gamma)/32.*cv1
       return
      else if (delta .le. 1e19) then
       cv=nionkb*(1.5+3.*hcv/gamma**2+1.5)
       return
      else if ((delta .gt. 1e19) .and. (delta .lt. 7e20)) then
       cv0(0)=1.5+3.*hcv/gamma**2+1.5
       i1=int(delta*2.e-20)
       cv=nionkb*(cv0(i1)+(delta*2.e-20-i1)*(cv0(i1+1)-cv0(i1)))
       return
      else
       delta1=delta*1.e-20
       cv=nionkb*cte/delta1**3
       return
      end if
     
      return
     
      end

c ************************************************************************
c ************************************************************************

      subroutine cvelec (t,rho,a,z,ikeep,cv)

c *******  checked on sept 4  1991 **************
     
      implicit real*8(a-h,k-z)
      parameter (isize=10000)
      parameter (pi = 3.14159625)
     
      dimension cd(0:3,0:3),cp(0:3,0:3),cu(0:3,0:3)
      dimension fkeep(0:isize)
     
      data ((cd(i,j),i=0,3),j=0,3)
     1         / 2.315472, 7.128660, 7.504998, 2.665350,
     2           7.837752,23.507934,23.311317, 7.987465,
     3           9.215560,26.834068,25.082745, 8.020509,
     4           3.693280,10.333176, 9.168960, 2.668248/
     
      data ((cp(i,j),i=0,3),j=0,3)
     1         / 2.315472, 6.748104, 6.564912, 2.132280,
     2           7.837752,21.439740,19.080088, 5.478100,
     3           9.215560,23.551504,19.015888, 4.679944,
     4           3.693280, 8.859868, 6.500712, 1.334124/
     
      data ((cu(i,j),i=0,3),j=0,3)
     1         / 3.473208,10.122156, 9.847368, 3.198420,
     2          16.121172,43.477194,37.852852,10.496830,
     3          23.971040,60.392810,47.782844,11.361074,
     4          11.079840,26.579604,19.502136, 4.002372/
     
      hb=1.054588e-27
      kb=1.380662e-16
      c=2.997924e10
      na=6.022045e23
      me=9.109e-28
     
      ne=na*rho*z/a
      pf=hb*(3.*pi**2*ne)**(1./3.)
      ef=dsqrt((me*c**2)**2+(pf*c)**2)-me*c**2
      tf=ef/kb
      xe=pf/(me*c)
      ae=xe**2/dsqrt(1.+xe**2)
      cvt=ne*kb**2*pi**2/(me*c**2)/ae
     
      t0=tf/50.
     
      if(t .le. .5*t0)then
       cv=cvt*t
       return
      end if
     
      nehat=ne/1.7595e30
     
c ***** calculate f at .9999*t
     
      t1=0.9999 * t/5.93e9

      if (fkeep(ikeep).eq.0.) then
       f1=1.
      else
       f1=fkeep(ikeep)
      end if

1111  f=abs(f1)
      g=t1*dsqrt(1.+f)

      sum1=0.
      do 100 j1=0,3
       do 101 j2=0,3
        sum1=sum1+cd(j1,j2)*f**j1*g**j2
101    continue
100   continue
      sum2=0.
      do 102 j1=1,3
       do 103 j2=0,3
        sum2=sum2+j1*cd(j1,j2)*f**(j1-1)*g**j2
103    continue
102   continue
      sum3=0.
      do 104 j1=0,3
       do 105 j2=1,3
        sum3=sum3+j2*cd(j1,j2)*f**j1*g**j2
105    continue
104   continue
     
      pf=1.+f
      pg=1.+g
      if(f.lt.1.)then
       nef=f*g**1.5*pf**(-4)*pg**(-1.5)*sum1
       nef1=(g**1.5*pf**(-4)/pg**1.5-
     1       3.25*f*g**1.5*pf**(-5)*pg**(-1.5)-
     2       .75*f*g**2.5*pf**(-5)*pg**(-2.5)) * sum1 +
     3       f*g**1.5*pf**(-4)*pg**(-1.5)*(sum2+0.5/pf*sum3)
      else
       nef=(f**.25/pf)**4*(g/pg)**1.5*sum1
       nef1=(f**.25/pf)**4*(g/pg)**1.5*
     1      ( (1./f-3.25/pf-0.75*g/(pf*pg))*sum1+sum2+0.5/pf*sum3 )
      end if
     
      f1=abs(f+(nehat-nef)/nef1)
     
      if ((abs(nef-nehat)/abs(nehat).gt.1.e-10).or.
     1    (abs(f1-f)/abs(f1).gt.1.e-12)) goto 1111

      fkeep(ikeep)=f1

c *****  calculate f at 1.0001*t
     
      t2=1.0001 * t/5.93e9
      f2=f1
     
1112  f=abs(f2)
      g=t2*dsqrt(1.+f)
     
      sum1=0.
      do 106 j1=0,3
       do 107 j2=0,3
        sum1=sum1+cd(j1,j2)*f**j1*g**j2
107    continue
106   continue
      sum2=0.
      do 108 j1=1,3
       do 109 j2=0,3
        sum2=sum2+j1*cd(j1,j2)*f**(j1-1)*g**j2
109    continue
108   continue
      sum3=0.
      do 110 j1=0,3
       do 111 j2=1,3
        sum3=sum3+j2*cd(j1,j2)*f**j1*g**j2
111    continue
110   continue
     
      pf=1.+f
      pg=1.+g
      if(f.lt.1)then
       nef=f*g**1.5*pf**(-4)*pg**(-1.5)*sum1
       nef1=(g**1.5*pf**(-4)/pg**1.5-
     1       3.25*f*g**1.5*pf**(-5)*pg**(-1.5)-
     2       .75*f*g**2.5*pf**(-5)*pg**(-2.5)) * sum1 +
     3       f*g**1.5*pf**(-4)*pg**(-1.5)*(sum2+0.5/pf*sum3)
      else
       nef=(f**.25/pf)**4*(g/pg)**1.5*sum1
       nef1=(f**.25/pf)**4*(g/pg)**1.5*
     1      ( (1./f-3.25/pf-0.75*g/pf/pg)*sum1+sum2+0.5/pf*sum3 )
      end if
     
     
      f2=abs(f+(nehat-nef)/nef1)
     
      if ((abs(nef-nehat)/abs(nehat).gt.1.e-10).or.
     1    (abs(f2-f)/abs(f2).gt.1.e-12)) goto 1112
     
c *****  calculate cv
     
      sumu2=0.
      g2=t2*dsqrt(1.+f2)
      do 120 j1=0,3
       do 121 j2=0,3
        sumu2=sumu2+cu(j1,j2)*f2**j1*g2**j2
121    continue
120   continue
      sumu1=0.
      g1=t1*dsqrt(1.+f1)
      do 122 j1=0,3
       do 123 j2=0,3
        sumu1=sumu1+cu(j1,j2)*f1**j1*g1**j2
123    continue
122   continue
      if(f1.lt.1)then
       u1=1.44e24*f1*g1**2.5/(1.+f1)**4/(1.+g1)**1.5*sumu1
       u2=1.44e24*f2*g2**2.5/(1.+f2)**4/(1.+g2)**1.5*sumu2
      else
       u1=1.44e24*(f1**.25/(1.+f1))**4*g1*(g1/(1.+g1))**1.5*sumu1
       u2=1.44e24*(f2**.25/(1.+f2))**4*g2*(g2/(1.+g2))**1.5*sumu2
      end if
     
      cv=(u2-u1)/(.0002*t)
     
      if (t .lt. 1.5*t0) then
       w1=(t-.5*t0)/t0
       w2=1.-w1
       cv=w1*cv+w2*cvt*t
      end if
     
      return
     
      end
     
