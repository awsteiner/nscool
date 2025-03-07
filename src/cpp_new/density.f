c *********************************************************************
      subroutine density(T,P,A,Z,Rho)
c************************************************************
c given T,P,A,Z, calculates Rho: uses Newton's method
c and Rho as input is taken as a first guess
c************************************************************
c ***** checked
       implicit real*8(a-h,k-z)
        eps=1.d-3
c        print *,'--------------------------------'
c        print '(1a20,0p1f20.8)','--------->',Rho
 100    Rho0=Rho
        call pressure(T,Rho0,A,Z,Pre0)
        Rho1=(1.d0+eps)*Rho0
        call pressure(T,Rho1,A,Z,Pre1)
        f=Pre0-P
        f1=(Pre1-Pre0)/(Rho1-Rho0)
        dRho=-f/f1
        Rho=Rho0+dRho
c        print '(1a20,0p2f20.8,1p3e20.8)','--------->',Rho,Rho1,Pre0,Pre1
        if (abs(dRho/Rho).lt.1.d-5) return
        goto 100
       return
      end
c *********************************************************************
c *********************************************************************
      subroutine pressure(T,Rho,A,Z,Pres)
c **** sept 1991 version, checked on sept 2
       implicit real*8(a-h,k-z)
        hb=1.054588d-27
        kb=1.380662d-16
        c=2.997924d10
        NA=6.022045d23
        me=9.109d-28
        pi=3.141592653d0
C **** Calculate the electron density and ionization
        ne=Rho*NA*Z/A
        nion=Rho*NA/A
        Zeff=ne/nion
C **** Calculate the ionic pressure
        gamma=2.273E5*Zeff**2*(Rho/A)**(1./3.)/T
        if (gamma.ge.210.) then
         Uion=1.5-0.895929*gamma+3225./gamma**2
        else
         Uion=-0.897744*gamma+0.95043*gamma**.25
     1        +0.18956/gamma**.25-0.81487
        end if
        Pion=nion*kb*T*(1.0d0+Uion/3.d0)      
C **** Get the electron pressure
        call P_electron(T,ne,Pel)
C *********************************
        Pres=Pel+Pion
       return
      end
C ********************************************************************
C ********************************************************************
      subroutine P_electron(T,ne,Pres) 
C*********************************************************************C
C  Calculates the pressure of a perfect electron gas (Fermi-Dirac)    C
C  From Eggleton, Faulkner & Flannery, A&A23 (1973), p. 325-330       C
C  for the approx. which gives thermodynamically consistent D,U & P   C
C                                                                     C
C      CHECKED on nov. 24 1990 and again on July 15 2005              C
C         that it gives the correct perfect gas and                   C
C    degenerate relativistic and non-relativistic limits :            C
C                                                                     C
C      PDEGNR=(3.*PI**2)**(2./3.)/5.*(HB*NE)*(HB/ME)*NE**(2./3.)      C
C      PDEGR=(3.*PI**2)**(1./3.)/4.*(HB*NE)*(  C  )*NE**(1./3.)       C
C      PPERF=NE*KB*T                                                  C
C*********************************************************************C
       implicit real*8(a-h,k-z)
       parameter (pi = 3.14159652d0,NA=6.022045d23)
       parameter(epsilon=1.d-12)
       dimension CD(0:3,0:3),CP(0:3,0:3),CU(0:3,0:3)
       dimension DS(0:3)
C *****
       DATA ((CD(I,J),I=0,3),J=0,3)
     1         / 2.315472, 7.128660, 7.504998, 2.665350,
     2           7.837752,23.507934,23.311317, 7.987465,
     3           9.215560,26.834068,25.082745, 8.020509,
     4           3.693280,10.333176, 9.168960, 2.668248/
       DATA ((CP(I,J),I=0,3),J=0,3)
     1         / 2.315472, 6.748104, 6.564912, 2.132280,
     2           7.837752,21.439740,19.080088, 5.478100,
     3           9.215560,23.551504,19.015888, 4.679944,
     4           3.693280, 8.859868, 6.500712, 1.334124/   
       DATA ((CU(I,J),I=0,3),J=0,3)
     1         / 3.473208,10.122156, 9.847368, 3.198420,
     2          16.121172,43.477194,37.852852,10.496830,
     3          23.971040,60.392810,47.782844,11.361074,
     4          11.079840,26.579604,19.502136, 4.002372/

C *****     
      NEHAT=NE/1.7595d30
C ***** CALCULATE F

      T1=T/5.93d9
      F1=1.D-3*(NE/NA)*(1.d7/T)**3
      IF ( (ABS(OLDNE-NE)/NE.LE.(5.D-1)).AND.
     1        (ABS(OLDT-T)/T.LE.(5.D-1)) )  F1=OLDF1
     
1111  F=ABS(F1)
      G=T1*SQRT(1.D0+F)
      PF=1.D0+F
      PG=1.D0+G

      DO J1=0,3
       DS(J1)=CD(J1,0)+(CD(J1,1)+CD(J1,2)*G+CD(J1,3)*G*G)*G
       DS(J1)=DS(J1)/PF**3
      END DO
      SUM1=DS(0)+(DS(1)+DS(2)*F+DS(3)*F*F)*F

      DO J1=1,3
       DS(J1)=CD(J1,0)+(CD(J1,1)+CD(J1,2)*G+CD(J1,3)*G*G)*G
       DS(J1)=DS(J1)/PF**3
      END DO
      SUM2=DS(1)+(2.*DS(2)+3.*DS(3)*F)*F

      DO J1=0,3
       DS(J1)=(CD(J1,1)+2.*CD(J1,2)*G+3.*CD(J1,3)*G*G)*G
       DS(J1)=DS(J1)/PF**3
      END DO
      SUM3=DS(0)+(DS(1)+DS(2)*F+DS(3)*F*F)*F

      IF(F.LT.1.)THEN
       COEF=F*G**1.5/PF/PG**1.5
       NEF=COEF*SUM1
       NEF1=(COEF/F-
     1       3.25*COEF/PF-
     2       .75*COEF*G/PF/PG) * SUM1 +
     3       COEF*(SUM2+0.5/PF*SUM3)
      ELSE
       COEF=(F/PF)*(G/PG)**1.5
       NEF=COEF*SUM1
       NEF1=COEF*
     1      ( (1./F-3.25/PF-0.75*G/(PF*PG))*SUM1+SUM2+0.5/PF*SUM3 )
      END IF

      F1=ABS(F+(NEHAT-NEF)/NEF1)
     
      IF ((ABS(NEF-NEHAT)/ABS(NEHAT).GT.EPSILON).OR.
     1    (ABS(F1-F)/ABS(F1).GT.EPSILON)) GOTO 1111

C *****  Calculate pressure
     
      G1=T1*SQRT(1.+F1)

      DO J1=0,3
       DS(J1)=CP(J1,0)+(CP(J1,1)+CP(J1,2)*G1+CP(J1,3)*G1*G1)*G1
      END DO
      SUMP1=DS(0)+(DS(1)+DS(2)*F1+DS(3)*F1*F1)*F1

      IF(F1.LT.1)THEN
       P1=1.44E24*F1*G1**2.5/(1.+F1)**4/(1.+G1)**1.5*SUMP1
      ELSE
       P1=1.44E24*(F1**.25/(1.+F1))**4*G1*(G1/(1.+G1))**1.5*SUMP1
      END IF
     
      PRES=P1
     
      OLDNE=NE
      OLDT=T
      OLDF1=F1
     
      RETURN
     
      END

C*********************************************************************
C*********************************************************************
C*********************************************************************

