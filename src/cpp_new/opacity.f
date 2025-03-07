      subroutine opacity(T,Rho,A,Z,kappa,iopacity)

c ******************************************************************c
c     temperature in K  density in gm/cm3  op given in cm2/gm       c
c*******************************************************************c
c     Uses Kramer opacity only, from Shapiro & Teukolsky 
c ******************************************************************c
      implicit real*8(a-h,k-z)
      if (iopacity.eq.0) then
         kappa=1.d200
      else
         if (Rho.ge.1.e14) then
            A=100.
            Z=32.
         end if
         kappa = 0.645d23*Z**3/A**2 * Rho/T**3.5
      end if
      return
      end
c ******************************************************************c


