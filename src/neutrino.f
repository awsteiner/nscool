      subroutine neutrino(irank,i,t,rho,a,z,qtot,
     1     qeebrem,qnpb,qplasma,qsynch,qbubble,qpair,qphoto,qbrem_nn,
     2     qmurca_nucl,qbrem_nucl,qmurca_hyp,qbrem_hyp,
     3     qdurca_np,qdurca_lap,qdurca_smn,qdurca_smla,qdurca_sms0,
     4     qfast,qdurca_q,qmurca_q,
     6     qpbf_n1s0,qpbf_n3p2,qpbf_p1s0,qpbf_q,
     7     debug,naa,nbfield2,rhodrip,rhocore,
     8     mstp,mstn,mstla,mstsm,msts0,mstsp,kfe,kfm,kfp,kfn,
     9     kfqu,kfqd,kfqs,bar,yelect,ymuon,fhad,theta_k,theta_p,v_ion,
     a     rhoexo,cexo,pexo,c_nu_str,p_nu_str,
     b     murca_increase,inu_durca,inu_eion,inu_plasma,inu_synch,
     c     inu_n1s0_pbf,inu_n3p2_pbf,inu_p_pbf,
     d     inu_bubble,inu_photo,inu_pair,
     e     idurca_np,idurca_lap,durca_ctrl_e,durca_ctrl_m,
     f     idurca_smn,idurca_smla,idurca_sms0,
     g     idurca_quqd,idurca_quqs,tcn,tcp,tcla,tcu,tcd,tcs,
     h     tcu1,tcu2,tcu3,tcd1,tcd2,tcd3,tcs1,tcs2,tcs3,isf,
     i     neebrem_logt,neebrem_nalpha,neebrem_n2,
     j     sf_lgtau1,sf_lgtau2,sf_lgr,sf_lgr2)
      
c     **** checked  august 21 1991 **************

      implicit real*8 (a-h,k-z)
      parameter (isize=10000)

      dimension idurca_np(0:isize),idurca_lap(0:isize),
     2     idurca_smn(0:isize),idurca_smla(0:isize),
     3     idurca_sms0(0:isize),
     4     idurca_quqd(0:isize),idurca_quqs(0:isize),
     5     durca_ctrl_e(0:isize),durca_ctrl_m(0:isize)
      
      dimension mstp(0:isize),mstn(0:isize),mstla(0:isize),
     2     mstsm(0:isize),msts0(0:isize),mstsp(0:isize)
      dimension kfe(0:isize),kfm(0:isize),kfp(0:isize),kfn(0:isize),
     1     kfqu(0:isize),kfqd(0:isize),kfqs(0:isize)
      dimension bar(0:isize),
     2     yelect(0:isize),ymuon(0:isize),
     6     fhad(0:isize),
     7     theta_k(0:isize),theta_p(0:isize),
     9     v_ion(0:isize)
      
      dimension tcn(0:isize),tcp(0:isize),tcla(0:isize),
     5     tcu(0:isize),tcd(0:isize),tcs(0:isize)

c     April 2004: seperate Tc for flavor AND color:
      dimension tcu1(0:isize),tcu2(0:isize),tcu3(0:isize),
     2     tcd1(0:isize),tcd2(0:isize),tcd3(0:isize),
     3     tcs1(0:isize),tcs2(0:isize),tcs3(0:isize)

      dimension nbfield2(0:isize)

      dimension neebrem_logt(56),neebrem_nalpha(56),neebrem_n2(56)
      dimension sf_lgtau1(35),sf_lgtau2(35)
      dimension sf_lgr(35,35),sf_lgr2(35,35)
      
      istrange=0

      if (debug.ge.2.) print *,'Entering subroutine `neutrino'' ',
     2     ' T, rho, A, Z = ',t,rho,a,z
c     *** ELECTRON-ELECTRON PAIR BREMSSTRAHLUNG:
      if (rho.lt.rhocore) then
         mu_el=kfe(i)*197.
         call neebrem(irank,T,mu_el,qeebrem,neebrem_logt,neebrem_nalpha,
     1        neebrem_n2)
      else
         qeebrem=0.0d0
      end if
c     *** ELECTRON-ION PAIR BREMSSTRAHLUNG:
      if (inu_eion.eq.1) then
         if (rho.lt.rhocore) then
            call npb_new(t,rho,qnpb)
         else
            qnpb=0.0d0
         end if
      else if (inu_eion.eq.2) then
         if (rho.lt.rhocore) then
            call npb(t,rho,a,z,qnpb)
         else
            qnpb=0.0d0
         end if
      else
         if (rho.lt.rhocore) then
            qnpb=0.0d0
            if (print_it.ne.1.) then
               print *,'No npb: Rho, Qnpb=',rho,qnpb
               print_it=1.
            end if
         else
            qnpb=0.0d0
         end if
      end if
c     *** PLASMA NEUTRINO:
      if (inu_plasma.eq.1) then
         if (rho.lt.rhocore) then
            call nplasma(t,rho,a,z,qplasma)
         else
            qplasma=0.0d0
         end if
      else if (inu_plasma.eq.-1) then
         if (rho.lt.rhocore) then
            call nplasma_old(t,rho,a,z,qplasma)
         else
            qplasma=0.0d0
         end if
      else
         qplasma=0.0d0
      end if
c     *** SYNCHROTRON NEUTRINO:
      if (inu_synch.eq.1) then
         if (rho.lt.rhocore) then
            call nsynch(t,nbfield2(i),kfe(i),qsynch)
         end if
      else
         qsynch=0.0d0
      end if
c     *** BUBBLE NEUTRINO:
      if (inu_bubble.eq.1) then
         if (rho.lt.rhocore) then
            call nbub(i,t,rho,a,z,qbubble,rhocore,tcn,isf)
         else
            qbubble=0.0d0
         end if
      else
         qbubble=0.0d0
      end if
c     *** NEUTRINO PAIR:
      if (inu_pair.eq.1) then
         if (rho.lt.rhocore) then
            call npair(t,rho,a,z,qpair)
         else
            qpair=0.0d0
         end if
      else
         qpair=0.0d0
      end if
c     **** PHOTO-NEUTRINO:
      if (inu_photo.eq.1) then
         if (rho.lt.rhocore) then
            call nphoto(t,rho,a,z,qphoto)
         else
            qphoto=0.0d0
         end if
      else
         qphoto=0.0d0
      end if
c     *** NN-BREMSTRAHLUNG in the inner crust:
      if ((rho.lt.rhocore).and.(rho.ge.rhodrip)) then
         call nubrem_crust_nn(i,t,v_ion(i),qbrem_nn,tcn,isf,kfn,mstn)
      else
         qbrem_nn=0.d0
      end if
c     *** URCA et al. PROCESSES:
      if (rho.ge.rhocore) then
         if (istrange.eq.0) then
            call numurca_nucl(i,t,qmurca_nucl,tcn,tcp,isf,
     1           mstn,mstp,kfe,kfm,kfn,kfp)
            qmurca_nucl=qmurca_nucl*(1.d0+murca_increase)
            qmurca_nucl=qmurca_nucl*fhad(i)
            call nubrem_nucl(i,t,qbrem_nucl,tcn,tcp,isf,
     1           kfn,kfp,mstn,mstp)
            qbrem_nucl=qbrem_nucl*(1.d0+murca_increase)
            qbrem_nucl=qbrem_nucl*fhad(i)
            call numurca_hyp(i,t,qmurca_hyp)
            qmurca_hyp=qmurca_hyp*fhad(i)
            call nubrem_hyp(i,t,qbrem_hyp)
            qbrem_hyp=qbrem_hyp*fhad(i)
            if (inu_durca.eq.1) then
               call nudurca_h(irank,i,t,rho,qdurca_np,qdurca_lap,
     1              qdurca_smn,qdurca_smla,qdurca_sms0,tcn,tcp,tcla,isf,
     2              bar,yelect,ymuon,mstp,mstn,mstla,mstsm,msts0,mstsp,
     3              durca_ctrl_e,durca_ctrl_m,idurca_lap,idurca_smla,
     4              idurca_smn,idurca_sms0,idurca_np,
     5              sf_lgtau1,sf_lgtau2,sf_lgr,sf_lgr2)
               qdurca_np=qdurca_np*fhad(i)
               qdurca_lap=qdurca_lap*fhad(i)
               qdurca_smn=qdurca_smn*fhad(i)
               qdurca_smla=qdurca_smla*fhad(i)
               qdurca_sms0=qdurca_sms0*fhad(i)
            else
               qdurca_np=0.0d0
               qdurca_lap=0.0d0
               qdurca_smn=0.0d0
               qdurca_smla=0.0d0
               qdurca_sms0=0.0d0
            end if
c     *** FAST neutrino emission:
            call nufast(i,t,rho,qfast,tcn,tcp,isf,bar,theta_k,theta_p,
     1           yelect,rhoexo,cexo,pexo,mstn,mstp,kfe)
            qfast=qfast*fhad(i)
c     *** QUARK processes:
            call nudurca_q(i,t,rho,qdurca_q,tcu1,tcu2,tcu3,
     1     tcd1,tcd2,tcd3,tcs1,tcs2,tcs3,kfe,kfm,kfqu,kfqd,kfqs,
     2     idurca_quqd,idurca_quqs)
            call numurca_q(i,t,rho,qmurca_q,kfqu,tcu,tcd)
            qdurca_q=qdurca_q*(1.d0-fhad(i))
            qmurca_q=qmurca_q*(1.d0-fhad(i))
            qstrange=0.d0
c     *** STRANGE QUARK MATTER processes:
         else if (istrange.eq.1) then
            qstrange=c_nu_str*(T/1.d9)**p_nu_str
            qmurca_nucl=0.0d0
            qbrem_nucl=0.0d0
            qmurca_hyp=0.0d0
            qbrem_hyp=0.0d0
            qdurca_np=0.0d0
            qdurca_lap=0.0d0
            qdurca_smn=0.0d0
            qdurca_smla=0.0d0
            qdurca_sms0=0.0d0
            qfast=0.0d0
            qdurca_q=0.0d0
            qmurca_q=0.0d0
         else
            print *,'neutrino: istrange not defined !'
            stop
         end if
      else
         qmurca_nucl=0.0d0
         qbrem_nucl=0.0d0
         qmurca_hyp=0.0d0
         qbrem_hyp=0.0d0
         qdurca_np=0.0d0
         qdurca_lap=0.0d0
         qdurca_smn=0.0d0
         qdurca_smla=0.0d0
         qdurca_sms0=0.0d0
         qfast=0.0d0
         qdurca_q=0.0d0
         qmurca_q=0.0d0
         qstrange=0.d0
      end if
c     *** PBF PROCESSES:
      if (istrange.eq.0) then
c     Neutrons 1S0:
         if ((inu_n1s0_pbf.eq.1).and.(i.gt.isf)) then
            call nu_1s0_pbf(t,tcn(i),mstn(i),kfn(i),qpbf_n1s0)
            qpbf_n1s0=qpbf_n1s0*fhad(i)
         else
            qpbf_n1s0=0.0d0
         end if
c     Neutron 3P2:
         if ((inu_n3p2_pbf.eq.1).and.(i.le.isf)) then
            call nu_n3p2_B_pbf(t,tcn(i),mstn(i),kfn(i),qpbf_n3p2)
            qpbf_n3p2=qpbf_n3p2*fhad(i)
         else
            qpbf_n3p2=0.0d0
         end if
c     Protons:
         if (inu_p_pbf.eq.1) then
            call nu_1s0_pbf(t,tcp(i),mstp(i),kfp(i),qpbf_p1s0)
            qpbf_p1s0=qpbf_p1s0*fhad(i)
         else
            qpbf_p1s0=0.0d0
         end if
c     Quarks: TO BE INCLUDED !!!!!!!!
         qpbf_q=0.0d0
         qpbf_q=qpbf_q*(1.d0-fhad(i))
      else
         qpbf_n1s0=0.0d0
         qpbf_n3p2=0.0d0
         qpbf_p1s0=0.0d0
         qpbf_q=0.0d0
      end if
c     *** ADDING EVERYTHING:
      qtot=qeebrem+qnpb+qplasma+qsynch+qbubble+qpair+qphoto+qbrem_nn+
     1     qmurca_nucl+qbrem_nucl+qmurca_hyp+qbrem_hyp+
     2     qdurca_np+qdurca_lap+qdurca_smn+qdurca_smla+qdurca_sms0+
     3     qfast+qdurca_q+qmurca_q+qstrange+
     4     qpbf_n1s0+qpbf_n3p2+qpbf_p1s0+qpbf_q
c*****
      if (debug.ge.2.) print *,'Exiting subroutine `neutrino'' '
      return

      end







