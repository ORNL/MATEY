import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import h5py
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re
from scipy.signal import correlate
from scipy.integrate import simpson

font = {'size'   : 22}
matplotlib.rc('font', **font)
class turbulence_descriptor():
    def turbulence_TKE_spectrum(self, uvec, L=2.0*np.pi):
        #https://turbulence.utah.edu/codes/turbogenpython/tkespec.py
        u1=uvec[:,:,:,0]
        u2=uvec[:,:,:,1]
        u3=uvec[:,:,:,2]
        # grid size and spacing 
        N = u1.shape[0]  # assuming cubic domain
        assert u1.shape[0]==u1.shape[1]==u1.shape[2]
        #assert L == 2.0*np.pi/1024*N    # physical size 
        dx = L/N #2*np.pi/N
        k0 = 2.0*np.pi/L
        #FFT for each velocity component
        U1 = np.fft.fftn(u1)/(N**3)
        U2 = np.fft.fftn(u2)/(N**3)
        U3 = np.fft.fftn(u3)/(N**3)

        # Energy for each (k1, k2, k3)
        E_k_3D = 0.5 * (np.abs(U1)**2 + np.abs(U2)**2 + np.abs(U3)**2)
        E_k_3D = np.fft.fftshift(E_k_3D)

        box_radius = int(np.ceil((np.sqrt(N**2+N**2+N**2))/2.)+1)
        centerx = N//2
        centery = N//2
        centerz = N//2

        #Ek = np.zeros(box_radius)+1e-20
        i_vals = np.arange(N)
        j_vals = np.arange(N) 
        k_vals = np.arange(N) 
        ii, jj, kk = np.meshgrid(i_vals, j_vals, k_vals, indexing='ij')
        rr = np.sqrt((ii - centerx)**2 + (jj - centery)**2 + (kk - centerz)**2)
        wn_array = np.rint(rr).astype(np.int32)
        wn_flat = wn_array.ravel()
        E_flat  = E_k_3D.ravel()
        Ek   = np.bincount(wn_flat, weights=E_flat)
        # Create wavenumber array
        #kvals = np.arange(box_radius) * k0
        kvals = np.arange(len(Ek)) * k0
        max_rr= 482 # truncated based on the wave number provided by the JHTDB website
        kvals= kvals[:max_rr+1]
        Ek   = Ek[:max_rr+1]
        return kvals, Ek

    def visualize_energy_spectrum(self, kvals, Ek, outputdir="./", filestr="",label="test"):
        fig = plt.figure(figsize=(10,8))
        # Plot energy spectrum on log-log axes
        plt.loglog(kvals[1:], Ek[1:], marker='o', label=label)  # skip zero-index to avoid k=0
        C = 2.0 
        slope_53 = C * (kvals**(-5.0/3.0))
        plt.loglog(kvals[1:], slope_53[1:],"m--", label=r'Reference slope $k^{-5/3}$')
        data = np.loadtxt('/lustre/orion/lrn037/proj-shared/zhangp/JHUTDB/isotropic1024fine/spectrum.txt', skiprows=2)
        kvals = data[:, 0]
        Ek = data[:, 1]
        plt.loglog(kvals[1:], Ek[1:],"k+", label=r'JHTDB spectrum')
        plt.xlabel(r'$k$')
        plt.ylabel(r'$E(k)$')
        plt.title('Energy Spectrum')
        plt.grid(True)  # optional, to help read the plot
        plt.legend()
        plt.savefig(outputdir+f"/energy_spectrum_{filestr}.png")

    def visualize_energy_spectrum_comp(self, kvalstar, Ektar, kvals, Ek, outputdir="./", filestr="",label="test"):
        fig = plt.figure(figsize=(10,8))
        # Plot energy spectrum on log-log axes
        plt.loglog(kvalstar[1:], Ektar[1:], marker='d', label="true")  # skip zero-index to avoid k=0
        plt.loglog(kvals[1:], Ek[1:], marker='o', markerfacecolor='none', label="pred")  # skip zero-index to avoid k=0
        C = 2.0 
        slope_53 = C * (kvals**(-5.0/3.0))
        plt.loglog(kvals[1:], slope_53[1:],"m--", label=r'Reference slope $k^{-5/3}$')
        data = np.loadtxt('/lustre/orion/lrn037/proj-shared/zhangp/JHUTDB/isotropic1024fine/spectrum.txt', skiprows=2)
        kvals = data[:, 0]
        Ek = data[:, 1]
        plt.loglog(kvals[1:], Ek[1:],"k+", label=r'JHTDB spectrum')
        plt.xlabel(r'$k$')
        plt.ylabel(r'$E(k)$')
        plt.title('Energy Spectrum')
        plt.grid(True)  # optional, to help read the plot
        plt.legend()
        plt.savefig(outputdir+f"/energy_spectrum_{filestr}.png")

    def visualize_energy_spectrum_threecomp(self, kvalstar, Ektar,  kvalsbase_c, Ekbase_c, kvalsbase, Ekbase, kvals, Ek, outputdir="./", filestr="",label=["base-64","base-32", "hierarchical-8"]):
        fig = plt.figure(figsize=(10,8))
        lwd=2.0
        kend=483
        # Plot energy spectrum on log-log axes
        #plt.loglog(kvalstar[1:kend], Ektar[1:kend], '-.', label="true", linewidth=lwd)  # skip zero-index to avoid k=0
        plt.loglog(kvalsbase_c[1:kend], Ekbase_c[1:kend], "g--", label="pred-"+label[0], linewidth=lwd)  
        plt.loglog(kvalsbase[1:kend], Ekbase[1:kend], "b--", label="pred-"+label[1], linewidth=lwd)  
        plt.loglog(kvals[1:kend], Ek[1:kend], "r--", label="pred-"+label[2], linewidth=lwd)  # skip zero-index to avoid k=0
        C = 2.0 
        slope_53 = C * (kvals**(-5.0/3.0))
        plt.loglog(kvals[1:], slope_53[1:],"m--", label=r'Reference slope $k^{-5/3}$')
        data = np.loadtxt('/lustre/orion/lrn037/proj-shared/zhangp/JHUTDB/isotropic1024fine/spectrum.txt', skiprows=2)
        kvals = data[:, 0]
        Ek = data[:, 1]
        plt.loglog(kvals, Ek,"k-", label=r'JHTDB spectrum', linewidth=lwd)
        for ps in [8, 32, 64]:
            plt.loglog([512/ps, 512/ps],[min(Ek[1:kend]), max(Ek[1:kend])],"k:")
        plt.xlabel(r'$k$')
        plt.ylabel(r'$E(k)$')
        plt.title('Energy Spectrum')
        plt.grid(True)  # optional, to help read the plot
        plt.legend()
        #plt.ylim(1e-4,15)
        #plt.savefig(outputdir+f"/energy_spectrum_{filestr}_zoomin.png")
        plt.savefig(outputdir+f"/energy_spectrum_{filestr}.png")

    def central_difference(self, f, dx, ax):
        """
        f:(nx, ny, nz)
        ax : int (0 -> x, 1 -> y, 2 -> z).
        Returns: dfdx: (nx, ny, nz), approximation of df/dx along axis
        """
        #assuming periodic condition in ax
        f_plus  = np.roll(f, -1, axis=ax)
        f_minus = np.roll(f,  1, axis=ax)
        dfdx    = (f_plus - f_minus) / (2.0 * dx)
        return dfdx

    def compute_enstrophy_and_dissipation(self, uvec, dx=0.00613592315, dy=0.00613592315, dz=0.00613592315, nu=0.000185):
        #the default values are from: https://turbulence.idies.jhu.edu/docs/isotropic/README-isotropic.pdf
        """
        u, v, w: (nx, ny, nz) 
        dx, dy, dz : 
        nu: Kinematic viscosity
        Returns: enstrophy; dissipation 
        """
        u = uvec[:,:,:,0]
        v = uvec[:,:,:,1]
        w = uvec[:,:,:,2]
        # du/dx, du/dy, du/dz
        dudx = self.central_difference(u, dx, ax=0)
        dudy = self.central_difference(u, dy, ax=1)
        dudz = self.central_difference(u, dz, ax=2)
        # dv/dx, dv/dy, dv/dz
        dvdx = self.central_difference(v, dx, ax=0)
        dvdy = self.central_difference(v, dy, ax=1)
        dvdz = self.central_difference(v, dz, ax=2)
        # dw/dx, dw/dy, dw/dz
        dwdx = self.central_difference(w, dx, ax=0)
        dwdy = self.central_difference(w, dy, ax=1)
        dwdz = self.central_difference(w, dz, ax=2)
        # omega_x = d(w)/dy - d(v)/dz
        omega_x = dwdy - dvdz
        # omega_y = d(u)/dz - d(w)/dx
        omega_y = dudz - dwdx
        # omega_z = d(v)/dx - d(u)/dy
        omega_z = dvdx - dudy
        # Enstrophy = 0.5 * (omega_x^2 + omega_y^2 + omega_z^2)
        enstrophy = 0.5 * (omega_x**2 + omega_y**2 + omega_z**2)
        
        # Strain-rate tensor S_ij = 0.5(du_i/dx_j + du_j/dx_i)
        S11 = dudx
        S22 = dvdy
        S33 = dwdz
        S12 = 0.5 * (dudy + dvdx)
        S13 = 0.5 * (dudz + dwdx)
        S23 = 0.5 * (dvdz + dwdy)
        
        Sij_sqr = (S11**2 + S22**2 + S33**2 
                + 2.0*(S12**2 + S13**2 + S23**2))
        # Then dissipation = 2 * nu * S_{ij}S_{ij} = 2 * nu * Sij_sqr
        dissipation = 2.0 * nu * Sij_sqr
        
        return enstrophy, dissipation

    def compute_joint_pdf(self, f1, f2, bins=200, range_f1=None, range_f2=None):
        values_f1 = f1.ravel()
        values_f2 = f2.ravel()
        if range_f1 is None:
            range_f1 = (values_f1.min(), 10.0)# values_f1.max())
        if range_f2 is None:
            range_f2 = (values_f2.min(), 10.0)#, values_f2.max())
        pdf, f1_edges, f2_edges = np.histogram2d(values_f1, values_f2, bins=bins, range=[range_f1, range_f2], density=True)
        f1_centers = 0.5 * (f1_edges[:-1] + f1_edges[1:])
        f2_centers = 0.5 * (f2_edges[:-1] + f2_edges[1:])
        return pdf, f1_centers, f2_centers

    def compute_pdf(self, f1, bins=200, range_f1=None, no_cutoff=False):
        values_f1 = f1.ravel()
        if range_f1 is None:
            if no_cutoff:
                range_f1=(values_f1.min(), values_f1.max())
            else:
                range_f1=(values_f1.min(), 10.0)

        pdf, f1_edges = np.histogram(values_f1, bins=bins, range=range_f1, density=True)
        f1_centers = 0.5 * (f1_edges[:-1] + f1_edges[1:])
        return pdf, f1_centers

    def visualize_jpdf_omg_eps(self, OMGtar, EPSItar, OMGpre, EPSIpre, outputname):
        # Compute joint PDF
        range_f1=(min(np.log10(OMGtar/OMGtar.mean()).min(),   np.log10(OMGpre/OMGpre.mean()).min()), 10.0)
        range_f2=(min(np.log10(EPSItar/EPSItar.mean()).min(), np.log10(EPSIpre/EPSIpre.mean()).min()), 10.0)
        pdf, f1_centers, f2_centers = self.compute_joint_pdf(np.log10(OMGtar/OMGtar.mean()), np.log10(EPSItar/EPSItar.mean()), 
                                                             range_f1=range_f1, range_f2=range_f2, bins=200)
        fig, axs = plt.subplots(1, 2, figsize=(15, 7))
        im=axs[0].contourf(f1_centers, f2_centers, pdf.T, cmap="Reds", levels=50)
        axs[0].set_xlabel(r'log10($\Omega/<\Omega>$)')
        axs[0].set_ylabel(r'log10($\varepsilon/<\varepsilon>$)')
        axs[0].set_title('True Joint PDF')
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical')

        pdfpre, f1_centers, f2_centers = self.compute_joint_pdf(np.log10(OMGpre/OMGpre.mean()), np.log10(EPSIpre/EPSIpre.mean()), 
                                                             range_f1=range_f1, range_f2=range_f2, bins=200)
        im=axs[1].contourf(f1_centers, f2_centers, pdfpre.T, cmap="Reds", levels=50)
        axs[1].set_xlabel(r'log10($\Omega/<\Omega>$)')
        axs[1].set_ylabel(r'log10($\varepsilon/<\varepsilon>$)')
        axs[1].set_title('Pred Joint PDF')
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical')
        
        plt.subplots_adjust(left=0.08, bottom=0.12, right=0.92, top=0.92, wspace=0.4, hspace=0.05)
        plt.savefig(outputname)
        plt.close()

    def visualize_jpdfcomp_omg_eps(self, OMGtar, EPSItar, OMGpre, EPSIpre, outputname):
        # Compute joint PDF
        pdf, f1_centers, f2_centers = self.compute_joint_pdf(np.log10(OMGtar/OMGtar.mean()), np.log10(OMGpre/OMGpre.mean()), bins=200)
        fig, axs = plt.subplots(1, 2, figsize=(15, 7))
      
        im=axs[0].contourf(f1_centers, f2_centers, pdf.T, cmap="Reds", levels=50)
        axs[0].set_xlabel(r'log10($\Omega/<\Omega>$) True')
        axs[0].set_ylabel(r'log10($\Omega/<\Omega>$) Pred')
        axs[0].set_title('Joint PDF')
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical')

        pdfpre, f1_centers, f2_centers = self.compute_joint_pdf(np.log10(EPSItar/EPSItar.mean()), np.log10(EPSIpre/EPSIpre.mean()), bins=200)
        im=axs[1].contourf(f1_centers, f2_centers, pdfpre.T, cmap="Reds", levels=50)
        axs[1].set_xlabel(r'True log10($\varepsilon/<\varepsilon>$)')
        axs[1].set_ylabel(r'Pred log10($\varepsilon/<\varepsilon>$)')
        axs[1].set_title('Joint PDF')
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical')
        
        plt.subplots_adjust(left=0.08, bottom=0.12, right=0.92, top=0.92, wspace=0.3, hspace=0.05)
        plt.savefig(outputname)
        plt.close()

    def visualize_pdf_omg_eps(self, OMGtar, EPSItar, OMGpre, EPSIpre, outputname):
        # Compute joint PDF
        fig, axs = plt.subplots(1, 2, figsize=(15, 7))
        pdf, f1_centers= self.compute_pdf(np.log10(OMGtar/OMGtar.mean()), bins=200)
        axs[0].plot(f1_centers, pdf,"-",label="True", linewidth=2.0)
        pdf, f1_centers= self.compute_pdf(np.log10(OMGpre/OMGpre.mean()), bins=200)
        axs[0].plot(f1_centers, pdf,"--",label="Pred", linewidth=2.0)
        axs[0].set_xlabel(r'log10($\Omega/<\Omega>$)')
        axs[0].set_ylabel('PDF')
        axs[0].legend()
        #axs[0].set_xscale("log")

        pdf, f1_centers= self.compute_pdf(np.log10(EPSItar/EPSItar.mean()), bins=200)
        axs[1].plot(f1_centers, pdf,"-",label="True", linewidth=2.0)
        pdf, f1_centers= self.compute_pdf(np.log10(EPSIpre/EPSIpre.mean()), bins=200)
        axs[1].plot(f1_centers, pdf,"--",label="Pred", linewidth=2.0)
        axs[1].set_xlabel(r'log10($\varepsilon/<\varepsilon>$)')
        axs[1].set_ylabel('PDF')
        axs[1].legend()
        #axs[1].set_xscale("log")
        
        plt.subplots_adjust(left=0.1, bottom=0.12, right=0.98, top=0.98, wspace=0.2, hspace=0.05)
        plt.savefig(outputname)
        plt.close()
    
    def visualize_pdf_omg_eps_comp(self, OMGtar, OMGpre, OMGprebase, OMGprehier, EPSItar,  EPSIpre, EPSIprebase, EPSIprehier, 
                                                      label=["base64","base32", "hierarchical8"], outputname=None):
        # Compute joint PDF
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        pdf, f1_centers= self.compute_pdf(np.log10(OMGtar/OMGtar.mean()), bins=200)
        axs[0].plot(f1_centers, pdf,"o",label="True", linewidth=2.0)
        pdf, f1_centers= self.compute_pdf(np.log10(OMGpre/OMGpre.mean()), bins=200)
        axs[0].plot(f1_centers, pdf,"g-",label=label[0], linewidth=2.0)
        pdf, f1_centers= self.compute_pdf(np.log10(OMGprebase/OMGprebase.mean()), bins=200)
        axs[0].plot(f1_centers, pdf,"b--",label=label[1], linewidth=2.0)
        pdf, f1_centers= self.compute_pdf(np.log10(OMGprehier/OMGprehier.mean()), bins=200)
        axs[0].plot(f1_centers, pdf,"r--",label=label[2], linewidth=2.0)
        axs[0].set_xlabel(r'log10($\Omega/<\Omega>$)')
        axs[0].set_ylabel('PDF')
        #axs[0].legend()
        #axs[0].set_xscale("log")

        pdf, f1_centers= self.compute_pdf(np.log10(EPSItar/EPSItar.mean()), bins=200)
        axs[1].plot(f1_centers, pdf,"o",label="True", linewidth=2.0)
        pdf, f1_centers= self.compute_pdf(np.log10(EPSIpre/EPSIpre.mean()), bins=200)
        axs[1].plot(f1_centers, pdf,"g-",label=label[0], linewidth=2.0)
        pdf, f1_centers= self.compute_pdf(np.log10(EPSIprebase/EPSIprebase.mean()), bins=200)
        axs[1].plot(f1_centers, pdf,"b--",label=label[1], linewidth=2.0)
        pdf, f1_centers= self.compute_pdf(np.log10(EPSIprehier/EPSIprehier.mean()), bins=200)
        axs[1].plot(f1_centers, pdf,"r--",label=label[2], linewidth=2.0)
        axs[1].set_xlabel(r'log10($\varepsilon/<\varepsilon>$)')
        axs[1].set_ylabel('PDF')
        axs[1].legend(fontsize=16)#20)
        #axs[1].set_xscale("log")
        
        plt.subplots_adjust(left=0.1, bottom=0.12, right=0.98, top=0.98, wspace=0.2, hspace=0.05)
        plt.savefig(outputname)
        plt.close()
    
    def TG_kineticenergies(self, rho, uvec, case="P1F2R32"):
        #from Murali
        r=rho
        u=uvec[:,:,:,0]
        v=uvec[:,:,:,1]
        w=uvec[:,:,:,2]

        E_H = 0.5 * (np.mean(u*u) + np.mean(v*v))
        E_V = 0.5 * np.mean(w*w)
        KE = E_H + E_V

        def extract_variable(filepath, variable_name):
            pattern = rf'\s*{re.escape(variable_name)}\s+([-\d.eE]+)'
            with open(filepath, 'r') as file:
                for line in file:
                    match = re.match(pattern, line)
                    if match:
                        return float(match.group(1))
            return None # If the variable isn’t found

        HDdir = "/lustre/orion/world-shared/stf006/muraligm/CFD135/data_iso/max_ent/binary_data"
        filepath = f"{HDdir}/{case}/global"
        accel = extract_variable(filepath, 'zAcceleration')
        rho0 = extract_variable(filepath, 'referenceDensity')
        drhobardz = extract_variable(filepath, 'densityGradient')

        print(f"accel: {accel}; rho0: {rho0}; drhobardz: {drhobardz}")

        b = r * (accel / rho0) 
        N = np.sqrt( -(accel / rho0) * drhobardz )
        PE = np.mean(b*b) / (2 * N**2)

        ET = KE + PE
        return KE, PE, ET
    
    def compute_3d_autocorrelation(self, field):
        field = field - np.mean(field)
        acorr = correlate(field, field, mode='full')
        acorr /= acorr.max()  # normalize
        return acorr

    def spherical_average(self, corr_3d, r_bins):
        nx, ny, nz = corr_3d.shape
        cx, cy, cz = nx//2, ny//2, nz//2

        x = np.arange(nx) - cx
        y = np.arange(ny) - cy
        z = np.arange(nz) - cz
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        R = np.sqrt(X**2 + Y**2 + Z**2)

        r_flat = R.ravel()
        c_flat = corr_3d.ravel()

        r_bin_indices = np.digitize(r_flat, r_bins)
        r_avg = []
        for i in range(1, len(r_bins)):
            values = c_flat[r_bin_indices == i]
            if values.size > 0:
                r_avg.append(np.mean(values))
            else:
                r_avg.append(0)

        r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
        return r_centers, np.array(r_avg)

    def compute_integral_length(self, field, dx, r_max_factor=0.5):
        corr = self.compute_3d_autocorrelation(field)
        nx = field.shape[0]
        r_max = r_max_factor * nx * dx
        r_bins = np.linspace(0, r_max, 100)
        r, R_r = self.spherical_average(corr, r_bins)
        R_trunc = R_r[r >0]
        r_trunc = r[r >0]
        return simpson(R_trunc, r_trunc)
    
    def extract_variable(self, filepath, variable_name):
        pattern = rf'\s*{re.escape(variable_name)}\s+([-\d.eE]+)'
        with open(filepath, 'r') as file:
            for line in file:
                match = re.match(pattern, line)
                if match:
                    return float(match.group(1))
        return None # If the variable isn’t found
    
    def calcualte_strainrates(self,u,v,w,dx,dy,dz):
        dudx = self.central_difference(u, dx, ax=0)
        dudy = self.central_difference(u, dy, ax=1)
        dudz = self.central_difference(u, dz, ax=2)
        # dv/dx, dv/dy, dv/dz
        dvdx = self.central_difference(v, dx, ax=0)
        dvdy = self.central_difference(v, dy, ax=1)
        dvdz = self.central_difference(v, dz, ax=2)
        # dw/dx, dw/dy, dw/dz
        dwdx = self.central_difference(w, dx, ax=0)
        dwdy = self.central_difference(w, dy, ax=1)
        dwdz = self.central_difference(w, dz, ax=2)
        # Strain-rate tensor S_ij = 0.5(du_i/dx_j + du_j/dx_i)
        S11 = dudx
        S22 = dvdy
        S33 = dwdz
        S12 = 0.5 * (dudy + dvdx)
        S13 = 0.5 * (dudz + dwdx)
        S23 = 0.5 * (dvdz + dwdy)
        
        Sij2 = (S11**2 + S22**2 + S33**2 + 2.0*(S12**2 + S13**2 + S23**2))
        return Sij2


    def integral_length_1d(self, field, spacing, axis):
        f = np.moveaxis(field, axis, 0)
        n, ny, nz = f.shape
        f -= f.mean()
        R_sum = np.zeros(n)
        n_planes = 0
        # Loop over planes perpendicular to the correlation direction
        for j in range(ny):
            for k in range(nz):
                sig = f[:, j, k]
                var = sig.var()
                if var == 0.0:
                    continue
                # FFT‑based auto‑correlation
                R_full = correlate(sig, sig, mode="full", method="fft")
                R = R_full[n-1:] / (var * n)   # keep Δ≥0, normalised
                R_sum += R[:n]                 # truncate to lag n‑1
                n_planes += 1

        R_mean = R_sum / n_planes

        # first zero crossing (or end of array if none)
        zc = np.argmax(R_mean <= 0) if np.any(R_mean <= 0) else n-1

        lags = np.arange(zc + 1) * spacing
        return float(simpson(R_mean[:zc + 1], lags))  

    def TG_nondimensional_numbers(self, rho, uvec, case="P1F2R32"):
        """
        Gn = epsilon / (nu N^2).  
        epsilon=dgs in mestats.  
        nu and N^2 are constants
        Frh= u/(NL).  
        Use u=urms and L=intls from vstats. 
        """
        #R32: 3200; Re64:6400; Re96: 9600
        Relib={}
        r=rho
        u=uvec[:,:,:,0]
        v=uvec[:,:,:,1]
        w=uvec[:,:,:,2]
        nx,ny,nz=u.shape

        HDdir = "/lustre/orion/world-shared/stf006/muraligm/CFD135/data_iso/max_ent/binary_data"
        filepath = f"{HDdir}/{case}/global"
        accel = self.extract_variable(filepath, 'zAcceleration')
        rho0 = self.extract_variable(filepath, 'referenceDensity')
        drhobardz = self.extract_variable(filepath, 'densityGradient')
        nu = self.extract_variable(filepath, 'kinematicViscosity')
        lx = self.extract_variable(filepath, 'xDomainSize')
        ly = self.extract_variable(filepath, 'yDomainSize')
        lz = self.extract_variable(filepath, 'zDomainSize')
        Re = int(case[-2:])*100

        print(f"Re: {Re};accel: {accel}; rho0: {rho0}; drhobardz: {drhobardz}; nu: {nu}; lx: {lx}; ly: {ly}; lz: {lz}: [nx,ny,nz]: {nx, ny, nz}")
 
        N = np.sqrt( -(accel / rho0) * drhobardz)
        #uhrms =  ((np.mean(u*u)-np.mean(u)**2 + np.mean(v*v)-np.mean(v)**2)/2.0)**0.5
        uhrms =  ((np.mean(u*u) + np.mean(v*v))/2.0)**0.5
        urms =  ((np.mean(u*u)+ np.mean(v*v) + np.mean(w*w))/3.0)**0.5

        dx=lx/nx; dy=ly/ny; dz=lz/nz        
        Sij2=self.calcualte_strainrates(u,v,w,dx,dy,dz)
        #dissipation = 2.0 * nu*np.mean(Sij2) #?
        #in paper: calulated from "np.mean(Sij2)/Re"; what is Re?
        dissipation = 2.0*np.mean(Sij2)/Re
        Gn = dissipation/(nu*N**2)     

        #L_u = self.compute_integral_length(u, dx)
        #L_v = self.compute_integral_length(v, dy)
        #Lu = self.integral_length_1d(u, dx, axis=0)   # correlate along x
        #Lv = self.integral_length_1d(v, dy, axis=1)
        #Lh = (Lu + Lv) / 2# 

        #Frh= 2*np.pi*uhrms/(N*Lh) 
        #calculate Frt, instead
        #Lt = urms**3 / dissipation
        Frt= dissipation/N/urms**2
        #2*np.pi*urms/(N*Lt)  #rm 2pi
        #epsilon/N/urms**2
        #Lt from epsi and urms:  Lt = urms**3 / dissipation
        return Gn, Frt
        #return Gn, Frh


       

        