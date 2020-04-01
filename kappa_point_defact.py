import numpy as np
import phonopy.file_IO as file_IO
from phonopy.structure.atoms import Atoms as PhonopyAtoms
from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp
from phonopy.structure.grid_points import get_qpoints
from phonopy.phonon.thermal_properties import mode_cv
from phonopy.units import THzToEv
from phonopy.units import *
from phonopy.phonon.dos import Dos
from phonopy.phonon.group_velocity import GroupVelocity
from constants import *

class thermal_conductivity():
    def __init__(self,primitive_cell,super_cell):
        self.primitive_cell = primitive_cell;
        self.super_cell = super_cell;

    def input_files(self,path_POSCAR,path_FORCECONSTANT):
        bulk = read_vasp(path_POSCAR)
        self.phonon = Phonopy(bulk,self.super_cell, primitive_matrix = self.primitive_cell)
        force_constants = file_IO.parse_FORCE_CONSTANTS(path_FORCECONSTANT)
        self.phonon.set_force_constants(force_constants)
        return self.phonon

    def set_mesh(self,mesh):
        self.phonon.set_mesh(mesh, is_eigenvectors=True)

    def set_point_defect_parameter(self,fi,defect_mass,defect_fc_ratio):

        self.mass = self.phonon.get_primitive().get_masses().sum()/self.phonon.get_primitive().get_number_of_atoms()
        self.cellvol = self.phonon.get_primitive().get_volume()
        self.mass_fc_factor = self.cellvol*10**-30/2.0/4/np.pi*(fi*((self.mass-defect_mass)/self.mass+2*defect_fc_ratio)**2)

    def kappa_separate(self,gamma, thetaD, T, P= 1, m =3, n =1):
        """
        nkpts: number of phonon k-points
        frequencies: list of eigenvalue-arrays as output by phonopy
        vgroup: group velocities
        T: temperature in K

        following parameters from Eq (70) in Tritt book, p.14
        P: proportionality factor of Umklapp scattering rate, defualt 1
        mass: mass in amu (average mass of unitcell)
        gamma: Gruneisen parameter, unitless
        thetaD: Debye temperature of relevance (acoustic modes) 283K for Mo3Sb7
        m = 3 by default
        n = 1 by default
        cellvol: unit cell volume in A^3
        """

        T=float(T)
        #self.phonon.set_mesh(mesh, is_eigenvectors=True)
        #self.mass = self.phonon.get_primitive().get_masses().sum()/self.phonon.get_primitive().get_number_of_atoms()
        self.cellvol = self.phonon.get_primitive().get_volume()
        self.qpoints, self.weigths, self.frequencies, self.eigvecs = self.phonon.get_mesh()

        nkpts = self.frequencies.shape[0]
        numbranches = self.frequencies.shape[1]

        self.group_velocity = np.zeros((nkpts,numbranches,3))
        for i in range(len(self.qpoints)):
            self.group_velocity[i,:,:] = self.phonon.get_group_velocity_at_q(self.qpoints[i])

        inv_nor_Um = np.zeros((nkpts,numbranches))
        inv_point_defect_mass_fc = np.zeros((nkpts,numbranches))
        inv_tau_total = np.zeros((nkpts,numbranches))

        kappa_Um = np.zeros((nkpts,numbranches))
        kappa_point_defect_mass_fc = np.zeros((nkpts,numbranches))
        kappa_total = np.zeros((nkpts,numbranches))
        self.kappaten = np.zeros((nkpts,numbranches,3,3))

        v_group = np.zeros((nkpts,numbranches))
        capacity = np.zeros((nkpts,numbranches))

        for i in range(nkpts):
            for j in range(numbranches):
                nu = self.frequencies[i][j]  # frequency (in Thz) of current mode
                velocity = self.group_velocity[i,j,:]  # cartesian vector of group velocity
                v_group = (velocity[0]**2 + velocity[1]**2 + velocity[2]**2)**(0.5) * THztoA
                capacity[i,j] = mode_cv(T, nu * THzToEv) * eV2Joule/(self.cellvol * Ang2meter**3)

                # tau inverse phonon-phonon, inverse lifetime
                inv_nor_Um[i,j] = P * hbar * gamma**2 / (self.mass*amu) / v_group**2  * (T**n)/thetaD * (2 * np.pi * THz*nu)**2 * np.exp((-1.0)*thetaD / m / T)

                # tau inverse point defect, inverse lifetime
                inv_point_defect_mass_fc[i,j] = (self.mass_fc_factor) * (2*np.pi*THz*nu)**4 / v_group**3

                # tau inverst, inverse lifetime
                inv_tau_total[i,j] = inv_nor_Um[i,j] + inv_point_defect_mass_fc[i,j]

                kappa_total[i,j] = (1.0/3.0)* self.weigths[i] * v_group**2 * capacity[i,j] / inv_tau_total[i,j] # 1/3 is for isotropic condition
                kappa_Um[i,j] = (1.0/3.0)* self.weigths[i] * v_group**2 * capacity[i,j] / inv_nor_Um[i,j] # 1/3 is for isotropic condition
                kappa_point_defect_mass_fc[i,j] = (1.0/3.0)* self.weigths[i] * v_group**2 * capacity[i,j] / inv_point_defect_mass_fc[i,j] # 1/3 is for isotropic condition
                self.kappaten[i,j,:,:] = self.weigths[i] * np.outer(velocity, velocity) * THztoA**2 * capacity[i,j] / inv_tau_total[i,j]
        #print(np.sum(inv_nor_Um, axis =1))
        #print(np.sum(inv_point_defect_mass_fc, axis =1))
        #print(np.sum(inv_tau_total, axis =1))
        self.tau = np.concatenate((inv_nor_Um,inv_point_defect_mass_fc,inv_tau_total),axis=1)
        self.kappa = [kappa_Um.sum()/self.weigths.sum(),kappa_point_defect_mass_fc.sum()/self.weigths.sum(),kappa_total.sum()/self.weigths.sum()]
        return self.tau, self.kappa, self.kappaten

def main():
    primitive_cell = np.array([[1,0,0],[0,1,0],[0,0,1]])
    super_cell = np.array([[2,0,0],[0,2,0],[0,0,5]])
    mesh = np.array([30,30,30])
    calculate_kappa = thermal_conductivity(primitive_cell,super_cell)

    path_POSCAR = 'POSCAR'
    path_FORCECONSTANT = 'FORCE_CONSTANTS'
    calculate_kappa.input_files(path_POSCAR,path_FORCECONSTANT)
    calculate_kappa.set_mesh(mesh)

    # mass is average atomic mass # need to calculate the debye temperature and
    gamma = 2.5
    thetaD = 200
    # need to check the mass factor # need to check the CV
    mass_w = 183.84
    defect_fc_ratio = -0.8659
    fraction = 0.01
    calculate_kappa.set_point_defect_parameter(fraction,mass_w,defect_fc_ratio)
    kappa_T =[]
    for T in np.arange(250,450,20,dtype=None):
        tau, kappa, kappaten = calculate_kappa.kappa_separate(gamma,thetaD,T, P= 1, m =3, n =1)
        kappa_T.append([T,kappa[0],kappa[2]])
    np.savetxt('kappa_t.txt',kappa_T,fmt = '%4.4f')

if __name__ == "__main__":
    main()
