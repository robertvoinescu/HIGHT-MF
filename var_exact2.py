import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from scipy import optimize as opt

def rotMatrix(theta,u):
    # rotate about arbitrary vector u
    from numpy import cos, sin
    u = u/LA.norm(u)
    ux,uy,uz = u
    mat = np.array([[cos(theta)+ux**2*(1-cos(theta)), ux*uy*(1-cos(theta))-u[2]*sin(theta), ux*uz*(1-cos(theta))+uy*sin(theta)],
                    [uy*ux*(1-cos(theta))+uz*sin(theta)       , cos(theta)+uy**2*(1-cos(theta)),   uy*uz*(1-cos(theta))-ux*sin(theta)],           
                    [uz*ux*(1-cos(theta))-uy*sin(theta), uz*uy*(1-cos(theta))+ux*sin(theta) , cos(theta)+uz**2*(1-cos(theta))]])
    return mat



def av_chi(chi,n_sites,coord):
    Total = 0
    for i in range(n_sites):
        Total += chi[coord+3*i,coord]
    return Total/n_sites




def genSus(coupling_dict,gtensor_dict,n_sites,beta):
    # need to solve: gvec = M*chi_vec

    assert len(gtensor_dict) == n_sites, 'not enough gtensors'

    M = np.zeros((3*n_sites,3*n_sites))
    # model Spin Hamiltonian couplings
    for i in range(n_sites):
        for j in range(n_sites): 
            if (i,j) in coupling_dict :
                M[3*i:3*(i+1),3*j:3*(j+1)] += coupling_dict[(i,j)]

    # self term
    for i in range(n_sites):
        M[3*i:3*(i+1),3*i:3*(i+1)] = 3/beta * np.eye(3)


    
    gvec = np.zeros((3*n_sites,3))
    for i in range(n_sites):
        if i in gtensor_dict:
            gvec[3*i:3*(i+1),0:3] += gtensor_dict[i]

    chi_vec = np.matmul(LA.inv(M),gvec)
    return chi_vec



kb = 1
T_sample = 1e6          # approximation only good for T>>high H_eff \approx n_site*max_coupling = 10*1e3 = 1e4
temp_domain = np.arange(T_sample-500,T_sample,10)

gtensor_dict = { 0 : np.eye(3),
                 1 : np.eye(3),
                 2 : np.eye(3),
                 3 : np.eye(3),
                 4 : np.eye(3),
                 5 : np.eye(3),
                 6 : np.eye(3),
                 7 : np.eye(3),
                 8 : np.eye(3),
                 9 : np.eye(3),
                 10: np.eye(3),
                 11: np.eye(3)}


# couplings in units of temperature
j1 = 1500*np.eye(3)
g1 = np.array([[-400,-800,300],[0,700,-50],[0,0,400-700]])
g1 = g1+g1.T
d1 = np.array([[0,-40,100],[0,0,-100],[0,0,0]])
d = d1-d1.T
# isotropic in this shceme has a *negative* sign
J1 = -j1+g1+d1


j2 = 500*np.eye(3)
g2 = np.array([[10,2,-10],[0,10,5],[0,0,-10-10]])
g2 = g2+g2.T
d2 = np.array([[0,-40,20],[0,0,80],[0,0,0]])
d2 = d2-d2.T
# isotropic in this scheme has a *negative* sign
J2 = -j2+g2+d2


# rotate one of the couplings
a0 = np.array([-1.51,-2,-.68])
a1 = np.array([-3.26,-2.02,-1.53])
u = a1-a0
J3 = np.einsum('ij,jk,kl',rotMatrix(-np.pi/2,u),J2,LA.inv(rotMatrix(-np.pi/2,u)))


coupling_dict = {( 0, 1) : J1,
                 ( 0, 8) : J3,
                 ( 0, 11) : J2,
                 ( 1, 2) : J1,
                 ( 2, 6) : J2,
                 ( 2, 9) : J3,
                 ( 3, 4) : J1,
                 ( 3, 8) : J2,
                 ( 3, 11) : J3,
                 ( 4, 5) : J1,
                 ( 5, 6) : J3,
                 ( 5, 9) : J2,
                 ( 6, 7) : J1,
                 ( 7, 8) : J1,
                 ( 9, 10) : J1,
                 ( 10, 11) : J1}




n_sites = 12


# Average Chi_ii over all sites
chi_sumx = np.array([av_chi(genSus(coupling_dict, gtensor_dict,n_sites,1/T/kb),n_sites,0) for T in temp_domain])
chi_sumy = np.array([av_chi(genSus(coupling_dict, gtensor_dict,n_sites,1/T/kb),n_sites,1) for T in temp_domain])
chi_sumz = np.array([av_chi(genSus(coupling_dict, gtensor_dict,n_sites,1/T/kb),n_sites,2) for T in temp_domain])

chi_sumx = [1/chi for chi in chi_sumx]
chi_sumy = [1/chi for chi in chi_sumy]
chi_sumz = [1/chi for chi in chi_sumz]

        
plt.style.use('ggplot')
fig,axs = plt.subplots(1,1)
axs.set_title('Analytic Approximation $1/\chi$')
axs.set(xlabel="Temperature (K) ")

from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(temp_domain,chi_sumx)
axs.plot(temp_domain,chi_sumx,label='$\\frac{1}{\chi_{xx}}$: $\\theta_x=$'+str(np.round(intercept/slope,4)))
axs.plot(temp_domain,temp_domain*slope+intercept,linestyle='--',color='black')

slope, intercept, r_value, p_value, std_err = stats.linregress(temp_domain,chi_sumy)
axs.plot(temp_domain,chi_sumy,label='$\\frac{1}{\chi_{yy}}$: $\\theta_y=$'+str(np.round(intercept/slope,4)))
axs.plot(temp_domain,temp_domain*slope+intercept,linestyle='--',color='black')

slope, intercept, r_value, p_value, std_err = stats.linregress(temp_domain,chi_sumz)
axs.plot(temp_domain,chi_sumz,label='$\\frac{1}{\chi_{zz}}$: $\\theta_z=$'+str(np.round(intercept/slope,4)))
axs.plot(temp_domain,temp_domain*slope+intercept,linestyle='--',color='black')


axs.legend()
plt.tight_layout()
plt.show()
