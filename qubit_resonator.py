import numpy as np
import qutip as qt

class QubitResonator:
    def __init__(self, wc_ghz=6.0, wq_ghz=5.2, g_mhz=100, Q=2000, N=10, t1_ns=None, t2_ns=None):
        self.wc = wc_ghz * 2 * np.pi
        self.wq = wq_ghz * 2 * np.pi
        self.g  = (g_mhz / 1000.0) * 2 * np.pi
        self.kappa = self.wc / Q
        self.N = N
        
        delta = self.wq - self.wc
        self.chi = (self.g ** 2) / delta
        
        self.a  = qt.tensor(qt.qeye(2), qt.destroy(N))
        self.n  = self.a.dag() * self.a
        
        self.sm = qt.tensor(qt.sigmam(), qt.qeye(N))
        self.sp = qt.tensor(qt.sigmap(), qt.qeye(N))
        self.sx = qt.tensor(qt.sigmax(), qt.qeye(N))
        self.sy = qt.tensor(qt.sigmay(), qt.qeye(N))
        self.sz = qt.tensor(qt.sigmaz(), qt.qeye(N))
        
        self.H0_jc = (self.wc * self.n) + (0.5 * self.wq * self.sz) + (self.g * (self.a.dag() * self.sm + self.a * self.sp))
        
        self.c_ops = []
        self.c_ops.append(np.sqrt(self.kappa) * self.a)
        
        if t1_ns is not None:
            self.gamma_1 = 1.0 / t1_ns
            self.c_ops.append(np.sqrt(self.gamma_1) * self.sm)
        else:
            self.gamma_1 = 0
            
        if t2_ns is not None:
            gamma_total = 1.0 / t2_ns
            gamma_pure = gamma_total - (self.gamma_1 / 2.0)
            gamma_pure = max(0.0, gamma_pure) 
            if gamma_pure > 0:
                self.c_ops.append(np.sqrt(gamma_pure / 2.0) * self.sz)

        p_ref_dbm = -100.0
        n_ref_photons = 0.1
        epsilon_ref = self.kappa * np.sqrt(n_ref_photons) / 2.0
        p_ref_watts = 10**(p_ref_dbm/10.0) / 1000.0
        self.conversion_factor = epsilon_ref / np.sqrt(p_ref_watts / self.wc)

    def _dbm_to_amp(self, power_dbm, freq_radns):
        p_watts = 10**(power_dbm / 10.0) / 1000.0
        return self.conversion_factor * np.sqrt(p_watts / freq_radns)

    def response(self, readout_freq_ghz, readout_dbm, qubit_drive_freq_ghz=None, qubit_drive_dbm=None):
        w_r = readout_freq_ghz * 2 * np.pi
        amp_r = self._dbm_to_amp(readout_dbm, w_r)
        d_c = self.wc - w_r
        
        if qubit_drive_freq_ghz is None:
            d_q = self.wq - w_r

            H = (d_c * self.n) + (0.5 * d_q * self.sz) + (self.g * (self.a.dag() * self.sm + self.a * self.sp))
            H += amp_r * (self.a + self.a.dag())
        else:
            w_q = qubit_drive_freq_ghz * 2 * np.pi
            amp_q = self._dbm_to_amp(qubit_drive_dbm, w_q)
            d_q = self.wq - w_q
            
            H = (d_c * self.n) + (self.chi * self.n * self.sz) + (0.5 * d_q * self.sz)
            H += amp_r * (self.a + self.a.dag())
            H += amp_q * self.sx

        rho_ss = qt.steadystate(H, self.c_ops)
        a_avg = qt.expect(self.a, rho_ss)
        n_avg = qt.expect(self.n, rho_ss)
        
        s21 = 1.0 - 1j * (self.kappa * a_avg / (2.0 * amp_r))
        return s21, n_avg

    def evolve(self, tlist, i_signal=None, q_signal=None):
        H_control = []
        if i_signal is not None: H_control.append([self.sx, i_signal])
        if q_signal is not None: H_control.append([self.sy, q_signal]) 
        
        H_total = [((self.wc - self.wq) * self.n) + (self.g * (self.a.dag() * self.sm + self.a * self.sp))] + H_control
        
        psi0 = qt.tensor(qt.basis(2, 0), qt.basis(self.N, 0))

        result = qt.mesolve(H_total, psi0, tlist, c_ops=self.c_ops, 
                            e_ops=[self.n, self.sz])
        return result
