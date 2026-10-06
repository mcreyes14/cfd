
import numpy as np
class ForwardTimeNumerical:
    def __init__(self, u0, lambda_, a,timestep_length):
        self.u0 = u0
        self.a = a
        self.lambda_ = lambda_
        self.timestep_length = timestep_length
        self.N = len(u0)
        self.u_old = u0.copy()
        self.u_solutions = np.zeros((self.N, timestep_length))

    def run_ftbs(self):
        for i in range(self.timestep_length):
            self.u_solutions[:, i] = self.u_old
            u_new = self.ftbsPeriodicBoundary(self.u_old, self.lambda_, self.a)
            self.u_old = u_new
        return self.u_solutions
    def run_ftfs(self):
        for i in range(self.timestep_length):
            self.u_solutions[:, i] = self.u_old
            u_new = self.ftfsPeriodicBoundary(self.u_old, self.lambda_, self.a)
            self.u_old = u_new
        return self.u_solutions

    def run_ftcs(self):
        for i in range(self.timestep_length):
            self.u_solutions[:, i] = self.u_old
            u_new = self.ftcsPeriodicBoundary(self.u_old, self.lambda_, self.a)
            self.u_old = u_new
        return self.u_solutions

    def run_laxWendroff(self):
        for i in range(self.timestep_length):
            self.u_solutions[:, i] = self.u_old
            u_new = self.laxWendroff(self.u_old, self.lambda_, self.a)
            self.u_old = u_new
        return self.u_solutions

    def run_btcs(self):
        for i in range(self.timestep_length):
            self.u_solutions[:, i] = self.u_old
            u_new = self.btcsPeriodicBoundary(self.u_old, self.lambda_, self.a)
            self.u_old = u_new
        return self.u_solutions

    def ftbsPeriodicBoundary(self, ui_n, lambda_,a):
        ui_n_next = np.copy(ui_n)
        N = len(ui_n)
        for unit in range(0,N):  # FTBS
            # Obtain averaged state at cell.
            ui = ui_n[unit]
            ui_minus = ui_n[unit - 1] # At i = 0, ui_minus = ui_n[-1] = ui_n[N]. Apply periodic boundary condition at the left boundary
            # flux function f(u). Linear advection equation: f(u) = u
            flux_minus = ui_minus
            flux_plus = ui
            
            # Update the solution using the conservative FVM formula
            ui_n_next[unit] = ui_n[unit] - lambda_ * a*(flux_plus - flux_minus) # Forward Time.

        return ui_n_next

    def ftfsPeriodicBoundary(self,ui_n, lambda_,a):
        ui_n_next = np.copy(ui_n)
        N = len(ui_n)
        for unit in range(0,N):  # FTFS
            # Obtain averaged state at cell.
            ui = ui_n[unit]
            if unit < N - 1:            
                ui_plus = ui_n[unit+1]
            else: # At i = N-1, ui_plus = ui_n[0]. Apply periodic boundary condition at the right boundary
                ui_plus = ui_n[0]
            # flux function f(u). Linear advection equation: f(u) = u
            flux_minus = ui
            flux_plus = ui_plus

            # flux function f(u). Linear advection equation: f(u) = u
            ui_n_next[unit] = ui_n[unit] - lambda_ * a*(flux_plus - flux_minus) # Forward Time

        return ui_n_next

    def ftcsPeriodicBoundary(self,ui_n, lambda_,a):
        ui_n_next = np.copy(ui_n)
        N = len(ui_n)
        for unit in range(0, N):  # FTCS
            # Obtain averaged state at cell.
            ui = ui_n[unit]
            ui_minus = ui_n[unit - 1] # At i = 0, ui_minus = ui_n[-1] = ui_n[N]. Apply periodic boundary condition at the left boundary
            if unit < N - 1:            
                ui_plus = ui_n[unit+1]
            else:
                ui_plus = ui_n[0]  # At i = N-1, ui_plus = ui_n[0]. Apply periodic boundary condition at the right boundary
            # flux function f(u)
            flux_minus = (ui+ui_minus)/2
            flux_plus = (ui_plus+ui)/2
            
            # flux function f(u). Linear advection equation: f(u) = u
            ui_n_next[unit] = ui_n[unit] - lambda_ * a*(flux_plus - flux_minus) # Forward Time

        return ui_n_next

    def laxWendroff(self,ui_n,lambda_,a):
        # linear advection equation: f(u) = u
        ui_n_next = np.copy(ui_n)
        N = len(ui_n)
        for unit in range(0, N):  # FTCS
            # Obtain averaged state at cell.
            ui = ui_n[unit]
            ui_minus = ui_n[unit - 1] # At i = 0, ui_minus = ui_n[-1] = ui_n[N]. Apply periodic boundary condition at the left boundary
            if unit < N - 1:            
                ui_plus = ui_n[unit+1]
            else:
                ui_plus = ui_n[0]  # At i = N-1, ui_plus = ui_n[0]. Apply periodic boundary condition at the right boundary
            # flux function f(u)
            flux_im = ui_minus
            flux_ip = ui_plus
            flux_i = ui
            
            # flux function f(u). Linear advection equation: f(u) = u
            ui_n_next[unit] = ui_n[unit] - 0.5*lambda_*(flux_ip - flux_im) 
            + 0.5 * (lambda_**2) * (a) * (flux_ip - 2*flux_i + flux_im) # Lax-Wendroff scheme

    def btcsPeriodicBoundary(self,ui_n, lambda_,a):
        ui_n_next = np.copy(ui_n)
        N = len(ui_n)

        M_ones = np.diag(np.full(N,1))
        M_r = np.diag(np.ones(N-1),1)*lambda_/2*a
        M_l = np.diag(np.ones(N-1),-1)*(-lambda_/2)*a
        Mfull = M_ones + M_r + M_l
        Mfull[0,N-1] = -lambda_/2
        Mfull[N-1,0] = lambda_/2
        Mfullinv = np.linalg.inv(Mfull)
        ui_n_next = Mfullinv @ ui_n

        return ui_n_next