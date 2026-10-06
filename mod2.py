# Conservative FVM
import numpy as np
import matplotlib.pyplot as plt
import keyboard

class ForwardTimeNumerical:
    def __init__(self, u0, lambda_, timestep_length):
        self.u0 = u0
        self.lambda_ = lambda_
        self.timestep_length = timestep_length
        self.N = len(u0)
        self.u_old = u0.copy()
        self.u_solutions = np.zeros((self.N, timestep_length))

    def run_ftbs(self):
        for i in range(self.timestep_length):
            self.u_solutions[:, i] = self.u_old
            u_new = self.ftbsPeriodicBoundary(self.u_old, self.lambda_)
            self.u_old = u_new
        return self.u_solutions
    def run_ftfs(self):
        for i in range(self.timestep_length):
            self.u_solutions[:, i] = self.u_old
            u_new = self.ftfsPeriodicBoundary(self.u_old, self.lambda_)
            self.u_old = u_new
        return self.u_solutions

    def run_ftcs(self):
        for i in range(self.timestep_length):
            self.u_solutions[:, i] = self.u_old
            u_new = self.ftcsPeriodicBoundary(self.u_old, self.lambda_)
            self.u_old = u_new
        return self.u_solutions

    def ftbsPeriodicBoundary(self, ui_n, lambda_):
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
            ui_n_next[unit] = ui_n[unit] - lambda_ * (flux_plus - flux_minus) # Forward Time.

        return ui_n_next

    def ftfsPeriodicBoundary(self,ui_n, lambda_):
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
            ui_n_next[unit] = ui_n[unit] - lambda_ * (flux_plus - flux_minus) # Forward Time

        return ui_n_next

    def ftcsPeriodicBoundary(self,ui_n, lambda_):
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
            ui_n_next[unit] = ui_n[unit] - lambda_ * (flux_plus - flux_minus) # Forward Time

        return ui_n_next


#----------------------------------------------------------  Definition/ Memory Allocation
#u0 = np.array([-1.0, 1.0, 1.0, -1.0])
#x = np.array([0.0, 1.0, 2.0, 3.0])
x = np.linspace(-1, 1, 21)[0:-1]
u0 = np.sin(np.pi*x)
lambda_ = 0.8 # define lambda

print("Initial condition:", len(u0))

# pre allocate memory for the solution at each time step
timestep_length = int(1e3)
u_old = u0.copy()
N = len(u0)
u_solutions = np.zeros((N, timestep_length))

#---------------------------------------------------------- PLOTTING/SOLUTION
fig, ax = plt.subplots()
line, = ax.plot(x, u0, marker='o')
plt.grid()
plt.xlabel('x')
plt.ylabel('u')
plt.title('FT')

u_solutions = ForwardTimeNumerical(u0, lambda_, timestep_length).run_ftbs() 

for i in range(timestep_length):
    line.set_ydata(u_solutions[:, i])
    fig.canvas.draw_idle()
    plt.pause(1e-16)
    if keyboard.is_pressed(49): # Pause
        print('Pausing...')
        while True:
            keyboard.wait(49)  
            plt.pause(0.1)
            break 
    elif keyboard.is_pressed(12):  # End
        print('Ending...')
        break


input("Press Enter to exit...")
plt.close()

#----------------------------------------------------------

