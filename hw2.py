from cfdNumerical import ForwardTimeNumerical
import numpy as np
import matplotlib.pyplot as plt
import keyboard

if __name__ == "__main__":
    
    lambda_ = 0.8 # define lambda dt/dx
    a = 1.0
    nx = 40
    x = np.linspace(0, 2, nx) # define the spatial grid
    u0 = np.sin(np.pi*x)
    # reconstruct time
    dx = x[1] - x[0]
    dt = lambda_ * dx

    print("Initial condition:", dx)

    # pre allocate memory for the solution at each time step
    total_time = 5.0
    u_old = u0.copy()
    N = len(u0)
    timestep_length = int(np.ceil(total_time / dt))
    final_time = timestep_length * dt
    print("Total time:", total_time, "Final time:", final_time, "Time step length:", timestep_length)

    # create solver
    solver = ForwardTimeNumerical(u0, lambda_, a,timestep_length)
    u_solutions = solver.run_btcs()  

    # plot results
    fig, ax = plt.subplots()
    line, = ax.plot(x,u0, marker = 'o')

    ax.grid()
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title('FTBS solution')

    output_interval = 0.5
    stoptime = 0
    for i in range(timestep_length):

        current_time = i*dt
        print('Current time:', current_time)
        error = abs(current_time - stoptime)
        print('Error:', error)
        if i == 0 or error < 1e-1:
            stoptime += output_interval
            line.set_ydata(u_solutions[:,i])
            ax.set_title(f'FTBS at t = {current_time:.2f} s')

            fig.canvas.draw()
            plt.pause(0.5)
    plt.show()

input("Press Enter to exit...")
plt.close()

    