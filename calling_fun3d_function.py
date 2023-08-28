import numpy as np




################################################################################
''' CALLING FUN3D '''

def fun3d_simulation_normx(x, out_name, subfolder):
    """ Modify input deck, run fun3d simulation, and read output deck to get results """
    
    import os
    import subprocess

    ############################# INPUT VARIABLES
    #            Mach, AoA, Altitude
    # 5
    ub = np.array([4.0,  8., 50_000.]) #6., 10., 25_000.
    lb = np.array([1.2, -5., 0.]) #-2.
    #ub = np.array([6.0,  10., 50_000.]) #6., 10., 25_000.
    #lb = np.array([1.2, -5., 0.]) #-2.

    x = x*(ub-lb) + lb

    ############################# FUNCTIONS FOR READING AND WRITING FILES

    from skaero.atmosphere import coesa
    from scipy import interpolate

    def edit_input(x):
        """ Edit fun3d.nml file """
        # unpack input x
        x = x.flatten() # only a single point
        x_mach = x[0]
        x_aoa = x[1]
        x_height = x[2]
        x_yaw = 0.0

        ###### extract atmospheric properties

        #[T, a, __, Rho] = atmoscoesa(x_height);
        h, T, p, rho = coesa.table(x_height)
        
        # Get speed of sound
        # Properties table: 
        # https://www.cambridge.org/us/files/9513/6697/5546/Appendix_E.pdf
        # Alternative calculation: https://www.grc.nasa.gov/www/BGH/sound.html
        t_table = np.array([200., 220., 240., 260., 280., 300., 320., 340.])
        cv_table = np.array([0.7153, 0.7155, 0.7158, 0.7162, 0.7168, 0.7177, 0.7188, 0.7202])
        cp_table = np.array([1.002, 1.003, 1.003, 1.003, 1.004, 1.005, 1.006, 1.007])
        gamma_table = cp_table/cv_table    
        
        interpolation_class = interpolate.interp1d(t_table, gamma_table, fill_value="extrapolate")
        
        gamma = interpolation_class(T)
        
        # # calorically perfect air:    
        # gamma_perfect = 1.4
        # a = np.sqrt(R*T*gamma_perfect)
        
        # calorically imperfect air:
        R = 286 # m^2/s^2/K
        a = np.sqrt(R*T*gamma)
        vel = a*x_mach
        
        # viscosity from Sutherland's Formula
        # (https://www.grc.nasa.gov/WWW/K-12/airplane/viscosity.html)
        S = 198.72/1.8 # R to K
        T0 = 518.7/1.8 # R to K
        mu0 = 3.62e-7 * 4.448222 * 1/0.3048**2 #lb-s/ft^2 to N-s/m^2 (est 1.716e-5)
        mu = mu0*(T/T0)**1.5*(T0+S)/(T+S);
        Len = 4.47 # Aircraft length (meters)
        Re = Len*vel*rho/mu
        

        # # Write variables
        # T       temperature = 221.65
        # rho     density = 0.039466     ! kg/m^3
        # Re      reynolds_number = 19932640.6964
        # x_mach  mach_number     = 0.95
        # vel     velocity = 283.5323    ! m/s
        # x_aoa   angle_of_attack = 10   ! degrees
        # none    angle_of_yaw = 0.0     ! degrees
        
        ###### Write atmospheric properties to file
        in_name = 'fun3d.txt'
        file1 = open(in_name, "r") # in_file
        lines = file1.readlines()
        file1.close()
        
        write_name = 'fun3d.nml';
        file2 = open(write_name, 'w') # out_file

        for line in lines:
            if 'mach_number' in line:
                file2.write('  mach_number     = '+str(x_mach)+'\n')
            elif 'angle_of_attack' in line:
                file2.write('  angle_of_attack = '+str(x_aoa)+'\n')
            elif 'angle_of_yaw' in line:
                file2.write('  angle_of_yaw = '+str(x_yaw)+'\n')
            elif 'density' in line:
                file2.write('  density = '+str(rho)+'\n') 
            elif 'temperature' in line and '_units' not in line:
                file2.write('  temperature = '+str(T)+'\n')
            elif 'velocity' in line:
                file2.write('  velocity = '+str(vel)+'\n') 
            elif 'reynolds_number' in line:
                file2.write('  reynolds_number = '+str(Re)+'\n') 
            else:
                file2.write(line)
        file2.close()
        return(None)


    import time
    from os.path import exists

    def read_output(out_name):
        # check output exists
        out_file = out_name+'_hist.dat'
        while True:
            output_exists = exists(out_file)
            if output_exists:
                print('------------- CFD simulation done ! ----------------')
                time.sleep(1)
                break
            else:
                print("No output deck found")
                time.sleep(0.1)


        # extract results
        file1 = open(out_file, "r")
        tline = file1.readline()
        tline = file1.readline()
        if 'R_6' in tline:
            cl_idx = 7
            cd_idx = 8
        else: # inviscid
            cl_idx = 6
            cd_idx = 7
       
        lines = file1.readlines()
        last_line = lines[-1]
        last_line = last_line.split(' ')
        last_line = [line for line in last_line if len(line.strip())>0] # remove elements with only spaces
        last_line = list(map(float, last_line))
        CL = last_line[cl_idx]
        CD = last_line[cd_idx]
        return(CL, CD)


    ############################# MODIFY FILE

    # subfolder = "GHV_494k_v" #"GHV_34k_v" #
    os.chdir(subfolder)
    edit_input(x)


    ############################# RUN MODIFIED FILE

    # path = os.getcwd()
    # print("path: ", path)

    # run fun3d
    subprocess.run(["nodet_mpi"])


    ############################# EXTRACT RESULTS

    #out_name = "GHV02_494k"
    CL, CD = read_output(out_name)

    os.chdir("../")

    return(CL, CD)


def viscous_simulation_cl_cd(xdat, out_name, subfolder):
    """ viscous lift to drag ratio """
    ndat = xdat.shape[0]
    CL = np.zeros([ndat, 1])
    CD = np.zeros([ndat, 1])
    for ii in range(ndat):
        xii = xdat[ii,:]
        CL[ii], CD[ii] = fun3d_simulation_normx(xii, out_name, subfolder)
    return(CL/CD)




""" Fun3d problem """


subfolder2 = "GHV_300k_v" 
out_name2  = "GHV02_300k" 

F = lambda x : -viscous_simulation_cl_cd(x, out_name2, subfolder2)


# run fun3d
x = np.array([[0.5, 0.5, 0.5]])
cl_cd_ratio = F(x)
print("cl_cd_ratio: ", cl_cd_ratio)
