import scipy.io
import control.matlab as ctrl
import numpy as np

class WTG():
    """description of class"""
    def get_tf(self,desired_windspeed,desired_azimuth,desired_input,desired_output):
        windspeed_ii = np.where(self.windspeeds == desired_windspeed)[0]
        desiredinput_ii = self.inputnames.index(desired_input)
        
        desiredoutput_ii = self.outputnames.index(desired_output)

        SS = self.SS[windspeed_ii[0]]
        desired_wtg = ctrl.ss2tf(SS.A,SS.B[:,desiredinput_ii],SS.C[desiredoutput_ii,:],SS.D[desiredoutput_ii,desiredinput_ii])
        return desired_wtg


    def __init__(self,matfile,azimuth_index):
        mat = scipy.io.loadmat(matfile)

        azimuths = mat['Azimuths'][0]
        windspeeds = mat['Windspeeds'][0]
        rotorspeeds = mat['RotorSpeeds'][0]
        gbx = mat['Gbx'][0][0]
        pitchangles = mat['PitchAngles']
        nblades = mat['NBlades']
        steadystate = mat['SteadyState']
        steadyinput = mat['SteadyInput']
        steadyoutput = mat['SteadyOutput']
        nomspeedarray = mat['NomSpeedArray']
        nomtorquearray = mat['NomTorqueArray']
        inputnames = [item.strip() for item in mat['SYSTURB']['inputname'][0][0]]
        outputnames =  [item.strip() for item in mat['SYSTURB']['outputname'][0][0]]
        statenames =  [item.strip() for item in mat['SYSTURB']['statename'][0][0]]

        A = mat['SYSTURB']['A'][0][0]
        B = mat['SYSTURB']['B'][0][0]
        C = mat['SYSTURB']['C'][0][0]
        D = mat['SYSTURB']['D'][0][0]

        WTG_SS = []
        for idx_wnd , each_windspeed in enumerate(windspeeds):
            if len(azimuths)>1:
                A_aux = A[:,:,idx_wnd,azimuth_index]
                B_aux = B[:,:,idx_wnd,azimuth_index]
                C_aux = C[:,:,idx_wnd,azimuth_index]
                D_aux = D[:,:,idx_wnd,azimuth_index]
            else:
                A_aux = A[:,:,idx_wnd]
                B_aux = B[:,:,idx_wnd]
                C_aux = C[:,:,idx_wnd]
                D_aux = D[:,:,idx_wnd]

            WTG_SS.append(ctrl.ss(A_aux,B_aux,C_aux,D_aux))


        rotor_rated_speed = max(rotorspeeds)
        gen_rated_speed = gbx * rotor_rated_speed
        rated_torque = np.empty(azimuths.shape)
        rated_torque = max(nomtorquearray[:,azimuth_index])
        
        windspeeds_at_rotor_rated_speed_index = np.where(rotorspeeds == rotor_rated_speed)[0]
        windspeeds_at_rated_torque_index = np.where(nomtorquearray[:,azimuth_index] == rated_torque)[0]

        windspeed_1 = windspeeds[0]
        windspeed_2 = windspeeds[windspeeds_at_rotor_rated_speed_index[0]]
        windspeed_rated = windspeeds[windspeeds_at_rated_torque_index[0]]
        windspeed_rated_p1 = windspeeds[windspeeds_at_rated_torque_index[1]]
        windspeed_cutout = windspeeds[-1]

        print('1st vertical wind speed is: '+ str(windspeed_1) +'m/s')
        print('2nd vertical wind speed is: '+ str(windspeed_2) +'m/s')
        print('Rated wind speed is: '+ str(windspeed_rated) +'m/s')
        print('Selected wind for pitch control to start is: '+ str(windspeed_rated_p1) +'m/s')
        print('Cut-out wind speed is: '+ str(windspeed_cutout) +'m/s')
        
        self.azimuths = azimuths
        self.windspeeds = windspeeds
        self.SS = WTG_SS
        self.inputnames = inputnames
        self.outputnames = outputnames
        self.statenames = statenames

        desired_input = 'Collective generator torque demand'
        desired_output = 'Generator speed'
        self.T_G_1 = self.get_tf(windspeed_1,azimuths[azimuth_index],desired_input,desired_output)


        



        