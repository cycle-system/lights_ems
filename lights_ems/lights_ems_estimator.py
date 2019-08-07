"""
Module that condense all the components of the EMS tested above
"""

# General Imports

import numpy as np
import math 
import pdb
from math import *
from matplotlib import pyplot as plt
from scipy import linalg
from sympy import symbols, solve
import sympy as sp
from scipy.optimize import fsolve
from datetime import datetime, timedelta
import json

# Sklearn imports

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
import inspect

# Get the predictions of GHI one day ahead

def oneDayPrediction(currentHour, GHI, a, b, g, R_input, Rules):
    
    # Variables Definition

    # Prediction Horizon
    h       = 24;
    # Maximun Delay: Corresponds one day before (1 hour sample time) 
    p       = 24; 
    
    # Number of Relevant inputs
    n       = a.shape[1];
    
    # Prepare data for model

    yesterday = GHI[currentHour-p:currentHour]
    today =  GHI[currentHour:currentHour+p]
    
    # Get prediction
    # Acumulate activation degree
    W       = np.ones(Rules, dtype=np.float128);
    # Individual Activatio Degree
    mu      = np.zeros([Rules,n], dtype=np.float128);

    # Array of predictions

    predictions = np.zeros(p);
    X_aux = np.flip(yesterday);
    X_in = np.zeros(a.shape[1]);

    for step in np.arange(0,h,1):

        i     = 0;
        
        # Select the relevant inputs

        D_in= np.isnan( X_aux*R_input );
        for z in np.arange(0,p,1):
            if D_in[z]  == False:
                X_in [i] = X_aux [z];
                i+=1;
                
        for r in np.arange(0,Rules,1):
            for j in np.arange(0,n,1):
                mu[r][j] = np.exp(-0.5*np.power(a[r][j]*(X_in[j]-b[r][j]),2)); 
                W[r]     = W[r]*mu[r][j];
                
        if np.sum(W)==0:
            Wn = W;
        else:
            Wn = np.divide(W,np.sum(W));

        xf = np.concatenate((1,X_in),axis=None);
        yr = np.dot(g,xf);  
        predictValue = np.dot(Wn,yr);

        if predictValue < 0:
            predictions[step] = 0;
        else:
            predictions[step] = predictValue

        # Shape of X_aux is 24, -1 for getting the last index... 

        X_aux = np.delete(X_aux,X_aux.shape[0]-1);
        X_aux = np.transpose(np.array([np.append(predictions[step],X_aux)]));
        
    return yesterday, today, predictions
    
# PV Estimator

# Inputs
#    Ga: GHI [Numpy Array]
#    Ta: Environment temperature [Numpy Array]  
#    Ga_0,T0_c,PM_max0,IM_sc0,VM_oc0,N_sm,N_pm: Datasheet information

# Output

#    pvPower: Output [Numpy Array]

def pvPowerGeneration(ghiPrediction, taPrediction, Ga_0,T0_c,PM_max0,IM_sc0,VM_oc0,N_sm,N_pm) :
    
    # GHI of the year
    
    Ga = ghiPrediction;
    Ta = taPrediction;
   
    # Initialization of model attributes
    
    TC     = np.transpose(np.array([np.zeros(Ga.shape[0])]));
    VCt_0  = np.transpose(np.array([np.zeros(Ga.shape[0])]));
    voc_0  = np.transpose(np.array([np.zeros(Ga.shape[0])]));
    FF     = np.transpose(np.array([np.zeros(Ga.shape[0])]));
    rs     = np.transpose(np.array([np.zeros(Ga.shape[0])]));
    RC_s   = np.transpose(np.array([np.zeros(Ga.shape[0])]));
    IC_sc  = np.transpose(np.array([np.zeros(Ga.shape[0])]));
    VC_oc  = np.transpose(np.array([np.zeros(Ga.shape[0])]));
    VCt    = np.transpose(np.array([np.zeros(Ga.shape[0])]));
    VMPP   = np.transpose(np.array([np.zeros(Ga.shape[0])]));
    current= np.transpose(np.array([np.zeros(Ga.shape[0])]));
    x      = np.transpose(np.array([np.zeros(Ga.shape[0])]));
    pvPower     = np.transpose(np.array([np.zeros(Ga.shape[0])]));
    current= np.transpose(np.array([np.zeros(Ga.shape[0])]));
    
    # Manufacturer's Information (Datasheet)

    #Irradiation in standard conditions w/m2
    Ga_0         = Ga_0;              
    #Cell temperature in standard conditions oC
    T0_c         = T0_c;           
    # Maximun power of the module   (watts)
    PM_max0      = PM_max0;          
    # Short circuit current of the module  (amperes)
    IM_sc0       = IM_sc0; 
    #Open circuit voltage of the module   (Volts)  
    VM_oc0       = VM_oc0; 
    #Numbers of cell in series  
    N_sm         = N_sm;  
    #Numbers of cell in parallel 
    N_pm         = N_pm;         
    
    #Constant Parameters
    
    # Cm2/w
    C2           = 0.03;   
    # mv/C
    C3           = -2.3e-3;  
    # Electron charge
    e            = 1.602e-19; 
    # Boltzmann constant 
    k            = 1.381e-23;     
    # correction factor or idealising factor
    m            = 1;
    
    # Equation to optimize
    
    def f(x):
        return (N_pm*IC_sc[i]*(1-(np.exp((VMPP[i]-(N_sm*VC_oc[i])+((x*RC_s[i]*N_sm)/N_pm))/(N_sm*VCt[i]))))-x)
    
    for i in np.arange(0,Ga.shape[0],1):

        # Absolut Cell temperature
        TC        [i]  = Ta+C2*Ga[i];
        # Thermal Voltage
        VCt_0     [i]  = np.divide((m*k*TC[i]),e);                                   
        voc_0     [i]  = VC_oc0/VCt_0 [i];
        # Fill Factor
        FF        [i]  = (voc_0[i]-np.log(voc_0[i]+0.72))/(voc_0[i]+1);   
        rs        [i]  = 1 - (FF[i]/FF0);
        #Equivalent serial resistance
        RC_s      [i]  = (rs[i]*VC_oc0)/IC_sc0; 
        #Short circuit current
        IC_sc     [i]  = C1*Ga[i];  
        #Open circuit voltage
        VC_oc     [i]  = VC_oc0 + C3*(TC[i]-T0_c); 
        #Thermal voltage of the single solar cell
        VCt       [i]  = np.divide((m*k*(273+TC[i])),e);                             
        VMPP      [i]  = VC_oc[i]*N_sm*0.8;
    
        #PV Current
        current [i]  = fsolve(f, 0.1)
        #PV Power
        pvPower[i]  = current[i]*VMPP[i]; 
 
    return pvPower
    
# EMS to decide the Energy Mixer command
    
def lightsEMS(voltageBatteries, powerLoads, initialSoc, Cn, PV):

    # Noise level (W)
    PN = 25;
    # SoC state
    SoC = initialSoc
    # Sampling time
    Ts = 1;
    # Get power of Load
    PL = max(powerLoads);
    
    # EB Definition
    EB = np.zeros(3);
    
    for i in np.arange(0,EB.shape[0],1):
        EB[i] = Cn[i]*SoC[i];

    # Get the battery with Max initial SoC and Voltage
        
    indexMaxSoc = np.where(SoC == max(SoC))[0];
    voltageMaxSoc = 0;

    for index in indexMaxSoc:
        voltageMaxSoc = max(voltageMaxSoc, voltageBatteries[index]);
        
    indexMaxVoltageSoc = np.where(voltageBatteries == voltageMaxSoc)[0][0];
    
    # Configure Energy Mixer command
    
    if indexMaxVoltageSoc == 0:
        emCommand = {
        "house 1" : [0,0],
        "house 2" : [0,0],
        "house 3" : [0,0],
        }
        
    if indexMaxVoltageSoc == 1:
        emCommand = {
        "house 1" : [0,1],
        "house 2" : [1,0],
        "house 3" : [0,0],
        }
        
    if indexMaxVoltageSoc == 2:
        emCommand = {
        "house 1" : [0,1],
        "house 2" : [0,0],
        "house 3" : [1,0],
        }
    
    emCommand = json.dumps(emCommand)
    
    # EB Evolution with PV Power (Default)
    
    for i in np.arange(0,SoC.shape[0],1):
        if SoC[i] < 100:
            EB[i] = Cn[i]*SoC[i] + Ts*PV[i];
    
    # Detect if Load is active and change EB evolution
    
    if PL > PN :
            
        EB[indexMaxVoltageSoc] = Cn[i]*SoC[indexMaxVoltageSoc] - Ts*PL;
        
    SoC = np.divide(EB,Cn);
    
    # Protection to avoid unreal SoC values
    
    for i in np.arange(0,SoC.shape[0],1):
        if SoC[i] > 1:
            SoC[i] = 1; 
        if SoC[i] < 0:
            SoC[i] = 0;
    
    return emCommand, SoC


"""
EMS for light Controller 
"""

class lightEmsEstimator(BaseEstimator):
    
    """ 
    A estimator based on rules wich contains the EMS configurations
    to administer a photovoltaic system with batteries using criteria of
    SoC, PmaxCh and PmaxDis.
    
    Parameters
    ----------
    ghi : np.array, default='None' - Radiation for one year  
    a : np.array, default='None' - (Std)^-1 of the clusters (TS Model)
    b : np.array, default='None' - Mean Value of the clusters (TS Model)
    g : np.array, default='None' - Parameters of the consecuents (TS Model)
    rInput : np.array, default='None' - Relevant inputs to the model 
    rules : int, default='None' - Rules number
    pvDatasheetArray : list, default='None' - Array of Datasheet parameters for the PV
        ga0: Irradiation in standard conditions w/m2
        t0c: Cell temperature in standard conditions oC
        pmMax0: Maximun power of the module   (watts)
        imSc0: Short circuit current of the module  (amperes)
        vmOc0: Open circuit voltage of the module   (Volts)  
        nSm: Numbers of cell in series  
        nPm: Numbers of cell in parallel 
    cn : np.array, default='None' - Nominal Capacity 
    """  
    
    def __init__(self,ghi,a,b,g,rInput,rules,pvDatasheetArray,cn):
        
        # Gather all the passed parameters
        args, _, _,values = inspect.getargvalues(inspect.currentframe())
        
        # Remove parameter 'self' 
        values.pop("self")

        # Init the attributes of the class
        for arg, val in values.items():
            setattr(self, arg, val)

    def fit(self, X, y = None):
        
        """
        Implementation of a fitting function. This section could help when needed to
        tune the rww parameter of SoC or other necessary.
        
        Features per row
        ----------
        X[0] -> List, v_bat - Present measure of voltage in the battery terminals
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
            
        Returns
        -------
        self : object
            Returns self.
        """
        
        # Check input and target vectors correctness
        # ToDo
        
        # Initialization values for SoC
        
        for x in X :
            
            self.soc=np.zeros(x.shape[0]);

            for i in np.arange(0,x.shape[0],1):
                if x[i] <= 11:
                    self.soc[i]=0;
                elif x[i]>=13.8:
                    self.soc[i]=1; 
                else:
                    self.soc[i]= 0.35714*x[i]-3.92857;
        
        # Configure parameters when fitted
        
        self.isFitted_ = True
        
        # `fit` should always return `self`
        return self

    def predict(self, X, y = None):
        
        """ 
        Implementation of a predicting function. With input features estimate
        the value of reference power for batteries.
        
        Features per row
        ----------
        X[0] -> List, vBat - Present measure of voltage in the battery terminals
        X[1] -> List, powerLoads - Present measure of Load 
        X[2] -> int, currentHour - Present hour 
        
        Outputs
        ----------
        y[0] -> json, emCommand - switching Command 
        y[1] -> np.array, pvEstimation - Estimation for the next 24 hours 
        y[2] -> np.array, soc - SoC for each battery 
        y[3] -> np.array, yesterday - measures 24 hours before of GHI 
        y[4] -> np.array, today - measures for today of GHI
             
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
            
        Returns
        -------
        y : ndarray, shape (n_samples, n_outputs)
            Returns an array of predictions
        """
        
        # Define the Output vector
        
        y = [];
        
        # Verification before prediction
        
        check_is_fitted(self, 'isFitted_')
        
        # Perform the prediction over the input array
        
        for x in X :
            
            # Define a vector of results
            
            y_reg = [];
            
            # Extract the values of the input array
            
            vBat = x[0];
            powerLoads = x[1];
            currentHour = x[2];
            
            # step 1: get GHI predictions for one day ahead
            
            yesterday,today,predictions=self.__computeOneDayPrediction (currentHour);
            
            # step 2: Estimation of PV power 
            
            taPrediction= 25;
            
            pvPower = np.zeros(self.pvDatasheetArray.shape[0]);
            pvEstimation = [];
            
            for i in np.arange(0,self.pvDatasheetArray.shape[0],1):
                
                ga0 = self.pvDatasheetArray[i][0];
                t0c = self.pvDatasheetArray[i][1];
                pmMax0 = self.pvDatasheetArray[i][2];
                imSc0 = self.pvDatasheetArray[i][3];
                vmOc0 = self.pvDatasheetArray[i][4];
                nSm = self.pvDatasheetArray[i][5];
                nPm = self.pvDatasheetArray[i][6];
                
                pvPowerOneDay= self.__computePvPowerGeneration(predictions,taPrediction,ga0,t0c,pmMax0,imSc0,vmOc0,nSm,nPm);
                pvPower[i]    = pvPowerOneDay[0];
                pvEstimation.append(pvPowerOneDay);
                    
                    
            # step 3: EMS for Lights
            
            emCommand, self.soc = self.__computeLightsEMS(vBat, powerLoads,pvPower)
            
            # Append results to output vector
            
            y_reg.append(emCommand);
            y_reg.append(pvEstimation);
            y_reg.append(self.soc);
            y_reg.append(yesterday);
            y_reg.append(today);
            
            # Append sub result to output vector
            
            y.append(y_reg);
        
        # Return the vector of outputs
        
        return y;
    
    """
    --------------------------------------------------------------------------------------------------
    Definition of specific EMS methods
    --------------------------------------------------------------------------------------------------
    """  
    
    def __computeOneDayPrediction(self,currentHour) : 
        return oneDayPrediction(currentHour, self.ghi, self.a, self.b, self.g, self.rInput, self.rules);
    
    def __computePvPowerGeneration(self,ghiPrediction,taPrediction,ga0,t0c,pmMax0,imSc0,vmOc0,nSm,nPm): 
        return pvPowerGeneration(ghiPrediction,taPrediction,ga0,t0c,pmMax0,imSc0,vmOc0,nSm,nPm);
        
    def __computeLightsEMS(self,voltageBatteries, powerLoads,pv): 
        return lightsEMS(voltageBatteries,powerLoads,self.soc,self.cn,pv);
