"""
Module that condense all the components of the EMS tested above
"""

# General Imports

import numpy as np
import math 
import pdb
from math import *
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
    w       = np.ones(Rules, dtype=np.float128);
    # Individual Activatio Degree
    mu      = np.zeros([Rules,n], dtype=np.float128);

    # Array of predictions

    predictions = np.zeros(p);
    xAux = np.flip(yesterday);
    xIn = np.zeros(a.shape[1]);

    for step in np.arange(0,h,1):

        i     = 0;
        
        # Select the relevant inputs

        dIn= np.isnan( xAux*R_input );
        for z in np.arange(0,p,1):
            if dIn[z]  == False:
                xIn[i] = xAux[z];
                i+=1;
                
        for r in np.arange(0,Rules,1):
            for j in np.arange(0,n,1):
                mu[r][j] = np.exp(-0.5*np.power(a[r][j]*(xIn[j]-b[r][j]),2)); 
                w[r]     = w[r]*mu[r][j];
                
        if np.sum(w)==0:
            wn = w;
        else:
            wn = np.divide(w,np.sum(w));

        xf = np.concatenate((1,xIn),axis=None);
        yr = np.dot(g,xf);  
        predictValue = np.dot(wn,yr);

        if predictValue < 0:
            predictions[step] = 0;
        else:
            predictions[step] = predictValue

        # Shape of X_aux is 24, -1 for getting the last index... 

        xAux = np.delete(xAux,xAux.shape[0]-1);
        xAux = np.transpose(np.array([np.append(predictions[step],xAux)]));
        
    return yesterday, today, predictions
    
# PV Estimator

# Inputs
#    ga: GHI [Numpy Array]
#    ta: Environment temperature [Numpy Array]  
#    ga0: Irradiation in standard conditions w/m2
#    t0C: Cell temperature in standard conditions oC
#    pmMax0: Maximun power of the module(watts)
#    imSC0: Short circuit current of the module  (amperes)
#    vmOC0: Open circuit voltage of the module   (Volts)  
#    nSM: Numbers of cell in series  
#    nPM: Numbers of cell in parallel 

# Output

#    pvPower: Output [Numpy Array]

def pvPowerGeneration(ghiPrediction, taPrediction, ga0,t0C,pmMax0,imSC0,vmOC0,nSM,nPM) :
    
    # GHI of the year
    
    ga = ghiPrediction;
    ta = taPrediction;
   
    # Initialization of model attributes
    
    tc     = np.transpose(np.array([np.zeros(ga.shape[0])]));
    vct0  = np.transpose(np.array([np.zeros(ga.shape[0])]));
    voc0  = np.transpose(np.array([np.zeros(ga.shape[0])]));
    ff     = np.transpose(np.array([np.zeros(ga.shape[0])]));
    rs     = np.transpose(np.array([np.zeros(ga.shape[0])]));
    rcS   = np.transpose(np.array([np.zeros(ga.shape[0])]));
    icSC  = np.transpose(np.array([np.zeros(ga.shape[0])]));
    vcOC  = np.transpose(np.array([np.zeros(ga.shape[0])]));
    tVC    = np.transpose(np.array([np.zeros(ga.shape[0])]));
    vmpp   = np.transpose(np.array([np.zeros(ga.shape[0])]));
    xCurrent      = np.transpose(np.array([np.zeros(ga.shape[0])]));
    pvPower     = np.transpose(np.array([np.zeros(ga.shape[0])]));
    current= np.transpose(np.array([np.zeros(ga.shape[0])]));
    
    # Manufacturer's Information (Datasheet)        
    
    #Constant Parameters
    
    # Cm2/w
    c2           = 0.03;   
    # mv/C
    c3           = -2.3e-3;  
    # Electron charge
    e            = 1.602e-19; 
    # Boltzmann constant 
    k            = 1.381e-23;     
    # correction factor or idealising factor
    correction            = 1;
    
    #Calculated parameters for cells 
    # Maximum power per cell
    pcMax0      = pmMax0/(nSM*nPM);  
    # Short circuit current per cell
    icSC0       = imSC0 / nPM;    
    # Open circuit voltage per cell
    vcOC0       = vmOC0 / nSM;          
    # Fill Factor 
    ff0          = pcMax0 /(vcOC0*icSC0); 
    c1           = icSC0 / ga0 ;
    
    # Equation to optimize
    
    def f(xCurrent):
        return (nPM*icSC[i]*(1-(np.exp((vmpp[i]-(nSM*vcOC[i])+((xCurrent*rcS[i]*nSM)/nPM))/(nSM*tVC[i]))))-xCurrent)
    
    for i in np.arange(0,ga.shape[0],1):

        # Absolut Cell temperature
        tc        [i]  = ta+c2*ga[i];
        # Thermal Voltage
        vct0     [i]  = np.divide((correction*k*tc[i]),e);                                   
        voc0     [i]  = vcOC0/vct0 [i];
        # Fill Factor
        ff        [i]  = (voc0[i]-np.log(voc0[i]+0.72))/(voc0[i]+1);   
        rs        [i]  = 1 - (ff[i]/ff0);
        #Equivalent serial resistance
        rcS      [i]  = (rs[i]*vcOC0)/icSC0; 
        #Short circuit current
        icSC     [i]  = c1*ga[i];  
        #Open circuit voltage
        vcOC     [i]  = vcOC0 + c3*(tc[i]-t0C); 
        #Thermal voltage of the single solar cell
        tVC       [i]  = np.divide((correction*k*(273+tc[i])),e);                             
        vmpp      [i]  = vcOC[i]*nSM*0.8;
    
        #PV Current
        current [i]  = fsolve(f, 0.1)
        #PV Power
        pvPower[i]  = current[i]*vmpp[i]; 
 
    return pvPower 
    
# EMS to decide the Energy Mixer command
    
def lightsEMS(voltageBatteries, powerLoads, initialSoc, cn, pv):

    # Noise level (W)
    pn = 25;
    # SoC state
    soc = initialSoc
    # Sampling time
    ts = 1;
    # Get power of Load
    pl = max(powerLoads);
    
    # EB Definition
    eb = np.zeros(3);
    
    for i in np.arange(0,eb.shape[0],1):
        eb[i] = cn[i]*soc[i];

    # Get the battery with Max initial SoC and Voltage
        
    indexMaxSoc = np.where(soc == max(soc))[0];
    voltageMaxSoc = 0;

    for index in indexMaxSoc:
        voltageMaxSoc = max(voltageMaxSoc, voltageBatteries[index]);
    
    indexMaxVoltageSoc = voltageBatteries.index(voltageMaxSoc);

    # Configure Energy Mixer command
    
    emCommand = None;
    
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
    
    for i in np.arange(0,soc.shape[0],1):
        if soc[i] < 100:
            eb[i] = cn[i]*soc[i] + ts*pv[i];
    
    # Detect if Load is active and change EB evolution
    
    if pl > pn :
            
        eb[indexMaxVoltageSoc] = cn[i]*soc[indexMaxVoltageSoc] - ts*pl;
        
    soc = np.divide(eb,cn);
    
    # Protection to avoid unreal SoC values
    
    for i in np.arange(0,soc.shape[0],1):
        if soc[i] > 1:
            soc[i] = 1; 
        if soc[i] < 0:
            soc[i] = 0;
    
    return emCommand, soc
    
"""
EMS for light Controller 
"""

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
import inspect

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
        Features per row
        ----------
        X[0] -> list, vBat - Present measure of voltage in the battery terminals
        X[1] -> list, powerLoads - Present measure of Load 
        X[2] -> int, currentHour - Present hour 
        
        Outputs
        ----------
        y[0] -> json str, emCommand - switching Command 
        y[1] -> list, pvEstimation - Estimation for the next 24 hours 
        y[2] -> list, soc - SoC for each battery 
        y[3] -> list, yesterday - measures 24 hours before of GHI 
        y[4] -> list, today - measures for today of GHI
             
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
            
        Returns
        -------
        y : ndarray, shape (n_samples, n_outputs)
            Returns an array of predictions
        """
        
        # Check input and target vectors correctness
        # ToDo
        
        # Initial value for initialization flag
        
        self.isInitialized_ = False;
        
        # Configure parameters when fitted
        
        self.isFitted_ = True;
        
        # `fit` should always return `self`
        return self

    def predict(self, X, y = None):
        
        """ 
        Implementation of a predicting function. With input features estimate
        the value of reference power for batteries.
        
        Features per row
        ----------
        X[0] -> str, vBat - Present measure of voltage in the battery terminals, values separated by ','
        X[1] -> str, powerLoads - Present measure of Load 
        X[2] -> int, currentHour - Present hour 
        
        Outputs
        ----------
        y[0] -> json str, emCommand - switching Command 
        y[1] -> list, pvEstimation - Estimation for the next 24 hours 
        y[2] -> list, soc - SoC for each battery 
        y[3] -> list, yesterday - measures 24 hours before of GHI 
        y[4] -> list, today - measures for today of GHI
             
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
            
            vBat = []
            powerLoads = []
            
            vBat = [float(i) for i in x[0].split(",")];
            powerLoads = [float(i) for i in x[1].split(",")];
            currentHour = int(x[2]); 
            
            # Define the first value of SoC, only the first time
            
            if(not self.isInitialized_):
                
                # Initialization values for SoC
        
                for x in X :

                    self.soc=np.zeros(len(vBat));

                    for i in np.arange(0,len(vBat),1):
                        if vBat[i] <= 11:
                            self.soc[i]=0;
                        elif vBat[i]>=13.8:
                            self.soc[i]=1; 
                        else:
                            self.soc[i]= 0.35714*vBat[i]-3.92857;
                
                self.isInitialized_ = True;            
            
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
            
            emCommand, self.soc = self.__computeLightsEMS(vBat,powerLoads,pvPower)
            
            # Append results to output vector
            
            y_reg.append(emCommand);
            y_reg.append(pvEstimation);
            y_reg.append(self.soc.tolist());
            y_reg.append(yesterday.tolist());
            y_reg.append(today.tolist());
            
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
