import numpy as np
import pandas as pd

class Haar(object):
	'''
	Implementation of the Haar Wavelet and its associated functions
	'''
	
	name = "Haar"
	
	def __init__(self, signal: np.array):
		self.signal = signal 
		self.__T = len(self.signal)
		
		# Init dictionnary of wavelet coefficients per resolution level
		self.__coeffDict = dict()
		# Init dictionnary of wavelet detail signals per resolution level
		self.__detailSignalsDict = dict()
		
		# Determine Time
		self.__t = np.linspace(0, self.__T - 1, self.__T)
		# Determine Shifts
		self.__k = np.linspace(-1, self.__T - 2, self.__T)


	#region WAVELET
	
	def __firstHalfIndicator(self, t: float) -> int:
		return ((0 <= t) and (t < 0.5)) * 1

	def __secondHalfIndicator(self, t: float) -> int:
		return ((0.5 <= t) and (t < 1)) * 1
	
	def motherFunction(self, t: float) -> int:
		return self.__firstHalfIndicator(t) - self.__secondHalfIndicator(t)
	
	#endregion
	
	#region GETTERS
	
	def getDetailSignalJ(self, j: int) -> np.array:
		try:
			return self.__detailSignalsDict[str(j)][2:self.__T - pow(2, -j)]
		except (KeyError):
			print("KEY ERROR: Detail signal not found in computed results.")
	
	def getWaveletTransformJ(self, j: int) -> np.array:
		try:
			return self.__coeffDict[str(j)][1:np.size(self.__coeffDict[j]) - 3]
		except (KeyError):
			print("KEY ERROR: Wavelet transform not found in computed results.")
	
	#endregion
	
	#region TRANSFORM
	
	def waveletTransform(self, j: int) -> np.array:

		# Determine number of shifts possible at this resolution level
		nb_k = int(np.ceil(self.__T / pow(2, -j)) + 2)
		# Init bounds of the Haar wavalet function support
		k1 = np.zeros(nb_k + 1)
		k2 = np.zeros(nb_k + 1)
		# Init results
		coeff = np.zeros(nb_k + 1)
		
		for i in range(0, nb_k +1):
		    k1[i] = self.__k[i] * pow(2, -j)
		    k2[i] = (self.__k[i] + 1) * pow(2, -j)
		    sum1, sum2 = 0, 0        
		    
		    for p in range(0, self.__T):
		        if self.__t[p] >= k1[i]:
		            sum1 += self.signal[p]
		        if self.__t[p] >= k2[i]:
		            sum2 += self.signal[p]

		    coeff[i] = pow(2, j / 2) * (sum1 - sum2)
		  
		self.__coeffDict[str(j)] = coeff
		
		return coeff
	
	def waveletDetailSignal(self, j: int) -> np.array:
		
		try:
			coeff = self.__coeffDict[str(j)]
		except(KeyError):
			self.__coeffDict[str(j)] = self.waveletTransform(j)
			coeff = self.__coeffDict[str(j)]
		
		# Determine number of shifts possible at this resolution level
		nb_k = int(np.ceil(self.__T / pow(2, -j)) + 2)

		t1 = np.zeros(self.__T-1)
		t2 = np.zeros(self.__T-1)
		
		detailSignal = np.zeros(self.__T)
		
		for i in range(1, self.__T):
		    t1[i - 1] = pow(2, j) * self.__t[i - 1] - 1
		    t2[i - 1] = pow(2, j) * self.__t[i - 1]
		    
		    sum1, sum2 = 0, 0
		    
		    for p in range(0, nb_k):
		        if(self.__k[p] >= t1[i-1]):
		            sum1 += coeff[p]
		        if(self.__k[p] >= t2[i-1]):
		            sum2 += coeff[p]
		    
		    detailSignal[i] = 2**(j/2) * (sum1 - sum2)
		
		self.__detailSignalsDict[str(j)] = detailSignal
		
		return detailSignal
	
	#endregion
