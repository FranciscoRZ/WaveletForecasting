import numpy as np
import pandas as pd
import math

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
		self.__approx = dict()
        # Init wavelet coeffcient
		self.__WT = dict()
        # Init inv WT
		self.__invWT = dict()
		
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
    
	def haarWaveletFunction(self, j: float, k: float, t: float) -> float:
		return pow(2, j / 2) * self.motherFunction(pow(2, j) * t - k)
	
	#endregion
	
	#region GETTERS

	def getApproxJ(self, j: float) -> np.array:
		try:
			return self.__approx[str(j)][2:]
		except (KeyError):
			print("KEY ERROR: Detail signal not found in computed results.")
	
	def getScaleCoeffJ(self, j: float) -> np.array:
		try:
			return self.__coeffDict[str(j)][1:np.size(self.__coeffDict[j]) - 2]
		except (KeyError):
			print("KEY ERROR: Wavelet transform not found in computed results.")

	def getSignalReconstitution(self, j: float) -> np.array:
		try:
			return self.__coeffDict[str(j)][1:np.size(self.__coeffDict[j]) - 2]
		except (KeyError):
			print("KEY ERROR: Wavelet transform not found in computed results.")
	#endregion
	
	#region TRANSFORM
	
	def minLen(self, allJs: np.array) -> int:
		min_len = self.__T
        
		for j in allJs:
			k_max = int(np.ceil(self.__T / pow(2, -j)))
            
			if((k_max - 1) * pow(2, -j) < min_len):
				min_len = int((k_max - 1) * pow(2, -j))
                
		return min_len
    
    
	def scaleCoeff(self, j: float, min_len: int) -> np.array:

		# Determine number of shifts possible at this resolution level
		nb_k = int(np.ceil(self.__T / pow(2, -j)) + 2)
		# Init bounds of the Haar wavalet function support
		k1 = np.zeros(nb_k + 1)
		k2 = np.zeros(nb_k + 1)
		# Init results
		coeff = np.zeros(nb_k)
		
		for i in range(0, nb_k):
		    k1[i] = self.__k[i] * pow(2, -j)
		    k2[i] = (self.__k[i] + 1) * pow(2, -j)
		    sum1, sum2 = 0, 0        
		    
		    for p in range(self.__T - min_len, self.__T):
		        if self.__t[p - (self.__T - min_len)] >= k1[i]:
		            sum1 += self.signal[p]
		        if self.__t[p - (self.__T - min_len)] >= k2[i]:
		            sum2 += self.signal[p]

		    coeff[i] = pow(2, j / 2) * (sum1 - sum2)
		  
		self.__coeffDict[str(j)] = coeff
		
		return coeff
	
	
	def approx(self, j: float, min_len: int) -> np.array:
		
		try:
			coeff = self.__coeffDict[str(j)]
		except(KeyError):
			self.__coeffDict[str(j)] = self.scaleCoeff(j, min_len)
			coeff = self.__coeffDict[str(j)]
		
		# Determine number of shifts possible at this resolution level
		nb_k = int(np.ceil(self.__T / pow(2, -j)) + 2)

		t1 = np.zeros(self.__T-1)
		t2 = np.zeros(self.__T-1)
		
		detailSignal = np.zeros(min_len + 2)
		
		for i in range(1, min_len + 2):
		    t1[i - 1] = pow(2, j) * self.__t[i - 1] - 1
		    t2[i - 1] = pow(2, j) * self.__t[i - 1]
		    
		    sum1, sum2 = 0, 0
		    
		    for p in range(0, nb_k):
		        if(self.__k[p] >= t1[i-1]):
		            sum1 += coeff[p]
		        if(self.__k[p] >= t2[i-1]):
		            sum2 += coeff[p]
		    
		    detailSignal[i] = 2**(j/2) * (sum1 - sum2)
		
		self.__approx[str(j)] = detailSignal
		
		return detailSignal
	
	
	def getWT(self, allJs: np.array) -> dict:
		for j in allJs:
			zjk = list()
			maxSteps = int(np.ceil(len(self.signal) / pow(2, -j)) + 2)
            
			for k in range(0, maxSteps -3):
				# Get wavelet function support
				first = math.floor(pow(2, -j) * k)
				second = math.ceil(pow(2, -j) * (k + 1))

				# Compute the local decomposition
				waveletCoeff = 0
                
				for t, sig in enumerate(self.signal[first:second]):
					t_wavelet = t + first
					waveletCoeff += sig * self.haarWaveletFunction(j, k, t_wavelet)
				
				zjk.append(waveletCoeff)
			
			self.__WT[str(j)] = zjk

    
	def getinvWT(self, allJs: np.array) -> dict:
		
		self.getWT(allJs)
        
		for j in allJs:
			results = []

			for t in range(0, len(self.signal)):
				res = 0
				for k in range(0, len(self.__WT[str(j)])):
					res += self.__WT[str(j)][k] * self.haarWaveletFunction(j, k, t)
				results.append(res)
            
			self.__invWT[str(j)] = results
            
		#return self.__invWT
        
	def signal_reconstruction(self, allJs: np.array) -> np.array:
		# Sum for all j<= j0
		self.getinvWT(allJs)
        
		z = self.__invWT[str(np.max(allJs)-1)]
		for j in range(np.min(allJs), np.max(allJs)):
			for i in range(0, len(self.__invWT[str(np.max(allJs))])):
				z[i] = z[i] + self.__invWT[str(j)][i]
                
		# with scaling function of j0
		min_len = self.minLen([np.max(allJs)])
		self.approx(int(np.max(allJs)), min_len)
		for i in range(0, len(self.__invWT[str(np.max(allJs))])):
			z[i] = z[i] + self.__approx[str(np.max(allJs))][i]
            
		return z[2:]
		
	#endregion
