import numpy as np

def MAD(S: np.array, S_hat: np.array) -> float:
	'''
	Computes the Mean Absolute Deviation (formula in part 
	4.2 of article)
	'''
	if len(S) != len(S_hat):
		print("ERROR: Both arrays must have same size")
		return
	return np.sqrt(np.sum(abs(S - S_hat)) / len(S))

def MAPE(S: np.array, S_hat: np.array) -> float:
	'''
	Computes the Mean Average Percentage Error (formula in part 
	4.2 of article)
	'''
	if len(S) != len(S_hat):
		print("ERROR: Both arrays must have same size")
		return
	return np.sqrt(np.sum(abs(S - S_hat) / (S * len(S))))
    
def RMSE(S: np.array, S_hat: np.array) -> float:
	'''
	Computes the Root Mean Square Error
	'''
	if len(S) != len(S_hat):
		print("ERROR: Arrays must have the same size")
		return
	return np.sqrt(np.sum(np.power(S - S_hat, 2)) / len(S))
