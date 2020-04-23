import abc
import numpy as np
import scipy.optimize as spo


class __AbstractDenoiser(abc.ABC):

    #region UNIVERSAL_THRESHOLD
    def estimate_noise_variance(self, WT: dict) -> float:
        max_scale = max([x[0] for x in WT.keys()])
        wavelet_coeffs_max_scale = [v for (k, v) in WT.items() if k[0]==max_scale]

        normal = np.random.normal(0,1,1000000)
        median_abs_normal = np.median(abs(normal))

        return np.median([abs(x) for x in wavelet_coeffs_max_scale]) / median_abs_normal

    def __get_signal_length(self, WT: dict) -> int:
        max_scale = max([k[0] for (k, _) in WT.items()])
        return 2 * max_scale

    def compute_universal_threshold(self, WT: dict) -> float:
        noise_variance = self.estimate_noise_variance(WT)
        T = self.__get_signal_length(WT)
        return pow(2 * np.log(T), 0.5) * noise_variance
    #endregion

    #region Denoising
    def suppress_by_scale(self, ts: list, threshold: float, method: str='hard'):
        ''' Applies the thresholding as specified in part 3.2.1 of article
        '''
        if method=='hard':
            return [x if abs(x) > threshold else 0 for x in ts]
        elif method=='soft':
            return [np.sign(x) * (abs(x) - threshold) if abs(x) > threshold else 0 for x in ts]
        return

    @abc.abstractmethod
    def compute_threshold(self, ts: list, universal_threshold: float, T: int) -> float:
        pass

    def denoise_coefficients(self, WT: dict, method: str='hard') -> dict:
        all_scales = set([k[0] for (k, v) in WT.items()])
        T = 2 * max(all_scales)
        WT_denoised = dict()
        universal_threshold = self.compute_universal_threshold(WT)
        for scale in all_scales:
            scale_coefficients = [v for (k, v) in WT.items() if k[0]==scale]
            threshold = self.compute_threshold(scale_coefficients, universal_threshold, T)
            scale_coefficients = self.suppress_by_scale(scale_coefficients,
                                                        threshold, method)
            WT_denoised.update({(scale, i):val for (i, val) in enumerate(scale_coefficients)})
        return WT_denoised
    #endregion

    
class SURE(__AbstractDenoiser):

    def __init__(self):
        super()
        pass

    def __count_below_threshold(self, ts: list, threshold: float) -> int:
        ''' Second part equation (18) in article
        '''
        return sum([1 if abs(x) <= threshold else 0 for x in ts])

    def __sum_min_threshold(self, ts: list, threshold: float) -> int:
        ''' Third part equation (18) in article
        '''
        return sum([pow(min(abs(x), threshold),2) for x in ts])

    def __compute_SURE(self, ts: list, threshold: float, T: int):
        ''' Equation (18) in article
        '''
        return T - self.__count_below_threshold(ts, threshold) \
                 + self.__sum_min_threshold(ts, threshold)

    def compute_threshold(self, ts: list, universal_threshold: float, T: int) -> float:
        ''' Equation (17) in article
        '''
        all_thresholds = np.linspace(0, universal_threshold, 1000)
        all_SURE_scores = [self.__compute_SURE(ts, x, T) for x in all_thresholds]
        min_idx = np.argmin(all_SURE_scores)
        return all_thresholds[min_idx]

    
class SURE_G(__AbstractDenoiser):
    ''' Implements SURE denoising as specified in page 120 of M. G. course material
    '''
    
    def __init__(self):
        super()
        pass
    
    def __compute_SURE_SI(self, y: float, threshold: float, sigma: float) -> float:
        ''' Implements SI function in SURE sum page 120 M. G. course material.
        '''
        if abs(y) >= threshold:
            return pow(threshold, 2) + pow(sigma, 2)
        return pow(y, 2) - pow(sigma, 2)
     
    def __compute_SURE(self, coeffs:list, threshold:float, sigma:float) -> float:
        ''' Implements SURE estimation of wavelet reconstruction error as in page 120 of M.G course material.
        '''
        return np.sum([self.__compute_SURE_SI(x, sigma, threshold) for x in coeffs])
    
    def compute_threshold(self, coeffs:list, univ_threshold:float, sigma:float) -> float:
        ''' Computes the SURE threshold as specified page 119 in M. G. course material.
        '''
        res = spo.minimize(lambda x: self.__compute_SURE(coeffs, x, sigma), x0=univ_threshold)
        return res.x[0]

    def denoise_coefficients(self, WT: dict, method: str='hard') -> dict:
        ''' Override mother method because of "T" which we don't use anymore
        '''
        all_scales = set([k[0] for (k, v) in WT.items()])
        sigma_estimate = self.estimate_noise_variance(WT) 
        WT_denoised = dict()
        universal_threshold = self.compute_universal_threshold(WT)
        for scale in all_scales:
            scale_coefficients = [v for (k, v) in WT.items() if k[0]==scale]
            threshold = self.compute_threshold(scale_coefficients, universal_threshold, sigma_estimate)
            scale_coefficients = self.suppress_by_scale(scale_coefficients,
                                                        threshold, method)
            WT_denoised.update({(scale, i):val for (i, val) in enumerate(scale_coefficients)})
        return WT_denoised
    
class SUREShrink(SURE):

    def __init__(self):
        super()
        pass

    def __use_universal_threshold(self, ts: list) -> bool:
        ''' Condition (19) in article
        '''
        left = sum([pow(x,2) - 1 for x in ts])
        right = np.log2(pow(len(ts), 1.5))
        return left <= right

    def compute_threshold(self, ts: list, universal_threshold: float, T: int) -> float:
        ''' Article equation (17)
        '''
        if self.__use_universal_threshold(ts):
            return universal_threshold
        else:
            return super().compute_threshold(ts, universal_threshold, T)

