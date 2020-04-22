import math
import numpy as np

class __UniversalThreshold():

    def __init__(self):
        pass

    def __estimate_noise_variance(self, WT: dict) -> float:
        max_scale = max([x[0] for x in WT.keys()])
        wavelet_coeffs_max_scale = [v for (k, v) in WT.items() if k[0]==max_scale]

        normal = np.random.normal(0,1,1000000)
        median_abs_normal = np.median(abs(normal))

        return np.median([abs(x) for x in wavelet_coeffs_max_scale]) / median_abs_normal

    def __get_signal_length(self, WT: dict) -> int:
        max_scale = max([k[0] for (k, _) in WT.items()])
        return 2 * max_scale

    def compute_universal_threshold(self, WT: dict) -> float:
        noise_variance = self.__estimate_noise_variance(WT)
        T = self.__get_signal_length(WT)
        return pow(2 * np.log(T), 0.5) * noise_variance

class SURE(__UniversalThreshold):

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

    def __compute_SURE(self, ts: list, threshold: float):
        ''' Equation (18) in article
        '''
        T = len(ts)
        return T - self.__count_below_threshold(ts, threshold) \
                 + self.__sum_min_threshold(ts, threshold)

    def __suppress_by_scale(self, ts: list, threshold: float, method: str='hard'):
        ''' Applies the thresholding as specified in part 3.2.1 of article
        '''
        if method=='hard':
            return [x if abs(x) > threshold else 0 for x in ts]
        elif method=='soft':
            return [np.sign(x) * (abs(x) - threshold) if abs(x) > threshold else 0 for x in ts]
        return

    def compute_threshold(self, ts: list, universal_threshold: float) -> float:
        ''' Equation (17) in article
        '''
        all_thresholds = np.linspace(0, universal_threshold, 1000)
        all_SURE_scores = [self.__compute_SURE(ts, x) for x in all_thresholds]
        min_idx = np.argmin(all_SURE_scores)
        return all_thresholds[min_idx]

    def denoise_coefficients(self, WT: dict, method='hard') -> dict:
        all_scales = set([k[0] for (k, v) in WT.items()])
        WT_denoised = dict()
        universal_threshold = self.compute_universal_threshold(WT)
        for scale in all_scales:
            scale_coefficients = [v for (k, v) in WT.items() if k[0]==scale]
            threshold = self.compute_threshold(scale_coefficients, universal_threshold)
            scale_coefficients = self.__suppress_by_scale(scale_coefficients,
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

    def compute_threshold(self, ts: list, universal_threshold: float) -> float:
        ''' Article equation (17)
        '''
        if self.__use_universal_threshold(ts):
            return universal_threshold
        else:
            return super().compute_threshold(ts, universal_threshold)
