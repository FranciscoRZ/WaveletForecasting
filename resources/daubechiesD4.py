import numpy as np

class DaubechiesD4(object):

    __h0 = (1 + np.sqrt(3))/(4 * np.sqrt(2))
    __h1 = (3 + np.sqrt(3))/(4 * np.sqrt(2))
    __h2 = (3 - np.sqrt(3))/(4 * np.sqrt(2))
    __h3 = (1 - np.sqrt(3))/(4 * np.sqrt(2))
    __g0 = __h3
    __g1 = -__h2
    __g2 = __h1
    __g3 = -__h0
    __ih0 = __h2
    __ih1 = __g2
    __ih2 = __h0
    __ih3 = __g0
    __ig0 = __h3
    __ig1 = __g3
    __ig2 = __h1
    __ig3 = __g1
    __n_for_transform = []
    WT = []

    def __init__(self, data):
        self.data = data.copy()
        self.N = len(data)

    def __fwd_step_transform(self, n: int):
        i = 0
        tmp = np.zeros(n)
        half = int(n/2)
        for j in range(0, n - 3, 2):
            tmp[i+half] = self.data[j] * self.__h0 + self.data[j+1] * self.__h1 \
                          + self.data[j+2] * self.__h2 + self.data[j+3] \
                          * self.__h3
            tmp[i] = self.data[j] * self.__g0 + self.data[j+1] \
                          * self.__g1 + self.data[j+2] * self.__g2 \
                          + self.data[j+3] * self.__g3
            i += 1
        tmp[i+half] = self.data[n-2] * self.__h0 + self.data[n-1] * self.__h1 \
                      + self.data[0] * self.__h2 + self.data[1] * self.__h3
        tmp[i] = self.data[n-2] * self.__g0 + self.data[n-1] \
                                  * self.__g1 + self.data[0] * self.__g2 \
                                  + self.data[1] * self.__g3
        for i in range(0, n):
            self.data[i] = tmp[i]

    def __inv_step_transform(self, n: int):
        j = 2
        tmp = np.zeros(n)
        half = int(n/2)
        tmp[0] = self.data[half-1]*self.__ih0 + self.data[n-1]*self.__ih1 \
                + self.data[0]*self.__ih2 + self.data[half]*self.__ih3
        tmp[1] = -(self.data[half-1]*self.__ig0 + self.data[n-1]*self.__ig1 \
                + self.data[0]*self.__ig2 + self.data[half]*self.__ig3)
        for i in range(0, half - 1):
            tmp[j] = self.data[i]*self.__ih0 + self.data[i + half]*self.__ih1 \
                    + self.data[i+1]*self.__ih2 + self.data[i + half + 1]*self.__ih3
            j += 1
            tmp[j] = -(self.data[i]*self.__ig0 + self.data[i + half]*self.__ig1 \
                    + self.data[i+1]*self.__ig2 + self.data[i + half + 1]*self.__ig3) 
            j += 1
        for i in range(0, n):
            self.data[i] = tmp[i]

    def transform(self):
        n = self.N
        while n > 3:
            self.__fwd_step_transform(n)
            self.WT.append(np.array(self.data))
            n /= 2
            n = int(n)
            self.__n_for_transform.append(n)

    def inv_transform(self):
        for n in self.__n_for_transform[::-1]:
            self.__inv_step_transform(n)
        self.__inv_step_transform(len(self.data))
