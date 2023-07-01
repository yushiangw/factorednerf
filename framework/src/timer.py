import numpy as np
import time 



class MyTimer():

    def __init__(self, verbose ):
        super().__init__()

        self.start=None
        self.msg=''
        self.verbose=verbose

    def tstart(self, msg):
        self.start=time.time()
        self.msg=msg

    def tend(self):
        sec = time.time()-self.start 

        if self.verbose:
            print(f'[{self.msg:25}]:\t{sec:10.2f}sec')
        self.msg=None

        return sec