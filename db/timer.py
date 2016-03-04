# http://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python:
# call as: 
# with Timer("timer_name"): 
#     do_code
import time
class Timer(object):
    def __init__(self, name=None):
        self.name = name
    
    def __enter__(self):
        self.tstart = time.time()
    
    def __exit__(self, type, value, traceback):
        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %.2f seconds' % (time.time() - self.tstart)
