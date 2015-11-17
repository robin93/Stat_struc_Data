#imports
import numpy as np
from scipy.stats import f
from numpy.linalg import inv
import os
cwd = os.getcwd()

#functions

def chapter_5(input_list):
	confidence_int = 0.95
	for choice in input_list:
		if choice == 1:
			data = np.loadtxt(os.path.join(cwd,'T5-12.dat'))
			x1_data = np.loadtxt(os.path.join(cwd,'T5-12.dat'))[:,0]
			x2_data = np.loadtxt(os.path.join(cwd,'T5-12.dat'))[:,1]
			print 'x1_data',x1_data
			print 'x2_data',x2_data
			print 'mean_of_length data',x1_data.mean()
			print 'mean of wing length data',x2_data.mean()
			n = x1_data.shape[0]
			print 'n',n
			p = 2
			cov_matrix = np.cov(data,rowvar=0)
			#http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.inv.html
			inv_of_cov = inv(cov_matrix)
			print 'covariance matrix\n',cov_matrix
			print 'inverse of covariance matrix\n',inv_of_cov
			print 'f.ppf(confidence_int,p,n-p)',f.ppf(confidence_int,p,n-p)
			print '(n-1)*p/(n*(n-p))',float((n-1)*p)/(n*(n-p))
			#http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.f.html
			c_square = (float((n-1)*p)/(n*(n-p)))*(f.ppf(confidence_int,p,n-p))
			print 'c_square',c_square

			#simulataneous confidence intervals
			a1,a2 = np.matrix('1;0'),np.matrix('0;1')
			print 'a1,a2',a1,a2






#execution
chapter_5([1])