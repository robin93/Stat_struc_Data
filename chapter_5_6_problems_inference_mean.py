#imports
import numpy as np
from scipy.stats import f,t
import scipy.stats as stats
import pylab
from numpy.linalg import inv
import os
cwd = os.getcwd()

#functions

def chapter_5(input_list):
	confidence_int = 0.975
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
			print 'f.ppf(confidence_int,p,n-p)',f.ppf(0.95,p,n-p)
			print '(n-1)*p/(n*(n-p))',float((n-1)*p)/(n*(n-p))
			#http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.f.html
			c_square = (float((n-1)*p)/((n-p)))*(f.ppf(0.95,p,n-p))
			print 'c_square',c_square

			#simulataneous confidence intervals
			a1,a2 = np.matrix('1;0'),np.matrix('0;1')
			#Tsquare confidence interval
			half_width1 = np.sqrt(((np.transpose(a1)*cov_matrix*a1)*c_square)/n)
			upperbound_1,lowerbound_1 = x1_data.mean() + half_width1 , x1_data.mean() - half_width1
			print 'T-square upperbound_1,lowerbound_1',upperbound_1,lowerbound_1
			half_width2 = np.sqrt((np.transpose(a2)*cov_matrix*a2)*(c_square)/n)
			upperbound_2,lowerbound_2 = x2_data.mean() + half_width2 , x2_data.mean() - half_width2
			print 'Tsquare upperbound_2,lowerbound_2',upperbound_2,lowerbound_2	

			#Bonferroni confidence interval
			half_width1 = np.sqrt((cov_matrix[0,0])/n)*(t.ppf(0.5+ (confidence_int)/2,n-1))
			upperbound_1,lowerbound_1 = x1_data.mean() + half_width1 , x1_data.mean() - half_width1
			print 'Bonferroni upperbound_1,lowerbound_1',upperbound_1,lowerbound_1
			half_width2 = np.sqrt((cov_matrix[1,1])/n)*(t.ppf(0.5+ (confidence_int)/2,n-1))
			upperbound_2,lowerbound_2 = x2_data.mean() + half_width2 , x2_data.mean() - half_width2
			print 'Bonferroni upperbound_2,lowerbound_2',upperbound_2,lowerbound_2

			stats.probplot(x1_data, dist="norm", plot=pylab)
			pylab.show()








#execution
chapter_5([1])