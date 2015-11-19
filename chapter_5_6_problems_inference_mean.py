#imports
import numpy as np
from scipy.stats import f,t,chi2,norm
import scipy.stats as stats
import pylab
from numpy.linalg import inv
import matplotlib.pyplot as plt
import os
cwd = os.getcwd()

#functions

def scatterplots(list1,list2):
	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	ax1.scatter(list1,list2,color='blue',s=60,edgecolor='none')#,ax1.grid(True),ax1.set_xlim([-0.8,0.8]),ax1.set_ylim([-0.8,0.8])#,ax1.set_aspect(1./ax1.get_data_ratio())
	plt.axhline(0, color='black')
	plt.axvline(0, color='black')
	plt.show()


def chapter_5(input_list):
	confidence_int = 0.975
	for choice in input_list:
		if choice == 20:
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

			print x1_data
			stats.probplot(x1_data, dist="norm", plot=pylab)
			pylab.show()

		elif choice == 22: 
			#http://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html
			data = np.loadtxt(os.path.join(cwd,'T6-10.dat'),usecols = (0,1,2))
			fuel_data,repair_data,capital_data = data[:,0],data[:,1],data[:,2]
			# #QQ plots before removal of outliers
			# stats.probplot(fuel_data, dist="norm", plot=pylab),pylab.show()
			# stats.probplot(repair_data, dist="norm", plot=pylab),pylab.show()
			# stats.probplot(capital_data, dist="norm", plot=pylab),pylab.show()
			# #scatter plots before removal of outliers
			# scatterplots(fuel_data,repair_data),scatterplots(capital_data,repair_data),scatterplots(fuel_data,capital_data)

			#removing outliers
			#http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.select.html
			#fuel_data2 = np.select([fuel_data<20],[fuel_data])
			outliers_removed_data = data[(data[:,0]<20)&(data[:,0]>6)&(data[:,1]<20)&(data[:,1]>3)&(data[:,2]<25)&(data[:,2]>5)]
			#fuel_data2 = fuel_data[(fuel_data < 20) & (fuel_data > 6)]   #http://stackoverflow.com/questions/3806878/subsetting-data-in-python
			fuel_data2,repair_data2,capital_data2 = outliers_removed_data[:,0],outliers_removed_data[:,1],outliers_removed_data[:,2]
			# stats.probplot(fuel_data2, dist="norm", plot=pylab),pylab.show()
			# stats.probplot(repair_data2, dist="norm", plot=pylab),pylab.show()
			# stats.probplot(capital_data2, dist="norm", plot=pylab),pylab.show()
			#repair_data2,capital_data2 = np.select([fuel_data<20],[fuel_data])

			#confidence interval calculations
			n = outliers_removed_data.shape[0]
			cov_matrix = np.cov(outliers_removed_data,rowvar=0)
			#bonferroni confidence interval
			half_width1 = np.sqrt((cov_matrix[0,0])/n)*(t.ppf(0.5+ (confidence_int)/2,n-1))
			upperbound_1,lowerbound_1 = fuel_data2.mean() + half_width1 , fuel_data2.mean() - half_width1
			print 'Bonferroni fuel data lowerbound_1,upperbound_1',lowerbound_1,upperbound_1
			half_width2 = np.sqrt((cov_matrix[1,1])/n)*(t.ppf(0.5+ (confidence_int)/2,n-1))
			upperbound_2,lowerbound_2 = repair_data2.mean() + half_width2 , repair_data2.mean() - half_width2
			print 'Bonferroni repair_data2 lowerbound_2,upperbound_2',lowerbound_2,upperbound_2
			half_width3 = np.sqrt((cov_matrix[2,2])/n)*(t.ppf(0.5+ (confidence_int)/2,n-1))
			upperbound_3,lowerbound_3 = capital_data2.mean() + half_width3 , capital_data2.mean() - half_width3
			print 'Bonferroni capital_data2 lowerbound_2,upperbound_2',lowerbound_3,upperbound_3
			
			#T square confidence interval calculations
			p = 3
			a1,a2,a3 = np.matrix('1;0;0'),np.matrix('0;1;0'),np.matrix('0;0;1')
			c_square = (float((n-1)*p)/((n-p)))*(f.ppf(0.95,p,n-p))
			half_width1 = np.sqrt(((np.transpose(a1)*cov_matrix*a1)*c_square)/n)
			upperbound_1,lowerbound_1 = fuel_data2.mean() + half_width1 , fuel_data2.mean() - half_width1
			print 'fuel data T-square upperbound_1,lowerbound_1',lowerbound_1,upperbound_1
			half_width2 = np.sqrt(((np.transpose(a2)*cov_matrix*a2)*c_square)/n)
			upperbound_2,lowerbound_2 = repair_data2.mean() + half_width2 , repair_data2.mean() - half_width2
			print 'repair_data2 T-square upperbound_1,lowerbound_1',lowerbound_2,upperbound_2
			half_width3 = np.sqrt(((np.transpose(a3)*cov_matrix*a3)*c_square)/n)
			upperbound_3,lowerbound_3 = capital_data2.mean() + half_width3 , capital_data2.mean() - half_width3
			print 'capital_data2 T-square upperbound_1,lowerbound_1',lowerbound_3,upperbound_3

		elif choice == 30:
			n = 50   #seen from the solutions
			confidence_int = 0.95
			means = np.matrix('0.766;0.508;0.438;0.161')
			cov_matrix = np.matrix('0.856,0.635,0.173,0.096;0.635,0.568,0.128,0.067;0.173,0.127,0.171,0.039;0.096,0.067,0.039,0.043')
			print 'means\n',means
			print 'cov_matrix\n',cov_matrix
			a1,a2,a3,a4 = np.matrix('1;0;0;0'),np.matrix('0;1;0;0'),np.matrix('0;0;1;0'),np.matrix('0;0;0;1')
			a5,a6= np.matrix('1;1;1;1'),np.matrix('1;-1;0;0')
			#simultaneous confidence intervals
			c_square = chi2.ppf(confidence_int,4)
			print 'c_square',c_square
			half_width1 = np.sqrt(((np.transpose(a1)*cov_matrix*a1)*c_square)/n)
			upperbound_1,lowerbound_1 = means[0,0] + half_width1 , means[0,0] - half_width1
			print 'Petroleum Chi-square lowerbound_1,upperbound_1',lowerbound_1,upperbound_1
			# half_width2 = np.sqrt(((np.transpose(a2)*cov_matrix*a2)*c_square)/n)
			# upperbound_2,lowerbound_2 = repair_data2.mean() + half_width2 , repair_data2.mean() - half_width2
			# print 'repair_data2 T-square upperbound_1,lowerbound_1',lowerbound_2,upperbound_2
			# half_width3 = np.sqrt(((np.transpose(a3)*cov_matrix*a3)*c_square)/n)
			# upperbound_3,lowerbound_3 = capital_data2.mean() + half_width3 , capital_data2.mean() - half_width3
			# print 'capital_data2 T-square upperbound_1,lowerbound_1',lowerbound_3,upperbound_3

			half_width5 = np.sqrt(((np.transpose(a5)*cov_matrix*a5)*c_square)/n)
			upperbound_1,lowerbound_1 = means.sum() + half_width5 , means.sum() - half_width5
			print 'Petroleum Chi-square lowerbound_5,upperbound_5',lowerbound_1,upperbound_1

			half_width6 = np.sqrt(((np.transpose(a6)*cov_matrix*a6)*c_square)/n)
			upperbound_1,lowerbound_1 = np.transpose(a6)*means + half_width6 , np.transpose(a6)*means - half_width6
			print 'Petroleum Chi-square lowerbound_5,upperbound_5',lowerbound_1,upperbound_1




			#Bonferroni confidence intervals
			c_square = t.ppf((1-(1 - confidence_int)/12),n-1)
			print 'c_square',c_square 
			half_width1 = c_square * np.sqrt(((np.transpose(a1)*cov_matrix*a1)/n))
			upperbound_1,lowerbound_1 = means[0,0] + half_width1 , means[0,0] - half_width1
			print 'Petroleum Bonferroni lowerbound_1,upperbound_1',lowerbound_1,upperbound_1

def chapter_6(input_list):
	for choice in input_list:
		if choice == 1:
			print "Already explained in the example 6.1"
		if choice == 7:
			means1 = np.matrix('204.4;556.6')
			means2 = np.matrix('130;355')
			cov_matrix1 = np.matrix('13825.3,23823.4;23823.4,73107.4')
			cov_matrix2 = np.matrix('8632,19616.7;19616.7,55964.5')
			print 'cov_matrix1',cov_matrix1
			print 'cov_matrix2',cov_matrix2
			n1,n2 = 45,55
			s_pooled = ((n1-1)*(cov_matrix1) + (n2-1)*(cov_matrix2))/float(n1 + n2 -2)
			print 's_pooled',s_pooled
			mean_diff = means1-means2
			print 'mean_diff',mean_diff
			T_square = (np.transpose(mean_diff)*(((n1*n2/(n2+n1))*(inv(s_pooled)))*mean_diff))
			print 'T_square',T_square
			c_square = (n1+n2-2)*(2)*f.ppf(0.95,2,n1+n2-2-1)/(n1+n2-2-1)
			print 'c_square',c_square
			a = (inv(s_pooled))*(mean_diff)
			print 'Linear Combination',a






#execution
#chapter_5([30])
chapter_6([7])