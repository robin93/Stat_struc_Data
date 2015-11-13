#Some resources:
#http://sebastianraschka.com/Articles/2014_pca_step_by_step.html



import numpy as np
import pandas as pd
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
cwd = os.getcwd()


#functions 
def question(input_list):
	for choice in input_list:
		if choice == 1:
			sigma = np.matrix('5,2;2,2')
			corr_mat = np.matrix('1,0.2;0.2,1')
			values, vectors = LA.eig(corr_mat)
			print 'Eigen values', values
			print 'Eigen vectors\n', vectors
			eigen_value1,eigen_value2 = values[0],values[1]
			print 'Proportion of variance explained by the first Eigen vector', float(eigen_value1/(eigen_value1+eigen_value2))*100
			print 'Proportion of variance explained by the first Eigen vector', float(eigen_value2/(eigen_value1+eigen_value2))*100

			#vectors for the covariance matrix
			values1, vectors1 = LA.eig(sigma)
			print 'Eigen values from the sigma matrix', values1
			print 'Eigen vectors from the sigma matrix\n', vectors1

			print 'correlation between pc1 and org1', (values1[0]*vectors1[0,0])/(np.sqrt(values1[0] *sigma[0,0]))
			print 'correlation between pc1 and org2', (values1[0]*vectors1[0,1])/(np.sqrt(values1[0] *sigma[1,1]))
			print 'correlation between pc2 and org1', (values1[1]*vectors1[1,0])/(np.sqrt(values1[1] *sigma[0,0]))
		elif choice == 2:
			sigma = np.matrix('2,0,0;0,4,0;0,0,4')
			values, vectors = LA.eig(sigma)
			print 'Eigen values', values
			print 'Eigen vectors\n',vectors
		elif choice ==6:
			data = np.loadtxt(os.path.join(cwd,'P1-4.DAT'))[:,[0,1]]  #http://stackoverflow.com/questions/4455076/numpy-access-an-array-by-column
			print data
			row_means = np.mean(data,axis =0)
			print 'row means', row_means
			cov_matrix = np.cov(data,rowvar=0)
			print 'cov matrix',cov_matrix
			values,vectors = LA.eig(cov_matrix)
			print 'Eigen Vectors\n',vectors
			print 'variances are \n',values
			print 'correlation between pc1 and org1', (values[0]*vectors[0,0])/(np.sqrt(values[0] *cov_matrix[0,0]))
			print 'correlation between pc1 and org2', (values[1]*vectors[0,1])/(np.sqrt(values[0] *cov_matrix[1,1]))
		elif choice ==7:
			data = np.loadtxt(os.path.join(cwd,'P1-4.DAT'))[:,[0,1]]
			cor_matrix = np.matrix('1,0.6869;0.6869,1')
			values,vectors = LA.eig(cor_matrix)
			print 'Eigen Vectors\n',vectors
			print 'variances are \n',values
			print 'correlation between pc1 and org1', (values[0]*vectors[0,0])/(np.sqrt(values[0] *cor_matrix[0,0]))
			print 'correlation between pc1 and org2', (values[1]*vectors[0,1])/(np.sqrt(values[0] *cor_matrix[1,1]))
			fig = plt.figure()
			ax1 = fig.add_subplot(121)
			ax1.scatter(data[:,0],data[:,1],color='blue',s=60,edgecolor='none')#,ax1.grid(True),ax1.set_xlim([-0.8,0.8]),ax1.set_ylim([-0.8,0.8])#,ax1.set_aspect(1./ax1.get_data_ratio())
			plt.axhline(0, color='black')
			plt.axvline(0, color='black')
			plt.show()
		elif choice == 10:
			data = np.loadtxt(os.path.join(cwd,'T8-4.DAT'))
			print data
			cov_matrix = np.cov(data,rowvar=0)
			print 'sample covariance matrix\n',cov_matrix
			values,vectors = LA.eig(cov_matrix)
			print 'Eigen Vectors\n',vectors
			print 'variances are \n',values
			print 'proportion of variance', values*100/np.sum(values)
			#confidence intervals for the eigenvalues
			observations = data.shape[0]
			n_columns = data.shape[1]
			confidence_fraction = 0.025
			print '# of observations',observations
			print '# of columns', n_columns
			print 'norm.ppf(1 - (confidence_fraction/n_columns))',norm.ppf(1 - (confidence_fraction/n_columns))
			print 'np.sqrt(2/observations)',np.sqrt(2/float(observations))
			print 'lower bounds', values/(    1 +    norm.ppf(1 - (confidence_fraction/n_columns))    *   np.sqrt(2/float(observations))  )
			print 'upper bounds', values/(    1 -    norm.ppf(1 - (confidence_fraction/n_columns))    *   np.sqrt(2/float(observations))  )
		elif choice == 13:
			data = np.loadtxt(os.path.join(cwd,'T1-7.DAT'))
			#print data
			cov_matrix = np.cov(data,rowvar=0)
			cor_matrix = np.corrcoef(data,rowvar=0)
			print 'covariance matrix\n',cov_matrix
			print 'correlation matrix\n',cor_matrix
			values,vectors = LA.eig(cov_matrix)
			print 'Eigen Vectors\n',vectors
			print 'variances are \n',values
			print 'proportion of variance', values*100/np.sum(values)
		elif choice == 18:
			#data = np.loadtxt(os.path.join(cwd,'T1-9.dat'))
			data = pd.read_table(os.path.join(cwd,'T1-9.dat'))
			np_data = data.iloc[:,1:8]
			cor_matrix = np.corrcoef(np_data,rowvar=0)
			print 'correlation matrix',cor_matrix
			values,vectors = LA.eig(cor_matrix)
			print 'Eigen Vectors\n',vectors
			print 'variances are \n',values
			print 'proportion of variance', values*100/np.sum(values)

			#standardize the variables
			row_means = np.mean(np_data,axis =0)
			row_std = np.std(np_data,axis=0)
			stand_data = (np_data-row_means)/row_std
			cor_matrix = np.corrcoef(stand_data,rowvar=0)
			print 'correlation matrix',cor_matrix
			values,vectors = LA.eig(cor_matrix)
			print 'Eigen Vectors\n',vectors
			print 'variances are \n',values
			print 'proportion of variance', values*100/np.sum(values)
			pc1 = np.matrix(vectors[0])
			stand_data = np.matrix(stand_data)
			print pc1
			first_pc_values = stand_data*np.transpose(pc1)
			print 'first_pc_values',first_pc_values
			s1 = pd.Series(first_pc_values,name='values')
			country = pd.concat([data.iloc[:,0],s1],axis =1)
			print country

















#execution
question([18])