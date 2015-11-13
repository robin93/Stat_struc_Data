#Some resources:
#http://sebastianraschka.com/Articles/2014_pca_step_by_step.html



import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
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







#execution
question([10])