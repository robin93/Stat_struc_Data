import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale

"""functions"""
def question(input_list):
	for choice in input_list:
		if choice == 1:
			y_matrix = np.matrix('15;9;3;25;7;13')
			z_matrix = np.matrix('1,10;1,5;1,7;1,19;1,11;1,8')
			print 'y_matrix\n', y_matrix
			print 'z_matrix\n',z_matrix
			#http://docs.scipy.org/doc/numpy/reference/routines.linalg.html
			beta_estimate = (np.linalg.inv(np.transpose(z_matrix)*z_matrix))*np.transpose(z_matrix)*y_matrix
			print 'beta_estimate\n',beta_estimate
			fitted_matrix = z_matrix*beta_estimate
			print 'fitted_matrix\n',fitted_matrix
			residuals = y_matrix - z_matrix*beta_estimate
			print 'residuals\n',residuals
			sum_of_residuals = np.transpose(residuals)*residuals
			print 'sum_of_residuals -- ',sum_of_residuals

		if choice == 2:
			y_matrix = np.matrix('15;9;3;25;7;13')
			z_matrix = np.matrix('10.0,2.0;5.0,3.0;7.0,3.0;19.0,6.0;11.0,7.0;18.0,9.0')
			row_means = np.mean(z_matrix,axis =0)
			row_std = np.std(z_matrix,axis=0)
			print 'row_means\n',row_means
			print 'row_std\n',row_std
			stand_z_matrix = (z_matrix-row_means)/row_std
			print 'stand_z_matrix\n',stand_z_matrix
			##http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler			
			#scikit_scaled_matrix = scale(z_matrix, axis=0)
			#print 'scikit_scaled_matrix\n',scikit_scaled_matrix
			standard_beta_estimate = (np.linalg.inv(np.transpose(stand_z_matrix)*stand_z_matrix))*np.transpose(stand_z_matrix)*y_matrix
			print 'standard_beta_estimate\n',standard_beta_estimate

			original_beta_estimate = (np.linalg.inv(np.transpose(z_matrix)*z_matrix))*np.transpose(z_matrix)*y_matrix
			print 'original_beta_estimate\n',original_beta_estimate




"""Execution"""
question([2])