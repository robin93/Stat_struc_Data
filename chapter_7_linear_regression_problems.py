import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm,t

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
		if choice == 9:
			z_matrix = np.matrix('1,-2;1,-1;1,0;1,1;1,2')
			y1,y2 = np.matrix('5;3;4;2;1'),np.matrix('-3;-1;-1;2;3')
			beta1 = (np.linalg.inv(np.transpose(z_matrix)*z_matrix))*np.transpose(z_matrix)*y1
			beta2 = (np.linalg.inv(np.transpose(z_matrix)*z_matrix))*np.transpose(z_matrix)*y2
			y1_est,y2_est = z_matrix*beta1,z_matrix*beta2
			residual1,residual2 = y1-y1_est,y2-y2_est
			Y,Y_est,residuals = np.concatenate((y1,y2),axis = 1),np.concatenate((y1_est,y2_est),axis = 1),np.concatenate((residual1,residual1),axis = 1)
			print 'Yt*Y\n',np.transpose(Y)*Y
			print 'Y_est_trans*Y_est + residuals\n',np.transpose(Y_est)*Y_est + np.transpose(residuals)*residuals
		if choice == 10:
			z_matrix = np.matrix('1,-2;1,-1;1,0;1,1;1,2')
			y1,y2 = np.matrix('5;3;4;2;1'),np.matrix('-3;-1;-1;2;3')
			beta1 = (np.linalg.inv(np.transpose(z_matrix)*z_matrix))*np.transpose(z_matrix)*y1
			print 'beta1\n',beta1
			beta2 = (np.linalg.inv(np.transpose(z_matrix)*z_matrix))*np.transpose(z_matrix)*y2
			y1_est,y2_est = z_matrix*beta1,z_matrix*beta2
			residual1,residual2 = y1-y1_est,y2-y2_est
			zo = np.matrix('1;0.5')
			n = 5
			r = 1
			multiplier = t.ppf(0.975,n-r-1)
			print 'multiplier',multiplier
			s_square = (np.transpose(y1-z_matrix*beta1)*(y1-z_matrix*beta1))/(n-r-1)
			print 's_square',s_square
			half_width = multiplier *np.sqrt((np.transpose(zo)*(np.linalg.inv(np.transpose(z_matrix)*z_matrix))*zo)*s_square)
			print 'np.linalg.inv(np.transpose(z_matrix)*z_matrix))\n',np.linalg.inv(np.transpose(z_matrix)*z_matrix)
			print 'half width',half_width
			upperbound, lowerbound = np.transpose(zo)*beta1 + half_width,np.transpose(zo)*beta1 - half_width
			print 'upperbound,lowerbound',upperbound,lowerbound

			half_width2 = multiplier *np.sqrt(1 + (np.transpose(zo)*(np.linalg.inv(np.transpose(z_matrix)*z_matrix))*zo)*s_square)
			upperbound, lowerbound = np.transpose(zo)*beta1 + half_width2,np.transpose(zo)*beta1 - half_width2
			print 'upperbound2,lowerbound2',upperbound,lowerbound






"""Execution"""
question([10])