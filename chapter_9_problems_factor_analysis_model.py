import numpy as np
import matplotlib.pyplot as plt

"""  functions """
def question(input_list):
	for choice in input_list:
		if choice == 1:
			rho = np.matrix('1.0,0.63,0.45;0.63,1.0,0.35;0.45,0.35,1.0')
			psi = np.matrix('0.19,0,0;0,0.51,0;0,0,0.75')
			L = np.matrix('0.9;0.7;0.5')
			Lt = np.transpose(L)
			print 'L*t(L)\n',L*Lt
			print 'psi\n',psi
			print 'LLt + psi\n',(L*Lt) + psi
			print 'actual rho\n',rho
		if choice == 2:
			rho = np.matrix('1.0,0.63,0.45;0.63,1.0,0.35;0.45,0.35,1.0')
			psi = np.matrix('0.19,0,0;0,0.51,0;0,0,0.75')
			L = np.matrix('0.9;0.7;0.5')
			Lt = np.transpose(L)
			h1,h2,h3 = np.sum((L*Lt)[0]),np.sum((L*Lt)[1]),np.sum((L*Lt)[2])
			print 'h1,h2,h3',h1,h2,h3
			print "Proportions of the variance of the first variable by the common factor",100*h1/(h1+psi[0,0])
			print "Proportions of the variance of the second variable by the common factor",100*h2/(h2+psi[1,1])
			print "Proportions of the variance of the third variable by the common factor",100*h3/(h3+psi[2,2])
		if choice == 3:
			lambda_1 = 1.96
			eigen_1 = np.matrix('0.625;0.593;0.507')
			L_hat = np.sqrt(lambda_1)*eigen_1
			print 'estimate of the loading vector using the first eigen vectors\n', L_hat
			rho = np.matrix('1.0,0.63,0.45;0.63,1.0,0.35;0.45,0.35,1.0')
			psi_estimate = np.diagonal(rho - L_hat*np.transpose(L_hat))
			print 'L_hat*np.transpose(L_hat)\n',L_hat*np.transpose(L_hat)
			print 'Estimate of psi using the principal component method is: \n',np.diag(psi_estimate)
			h_new1 = np.sum(L_hat*np.transpose(L_hat)[0])
			print 'Proportions of the variance of the first variable by the common factor : ',h_new1/(h_new1+np.diag(psi_estimate)[0,0])
		if choice ==4:
			rho = np.matrix('1.0,0.63,0.45;0.63,1.0,0.35;0.45,0.35,1.0')
			print 'original rho matrix\n',rho
			psi = np.matrix('0.19,0,0;0,0.51,0;0,0,0.75')
			print 'original psi matrix\n',psi
			reduced_rho = rho - psi
			print 'reduced_rho matrix\n',reduced_rho
			modified_loading_vector = np.sqrt(np.diag(reduced_rho))
			print 'modified_loading_vector\n',modified_loading_vector
		if choice == 9:
			L1 = [0.64,0.50,0.46,0.17,-0.29,-0.29,-0.49,-0.52,-0.60]
			L2 = [0.02,-0.06,-0.24,0.74,0.66,-0.08,0.20,-0.03,-0.17]
			print L1,'\n',L2
			fig = plt.figure()
			ax1 = fig.add_subplot(121)
			ax1.scatter(L1,L2,color='blue',s=60,edgecolor='none'),ax1.grid(True),ax1.set_xlim([-0.8,0.8]),ax1.set_ylim([-0.8,0.8])#,ax1.set_aspect(1./ax1.get_data_ratio())
			plt.axhline(0, color='black')
			plt.axvline(0, color='black')
			plt.show()
			

""" Execution"""
question([9])
