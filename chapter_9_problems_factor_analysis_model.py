import numpy as np

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
""" Execution"""
question([1])
