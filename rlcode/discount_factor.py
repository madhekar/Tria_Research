import numpy as np
import matplotlib.pyplot as plt

labelsize = 12

width = 5
height = width / 1.618

plt.rc('font', family='serif')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=labelsize)
plt.rc('ytick', labelsize=labelsize)
plt.rc('axes', labelsize=labelsize)

gammas = np.arange(0,1,1.0/100)

i = 1
def animator(gamma, i):
	fig1, ax = plt.subplots()
	weight_terms = [np.power(gamma, k) for k in range(0,50)]
	plt.stem(weight_terms, use_line_collection= True)
	plt.xticks([])
	plt.title('$\gamma$ = {}'.format(round(gamma,2)))
	plt.ylabel('Weight value')
	plt.xlabel('Gamma power sequence '+ '$\gamma^0, \gamma^1, \gamma^2,\gamma^3, ...,\gamma^{49}$')
	fig1.set_size_inches(width, height)
	plt.savefig('dynamicGammaSequence'+str("{0:0=4d}".format(i))+'.png', dpi = 200)
	plt.close()
	
for gamma in gammas:
	animator(gamma, i)
	print('Saving frame: ', i)
	i+=1
