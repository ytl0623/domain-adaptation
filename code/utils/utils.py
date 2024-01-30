import numpy as np
import matplotlib.pyplot as plt
import torch
import os
#from google.colab import output

def plotLosses(class_loss, source_loss, target_loss, n_epochs=30, show=False) : 
	epochs = range(n_epochs)
	
	plt.figure()
	plt.xlabel('Iterations')
	plt.ylabel('Loss')
	plt.plot(epochs, class_loss, 'b--', label="classifier")
	plt.plot(epochs, source_loss, 'g--', label="discriminator source")
	plt.plot(epochs, target_loss, 'r--', label="discriminator target")
	plt.legend()
	
	if show: 
		plt.savefig('losses.png', dpi=250)
	return

def plotImageDistribution(data1, data2, dataset_names, classes_names, show=False):
	# concatenate datasets
	data = np.concatenate((data1, data2))
	
	# count element per class
	unique, counts = np.unique(data, return_counts=True)
	
	# for each domain
	unique, counts1 = np.unique(data1, return_counts=True)
	unique, counts2 = np.unique(data2, return_counts=True)

	if show: 
		print("------ Some statistics ------")
		print('Total images:', np.sum(counts))
		print('Number of classes:', len(unique))
		print('Classes:', unique)
		print('Classes Names:', classes_names) 
		print()
		print('Total images per class:', counts)
		print('Mean images per class:', counts.mean())
		print('Std images per class:', counts.std())
		print()
		print('Total images per domain/dataset:')
		print(f"ADC Dataset: {len(data1)}")
		print(f"DWI Dataset: {len(data2)}")
		print()
		print('Element per class for each domain:')
		for name,count in zip(dataset_names,[counts1,counts2]) : 
			print(f'{name}_dataset: {count}')

	fig, ax = plt.subplots(figsize=(10,7))

	width=0.18

	plt.bar(unique-2*(width)+(width/2), counts1, width=width, color='#FF8F77', linewidth=0.5, label='ADC')
	plt.bar(unique-(width/2), counts2, width=width, color='#FFDF77', linewidth=0.5, label='DWI')

	ax.set_xticks(unique)
	classes = ['0', '1']
	ax.set_xticklabels(classes)

	plt.grid(alpha=0.2, axis='y')

	plt.legend()
	if show: 
		plt.show()
	plt.savefig('distribution.png', dpi = 250)
	return
