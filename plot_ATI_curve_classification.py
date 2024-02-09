import numpy as np
from sklearn.metrics import mean_absolute_error,f1_score
import matplotlib.pyplot as plt
data_folder = 'G:/Downstream_task/AF/'

# x_test = np.load(data_folder + 'standardized_x_test.npy')
quality_index = np.load(data_folder + 'test_quality_index2.npy')
true_label = np.load(data_folder + 'stanford_test_label.npy')
predicted_label = np.load(data_folder + 'predicted_labels2.npy')

quality_bins = {}
for threshold in np.arange(0.1,1,0.1):
    quality_bins[threshold] = [[],[]]
#
for index, quality in enumerate(quality_index):
    percentage_of_artifacts = np.sum(quality<=0.5)/len(quality)
    # plt.plot(x_test[index])
    # plt.show()
    # print('hello')
    for threshold in np.arange(0.1,1,0.1):
        if percentage_of_artifacts < threshold:
            quality_bins[threshold][0].append(true_label[index])
            quality_bins[threshold][1].append(predicted_label[index])


for threshold in np.arange(0.1,1,0.1):
    quality_bins[threshold][0] = np.array(quality_bins[threshold][0])
    quality_bins[threshold][1] = np.array(quality_bins[threshold][1])
    print('threshold: %f sample number: %f F1 Score: %f'%(threshold,len(quality_bins[threshold][0]), f1_score(quality_bins[threshold][0], quality_bins[threshold][1], average='weighted')))

print('threshold: %f sample number: %f  F1 Score: %f'%(1.0,len(true_label),f1_score(true_label, predicted_label, average='weighted')))