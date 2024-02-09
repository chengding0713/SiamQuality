import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
data_folder = 'G:/Downstream_task/blood_pressure/'

# x_test = np.load(data_folder + 'X_test_mimic.npy')
quality_index = np.load(data_folder + 'test_quality_index_vital.npy')
true_label = np.load(data_folder + 'y_test_vital_dbp.npy')
predicted_label = np.load(data_folder + 'predicted_labels_vital_dbp.npy')

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
    print('threshold: %f sample number: %f MAE: %f'%(threshold,len(quality_bins[threshold][0]), mean_absolute_error(quality_bins[threshold][0], quality_bins[threshold][1])))

print('threshold: %f sample number: %f MAE: %f'%(1.0,len(true_label),mean_absolute_error(true_label, predicted_label)))