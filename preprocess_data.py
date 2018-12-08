import numpy as np
import librosa as lb 
import matplotlib.pyplot as plt

SR = 22050
N_FFT = 512
HOP_LENGTH = N_FFT // 2
N_MELS = 64 

def log_melspectrogram(data, log=True, plot=False):

	melspec = lb.feature.melspectrogram(y=data, hop_length = HOP_LENGTH, n_fft = N_FFT, n_mels = N_MELS)

	if log:
		melspec = lb.power_to_db(melspec**2)

	if plot:
		melspec = melspec[np.newaxis, :]
		plt.imshow(melspec.reshape((melspec.shape[1],melspec.shape[2])))
		plt.savefig('melspec.png')
	
	return melspec        

def batch_log_melspectrogram(data_list, log=True):
	melspecs = np.asarray([dp.log_melspectrogram(i, log=log) for i in data_list])
	melspecs = melspecs.reshape(spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[2], 1) #this line may or may not be neccesary idk
	
	return melspecs

# training = np.load('gtzan/gtzan_tr.npy')
# data_tr = np.delete(training, -1, 1)
# label_tr = training[:,-1]

# epoch = 200

# spects = log_melspectrogram(data_tr[0], plot=True)



