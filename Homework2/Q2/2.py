import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.io import wavfile

def conv_func(data, filter):
    output = np.zeros(len(data) + len(filter) - 1)
    
    for i in range(len(data) + len(filter) - 1):

      for j in range(len(filter)):
        if((i-j) < 0 or (i-j) >= len(data)):
          continue
        else:
          output[i] += filter[j] * data[i-j]

    return output

def show_spectrum(input_data, n, rate, path, type):
  plt.clf()
  fourier_transform = np.fft.fft(input_data)
  frequency_values = np.fft.fftfreq(len(input_data), 1./ rate)
  
  if(n == -1):
    plt.plot(frequency_values, np.abs(fourier_transform))
  else:
    plt.plot(frequency_values[:int(n)], np.abs(fourier_transform[:int(n)]))
  if(type == 0):
    plt.xlim(0, 2000)   # Limit x-axis from 2 to 8
    plt.ylim(0, 27000) 
  elif(type == 1):
    plt.xlim(-2500, 2500)   # Limit x-axis from 2 to 8
    plt.ylim(-0.1, 1.1)   
  plt.savefig(path)

def low_pass_filter(fs, N, cut_off_f):
  fc = cut_off_f / fs
  middle = int((N-1)/2)

  filter = np.zeros((N))

  for i in range(-middle, middle+1):
    tmp = i + middle
    if(i==0):
      filter[tmp] = 2*fc
    else:
      filter[tmp] = math.sin(2*math.pi*fc*i) / (math.pi*i)
  for i in range(N):
    filter[i] = filter[i] * (0.42 - 0.5 * math.cos(2*math.pi*i/(N-1)) + 0.08 * math.cos(4*math.pi*i/(N-1)))
  return filter

def high_pass_filter(fs, N, cut_off_f):
  fc = cut_off_f / fs
  middle = int((N-1)/2)

  filter = np.zeros((N))

  for i in range(-middle, middle+1):
    tmp = i + middle
    if(i==0):
      filter[tmp] = 1-2*fc
    else:
      filter[tmp] = -math.sin(2*math.pi*fc*i) / (math.pi*i)
  for i in range(N):
    filter[i] = filter[i] * (0.42 - 0.5 * math.cos(2*math.pi*i/(N-1)) + 0.08 * math.cos(4*math.pi*i/(N-1)))
  return filter

def band_pass_filter(fs, N, cut_off_f1, cut_off_f2):
  fc1 = cut_off_f1 / fs
  fc2 = cut_off_f2 / fs
  middle = int((N-1)/2)

  filter = np.zeros((N))

  for i in range(-middle, middle+1):
    tmp = i + middle
    if(i==0):
      filter[tmp] = 2*fc2 - 2*fc1
    else:
      filter[tmp] = math.sin(2*math.pi*fc2*i) / (math.pi*i) - math.sin(2*math.pi*fc1*i) / (math.pi*i)
  for i in range(N):
    filter[i] = filter[i] * (0.42 - 0.5 * math.cos(2*math.pi*i/(N-1)) + 0.08 * math.cos(4*math.pi*i/(N-1)))
  return filter

def apply_filter(input_data, rate, window_size, f1, f2, type_of_filter, path):
  if(type_of_filter == 'lowpass'):
    filter = low_pass_filter(rate, window_size, f1)
  elif(type_of_filter == 'highpass'):
    filter = high_pass_filter(rate, window_size, f2)
  elif(type_of_filter == 'bandpass'):
    filter = band_pass_filter(rate, window_size, f1, f2)
  filter_signal = conv_func(input_data, filter)
  wavfile.write(path, rate, filter_signal.astype(np.float32))
  return filter, filter_signal

def plot_shape(filter, path):
  plt.clf()
  plt.plot(filter)
  plt.savefig(path)

def reduce_rate(input_data, new_rate, old_rate, path):
    new_samples = len(input_data) * new_rate / old_rate
    new_samples = math.floor(new_samples)
    output_data = np.zeros((new_samples))
    for i in range(new_samples):
        start_index = i * old_rate / new_rate
        end_index = (i+1) * old_rate / new_rate
        start_index = math.floor(start_index)
        end_index = math.floor(end_index)
        if(end_index > len(input_data)):
            end_index = len(input_data)
        tmp = 0
        for j in range(start_index, end_index):
            tmp += input_data[j]
        tmp = tmp / (end_index - start_index)
        output_data[i] = tmp
    wavfile.write(path, new_rate, output_data.astype(np.float32))
    return
  
# Read audio signal
org_rate, org_data = wavfile.read('Q2/HW2_Mix.wav')

# Show the spectrum of input audio signal
show_spectrum(org_data, 30000, org_rate, "Q2/output/input.png",0)

# Create filter and apply filter to audio signal
filter1, filter_signal1 = apply_filter(org_data, org_rate, 1751, 400, 720, 'lowpass', "Q2/output/Lowpass_400.wav")
filter2, filter_signal2 = apply_filter(org_data, org_rate, 1751, 400, 720, 'highpass', "Q2/output/Highpass_720.wav")
filter3, filter_signal3 = apply_filter(org_data, org_rate, 1751, 400, 720, 'bandpass', "Q2/output/Bandpass_400_720.wav")

# Show the spectrum of audio signal after apply filter
show_spectrum(filter_signal1, 30000, org_rate, "Q2/output/output_by_Lowpass.png",0)
show_spectrum(filter_signal2, 30000, org_rate, "Q2/output/output_by_Highpass.png",0)
show_spectrum(filter_signal3, 30000, org_rate, "Q2/output/output_by_Bandpass.png",0)

# Show the filter spectrum
show_spectrum(filter1, -1, org_rate, "Q2/output/Lowpass_spectrum.png",1)
show_spectrum(filter2, -1, org_rate, "Q2/output/Highpass_spectrum.png",1)
show_spectrum(filter3, -1, org_rate, "Q2/output/Bandpass_spectrum.png",1)

# Show the shape of filter
plot_shape(filter1, "Q2/output/Lowpass_shape.png")
plot_shape(filter2, "Q2/output/Highpass_shape.png")
plot_shape(filter3, "Q2/output/Bandpass_shape.png")

# Reduce the sample rate of audio signal
reduce_rate(filter_signal1, 2000, org_rate, "Q2/output/Lowpass_400_2kHZ.wav")
reduce_rate(filter_signal2, 20000, org_rate, "Q2/output/Highpass_720_2kHZ.wav")
reduce_rate(filter_signal3, 2000, org_rate, "Q2/output/Bandpass_400_720_2kHZ.wav")

# One echo
tmp4 = np.zeros((len(filter_signal1) - 3200))
for i in range(3200, len(filter_signal1)):
  tmp4[i-3200] = filter_signal1[i] + 0.8 * filter_signal1[i-3200]
wavfile.write("Q2/output/Echo_one.wav", org_rate, tmp4.astype(np.float32))

# Multiple Echo
tmp5 = np.zeros((len(filter_signal1)))
for i in range(3200):
  tmp5[i] = filter_signal1[i]
for i in range(3200, len(filter_signal1)):
  tmp5[i] = filter_signal1[i] + 0.8 * tmp5[i-3200]
wavfile.write("Q2/output/Echo_multiple.wav", org_rate, tmp5.astype(np.float32))