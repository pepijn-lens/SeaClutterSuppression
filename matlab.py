import scipy.io
import numpy as np

# Replace 'your_file.mat' with the path to your .mat file
mat_file_path = '/Users/pepijnlens/Documents/transformers/MaritimeRadarPPI/MaritimeRadarPPI.mat'

# Load the .mat file
mat = scipy.io.loadmat(mat_file_path)

# Display the conten
# print(mat.keys())


    # plot the first image
import matplotlib.pyplot as plt
import os 

plt.figure()
plt.imshow(mat['img1'])
plt.savefig('images/img1.png')

if not os.path.exists('images'):
    os.makedirs('images')

for i in range(1, 85):
    plt.figure()
    plt.imshow(mat['resp' + str(i)])
    plt.savefig('images/resp' + str(i) + '.png')
    plt.close()


