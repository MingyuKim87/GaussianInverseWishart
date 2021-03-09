import matplotlib.pyplot as plt
import torch
import numpy as np


###########################
#######데이터 불러오기######
###########################
# 아웃라이어 정보 불러오기: 데이터에 이상이 있다고 판단되는 날들
# 2000년 1월 1일이 index 0으로 기준점이 됨; 하루 증가시 1증가

def plot_test(x, y, file_name='/test_test'):
	'''
        Plots the predictive distribution (mean, var) and context points

        Args:
            All arguments are "NP arrays"
            target_x : [num_points, x_size(dimension)] 
            target_y : [num_points, y_size(dimension)] 
    '''
    
	# type cast : torch.Tensor -> numpy array
	x = x.numpy()
	y = y.numpy()
	
	result_path = "."
    
	RESULT_PATH = result_path
	FILE_NAME = RESULT_PATH + file_name

	for i in range(y.shape[-1]):
		plt.plot(x, y[:,i], 'b*', markersize=2)
    
		# Make a plot pretty
		plt.title("data_set_preview", fontweight='bold', loc='center')
		#plt.xlabel("{} iterations".format(iteration))

		plt.grid(False)
		ax = plt.gca()
		ax.set_facecolor('white')

		plt.savefig(FILE_NAME + "_{}.png".format(i) , dpi=300)

		plt.clf()

def standardize(x, y):
	'''
		Transform original x, y to "z transform"
		all compoenents are torch.Tensor
		
		Args:
			x : vector [num_point, ]
			y : vector [num_point, 5]

		Returns:
			temp_x : [num_point, ]
			temp_y
			(mu_x, std_x) 
			(mu_y, std_y)
	'''

	# type cast
	x = x.float()
	y = y.float()
	
	# for X
		# Caculate parameters
	mu_x, sigma_x = x.mean(), x.std()
            
		# Exceptional cases
	sigma_x[sigma_x==0] = 1.0

	# for Y
		# calculate parameters
	mu_y, sigma_y = y.mean(-2, keepdim=True), y.std(-2, keepdim=True)

		# Exceptional cases
	sigma_y[sigma_y==0] = 1.0
    
		# Standardize 
	x = (x - mu_x) / (sigma_x + 1e-5)
	y = (y - mu_y) / (sigma_y + 1e-5)

	return x, y, (mu_x, sigma_x), (mu_y, sigma_y)

def reverse_standarize(x, y, x_std_params, y_std_params):
	'''
		Get back all values based on std_params

		Args : 
			standardized context_x, context_y, target_x, target_y 
			x_std_params : (mu_x, sigma_x)
			y_std_parms : (mu_y, sigma_y)

		Returns:
			original context_x, context_y, target_x, target_y
	'''
	
	# params
	mu_x, sigma_x = x_std_params
	mu_y, sigma_y = y_std_params

	# get back to originals
	x = x * (sigma_x + 1e-5) + mu_x
	y = y * (sigma_y + 1e-5) + mu_y

	return x, y

def load_global_stat(global_file_path, outlier_file_path):
	'''
		Args:
			e.g)
			outliers = np.loadtxt('./Outliers.csv', delimiter=',', dtype=np.float32, skiprows=1)
			global_file_path = './GlobalInformation.csv'
	'''
	
	outliers = np.loadtxt(outlier_file_path, delimiter=',', dtype=np.float32, skiprows=1)
	global_SIC = np.loadtxt(global_file_path, delimiter=',', dtype=np.float32, skiprows=1)

	# Make x values (time)
	x = np.array(np.arange(0, global_SIC.shape[0]))

	# Filter outliers
	x = np.delete(x, np.squeeze(outliers).astype(np.int32), 0)
	y = np.delete(global_SIC, np.squeeze(outliers).astype(np.int32), 0)

	# Slicing y values
	y = y[:,4:]

	# continous
	x, y = torch.from_numpy(x), torch.from_numpy(y)
	x, y = x.contiguous(), y.contiguous()

	return x, y

if __name__ == "__main__":
	# data import
	global_file_path = './Data/GlobalInformation.csv'
	outlier_file_path = "./Data/Outliers.csv"

	# import data
	x, y = load_global_stat(global_file_path, outlier_file_path)

	# print
	print(x[:10])
	print(y[:10,:])

	# standardize
	x, y, x_params, y_params = standardize(x,y)

	# plot
	plot_test(x, y)

	# print
	print(x[:10])
	print(y[:10,:])

	# reverse standardize
	x, y = reverse_standarize(x, y, x_params, y_params)

	# print
	print(x[:10])
	print(y[:10,:])


	

	





