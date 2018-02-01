import os
import sys
import h5py
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

TRAIN_PATH = 'imgs/train/'
TEST_PATH = 'imgs/test/'

def driver_id_for_classes():
	driver_info = pd.read_csv("driver_imgs_list.csv")
	subjects = driver_info['subject']		
	classes = driver_info['classname']
	imgs = driver_info['img']		
	# 存储司机
	ever_driver = {}
	# 切分不同的司机和类别
	drivers = []
	driver_classes = []
	driver_imgs = []
	for index in range(len(subjects)):
 		if subjects[index] in ever_driver.keys():
 			ever_driver[subjects[index]].append((classes[index],imgs[index]))
 		else:
 			ever_driver[subjects[index]] = [(classes[index],imgs[index])]
	# 绘制图形
	#draw(ever_driver)
	# print(ever_driver.items())
	return ever_driver # {id:[(classesName,ImageName)]}
# 根据司机 ID 进行分类
def driver_id_save():
	key_2_number = {'c0':0,'c1':1,'c2':2,'c3':3,'c4':4,'c5':5,'c6':6,'c7':7,'c8':8,'c9':9}
	drivers = driver_id_for_classes()
	validation_DriverImageName = []
	validation_key = []
	train_DriverImageName = []
	trian_key = []
	ids = drivers.keys()
	driver_count = len(ids) #总共多少司机
	# 取 /2 /3 的位置的司机
	validation_id = [int(driver_count / 2),int(driver_count / 3)]
	index = 0 
	# 寻找图片,保存图片名
	for driver_id in ids:
		# index 代表去第几个位置的司机
		for driver_class,driver_imageName in drivers[driver_id]:
			if index in validation_id:
				# valid
				validation_DriverImageName.append('imgs/train/' + str(driver_class) + '/' + str(driver_imageName))
				validation_key.append(key_2_number[str(driver_class)])
			else:
				# train
				train_DriverImageName.append('imgs/train/' + str(driver_class) + '/' + str(driver_imageName))
				trian_key.append(key_2_number[str(driver_class)])
		index += 1
	print("验证集数量:",len(validation_DriverImageName),",id为:",list(drivers)[validation_id[0]],",",list(drivers)[validation_id[1]])
	print("测试集数量:",len(train_DriverImageName))
	# return 
	# 读取和保存
	# 创建文件
	train_file = h5py.File('imgs/train_with_driver_id.h5','w')
	# 读取数据集
	# 验证集
	validation_images = np.zeros((len(validation_DriverImageName),480,640,3),dtype=np.uint8)
	for i in range(len(validation_DriverImageName)):
		validation_images[i] = cv2.imread(validation_DriverImageName[i])
	validation_data_set = train_file.create_dataset('valid',validation_images.shape,data = validation_images)
	validation_key_set = train_file.create_dataset('valid_class',np.array(validation_key).shape,data = np.array(validation_key))
	print("(val)保存之后图片数组的shape:",validation_images.shape,validation_data_set.shape,np.array(validation_key).shape,validation_key_set.shape)
	del validation_images
	del validation_data_set
	del validation_key
	del validation_key_set
	# 训练集
	train_images = np.zeros((len(train_DriverImageName),480,640,3),dtype=np.uint8)
	for i in range(len(train_DriverImageName)):
		train_images[i] = cv2.imread(train_DriverImageName[i])
	train__data_set = train_file.create_dataset('train',train_images.shape,data = train_images)
	train_key_set = train_file.create_dataset('train_class',np.array(trian_key).shape,data = np.array(trian_key))
	print("(train)保存之后DataSet的shape:",train_images.shape,train__data_set.shape,np.array(trian_key).shape,train_key_set.shape)
	del train_images 
	del train__data_set
	del trian_key
	del train_key_set
	# 关闭文件
	train_file.close()

# pre-handing data
def normalize(images,labels):
	number_of_classes = 10
	# mean-zero
	for i in range(len(images)):
		pass
	return keras.utils.to_categorical(labels, number_of_classes)
def driver_class_labels():
	'''
	return array of the labels
	'''
	#np.string_('c0'),np.string_('c1'),np.string_('c2'),np.string_('c3'),np.string_('c4'),np.string_('c5'),np.string_('c6'),np.string_('c7'),np.string_('c8'),np.string_('c9')
	return np.array([0,1,2,3,4,5,6,7,8,9])
def driver_class_label_count():
	'''
	The data pre-handle
	'''
	return np.array([2489, 2267, 2317, 2346, 2326, 2312, 2325, 2002, 1911, 2129]) # 读取文件之后,保存下来了
def draw_classes_count():
	'''
	draw count of ever class
	'''
	Y = [2489, 2267, 2317, 2346, 2326, 2312, 2325, 2002, 1911, 2129]
	X = ('0' '1' '2' '3' '4' '5' '6' '7' '8' '9')
	#('c0' 'c1' 'c2' 'c3' 'c4' 'c5' 'c6' 'c7' 'c8' 'c9')

	fig,ax = plt.subplots()

	bar_width = 0.35

	plt.xlabel('classes-----C')

	plt.ylabel('num of class')

	plt.title('driver status train_set_num')

	count = np.arange(10)

	opacity = 0.5

	plt.bar(count,Y,bar_width,color='rgby')  

	ax.set_xticks(count)

	ax.set_xticklabels(X)
	
	plt.ylim(0,np.max(Y) + 200)

	for x,y in zip(X,Y):
		plt.text(x,y+0.05, '%d' % y, ha='center', va= 'bottom')

	plt.legend()

	plt.tight_layout()

	plt.show()

# 一次性保存,无法实现,需要分成五次实现
def save_test_to_h5File(h5Name):
#get the imageNames in the indicate path
	fileNames = os.listdir(TEST_PATH)
	imgNames = []
	h5file = h5py.File(h5Name,'w')
	set_count = 5
	ever_set_counts = np.zeros(set_count)
	# appending the fully name
	print("load test imageName")
	for index in range(len(fileNames)):
		if 'jpg' in fileNames[index]:
			imgNames.append(TEST_PATH + fileNames[index])
	print("end load test imageName")

	test_set_size = int(len(imgNames) / set_count)

	for count in range(set_count - 1):
		print("starting load test Image with set count :",count)
		imagsData = np.zeros((test_set_size,480,640,3),dtype=np.uint8)
		for imgName_index in range(test_set_size):
			im = cv2.imread(imgNames[count * test_set_size + imgName_index])
			imagsData[imgName_index] = im
		print("load test Image with count:",count)
		print("write to h5file with set_count:",count)
		test_set = h5file.create_dataset('test_image_set_' + str(count),imagsData.shape,data = imagsData)
		print("imageData.shape:",imagsData.shape," dataset.shape:",test_set.shape)
		print("write done h5file with set_count:",count)
		print("test image set use the memory size is :",(sys.getsizeof(imagsData) / np.power(1024,3))," G.")
		ever_set_counts[count] = (imagsData.shape)[0]
		del imagsData
		del test_set
	# the last set
	print("load last test set")
	imagsData = np.zeros((len(imgNames) - ((set_count - 1) * test_set_size),480,640,3),dtype=np.uint8)
	for imgNameIndex in range(len(imgNames) - ((set_count - 1) * test_set_size),):
		imagsData[imgNameIndex] = cv2.imread(imgNames[(set_count - 1) * test_set_size + imgNameIndex])
	ever_set_counts[-1] = (imagsData.shape)[0]
	print("write the last test set")
	test_set = h5file.create_dataset('test_image_set_' + str(set_count - 1),imagsData.shape,data = imagsData)
	test_set_sizes = h5file.create_dataset('test_set_sizes',ever_set_counts.shape,data = ever_set_counts)
	print("write done h5file with set_count:",(set_count - 1))
	print("imageData.shape:",imagsData.shape," dataset.shape:",test_set.shape)
	print("test image set use the memory size is :",(sys.getsizeof(imagsData) / np.power(1024,3))," G.")
	h5file.close()
	print("test image write done")
#保存成多个文件
def save_test_set_to_h5File_All(h5Names):
	#get the imageNames in the indicate path
	fileNames = os.listdir(TEST_PATH)
	imgNames = []
	# appending the fully name
	for index in range(len(fileNames)):
		if 'jpg' in fileNames[index]:
			imgNames.append(TEST_PATH + fileNames[index])
	# sets count
	set_size = int(len(imgNames) / len(h5Names))

	set_number = len(h5Names)

	previousCount = 0

	for h5Name in h5Names:
		imagsData = []
		h5file = h5py.File('imgs/' + h5Name,'w')
		if previousCount == set_number - 1 :
			#last
			for image in imgNames[(previousCount) * set_size:]:
				img = cv2.imread(image)
				imagsData.append(img)
		else:
			for image in imgNames[previousCount * set_size : (previousCount + 1) * set_size]:
				img = cv2.imread(image)
				imagsData.append(img)
		image_data_arr = np.array(imagsData)
		image_dataset = h5file.create_dataset('train_data_set',image_data_arr.shape,data = image_data_arr)
		print(image_data_arr.shape,image_dataset.shape)
		h5file.close()
		previousCount += 1

# 保存训练集图片
def save_train_set(h5Name):
	# total labels
	classes = os.listdir(TRAIN_PATH)
	# All image Data
	images_data = []
	# classes
	train_labels = driver_class_labels()
	# 
	string_category = ['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
	# labels 
	labels = np.repeat(driver_class_labels(),driver_class_label_count())
	#存放所有图片名字
	imgNames = []
	#ever label
	for label in classes:
		if label in string_category:
			print("-------------------------start loading:",label,"imageName-------------------------")
			files_of_label = os.listdir(TRAIN_PATH + label)
			for fileName in files_of_label:
				if 'jpg' in fileName:
					imgNames.append(TRAIN_PATH + label + '/' + fileName)
			print("end loading: ",label,"imageName")
	# all image of label
	# shuffle the data with ImageName
	train_data_key = list(zip(imgNames,labels))
	np.random.shuffle(train_data_key)
	imgNames = np.array([p[0] for p in train_data_key])
	labels = np.array([p[1] for p in train_data_key])
	# release memory
	del train_data_key
	# read all Image
	for imgName in imgNames:
		img = cv2.imread(imgName)
		images_data.append(img)
	print("end load all train image-------------------------")
	# 
	# Create the nd array
	image_data_arr = np.array(images_data)
	del images_data
	print("-------------------------------------all train image load in memory-------------------------------------")
	#print("num:",labels.shape[0])
	# 打乱数据
	#train_data_index = np.arange(labels.shape[0])
	print("train image use the memory size is :",(sys.getsizeof(image_data_arr) / np.power(1024,3))," G.")
	print("count:",len(labels),image_data_arr.shape)
	#train_Data = list(zip(image_data_arr,labels))
	#del image_data_arr
	#np.random.shuffle(train_Data)
	#X = np.array([p[0] for p in train_Data])
	#Y = np.array([p[1] for p in train_Data])
	#del trian_Data
	#del labels
	# create the cache file
	h5file = h5py.File(h5Name,'w')
	# create dataset
	image_data_set = h5file.create_dataset('train_images',image_data_arr.shape,data = image_data_arr)
	image_labels = h5file.create_dataset('train_labels',labels.shape,data = labels)
	# 打印输出结果
	print("image_set:",image_data_arr.shape,image_data_set.shape)
	print("image_label:",labels.shape,image_labels.shape)
	h5file.close()
	print("train image write done")
# loading
def load_h5File(h5Name):
	'''
	path The path of h5file
	h5Name the h5 file name
	return the pixel array of images
	'''
	h5file = h5py.File(h5Name,'r')
	print(h5file)
	print(h5file['train_num_set_0'].dtype)
if __name__ == '__main__':
	save_train_set('imgs/train_set.h5')
	save_test_to_h5File('imgs/test_set.h5')
