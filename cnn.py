import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import sys
import cv2
import keras
from keras.layers import Input,Lambda, Dense, Dropout, Flatten,Conv2D, MaxPooling2D,Input,Conv3D,GlobalAveragePooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.models import Sequential,Model
from keras.optimizers import Adadelta,SGD,Adam
from keras.losses import categorical_crossentropy
from keras import backend as K
from keras.utils import plot_model
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input as vgg_preprocess_input
from keras.applications.resnet50 import preprocess_input as res_preprocess_input
from keras.applications.inception_v3 import preprocess_input as inc_preprocess_input
from keras import regularizers
import os
# Global Valuegithub-scikit:
number_of_classes = 10
batch_size = 64 # 128,256
epochs = 20
TEST_PATH = 'imgs/test/'

# 映射
class_mapping = {'0':'safe driving','1':'right write','2':'right call','3':'left write','4':'left call',\
	'5':'tune_radio','6':'drinking','7':'take something in back','8':'clean hair','9':'talk with others'}
# save model to h5file
def save_model(model,modelname,isSaveWeights = False):
	if isSaveWeights:
		model.save_weights(modelname)
	else:
		model.save(modelname)  # creates a HDF5 file 'my_model.h5'
	del model  # deletes the existing model

# load model from h5file
def load_model(modelname,model_weights_name,isLoadWeights = False,model = None,isLoadweightForName = False):
	model = None
	if isLoadWeights:
		model = load_model(modelname)
		if isLoadweightForName:
			model.load_weights(model_weights_name)
		else:
			model.load_weights(model_weights_name,by_name=True) 
	else:
		model = load_model(modelname)
	return model
	
    
# 归一化
def resize_only(images,des_size):
	resize_images = np.zeros((images.shape[0],des_size[0],des_size[1],3),dtype = np.uint8)
	for i in range(images.shape[0]):
		img = cv2.resize(images[i],des_size)
		resize_images[i] = img
		# BGR->RGB 因为 processInput 会处理一次 RGB->BGR 格式, H5文件中通过 openCV 读取,默认是 BGR 的形式
		resize_images[i] = resize_images[i][:,:,::-1]
	return resize_images

def resize(images,des_size,process_input):
	resize_images = np.zeros((images.shape[0],des_size[0],des_size[1],3),dtype = np.float64)
	for i in range(images.shape[0]):
		img = cv2.resize(images[i],des_size)
		resize_images[i] = img
		# BGR->RGB 因为 processInput 会处理一次 RGB->BGR 格式, H5文件中通过 openCV 读取,默认是 BGR 的形式
		resize_images[i] = resize_images[i][:,:,::-1]
	# 归一化
	resize_images = process_input(resize_images,data_format = 'channels_last')
	return resize_images


# 640 * 480 --> 300 * 300
def resizeImage(images,des_size):
	normalize_images = np.zeros((images.shape[0],des_size[0],des_size[1],3),dtype = np.int8)
	for i in range(images.shape[0]):
		normalize_image = cv2.resize(images[i],des_size)
		normalize_image = normalize_image.astype(np.float64,copy=False)
		# 归一化
		normalize_image = (normalize_image - np.average(normalize_image)) / normalize_image.std()
		#添加
		normalize_images[i] = normalize_image
	print("normalize done!")
	#normalize_images[:int(images.shape[0] * 0.8)],normalize_images[int(images.shape[0] * 0.8):],normalize_images.shape[0]
	return normalize_images,normalize_images.shape[0]

# read the test Image
def load_test_set(testfile,index,size,process_input=None):
	test_set_image = resize_only(testfile[index][:],size)
	return test_set_image

# read the train image
def load_train_set(size):
	train_file = h5py.File('imgs/train_with_driver_id.h5')
	# load data
	train_labels = train_file['train_labels'][:]
	# split
	#trian_image,train_label,valid_image,valid_label = train_test_split(trian_images,train_labels,test_size = 0.2,random_state=42)
	# already shuffle data
	train_image,totalCount = resizeImage(train_file['train_images'],size)
	valid_image = train_image[int(train_image.shape[0] * 0.8):]
	train_image = train_image[:int(train_image.shape[0] * 0.8)]
	train_label = train_labels[:int(totalCount * 0.8)]
	valid_label = train_labels[int(totalCount * 0.8):]
	del train_labels
	# handle it
	valid_label = keras.utils.to_categorical(valid_label, number_of_classes)
	train_label = keras.utils.to_categorical(train_label, number_of_classes)
	# print info
	print("train_valid_image_label(shape):",
		train_image.shape,train_image.dtype,valid_image.shape,train_label.shape,valid_label.shape)
	print("train_image use the memory size is :",(train_image.nbytes / np.power(1024,3))," G.")
	# + sys.getsizeof() + sys.getsizeof() + sys.getsizeof())
	print("valid_image use the memory size is :",((valid_image.nbytes / np.power(1024,3))," G."))
	print("valid_label use the memory size is :",((valid_label.nbytes / np.power(1024,3))," G."))
	print("train_label use the memory size is :",((train_label.nbytes / np.power(1024,3))," G."))
	# close
	train_file.close()
	# return the result
	return (train_image,train_label,valid_image,valid_label,totalCount)


# data generator 
def data_generator(X,Y,batch_size = batch_size):
	if X.shape[0] != Y.shape[0]:
		raise "X,Y 的数据个数必须一样"
	data_count = X.shape[0]
	batch_count = data_count // batch_size
	for batch_index in range(batch_count):
		if batch_index == batch_count - 1:
			# the last batch
			yield X[(batch_index + 1) * batch_size:],Y[(batch_index + 1) * batch_size:]
		else:
			yield X[(batch_index) * batch_size: (batch_index + 1) * batch_size], \
				Y[(batch_index) * batch_size:(batch_index + 1) * batch_size]
                
                
# predict
def predict(model,size):
	test_file = h5py.File('imgs/test_set.h5')
	# all test labels
	all_labels = []
	# has 5 test set
	for i in range(5):
		test_image = load_test_set(test_file,'test_image_set_'+str(i),size)
		labels = model.predict(test_image) # (set_size * 10)
		all_labels.append(labels)
	# all_labels ->
	nd_array_labels = []
	for row in range(len(all_labels)):
		for col in range(len(all_labels[row])):
			nd_array_labels.append(all_labels[row][col])
		#= np.append(all_labels,np.array(labels))# 5 * (set_size * 10) --> total_set * 10
	nd_array_labels = np.array(nd_array_labels)
	print(nd_array_labels.shape)
		# scores
	return nd_array_labels #all_labels.reshape(all_labels.shape[0] * all_labels.shape[1],all_labels[2])



# output to csv after predict
def predict_output(model,size,csv_filename):
	predicts = predict(model,size)
	# 输出的是值
	fileNames = os.listdir('imgs/test/')
	imgNames = []
	output_data = []
	for index in range(len(fileNames)):
		if 'jpg' in fileNames[index]:
			# 不需要 imgs/test/*.jpg --> 文件名就行
			imgNames.append(fileNames[index])
	#(img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9)
	predicts = np.array(predicts)
	print(predicts.shape)
	data = {'img':imgNames,'c0':predicts[:,0],'c1':predicts[:,1],'c2':predicts[:,2],'c3':predicts[:,3],'c4':predicts[:,4],'c5':predicts[:,5],'c6':predicts[:,6],'c7':predicts[:,7],'c8':predicts[:,8],'c9':predicts[:,9]}
	# print(data)
	for row in range(len(predicts)):
		output_data.append([imgNames[row],predicts[row][0],predicts[row][1],predicts[row][2],predicts[row][3],predicts[row][4],predicts[row][5],predicts[row][6],predicts[row][7],predicts[row][8],predicts[row][9]])
	# print(np.array(output_data).shape)
	dataFrame = pd.DataFrame(data = data)#(data = np.array(output_data),columns=['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9'])
	# print(dataFrame)
	dataFrame.to_csv(csv_filename,index=False)
    
    
    
# None * 640 * 480 * 3, simple_conv_nn
def buildMutliCNN():
	# split
	#trian_image,train_label,valid_image,valid_label = train_test_split(trian_images,train_labels,test_size = 0.2,random_state=42)
	# already shuffle data
	# train_data
	train_file = h5py.File('imgs/train_with_driver_id.h5')
	train_images = np.array(train_file['train'][:])
	train__label = train_file['train_class'][:]
	valid_images = np.array(train_file['validation'][:])
	valid_label = train_file['validation_class'][:]
	# 传入
	#all_images = resize(images,(224,224),res_preprocess_input)
	train_images = resize_only(train_images,(100,100))
	valid_images = resize_only(valid_images,(100,100))
	valid_label = keras.utils.to_categorical(valid_label, number_of_classes)
	train_label = keras.utils.to_categorical(train_label, number_of_classes)
	print(train_images.shape,valid_images.shape)
	# NN
   	# image size 300 * 300(宽高)
	input_img_size = (100, 100, 3)
	# Model
	model = Sequential()
	# Conv layer
	# C1
	input_conv_layer_1 = Conv2D(64, kernel_size = (4, 4), strides = 1,padding='valid', activation='relu',data_format = 'channels_last',input_shape=input_img_size)
	output_conv_layer_1 = MaxPooling2D(pool_size = (2, 2), strides=(1, 1), padding='valid')
	model.add(input_conv_layer_1)
	model.add(output_conv_layer_1)
	# C2
	input_conv_layer_2 = Conv2D(128, kernel_size = (3, 3), strides = 1, padding='valid', activation='relu')
	output_conv_layer2 = MaxPooling2D(pool_size = (2,2), strides = (1,1), padding='valid')
	model.add(input_conv_layer_2)
	model.add(output_conv_layer2)
	# C3
	input_conv_layer_3 = Conv2D(256,kernel_size = (3,3), strides = 1, padding='valid',activation='relu')
	output_conv_layer_3 = MaxPooling2D(pool_size = (2,2), strides = (1,1), padding='valid')
	model.add(input_conv_layer_3)
	model.add(output_conv_layer_3)
	# Fully Conn
	flatten = Flatten()
	fully_conn_layer = Dropout(0.5)
	model.add(flatten)
	model.add(fully_conn_layer)
	# Output,softmax
	output_layer = Dense(number_of_classes,activation='softmax')
	model.add(output_layer)
	#Loss
	loss = categorical_crossentropy

	# Optimizer
	optimizer = Adadelta(lr = 0.001)

	#fit
	model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])
	# generator
	#train_generator = data_generator(train_image,train_label)
	#valid_generator = data_generator(valid_image,valid_label)
	trian_datagen = ImageDataGenerator()
	valid_datagen = ImageDataGenerator()
	# fit
	history = model.fit_generator(trian_datagen.flow(train_image,train_label,batch_size=batch_size),(train_image.shape[0] // batch_size),
		epochs=5,validation_data=valid_datagen.flow(valid_image,valid_label,batch_size=batch_size),validation_steps=((valid_image.shape[0] // batch_size)))
	'''
	history = model.fit(train_image, train_label,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(valid_image, valid_label))
	''' 
	# save_model
	save_model(model,'imgs/simple_conv_nn_weights.h5',isSaveWeights = True)
	# print info of the NN
	plot_model(model, to_file='model.png')
	with open("simple_conv_nn_val_train_loss_acc.txt","w") as f:
		f.write(str(history.history))
	print("history:",history.history)
	#predict,需要转换为 test
	score = model.evaluate(valid_image, valid_label,batch_size = batch_size, verbose=1)
	print('loss:',score[0])
	print('accuracy:',score[1])
    
    
# 加载损失日志
def load_train_valid_loss(filename):
	history = None
	if os.path.exists(filename):
		with open(filename,'r') as f:
			values = f.read()
			history = eval(values)
	return history


def draw_loss_acc(filename):
	history = load_train_valid_loss(filename)
	acc = history['acc']
	loss = history['loss']
	val_loss = history['val_loss']
	val_acc = history['val_acc']
	# draw
	figure,(ax1,ax2) = plt.subplots(2,1)

	# loss
	ax1.plot(acc)
	ax1.plot(val_acc)
	ax1.set_title('model accuracy')
	ax1.set_ylabel('accuracy')
	ax1.set_xlabel('epoch')
	ax1.legend(['train', 'valid'], loc='upper left')
	ax1.set_xticks(np.arange(21))
	# summarize history for loss
	ax2.plot(loss)
	ax2.plot(val_loss)
	ax2.set_title('model loss')
	ax2.set_ylabel('loss')
	ax2.set_xlabel('epoch')
	ax2.legend(['train', 'valid'], loc='upper left')
	ax2.set_xticks(np.arange(21))
	plt.show()
    
    
# Transfer Learning
def VGG16Transfer():
	print("Start VGG16-Model")
	# 传入
	'''
	all_images = resize(images,(224,224),vgg_preprocess_input)
	'''
	# train_data
	train_file = h5py.File('imgs/train_with_driver_id.h5')
	train_image = np.array(train_file['train'][:])
	train_label = train_file['train_class'][:]
	valid_image = np.array(train_file['valid'][:])
	valid_label = train_file['valid_class'][:]
	# 传入
	#all_images = resize(images,(224,224),res_preprocess_input)
	train_image = resize_only(train_image,(224,224))
	valid_image = resize_only(valid_image,(224,224))
	valid_label = keras.utils.to_categorical(valid_label, number_of_classes)
	train_label = keras.utils.to_categorical(train_label, number_of_classes)
	print(train_image.shape,valid_image.shape)
	# 224 * 224
	input_tensor = Input(shape=(224, 224, 3))
	precess_input = Lambda(vgg_preprocess_input)(input_tensor)
	vggModel = keras.applications.vgg16.VGG16(include_top=False, \
		weights='imagenet',input_tensor=precess_input,input_shape=(224,224,3))
	# get the source output
	# 假设时输出层的输入结构
	print("input_layer:",vggModel.input)
	output_layer = vggModel.output
	print("out_layer is:",output_layer)
	flatten = Flatten()(output_layer) #GlobalAveragePooling2D
	# full connected
	fc_cus_layer_1 = Dense(4096,activation='relu')(flatten)
	fc_cus_layer_1 = Dropout(0.5)(fc_cus_layer_1)
	# full connected 2
	fc_cus_layer_2 = Dense(4096,activation='relu')(fc_cus_layer_1)
	# fc_cus_layer_2 = Dropout(0.5)(fc_cus_layer_2)
	# custom output layer,正则项  --> kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)
	prediction_layer = Dense(number_of_classes, activation='softmax',name='vgg_out_put')(fc_cus_layer_2) 
	print("output layer:",prediction_layer)
	# fine-tune model
	model = Model(inputs=vggModel.input, outputs=prediction_layer)
	print("model_input:",model.input)
	# lock the layers(conv)
	# print(len(model.layers))
	for layer in model.layers[:10]:
		layer.trainable = False
	for layer in model.layers[10:]:
		layer.trainable = True
	# compile
	model.compile(optimizer=Adadelta(lr=0.001), loss='categorical_crossentropy',metrics=['accuracy'])
	# fit
	trian_datagen = ImageDataGenerator()
	valid_datagen = ImageDataGenerator()
	# fit
	history = model.fit_generator(trian_datagen.flow(train_image,train_label,batch_size=batch_size),(train_image.shape[0] // batch_size),
		epochs=epochs,validation_data=valid_datagen.flow(valid_image,valid_label,batch_size=batch_size),validation_steps=((valid_image.shape[0] // batch_size)))
	'''
	#开放后四层
	for layer in model.layers[:len(model.layers) - 8]:
		layer.trainable = False
	for layer in model.layers[len(model.layers) - 8:]:
		layer.trainable = True
	#重新编译
	model.compile(optimizer=Adadelta(), loss='categorical_crossentropy',metrics=['accuracy'])
	#继续训练
	# fit
	history2 = model.fit_generator(trian_datagen.flow(train_image,train_label,batch_size=batch_size),(train_image.shape[0] // batch_size),
		epochs=15,validation_data=valid_datagen.flow(valid_image,valid_label,batch_size=batch_size),validation_steps=((valid_image.shape[0] // batch_size))) 
	'''
	#predict,需要转换为 test
	score = model.evaluate(valid_image, valid_label,batch_size = batch_size, verbose=1)
	print('VGG16 loss:',score[0])
	print('VGG16 accuracy:',score[1])
	# save_model
	save_model(model,'imgs/vgg_conv_nn_weights.h5',isSaveWeights = True)
	# operation logs(dict) of history object
	with open("vgg_16_train_loss_acc.txt","w") as f:
		print(history.history)
		f.write(str(history.history))
	'''
		history = {}
		for key in history1.history.keys():
			history[key] = history1.history[key]
			for value in history2.history[key]:
				history[key].append(value)
	'''
	# print info of the NN
	#plot_model(model, to_file='vgg16Model.png',show_shapes=True)

    
def ResNet50():
	print('start ResNet50')
	input_tensor = Input(shape=(224, 224, 3))
	precess_input = Lambda(res_preprocess_input)(input_tensor)
	#  224 x 224
	resModel = keras.applications.resnet50.ResNet50(include_top=True,input_tensor=precess_input,weights='imagenet',input_shape=(224,224,3))
	# 只训练最后一层
	resModel = Model(inputs=resModel.input,outputs=resModel.layers[-2].output)
	# origin output
	origin_output = resModel.output
	# origin_output = Dropout(0.5)(origin_output)
	# print(resModel.layers)
	# full connected
	# flatten = GlobalAveragePooling2D()(origin_output)
	# flatten = Dropout(0.5)(flatten)
	# 可以考虑加一个 dropout 层
	# fc_cus_layer = Dense(1000,activation='relu')(flatten)
	# print(fc_cus_layer)
	# custom output layer
	# kernel_regularizer=regularizers.l2(0.01) ,kernel_regularizer=regularizers.l2(0.1),kernel_regularizer=regularizers.l2(0.05)
	prediction_layer = Dense(number_of_classes, activation='softmax',name='res_out_put')(origin_output)#resModel.layers[-1].output) # origin_output
	# print(prediction_layer)
	# fine-tune model
	model = Model(inputs=resModel.input, outputs=prediction_layer)
	print(model.layers)
	# train_data
	train_file = h5py.File('imgs/train_with_driver_id.h5')
	train_image = np.array(train_file['train'][:])
	train_label = train_file['train_class'][:]
	valid_image = np.array(train_file['valid'][:])
	valid_label = train_file['valid_class'][:]
	# 传入
	#all_images = resize(images,(224,224),res_preprocess_input)
	train_image = resize_only(train_image,(224,224))
	valid_image = resize_only(valid_image,(224,224))
	valid_label = keras.utils.to_categorical(valid_label, number_of_classes)
	train_label = keras.utils.to_categorical(train_label, number_of_classes)
	print(train_image.shape,valid_image.shape)
	# lock the layers(conv)
	for layer in resModel.layers:
		layer.trainable = False
	# compile
	model.compile(optimizer=Adadelta(), loss='categorical_crossentropy',metrics=['accuracy'])
	# fit
	trian_datagen = ImageDataGenerator()#rotation_range=90.,width_shift_range=0.1,height_shift_range=0.1)
	valid_datagen = ImageDataGenerator()#rotation_range=90.,width_shift_range=0.1,height_shift_range=0.1)
	# fit
	history1 = model.fit_generator(trian_datagen.flow(train_image,train_label,batch_size=batch_size,seed=1),(train_image.shape[0] // batch_size),
		epochs=5,validation_data=valid_datagen.flow(valid_image,valid_label,batch_size=batch_size,seed=1),validation_steps=((valid_image.shape[0] // batch_size)))
	# 第二次训练
	# 开放接下来所有层  
	for layer in model.layers[:]:
		layer.trainable = True
	# SGD
	sgd = SGD(lr=0.001, momentum=0.9)
	# Adam
	adam = Adam(lr=0.001)
	#Adadelta()
	model.compile(optimizer=Adadelta(), loss='categorical_crossentropy',metrics=['accuracy'])
	# 提前停止
	early = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=5, verbose=0, mode='auto')
	history2 = model.fit_generator(trian_datagen.flow(train_image,train_label,batch_size=batch_size,seed=1),(train_image.shape[0] // batch_size),
		epochs=15,validation_data=valid_datagen.flow(valid_image,valid_label,batch_size=batch_size,seed=1),validation_steps=((valid_image.shape[0] // batch_size)),callbacks=[early])
	#predict,需要转换为 test
	score = model.evaluate(valid_image, valid_label,batch_size = batch_size, verbose=1)
	# save_model
	save_model(model,'imgs/resNet50_conv_nn_weights.h5',isSaveWeights = True)
	print('resNet50 loss:',score[0])
	print('resNet50 accuracy:',score[1])
	# operation logs(dict) of history object
	with open("resNet50_train_loss_acc.txt","w") as f:
		print(history2.history)
		f.write(str(history2.history))
		'''
		history = {}
		for key in history1.history.keys():
			history[key] = history1.history[key]
			for value in history2.history[key]:
				history[key].append(value)
		print(history)
		f.write(str(history))
		'''
	# print info of the NN
	#plot_model(model, to_file='resNet50Model.png',show_shapes=True)
    
    
def InceptionV3():
	input_tensor = Input(shape=(299, 299, 3))
	precess_input = Lambda(inc_preprocess_input)(input_tensor)
	# 299 x 299
	inceptionV3Model = keras.applications.inception_v3.InceptionV3(weights='imagenet',input_tensor=precess_input, include_top=False,input_shape=(299,299,3))
	# origin output
	origin_output = inceptionV3Model.output
	# custom output layer
	pooling_layer = GlobalAveragePooling2D()(origin_output)
	fc_layer = Dense(1024, activation='relu')(pooling_layer)
	prediction_layer = Dense(number_of_classes, activation='softmax',name='inc_out_put')(fc_layer)
	print(origin_output)
	print(pooling_layer)
	print(fc_layer)
	print(prediction_layer)
	#return 
	# fine-tune model
	model = Model(inputs=inceptionV3Model.input, outputs=prediction_layer)
	# lock the layers(conv)
	for layer in inceptionV3Model.layers:
		layer.trainable = False
	# train_data
	train_file = h5py.File('imgs/train_with_driver_id.h5')
	train_image = np.array(train_file['train'][:])
	train_label = train_file['train_class'][:]
	valid_image = np.array(train_file['valid'][:])
	valid_label = train_file['valid_class'][:]
	# 传入
	#all_images = resize(images,(224,224),res_preprocess_input)
	train_image = resize_only(train_image,(299,299))
	valid_image = resize_only(valid_image,(299,299))
	valid_label = keras.utils.to_categorical(valid_label, number_of_classes)
	train_label = keras.utils.to_categorical(train_label, number_of_classes)
	print(train_image.shape,valid_image.shape)
	# compile
	model.compile(optimizer=Adadelta(), loss='categorical_crossentropy',metrics=['accuracy'])
	# 合适的 epochs 之后,可以适当开放一下接下来的几层
	trian_datagen = ImageDataGenerator()
	valid_datagen = ImageDataGenerator()
	# fit
	history1 = model.fit_generator(trian_datagen.flow(train_image,train_label,batch_size=batch_size),(train_image.shape[0] // batch_size),
		epochs=5,validation_data=valid_datagen.flow(valid_image,valid_label,batch_size=batch_size),validation_steps=((valid_image.shape[0] // batch_size)))
	#predict,需要转换为 test
	# 下面15代
	for layer in model.layers:
		layer.trainable = True
	# compile
	model.compile(optimizer=Adadelta(), loss='categorical_crossentropy',metrics=['accuracy'])
	# 提前停止
	early = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=5, verbose=0, mode='auto')
	# fit
	history2 = model.fit_generator(trian_datagen.flow(train_image,train_label,batch_size=batch_size),(train_image.shape[0] // batch_size),
		epochs=15,validation_data=valid_datagen.flow(valid_image,valid_label,batch_size=batch_size),validation_steps=((valid_image.shape[0] // batch_size)),callbacks=[early])
	score = model.evaluate(valid_image, valid_label,batch_size = batch_size, verbose=1)
	# save_model
	save_model(model,'imgs/inceptionV3Model_conv_nn_weights.h5',isSaveWeights = True)	
	# operation logs(dict) of history object
	print('inc loss:',score[0])
	print('inc accuracy:',score[1])
	with open("InceptionV3_train_loss_acc.txt","w") as f:
		history = {}
		for key in history1.history.keys():
			history[key] = history1.history[key]
			for value in history2.history[key]:
				history[key].append(value)
		print(history)
		f.write(str(history))
	# print info of the NN
	#plot_model(model, to_file='inceptionV3Model.png',show_shapes=True)
    
    
# 搭建 Model-CAM 模型
def VGG_16_CAM():
	input_tensor = Input(shape=(244, 244, 3))
	precess_input = Lambda(vgg_preprocess_input)(input_tensor)
	# 传入
	#all_images = resize(images,(224,224),vgg_preprocess_input)
	# train_data
	train_file = h5py.File('imgs/train_with_driver_id.h5')
	train_image = np.array(train_file['train'][:])
	train_label = train_file['train_class'][:]
	valid_image = np.array(train_file['valid'][:])
	valid_label = train_file['valid_class'][:]
	# 传入
	#all_images = resize(images,(224,224),res_preprocess_input)
	train_image = resize_only(train_image,(224,224))
	valid_image = resize_only(valid_image,(224,224))
	valid_label = keras.utils.to_categorical(valid_label, number_of_classes)
	train_label = keras.utils.to_categorical(train_label, number_of_classes)
	print(train_image.shape,valid_image.shape)
	# 224 * 224
	vggModel = keras.applications.vgg16.VGG16(include_top=False, \
		 weights='imagenet',input_tensor=precess_input,input_shape=(224,224,3))
	# get the source output
	# 假设时输出层的输入结构
	output_layer = vggModel.output
	# 换成池化层
	output = GlobalAveragePooling2D()(output_layer)
	# custom output layer
	prediction_layer = Dense(number_of_classes, activation='softmax',name='vgg_cam_out_put')(output)
	# fine-tune model
	model = Model(inputs=vggModel.input, outputs=prediction_layer)
	# lock the layers(conv)
	for layer in vggModel.layers:
		layer.trainable = False
	# compile
	model.compile(optimizer=Adadelta(), loss='categorical_crossentropy',metrics=['accuracy'])
	# fit
	trian_datagen = ImageDataGenerator()
	valid_datagen = ImageDataGenerator()
	# fit
	history = model.fit_generator(trian_datagen.flow(train_image,train_label,batch_size=batch_size),(train_image.shape[0] // batch_size),
		epochs=epochs,validation_data=valid_datagen.flow(valid_image,valid_label,batch_size=batch_size),validation_steps=((valid_image.shape[0] // batch_size)))
	# 展示?
	# score = model.evaluate(valid_image, valid_label,batch_size = batch_size, verbose=1)
	# print('resNet50 loss:',score[0])
	# print('resNet50 accuracy:',score[1])
	# Save
	save_model(model,'imgs/vgg_CAM_weights.h5',isSaveWeights = True)
	#plot_model(model, to_file='vgg_CAM.png',show_shapes=True)
    
def Inception_CAM():
	input_tensor = Input(shape=(299, 299, 3))
	precess_input = Lambda(inc_preprocess_input)(input_tensor)
	# 299 x 299
	inceptionV3Model = keras.applications.inception_v3.InceptionV3(weights='imagenet',input_tensor=precess_input, include_top=False,input_shape=(299,299,3))
	# origin output
	origin_output = inceptionV3Model.output
	# custom output layer
	pooling_layer = GlobalAveragePooling2D()(origin_output)
	prediction_layer = Dense(number_of_classes, activation='softmax',name='inc_out_put')(pooling_layer)
	#return 
	# fine-tune model
	model = Model(inputs=inceptionV3Model.input, outputs=prediction_layer)
	# lock the layers(conv)
	for layer in inceptionV3Model.layers:
		layer.trainable = False
	# train_data
	train_file = h5py.File('imgs/train_with_driver_id.h5')
	train_image = np.array(train_file['train'][:])
	train_label = train_file['train_class'][:]
	valid_image = np.array(train_file['valid'][:])
	valid_label = train_file['valid_class'][:]
	# 传入
	#all_images = resize(images,(224,224),res_preprocess_input)
	train_image = resize_only(train_image,(299,299))
	valid_image = resize_only(valid_image,(299,299))
	valid_label = keras.utils.to_categorical(valid_label, number_of_classes)
	train_label = keras.utils.to_categorical(train_label, number_of_classes)
	print(train_image.shape,valid_image.shape)
	# compile
	model.compile(optimizer=Adadelta(), loss='categorical_crossentropy',metrics=['accuracy'])
	# 合适的 epochs 之后,可以适当开放一下接下来的几层
	trian_datagen = ImageDataGenerator()
	valid_datagen = ImageDataGenerator()
	# fit
	history = model.fit_generator(trian_datagen.flow(train_image,train_label,batch_size=batch_size),(train_image.shape[0] // batch_size),
		epochs=epochs,validation_data=valid_datagen.flow(valid_image,valid_label,batch_size=batch_size),validation_steps=((valid_image.shape[0] // batch_size)))
	# Save
	save_model(model,'imgs/inception_CAM_weights.h5',isSaveWeights = True)
	#plot_model(model, to_file='inception_CAM.png',show_shapes=True)
    
def ResNet_CAM():
	input_tensor = Input(shape=(224, 224, 3))
	precess_input = Lambda(res_preprocess_input)(input_tensor)
	#  224 x 224
	resModel = keras.applications.resnet50.ResNet50(include_top=False,input_tensor=precess_input, weights='imagenet',input_shape=(224,224,3))
	# origin output
	origin_output = resModel.output
	# 根据培文导师说 CAM 相当于 激活特征的一个1*1卷积操作
	# one_conv = Conv2D(1,(1,1))(origin_output)
	output = GlobalAveragePooling2D()(origin_output)
	# custom output layer
	prediction_layer = Dense(number_of_classes, activation='softmax',name='res_out_put')(output) # one_conv
	print(prediction_layer)
	# fine-tune model
	model = Model(inputs=resModel.input, outputs=prediction_layer)
	# lock the layers(conv)
	for layer in resModel.layers:
		layer.trainable = False
	# train_data
	train_file = h5py.File('imgs/train_with_driver_id.h5')
	train_image = np.array(train_file['train'][:])
	train_label = train_file['train_class'][:]
	valid_image = np.array(train_file['valid'][:])
	valid_label = train_file['valid_class'][:]
	# 传入
	#all_images = resize(images,(224,224),res_preprocess_input)
	train_image = resize_only(train_image,(224,224))
	valid_image = resize_only(valid_image,(224,224))
	valid_label = keras.utils.to_categorical(valid_label, number_of_classes)
	train_label = keras.utils.to_categorical(train_label, number_of_classes)
	print(train_image.shape,valid_image.shape)
	# compile
	model.compile(optimizer=Adadelta(), loss='categorical_crossentropy',metrics=['accuracy'])
	# fit
	trian_datagen = ImageDataGenerator()
	valid_datagen = ImageDataGenerator()
	# fit
	history1 = model.fit_generator(trian_datagen.flow(train_image,train_label,batch_size=batch_size),(train_image.shape[0] // batch_size),
		epochs=5,validation_data=valid_datagen.flow(valid_image,valid_label,batch_size=batch_size),validation_steps=((valid_image.shape[0] // batch_size)))
	# 开放接下来所有层    
	for layer in model.layers:
		layer.trainable = True
	model.compile(optimizer=Adadelta(lr=0.001), loss='categorical_crossentropy',metrics=['accuracy'])
	# 提前停止
	early = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=5, verbose=0, mode='auto')
	history2 = model.fit_generator(trian_datagen.flow(train_image,train_label,batch_size=batch_size),(train_image.shape[0] // batch_size),
		epochs=15,validation_data=valid_datagen.flow(valid_image,valid_label,batch_size=batch_size),validation_steps=((valid_image.shape[0] // batch_size)),callbacks=[early])
	#Save
	save_model(model,"imgs/resnet_CAM_weights.h5",isSaveWeights = True)
	# plot_model(model, to_file='resnet_CAM.png',show_shapes=True)

# 自己从头搭建网络结构,然后加载权重
def build_simple_conv():
	input_img_size = (100, 100, 3)
	# Model
	model = Sequential()
	# Conv layer
	# C1
	input_conv_layer_1 = Conv2D(64, kernel_size = (4, 4), strides = 1,padding='valid', activation='relu',data_format = 'channels_last',input_shape=input_img_size)
	output_conv_layer_1 = MaxPooling2D(pool_size = (2, 2), strides=(1, 1), padding='valid')
	model.add(input_conv_layer_1)
	model.add(output_conv_layer_1)
	# C2
	input_conv_layer_2 = Conv2D(128, kernel_size = (3, 3), strides = 1, padding='valid', activation='relu')
	output_conv_layer2 = MaxPooling2D(pool_size = (2,2), strides = (1,1), padding='valid')
	model.add(input_conv_layer_2)
	model.add(output_conv_layer2)
	# C3
	input_conv_layer_3 = Conv2D(256,kernel_size = (3,3), strides = 1, padding='valid',activation='relu')
	output_conv_layer_3 = MaxPooling2D(pool_size = (2,2), strides = (1,1), padding='valid')
	model.add(input_conv_layer_3)
	model.add(output_conv_layer_3)
	# Fully Conn
	flatten = Flatten()
	fully_conn_layer = Dropout(0.5)
	model.add(flatten)
	model.add(fully_conn_layer)
	# Output,softmax
	output_layer = Dense(number_of_classes,activation='softmax')
	model.add(output_layer)
	model.load_weights('imgs/simple_conv_nn_weights.h5')
	return model

def build_VGG_16():
	input_tensor = Input(shape=(224, 224, 3))
	precess_input = Lambda(vgg_preprocess_input)(input_tensor)
	vggModel = keras.applications.vgg16.VGG16(include_top=False, \
		 weights='imagenet',input_tensor=precess_input,input_shape=(224,224,3))
	# get the source output
	output_layer = vggModel.output
	flatten = Flatten()(output_layer)
	# full connected
	fc_cus_layer_1 = Dense(4096,activation='relu')(flatten)
	fc_cus_layer_1 = Dropout(0.5)(fc_cus_layer_1)
	# full connected 2
	fc_cus_layer_2 = Dense(4096,activation='relu')(fc_cus_layer_1)
	#fc_cus_layer_2 = Dropout(0.5)(fc_cus_layer_2)
	# custom output layer
	prediction_layer = Dense(number_of_classes, activation='softmax',name='vgg_out_put')(fc_cus_layer_2)
	# fine-tune model
	model = Model(inputs=vggModel.input, outputs=prediction_layer)
	model.load_weights('imgs/vgg_conv_nn_weights.h5')
	return model

def build_ResNet():
	input_tensor = Input(shape=(224, 224, 3))
	precess_input = Lambda(res_preprocess_input)(input_tensor)
	resModel = keras.applications.resnet50.ResNet50(include_top=False,input_tensor=precess_input, weights='imagenet',input_shape=(224,224,3))
	# origin output
	origin_output = resModel.output
	# full connected
	flatten = Flatten()(origin_output)
	prediction_layer = Dense(number_of_classes, activation='softmax',name='res_out_put')(flatten)
	# fine-tune model
	model = Model(inputs=resModel.input, outputs=prediction_layer)
	model.load_weights('imgs/resNet50_conv_nn_weights.h5')
	return model

def build_Inception():
	input_tensor = Input(shape=(299, 299, 3))
	precess_input = Lambda(inc_preprocess_input)(input_tensor)
	inceptionV3Model = keras.applications.inception_v3.InceptionV3(weights='imagenet',input_tensor=precess_input, include_top=False,input_shape=(299,299,3))
	# origin output
	origin_output = inceptionV3Model.output
	pooling_layer = GlobalAveragePooling2D()(origin_output)
	fc_layer = Dense(1024, activation='relu')(pooling_layer)
	prediction_layer = Dense(number_of_classes, activation='softmax',name='inc_out_put')(fc_layer)
	model = Model(inputs=inceptionV3Model.input, outputs=prediction_layer)
	model.load_weights('imgs/inceptionV3Model_conv_nn_weights.h5')
	return model

def build_VGG_CAM():
	input_tensor = Input(shape=(224, 224, 3))
	precess_input = Lambda(vgg_preprocess_input)(input_tensor)
	vggModel = keras.applications.vgg16.VGG16(include_top=False, \
		 weights='imagenet',input_tensor=precess_input,input_shape=(224,224,3))
	# get the source output
	# 假设时输出层的输入结构
	output_layer = vggModel.output
	# 换成池化层
	output = GlobalAveragePooling2D()(output_layer)
	# custom output layer
	prediction_layer = Dense(number_of_classes, activation='softmax',name='vgg_cam_out_put')(output)
	# fine-tune model
	model = Model(inputs=vggModel.input, outputs=prediction_layer)
	model.load_weights('imgs/vgg_CAM_weights.h5')
	return model

def build_Inception_CAM():
	input_tensor = Input(shape=(299, 299, 3))
	precess_input = Lambda(inc_preprocess_input)(input_tensor)
	# 299 x 299
	inceptionV3Model = keras.applications.inception_v3.InceptionV3(weights='imagenet',input_tensor=precess_input, include_top=False,input_shape=(299,299,3))
	# origin output
	origin_output = inceptionV3Model.output
	pooling_layer = GlobalAveragePooling2D()(origin_output)
	prediction_layer = Dense(number_of_classes, activation='softmax',name='inc_out_put')(pooling_layer)
	model = Model(inputs=inceptionV3Model.input, outputs=prediction_layer)
	model.load_weights('imgs/inception_CAM_weights.h5')
	return model


# 不单独训练了,直接采用训练好的 ResNet 结构做操作
def build_ResNet_CAM():
	input_tensor = Input(shape=(224, 224, 3))
	precess_input = Lambda(res_preprocess_input)(input_tensor)
	resModel = keras.applications.resnet50.ResNet50(include_top=False,input_tensor=precess_input, weights='imagenet',input_shape=(224,224,3))
	# full connected
	# origin output
	origin_output = resModel.output
	flatten = Flatten()(origin_output)
	prediction_layer = Dense(number_of_classes, activation='softmax',name='res_out_put')(flatten)
	full_model = Model(inputs=resModel.input, outputs=prediction_layer)
	full_model.load_weights('imgs/resNet50_conv_nn_weights.h5')
	return full_model # CAM,model
	'''
	input_tensor = Input(shape=(224, 224, 3))
	precess_input = Lambda(res_preprocess_input)(input_tensor)
	# 不需要预设置权值,直接可以获取保存训练好的权值
	resModel_cam = keras.applications.resnet50.ResNet50(include_top=False, weights=None,input_shape=(224,224,3))
	#获取卷积 Model
	resModel_cam.load_weights('imgs/resnet_CAM_weights.h5',by_name=True)
	# 获取权值 Model
	full_model = keras.applications.resnet50.ResNet50(include_top=False, weights=None,input_shape=(224,224,3))
	# origin output
	origin_output = full_model.output
	output = GlobalAveragePooling2D()(origin_output)
	# custom output layer
	prediction_layer = Dense(number_of_classes, activation='softmax',name='res_out_put')(output)
	# fine-tune model
	model = Model(inputs=full_model.input, outputs=prediction_layer)
	# 获取分类权值
	category_weights = model.layers[-1].get_weights()
	# 加载
	model.load_weights('imgs/resnet_CAM_weights.h5')
	'''
# CAM
# 训练之后的模型
# 可视化 Class Activation Mapping
def CAM(origin_image,conv_outs,weights,classes_out):
	'''
	origin_image: 原图片
	trained_model: 训练好的模型
		- VGG-CAM
		- ResNet-CAM
		- Inception-CAM
	'''
	'''
	# 获取权值
	last_weights = trained_model.layers[-2].get_weights()
	# 获取卷积层输出
	get_last_conv_output = K.function([trained_model.layers[0].input],
                                  [trained_model.layers[-3].output])
	get_last_conv_output = get_last_conv_output([origin_image])[0]
	# CAM 操作
	print(get_last_conv_output,last_weights)
	'''
	# 原图是 BGR 格式的
	cam = np.zeros((conv_outs.shape[0],conv_outs.shape[1]))
	for row,col in enumerate(weights[:,number_of_classes-1]): # 2048,9
		cam += col * conv_outs[:,:,row]
	# 归一化
	cam -= cam.min()
	cam /= cam.max()
	cam -= 0.2
	cam /= 0.8
	cam = cv2.resize(cam,(224,224))
	cam = np.uint8(255 * cam)
	#彩图
	hotMap = cv2.applyColorMap(cam,cv2.COLORMAP_JET)
	# 裁剪
	hotMap[np.where(hotMap < 0.2)] = 0
	#与原图权值相加
	out = cv2.addWeighted(origin_image,0.8,hotMap,0.4,0)
	# 显示
	index_class = classes_out.argmax()

	plt.title("pred:" + class_mapping[str(index_class)])
	plt.imshow(out) # out[:,:,::-1]
	plt.show()

if __name__ == '__main__':
	'''
	filePath = 'imgs/test/img_1.jpg'
	image = cv2.imread(filePath)
	image = image.astype(np.float64)
	image = cv2.resize(image,(224,224))
	image_data = vgg_preprocess_input(image)
	model = keras.applications.vgg16.VGG16(include_top=False,weights='imagenet',input_shape=(224,224,3))
	CAM(np.array([image_data]),model)
	'''
	#draw_loss_acc('vgg_16_train_loss_acc.txt')
	#buildMutliCNN()
	# VGG16Transfer()
	#ResNet50()
	#InceptionV3()
