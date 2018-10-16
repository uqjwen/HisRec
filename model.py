import sys
import os
import numpy as np 
import tensorflow as tf 
import keras
from keras import backend as K
from keras.regularizers import l1, l2
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Dropout
from keras.layers import Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
import pickle
from time import time
# from metrics import precision_k_curve,recall_k_curve,ndcg_k_curve
from utils import precision_k_curve, recall_k_curve, ndcg_k_curve, cos_sim, map_k_curve
from tensorflow.contrib import rnn
from tensorflow.python.ops import control_flow_ops  
from tensorflow.python.training import moving_averages  
from tensorflow.python.training.moving_averages import assign_moving_average

# def bn(x, is_training):
# 	BN_DECAY = 0.9
# 	BN_EPSILON = 1e-5
# 	x_shape = x.get_shape()  
# 	params_shape = x_shape[-1:]  
  
# 	axis = list(range(len(x_shape) - 1))  
  
# 	beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer())  
# 	gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer())  
  
# 	moving_mean = tf.get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer(), trainable=False)  
# 	moving_variance = tf.get_variable('moving_variance', params_shape, initializer=tf.ones_initializer(), trainable=False)  
  
# 	# These ops will only be preformed when training.  
# 	mean, variance = tf.nn.moments(x, axis)  
# 	update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)  
# 	update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY)  
# 	tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)  
# 	tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)  
  
# 	mean, variance = control_flow_ops.cond(  
# 		is_training, lambda: (mean, variance),  
# 		lambda: (moving_mean, moving_variance))  
  
# 	return tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)  


class Model():

	def __init__(self,batch_size,layers,
				num_user,num_item,hidden_size, user_profile):
		print("Building model...")
		self.num_user = num_user
		self.num_item = num_item

		self.user_input = tf.placeholder(tf.int32,shape=[None])#shape=None
		self.item_input = tf.placeholder(tf.int32, shape=[None])

		self.prediction = tf.placeholder(tf.float32, shape=[None,1])




		self.user_embedding_matrix = tf.Variable(
			tf.random_uniform([num_user, hidden_size],-1.0,1.0))


		self.item_embedding_matrix = tf.Variable(
			tf.random_uniform([num_item, hidden_size],-1.0,1.0))

		
		
		user_latent = tf.nn.embedding_lookup(self.user_embedding_matrix, self.user_input)
		item_latent = tf.nn.embedding_lookup(self.item_embedding_matrix, self.item_input)





		sim1 = tf.reduce_sum(tf.multiply(user_latent, item_latent), axis=1, keep_dims = True)
		vector1 = tf.concat([user_latent, item_latent,sim1, tf.multiply(user_latent, item_latent) ], axis=1)
		vector1 = tf.layers.batch_normalization(vector1)

		for i in range(len(layers)):
			hidden = Dense(layers[i], activation='relu',kernel_initializer = 'lecun_uniform',name='v1_ui_hidden_' + str(i))
			vector1 = hidden(vector1)
		self.logits_1 = Dense(1, kernel_initializer='lecun_uniform', name = 'prediction')(vector1)




		self.logits = self.logits_1
		# self.logits = tf.cond(self.texting, lambda:temp_logits+self.logits_2, lambda:temp_logits)


		self.pred = tf.nn.sigmoid(self.logits)





		# self.loss = tf.reduce_mean(tf.sigmoid_cross_entropy_with_logits(logits,prediction))	
		self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.logits, labels = self.prediction))



		reg_error = tf.nn.l2_loss(self.user_embedding_matrix)+\
					tf.nn.l2_loss(self.item_embedding_matrix)


		reg_rate = float(sys.argv[2])
		self.cost = self.loss+\
					0.0001*reg_error+\
					reg_rate*self.attentive_embedding(self.user_input, self.user_embedding_matrix, user_profile)+\
					reg_rate*self.regularization(self.user_input, self.user_embedding_matrix)

		self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.cost)
		# self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
	def regularization(self, user_input, user_embedding_matrix):
		f_matrix = self.load_friend().astype(np.float32)
		s_f_matrix = np.sum(f_matrix,axis=1, keepdims= 1)
		s_f_matrix[s_f_matrix<=0] = 1
		n_f_matrix = f_matrix/s_f_matrix




		batch_u_embedding = tf.nn.embedding_lookup(user_embedding_matrix, user_input)
		batch_f_matrix = tf.nn.embedding_lookup(n_f_matrix, user_input)

		batch_f_embedding = tf.matmul(batch_f_matrix, user_embedding_matrix)


		reg = tf.reduce_sum(tf.square(batch_f_embedding - batch_u_embedding))

		return reg


	def attentive_embedding(self, user_input, user_embedding_matrix, user_profile):
		f_matrix = self.load_friend().astype(np.float32)

		batch_user_profile = tf.nn.embedding_lookup(user_profile, user_input) ## batch*profile 1024*300   user_profile 7761*300
		batch_user_friend = tf.nn.embedding_lookup(f_matrix, user_input) ## batch*#user 1024*7761
		profile_size = user_profile.shape[-1]



		s_matrix = np.ones((self.num_user, self.num_user)) - np.eye(self.num_user)
		batch_s_matrix = tf.nn.embedding_lookup(s_matrix.astype(np.float32), user_input)

		# weight = tf.Variable(tf.truncated_normal([profile_size], stddev))

		expand_batch_user_profile = tf.expand_dims(batch_user_profile, 1)
		tile_batch_user_profile = tf.tile(expand_batch_user_profile,[1,self.num_user, 1])

		# expand_user_profile = tf.expand_dims(user_profile,0)
		# batch_size = batch_user_profile.shape.as_list()[0]
		# print(expand_batch_user_profile.shape, batch_size)
		# tile_user_profile = tf.tile(expand_user_profile,[batch_size,1,1])

		w = tf.get_variable('w', shape=[profile_size, profile_size], initializer = tf.contrib.layers.xavier_initializer())
		v = tf.get_variable('v', shape=[profile_size,1],initializer = tf.contrib.layers.xavier_initializer())

		element_dot = tf.reshape(tf.multiply(tile_batch_user_profile, user_profile), [-1,profile_size])
		element_dot_w = tf.nn.tanh(tf.matmul(element_dot, w))
		element_dot_v = tf.matmul(element_dot_w,v)


		user2user = tf.reshape(element_dot_v,[batch_size, self.num_user])
		# user2user = tf.multiply(user2user, batch_s_matrix)
		usre2user = tf.multiply(user2user, batch_user_friend)






		# w = tf.get_variable('w',shape=[profile_size],initializer=tf.contrib.layers.xavier_initializer())
		# b = tf.Variable(tf.constant(0.1,shape=[1]))

		# temp = tf.multiply(batch_user_profile, w) ##temp batch*#profile 1024*300

		# user2user = tf.matmul(temp, np.transpose(user_profile))+b  ##user2user batch*#user 1024*7761


		# user_weight = tf.nn.softmax(tf.multiply(user2user, batch_user_friend))
		user_weight = tf.nn.softmax(5*user2user)

		batch_weight_embedding = tf.matmul(user_weight, user_embedding_matrix)


		batch_user_embedding = tf.nn.embedding_lookup(user_embedding_matrix, user_input)

		reg = tf.reduce_sum(tf.square(batch_user_embedding - batch_weight_embedding))

		return reg

		

	def save_friend(self,f_matrix):
		fr = open('friends.pkl', 'wb')
		data = {}
		data['num_user'] = len(f_matrix)
		f_dic = dict([(i, np.where(f_matrix[i]>0)[0]) for i in range(len(f_matrix)) if len(np.where(f_matrix[i]>0)[0])>0])
		data['f_dic'] = f_dic
		pickle.dump(data, fr)
		fr.close()

	def load_friend_origin(self):
		print('social reg loading...')
		# fr = open('friends_new.pkl','rb')
		# data = pickle.load(fr)
		# fr.close()

		# num_user = data['num_user']
		# f_dic = data['f_dic']
		# f_matrix = np.zeros((num_user, num_user))
		# for i in f_dic.keys():
		# 	f_matrix[i][f_dic[i]] = 1
		# return f_matrix
		f_matrix = np.load('./friends_new.npy')
		row,col = np.where(f_matrix!=0)
		for r,c in zip(row,col):
			rand = np.random.random()
			if rand>0.68:
				f_matrix[r][c] = 0
				col_rand = np.random.randint(len(f_matrix[0]))
				f_matrix[r][col_rand] = 1
		np.save('friends_new1', f_matrix)
		return f_matrix



	def load_friend(self):
		print('social reg loading...')
		f_matrix = np.load('./s_matrix_1.npy')
		return f_matrix
		# fr = open('friends.pkl','rb')
		# data = pickle.load(fr)
		# fr.close()

		# num_user = data['num_user']
		# f_dic = data['f_dic']
		# f_matrix = np.zeros((num_user, num_user))
		# for i in f_dic.keys():
		# 	f_matrix[i][f_dic[i]] = 1
		# return f_matrix


class Data_Loader():
	def __init__(self, batch_size):
		print("data loading...")
		pickle_file = open("./data40.pkl",'rb')



		self.data = pickle.load(pickle_file)
		self.R_m = self.data['ratings']
		self.num_user = self.data['num_user']
		self.num_item = self.data['num_item']
		self.batch_size = batch_size
		# self.user_profile = np.random.random((self.num_user,200)).astype(np.float32)
		self.user_profile = np.load('./user_profile.npy')


	def reset_data(self):

		print("resetting data...")
		u_input = self.data['train_user'][:]
		i_input = self.data['train_item'][:]
		item_num = self.data['num_item']
		ui_label = self.data['train_label'][:]
		negative_samples_num = 6
		for u in set(u_input):
			all_item = range(item_num)
			# positive = np.array(self.data['train_item'])[np.where(np.array(self.data['train_user'])==u)[0]]
			missing_values = list(set(all_item)-set(self.R_m[u]))
			# missing_values = list(set(all_item) - set(positive))

			u_input.extend([u]*negative_samples_num)
			negative_samples = np.random.choice(missing_values,negative_samples_num, replace=False)
			i_input.extend(list(negative_samples))
			ui_label.extend([0]*negative_samples_num)

		p_index = np.random.permutation(range(len(u_input)))
		self.u_input = np.array(u_input)[p_index]
		self.i_input = np.array(i_input)[p_index]
		self.ui_label = np.array(ui_label)[p_index]
		self.train_size = len(u_input)



	def reset_pointer(self):
		self.pointer = 0

	def next_batch(self):
		start = self.pointer*self.batch_size
		end = (self.pointer+1)*self.batch_size

		self.pointer+=1
		# return self.u_input[start:end], self.i_input[start:end], self.ui_label[start:end]
		item_index = self.i_input[start:end]
		user_index = self.u_input[start:end]



		return self.u_input[start:end],\
		self.i_input[start:end],\
		self.ui_label[start:end]






def get_data(u,data_loader):
	data = data_loader.data
	c_item = range(data['num_item'])
	train_user = np.array(data['train_user'])
	train_item = np.array(data['train_item'])

	test_user = np.array(data['test_user'])
	test_item = np.array(data['test_item'])
	test_label = np.array(data['test_label'])



	selected_test_item = test_item[np.where(test_user==u)[0]]
	negative_items = list(set(c_item)-set(data_loader.R_m[u]))

	rest_num = data_loader.batch_size-len(selected_test_item)
	# rest_num = len(selected_test_item)*500
	add_negative_items = np.random.choice(negative_items, rest_num, replace=False)

	items = np.concatenate((selected_test_item, add_negative_items))
	labels = np.array([1]*len(selected_test_item)+[0]*len(add_negative_items))
	users = np.array([u]*len(items))

	return users, items, labels

# def test(data_loader, model, checkpoint_dir):
# 	with tf.Session() as sess:
# 		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
# 		saver = tf.train.Saver(tf.global_variables())
# 		if ckpt and ckpt.model_checkpoint_path:
# 			saver.restore(sess, ckpt.model_checkpoint_path)
# 			print(" [*] Loaded parameters success!!!")
# 		else:
# 			print(' [!] loaded parameters failed')


def test(batch_size, data_loader, model):
	with tf.Session() as sess:
		# checkpoint_dir = './'+sys.argv[0].split('.')[0]+'_/'
		checkpoint_dir = './'+sys.argv[2]+'_'+sys.argv[3]+'/'
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		saver = tf.train.Saver(tf.global_variables())
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print (' [*] Loaded parameters success!!!')
		else:
			print (' [!] Loaded parameters failed...')
			return

		# user_embedding, group_embedding = sess.run([model.user_embedding, model.group_embedding])
		# np.savetxt(checkpoint_dir+'user_embedding', user_embedding)
		# np.savetxt(checkpoint_dir+'group_embedding', group_embedding)


		res_matrix = [[],[],[]]
		max_k=40
		metrics_num = 3
		f = [precision_k_curve,recall_k_curve,ndcg_k_curve]
		for user in range(data_loader.num_user):
			# u,i,u_text,i_text, item_adj,y_true = get_data(u,data_loader)
			u,i, y_true = get_data(user,data_loader)
			if np.sum(y_true)==0:
				continue
			y_pred = sess.run([model.pred], feed_dict = {model.user_input:u,
														model.item_input:i,
														model.prediction:y_true.reshape((-1,1))})
			for i in range(metrics_num):
				res = f[i](y_true.flatten(),y_pred[0].flatten(),max_k)
				res_matrix[i].append(res[:])
			if (user+1)%1000==0:
				print (np.mean(np.array(res_matrix),axis=1))
			sys.stdout.write("\ruser: "+str(user))
			sys.stdout.flush()
		print (np.mean(np.array(res_matrix),axis=1))
		
		res = np.mean(np.array(res_matrix), axis=1)
		np.savetxt(checkpoint_dir+"res.dat", res, fmt = "%.5f", delimiter = '\t')
		# f = open("res_tf_social.pkl",'wb')
		# pickle.dump(res_matrix,f)
		# f.close()
			

def train(batch_size,data_loader, model):
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		saver = tf.train.Saver(tf.global_variables())
		# checkpoint_dir = 'model'
		# checkpoint_dir = sys.argv[2]
		# checkpoint_dir = './'+sys.argv[0].split('.')[0]+'/'
		checkpoint_dir = './'+sys.argv[2]+'_'+sys.argv[3]+'/'
		print(checkpoint_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print (' [*] Loaded parameters success!!!')
		else:
			print (' [!] Loaded parameters failed...')

		# save_dir = './'+sys.argv[0].split('.')[0][:-2]+'_'+sys.argv[2]+'/'
		# user_embedding = np.genfromtxt(save_dir+'user_embedding')
		# group_embedding = np.genfromtxt(save_dir+'group_embedding')
		# sess.run(tf.assign(model.user_embedding, user_embedding))
		# sess.run(tf.assign(model.group_embedding, group_embedding))

		epochs = 100
		data_loader.reset_data()
		for i in range(epochs):
			total_batch = int(data_loader.train_size/batch_size)
			data_loader.reset_pointer()
			for b in range(total_batch):
				u_input, i_input, ui_label = data_loader.next_batch()
				train_loss, _ = sess.run([model.cost, model.train_op], feed_dict={model.user_input: u_input,
																					model.item_input:i_input,
																					model.prediction:ui_label.reshape((-1,1))})
				sys.stdout.write('\r {}/{} epoch, {}/{} batch, train loss:{}'.\
								format(i,epochs,b,total_batch,train_loss))

				if(i*total_batch+b+1)%20000==0 or \
					(i==epochs-1 and b == total_batch-1):
					saver.save(sess, checkpoint_dir+'model.ckpt', global_step = i*total_batch+b)




if __name__ == '__main__':
	batch_size = 128
	# if len(sys.argv)>1 and sys.argv[1] == 'test':
	# 	if sys.argv[3] == 'True':
	# 		batch_size = 1200
	# 	else:
	# 		batch_size = 1200
		# batch_size = 600
	epochs = 100
	reg_rate = float(sys.argv[2])
	hidden_size = int(sys.argv[3])
	data_loader = Data_Loader(batch_size = batch_size)
# self,f_matrix, batch_size, num_user,
# 				vocab_size, seq_length,
# 				filter_sizes, num_filters,
# 				num_item, embedding_size,


	layers = eval('[64, 16]')
	model = Model(batch_size = batch_size,
				layers = layers,
				num_user = data_loader.num_user,
				num_item = data_loader.num_item,
				hidden_size = hidden_size,
				user_profile = data_loader.user_profile)
	if sys.argv[1]=="test":
		test(1024, data_loader, model)
	else:
		train(batch_size, data_loader, model)
