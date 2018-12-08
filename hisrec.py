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
from utils import precision_k_curve, recall_k_curve, ndcg_k_curve,hr_k_curve, cos_sim, map_k_curve
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

	def __init__(self,f_matrix, batch_size,layers,
				num_user,num_item,hidden_size,user_profile):
		print("Building model...")
		self.num_user = num_user
		self.num_item = num_item

		self.user_input = tf.placeholder(tf.int32,shape=[None])#shape=None
		self.item_input = tf.placeholder(tf.int32, shape=[None])

		self.prediction = tf.placeholder(tf.float32, shape=[None,1])




		self.user_embedding = tf.Variable(
			tf.random_uniform([num_user, hidden_size],-1.0,1.0))


		self.item_embedding = tf.Variable(
			tf.random_uniform([num_item, hidden_size],-1.0,1.0))

		# self.mlp_user_embedding = tf.Variable(
		# 	tf.random_uniform([num_user, hidden_size], -1.0,1.0))
		# self.mlp_item_embedding = tf.Variable(
		# 	tf.random_uniform([num_item, hidden_size],-1.0,1.0))

		
		
		user_latent = tf.nn.embedding_lookup(self.user_embedding, self.user_input)
		item_latent = tf.nn.embedding_lookup(self.item_embedding, self.item_input)

		# mlp_user_latent = tf.nn.embedding_lookup(self.mlp_user_embedding, self.user_input)
		# mlp_item_latent = tf.nn.embedding_lookup(self.mlp_item_embedding, self.item_input)


		sim1 = tf.reduce_sum(tf.multiply(user_latent, item_latent), axis=1, keep_dims = True)
		vector1 = tf.concat([user_latent, item_latent,sim1, tf.multiply(user_latent, item_latent) ], axis=1)
		vector1 = tf.layers.batch_normalization(vector1)

		for i in range(len(layers)):
			hidden = Dense(layers[i], activation='relu',kernel_initializer = 'lecun_uniform',name='v1_ui_hidden_' + str(i))
			vector1 = hidden(vector1)
		self.logits = Dense(1, kernel_initializer='lecun_uniform', name = 'prediction')(vector1)

		# mf_vector = tf.multiply(user_latent, item_latent)

		# sim = tf.reduce_sum(tf.multiply(mlp_user_latent, mlp_item_latent),axis=1, keep_dims=True)
		# mlp_vector = tf.concat([mlp_user_latent, mlp_item_latent,sim],axis=1)
		# # mlp_vector = tf.layers.batch_normalization(mlp_vector)
		# for i in range(len(layers)):
		# 	hidden = Dense(layers[i], kernel_initializer = 'lecun_uniform',activation='relu', name="layer%d"%i)
		# 	mlp_vector = hidden(mlp_vector)

		# # self.logits = self.logits_1
		# # self.logits = tf.cond(self.texting, lambda:temp_logits+self.logits_2, lambda:temp_logits)
		# predict_vector = tf.concat([mf_vector, mlp_vector], axis=1)
		# self.logits = Dense(1,kernel_initializer='lecun_uniform', name='prediction')(predict_vector)



		self.pred = tf.nn.sigmoid(self.logits)





		# self.loss = tf.reduce_mean(tf.sigmoid_cross_entropy_with_logits(logits,prediction))	
		self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.logits, labels = self.prediction))



		reg_error = tf.nn.l2_loss(self.user_embedding)+\
					tf.nn.l2_loss(self.item_embedding)
					# tf.nn.l2_loss(self.mlp_item_embedding)+\
					# tf.nn.l2_loss(self.mlp_user_embedding)

		self.cost = self.loss\
					+0.0001*reg_error\
					+float(sys.argv[2])*self.regularization(f_matrix,self.user_input, self.user_embedding)\
					+float(sys.argv[3])*self.attentive_embedding(self.user_input, self.user_embedding, user_profile)
					# +0.001*self.regularization(f_matrix,self.user_input, self.user_embedding)\
					# +0.001*self.attentive_embedding(self.user_input, self.user_embedding, user_profile)

		self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.cost)
		# self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)


		
	def regularization(self,f_matrix, user_input, user_embedding):
		# f_matrix = self.load_friend().astype(np.float32)
		f_matrix = self.load_friend().astype(np.float32)
		

		# batch_f_matrix = tf.nn.embedding_lookup(f_matrix, user_input)
		batch_f_matrix = tf.nn.embedding_lookup(f_matrix, user_input)
		sum_bfm = tf.reduce_sum(batch_f_matrix, axis = 1, keep_dims = True)
		n_sum_bfm = batch_f_matrix/(sum_bfm+10e-10)
		ref_f = tf.matmul(n_sum_bfm, user_embedding)


		batch_embedding = tf.nn.embedding_lookup(user_embedding, user_input)
		reg = tf.reduce_sum(tf.square(batch_embedding - ref_f))
		return reg




	def attentive_embedding(self, user_input, user_embedding, user_profile):
		f_matrix = self.load_friend().astype(np.float32)

		batch_profile = tf.nn.embedding_lookup(user_profile, user_input) ## batch*profile 1024*300   user_profile 7761*300
		batch_friend = tf.nn.embedding_lookup(f_matrix, user_input) ## batch*#user 1024*7761
		profile_size = user_profile.shape[-1]
		batch_size = user_input.shape.as_list()[0]
		embedding_size = user_embedding.shape.as_list()[-1]


		neighbor = np.load('neighbor.npy').astype(np.int32)
		n_neighbor = neighbor.shape[-1]
		n_index = tf.nn.embedding_lookup(neighbor, user_input) #batch 20

		n_index = tf.reshape(n_index, [-1,1]) #batch*20
		n_embedding = tf.nn.embedding_lookup(user_embedding, n_index) #batch*20 embedding_size

		n_profile = tf.nn.embedding_lookup(user_profile, n_index) #batch*20 profile_size
		n_profile = tf.reshape(n_profile, [-1,n_neighbor, profile_size])

		batch_profile = tf.expand_dims(batch_profile, 1)
		batch_profile = tf.tile(batch_profile,[1,n_neighbor,1])

		merge = tf.multiply(batch_profile,n_profile) 
		merge = tf.reshape(merge, [-1,profile_size]) # batch*20 profile_size

		layer1 = Dense(int(profile_size/2), activation='relu',kernel_initializer = 'lecun_uniform',name='sim1')
		layer2 = Dense(1, kernel_initializer='lecun_uniform', name = 'sim2')


		merge = layer2(layer1(merge)) #batch*20 1

		user2neighbor = tf.reshape(merge,[-1,n_neighbor])

		user2neighbor = tf.nn.softmax(5*user2neighbor)

		user2neighbor = tf.expand_dims(user2neighbor,-1)
		user2neighbor = tf.tile(user2neighbor,[1,1,embedding_size])

		n_embedding = tf.reshape(n_embedding, [-1,n_neighbor,embedding_size])

		n_weight = tf.reduce_sum(tf.multiply(user2neighbor, n_embedding),axis=1)

		b_embedding = tf.nn.embedding_lookup(user_embedding, user_input)
		reg = tf.reduce_sum(tf.square(b_embedding - n_weight))




		# expand_batch_user_profile = tf.expand_dims(batch_user_profile, 1)
		# batch_profile = tf.tile(expand_batch_user_profile,[1,self.num_user, 1])


		# w1 = tf.get_variable('w1', shape=[profile_size, profile_size], initializer = tf.contrib.layers.xavier_initializer())
		# w2 = tf.get_variable('w2', shape=[profile_size, profile_size], initializer = tf.contrib.layers.xavier_initializer())
		# v = tf.get_variable('v', shape=[profile_size,1],initializer = tf.contrib.layers.xavier_initializer())



		# layer1 = Dense(int(profile_size/2), activation='relu',kernel_initializer = 'lecun_uniform',name='sim1')
		# layer2 = Dense(1, kernel_initializer='lecun_uniform', name = 'sim2')





		# merge = tf.reshape(tf.multiply(batch_profile, user_profile),[-1,profile_size]) ##batch*user, profile_size

		# merge = layer2(layer1(merge))

		# batch2user = tf.reshape(merge,[-1,self.num_user])
		# batch2user = tf.multiply(batch2user, batch_user_friend)
		# user_weight = tf.nn.softmax(5*batch2user)
		# weight_neighbor = tf.matmul(user_weight, user_embedding)
		# batch_user_embedding = tf.nn.embedding_lookup(user_embedding, user_input)
		# reg = tf.reduce_sum(tf.square(batch_user_embedding - weight_neighbor))





		# batch_size,_,profile_size
		# batch_profile = tf.reshape(batch_profile,[-1, profile_size])
		# batch_profile = tf.matmul(batch_profile, w1)
		# batch_profile = tf.reshape(batch_profile, [-1,self.num_user,profile_size])

		# all_profile = tf.matmul(user_profile, w2)

		# inter = tf.nn.tanh(batch_profile+all_profile)  #batch user profile

		# inter = tf.reshape(inter,[-1,profile_size])
		# inter = tf.squeeze(tf.matmul(inter,v)) #batch*user,1
		# batch2user = tf.reshape(inter,shape=[-1,self.num_user])  #batch_user

		# batch2user = tf.multiply(batch2user, batch_user_friend)
		# user_weight = tf.nn.softmax(5*batch2user)
		# weight_neighbor = tf.matmul(user_weight, user_embedding)
		# batch_user_embedding = tf.nn.embedding_lookup(user_embedding, user_input)
		# reg = tf.reduce_sum(tf.square(batch_user_embedding - weight_neighbor))









		# element_dot = tf.reshape(tf.multiply(tile_batch_user_profile, user_profile), [-1,profile_size])
		# element_dot_w = tf.nn.tanh(tf.matmul(element_dot, w))
		# element_dot_v = tf.matmul(element_dot_w,v)
		# user2user = tf.reshape(element_dot_v,[batch_size, self.num_user])
		# usre2user = tf.multiply(user2user, batch_user_friend)

		# user_weight = tf.nn.softmax(5*user2user)

		# batch_weight_embedding = tf.matmul(user_weight, user_embedding)

		# batch_user_embedding = tf.nn.embedding_lookup(user_embedding, user_input)

		# reg = tf.reduce_sum(tf.square(batch_user_embedding - batch_weight_embedding))

		return reg

	def save_friend(self,f_matrix):
		fr = open('friends.pkl', 'wb')
		data = {}
		data['num_user'] = len(f_matrix)
		f_dic = dict([(i, np.where(f_matrix[i]>0)[0]) for i in range(len(f_matrix)) if len(np.where(f_matrix[i]>0)[0])>0])
		data['f_dic'] = f_dic
		pickle.dump(data, fr)
		fr.close()

	def load_friend(self):
		print('social reg loading...')
		fr = open('friends.pkl','rb')
		data = pickle.load(fr)
		fr.close()

		num_user = data['num_user']
		f_dic = data['f_dic']
		f_matrix = np.zeros((num_user, num_user))
		for i in f_dic.keys():
			f_matrix[i][f_dic[i]] = 1
		return f_matrix



class Data_Loader():
	def __init__(self, batch_size):
		print("data loading...")
		# pickle_file = open("data.pkl",'rb')
		pickle_file = open('data.pkl','rb')



		self.data = pickle.load(pickle_file)
		self.R_m = self.data['ratings']
		self.num_user = self.data['num_user']
		self.num_item = self.data['num_item']
		self.batch_size = batch_size
		# self.user_profile = np.random.random((self.num_user,200)).astype(np.float32)
		self.user_profile = np.load('user_profile.npy').astype(np.float32)

		friends = self.data['friends']
		f_matrix = np.zeros((self.num_user, self.num_user))

		social = self.data['friends']
		self.f_matrix = np.zeros((self.num_user, self.num_user))
		for key in social:
			for u in social[key]:
				self.f_matrix[u][social[key]] = 1


	def reset_data(self):

		print("resetting data...")
		u_input = self.data['train_user'][:]
		i_input = self.data['train_item'][:]
		item_num = self.data['num_item']
		ui_label = self.data['train_label'][:]
		negative_samples_num = 5
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








def get_data(user,item,data_loader):
	data = data_loader.data 
	# c_item = range(data['num_item'])
	num_item = data['num_item']
	negative_items = 100
	u = [user]*negative_items
	i = [item]+np.random.randint(0,num_item,(negative_items-1)).tolist()
	ui_label = [1]+[0]*(negative_items-1)
	pmtt = np.random.permutation(negative_items)
	return np.array(u),\
			np.array(i)[pmtt],\
			np.array(ui_label)[pmtt]


def test(data_loader, model):
	with tf.Session() as sess:
		# checkpoint_dir = './'+sys.argv[0].split('.')[0]+'_/'
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


		res_matrix = [[],[]]
		max_k=10
		metrics_num = 2
		f = [hr_k_curve,ndcg_k_curve]
		count = 0
		for user, item in zip(data_loader.data['test_user'], data_loader.data['test_item']):
			# u,i,u_text,i_text, item_adj,y_true = get_data(u,data_loader)
			u,i, ui_label = get_data(user,item,data_loader)

			y_pred = sess.run([model.pred], feed_dict = {model.user_input:u,
														model.item_input:i,
														model.prediction:ui_label.reshape((-1,1))})
			for i in range(metrics_num):
				res = f[i](ui_label.flatten(),y_pred[0].flatten(),max_k)
				res_matrix[i].append(res[:])

			count+=1
			if (count)%3000==0:
				print (np.mean(np.array(res_matrix),axis=1))
			sys.stdout.write("\ruser: "+str(user))
			sys.stdout.flush()
		print (np.mean(np.array(res_matrix),axis=1))
		
		res = np.mean(np.array(res_matrix), axis=1).T
		np.savetxt(checkpoint_dir+"res.dat", res, fmt = "%.5f", delimiter = '\t')
		# f = open("res_tf_social.pkl",'wb')
		# pickle.dump(res_matrix,f)
		# f.close()
def sample(u,i):
	user = []
	item = []
	for usr,itm in zip(u,i):
		rand = np.random.random()
		if rand<0.1:
			user.append(usr)
			item.append(itm)
	return user,item

def val(data_loader,sess, model, tv_user, tv_item):
	res_matrix = [[],[]]
	max_k=10
	metrics_num = 2
	f = [hr_k_curve,ndcg_k_curve]
	for user, item in zip(tv_user,tv_item):
		# u,i,u_text,i_text, item_adj,y_true = get_data(u,data_loader)
		u,i, ui_label = get_data(user,item,data_loader)
		y_pred = sess.run([model.pred], feed_dict = {model.user_input:u,
													model.item_input:i,
													model.prediction:ui_label.reshape((-1,1))})
		for i in range(metrics_num):
			res = f[i](ui_label.flatten(),y_pred[0].flatten(),max_k)
			res_matrix[i].append(res[:])

	res = np.mean(np.array(res_matrix), axis=1).T
	return res[-1,0]

def train(batch_size,data_loader, model):
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		saver = tf.train.Saver(tf.global_variables())
		# checkpoint_dir = './'+sys.argv[0].split('.')[0]+'_/'

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
		# user_embedding = np.load('./NeuMf_/user_embedding.npy')
		# item_embedding = np.load('./NeuMf_/item_embedding.npy')
		# sess.run(tf.assign(model.user_embedding, user_embedding))
		# sess.run(tf.assign(model.item_embedding, item_embedding))


		tv_user,tv_item = sample(data_loader.data['val_user'], data_loader.data['val_item'])
		best_hr_10 = 0
		epochs_1 = 10
		epochs_2 = 50
		for i in range(epochs_1):
			data_loader.reset_data()
			total_batch = int(data_loader.train_size/batch_size)
			for e in range(epochs_2):
				data_loader.reset_pointer()
				for b in range(total_batch):
					iterations = i*epochs_2*total_batch+e*total_batch+b
					u_input, i_input, ui_label = data_loader.next_batch()
					train_loss, _ = sess.run([model.cost, model.train_op], feed_dict={model.user_input: u_input,
																						model.item_input:i_input,
																						model.prediction:ui_label.reshape((-1,1))})
					sys.stdout.write('\r {}/{} epoch, {}/{} batch, train loss:{}'.\
									format(i,e,b,total_batch,train_loss))

					if(iterations)%5000==0:
						hr_10 = val(data_loader, sess, model, tv_user, tv_item)
						if hr_10>best_hr_10:
							print('\n', hr_10)
							best_hr_10 = hr_10
							saver.save(sess, checkpoint_dir+'model.ckpt', global_step = iterations)




checkpoint_dir = './'+sys.argv[0].split('.')[0]+'_' + sys.argv[2]+'_'+sys.argv[3]+'/'

if __name__ == '__main__':
	batch_size = 256
	# if len(sys.argv)>1 and sys.argv[1] == 'test':
	# 	if sys.argv[3] == 'True':
	# 		batch_size = 1200
	# 	else:
	# 		batch_size = 1200
		# batch_size = 600
	epochs = 100
	data_loader = Data_Loader(batch_size = batch_size)
# self,f_matrix, batch_size, num_user,
# 				vocab_size, seq_length,
# 				filter_sizes, num_filters,
# 				num_item, embedding_size,


	layers = eval('[64,16]')
	model = Model(
				f_matrix = data_loader.f_matrix,
				batch_size = batch_size,
				layers = layers,
				num_user = data_loader.num_user,
				num_item = data_loader.num_item,
				hidden_size = 64,
				user_profile = data_loader.user_profile)
	if sys.argv[1]=="test":
		test(data_loader, model)
	else:
		train(batch_size, data_loader, model)
