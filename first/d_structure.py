import numpy as np 
import pickle

def d_similarity():
	fr = open('s_matrix_1.pkl','rb')
	dic = pickle.load(fr)
	num_users = dic['num_user']
	data = np.zeros((num_users, num_users))
	for key in dic.keys():
		if key == 'num_user':
			continue
		row,col = key.split(',')
		data[int(row)][int(col)] = float(dic[key])
	np.save('s_matrix_1',data)
if __name__ == '__main__':
	d_similarity()
