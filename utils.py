from sklearn.metrics import average_precision_score
import numpy as np 
def cos_sim(a,b):
	num = a.dot(b)
	if len(a.shape)>1:
		denom = np.linalg.norm(a,axis=1)*np.linalg.norm(b)
	else:
		denom = np.linalg.norm(a)*np.linalg.norm(b)
	cos = num/denom
	sim = 0.5+0.5*cos
	return sim
def recall_k_curve(y_true,y_pred,max_k):
	total = np.sum(y_true)
	res = []
	recommend = np.argsort(-y_pred)
	for k in range(max_k):
		step = np.sum(y_true[recommend[:k+1]])
		res.append(step*1.0/total)
	return res 

def precision_k_curve(y_true,y_pred,max_k):
	recommend = np.argsort(-y_pred)
	res = []
	for k in range(max_k):
		step = np.sum(y_true[recommend[:k+1]])
		res.append(step/(k+1))
	return res 

def ndcg(y_true, y_pred):
	dcg = 0.0
	# recommend = np.argsort(-y_pred)
	for i,item in enumerate(y_true):
		if item==1:
			dcg += 1/(np.log(i+2)/np.log(2))
	idcg = 0.0
	for i in range(min(len(y_pred), len(y_true))):
		idcg += 1/(np.log(i+2)/np.log(2))

	ndcg = dcg/idcg

	return ndcg
def ndcg_k_curve(y_true, y_pred, max_k):
	res = []
	recommend = np.argsort(-y_pred)

	for i in range(max_k):
		res.append(ndcg(y_true[recommend][:i+1], y_pred[recommend][:i+1]))
	return res
	



def average_precision(y_true, y_pred):
	assert len(y_true) == len(y_pred)
	score = 0.0
	hit = 0
	for i,item in enumerate(y_true):
		if item == 1:
			hit+=1.0
			score+=hit/(i+1)
	return 0 if hit == 0 else score/hit

def map_k_curve(y_true,y_pred,max_k):
	res = []
	recommend = np.argsort(-y_pred)
	for i in range(max_k):
		res.append(average_precision(y_true[recommend][:i+1], y_pred[recommend][:i+1]))
	return res

