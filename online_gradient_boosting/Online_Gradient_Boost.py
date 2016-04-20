import numpy as np
from sklearn.tree import DecisionTreeRegressor
import theano.tensor as T
import theano
from IPython import embed
import cPickle
import gzip
from multiprocessing import Pool
from FeedForward_NN import FeedForward_NN_regression
import copy

class Weak_learner(object):
    #class of a weak learner.
    #it contains the hypothesis, e.g., regression tree. 
    #it contains the aggregated datasets X, and Y, in case using follow the leader. 
	def __init__(self, learner, index):
		self.learner = copy.deepcopy(learner);
		self.datasetX = [];
		self.datasetY = [];
		self.id = index;

	#train in batch.
	def train(self, max_down_samples = None):
		dataX = np.array(self.datasetX);
		dataY = np.array(self.datasetY);		
		if dataX.ndim == 1: #only one sample
			dataX = dataX.reshape([1,-1]);
			dataY = dataY.reshape([1,-1]);
		#downsample:
		if max_down_samples is None or max_down_samples > len(dataX):
			num_down_samples = len(dataX); 
		else:
			num_down_samples = max_down_samples;
		perm = np.random.choice(len(dataX), num_down_samples, replace = False );
		self.learner.fit(dataX, dataY); #batch update, essentially follow the (regularized) leader
	
	#train in online fashion. 
	def online_train(self, x, y, learning_rate = 1e-3):
		self.learner.online_fit(x, y, lr = learning_rate);
	
	def predict(self, X):
		if X.ndim == 1: #only have one sample:
			tmpX = X.reshape([1,-1]);
			Y = self.learner.predict(tmpX);
        
		if X.ndim == 1:
			return Y[0];
		else:
			return Y;
   
   	def collect_data(self, x, y):
		self.datasetX.append(x);
		self.datasetY.append(y);


#################################### Binary Classifier #########################################
class Online_Gradient_Boost_binary_class(object):
	#implementation of binary classifier
	#labels {-1,1}
	def __init__(self, dx, learner, num_wl = 10, lr = 1e-2):
		self.weak_learners = [];
		self.num_wl = num_wl;
		self.lr = lr;
		self.aggregated_loss = 0.
		self.iter = 1;
		for i in xrange(num_wl):
			wlearner = Weak_learner(learner, index = i);
			self.weak_learners.append(wlearner);
			
	def loss(self, p, y):
		nll = np.log(1. + np.exp(-2.*y*p));
		return nll;
	
	def _linear_combine_wl(self, x):
		pred = 0.;
		for i in xrange(0,len(self.weak_learners)):
			pred = pred - self.lr * self.weak_learners[i].predict(x);
		return pred;
	
	def online_train(self, x, y, max_down_samples = None, N_job = 7):
		pred = 0.;
		for i in xrange(0, len(self.weak_learners)):
			#compute gradient:
			gradient_target = -2.*y / (1 + np.exp(2.*y*pred));
			self.weak_learners[i].collect_data(x, gradient_target);
			pred = pred - self.lr * self.weak_learners[i].predict(x);
			if np.mod(self.iter,1) == 0 and self.iter >= 10: #starting doing follow the leader with at least 10 examples. 
				self.weak_learners[i].train(max_down_samples);
		
		self.aggregated_loss = self.aggregated_loss + self.loss(pred, y);
		print "the average negative log likelihood is {}, at iter {}".format(self.aggregated_loss / self.iter, self.iter);
		self.iter = self.iter + 1;
	
	def fit(self, train_X, train_y, n_epoch = 1):
		for e in xrange(n_epoch):
			for i in xrange(train_X.shape[0]):
				self.online_train(train_X[i], train_y[i], max_down_samples = 40000);
	
	def predict(self, X):
		y = [];
		if X.ndim == 1:
			X = X.reshape([1,-1]);
		for i in xrange(0,X.shape[0]):
			x = X[i];
			pred = self._linear_combine_wl(x);
			p_1 = 1./(1.+np.exp(-2.*pred));
			if p_1 >= 0.5:
				y.append(1);
			else:
				y.append(-1);
		return y;
				

####################################### Multi-class ######################################					
class Online_Gradient_Boost_multi_class(object):
	#implementation of online gradient boosting.
	def __init__(self, dx, k, learner, num_wl = 10, lr = 1e-2, wl_lr = 1e-3, initializer = None):
		self.weak_learners = [];
		self.num_wl = num_wl;
		self.dx = dx;
		self.k = k;
		self.aggregated_loss = 0.;
		self.iter = 1;
		self.lr = lr;
		self.wl_lr = wl_lr;
		for i in xrange(num_wl):
			wlearner = Weak_learner(learner, index = i);
			self.weak_learners.append(wlearner); 
		
		self.initializer = initializer;
		
		#start theano to build stuff. 
		self.sym_pred = T.vector('x'); #this represents the sum of weak learner's predictions.
		self.sym_y_one_hot = T.vector('y');
		#compute the negative log likelihood
		#compute the gradient wrt the prediction vector 
		self.prob_vector = T.nnet.softmax(self.sym_pred)[0]; #take the first row. 
		self.sym_nll = -T.log(T.dot(self.prob_vector,self.sym_y_one_hot)); #negative log likelihood.
		grad_wrt_pred = T.grad(cost = self.sym_nll, wrt = self.sym_pred);  
		#embed()
		self.loss = theano.function(inputs = [self.sym_y_one_hot, self.sym_pred], 
									outputs = self.sym_nll, allow_input_downcast=True); #return a number.
		self.compute_gradient = theano.function(inputs = [self.sym_y_one_hot, self.sym_pred], 
									outputs = grad_wrt_pred, allow_input_downcast=True); #returns a vector.
		self.compute_prob_vector = theano.function(inputs = [self.sym_pred], 
									outputs = self.prob_vector, allow_input_downcast=True); #returns a vector.
		
	def _linear_combine_wl(self, x):
		if self.initializer is None:
			pred = np.zeros(self.k);
		else:
			pred = self.initializer.predict_before_softmax(x); #the pred before feeing to softmax layer!!
				
		for i in xrange(0,len(self.weak_learners)):
			pred = pred - self.lr * self.weak_learners[i].predict(x);	
		return pred;
		
	def online_train(self, x, y, max_down_samples = None): #dim(x) = dx, y is a k-dim one-hot representation. 
		if self.initializer is None:
			pred = np.zeros(self.k); #start from zeros. 
		else:
			pred = self.initializer.predict_before_softmax(x); #the pred before feeding to softmax layer!!
		for i in xrange(len(self.weak_learners)):
			grad_target = self.compute_gradient(y, pred);
			self.weak_learners[i].collect_data(x, grad_target);
			pred = pred - self.lr * self.weak_learners[i].predict(x);
			
			self.weak_learners[i].online_train(x, grad_target, learning_rate = self.wl_lr); #train in online fashion.
				
			#if np.mod(self.iter, 1) == 0 and self.iter >= 10:
				#train weak learner:
			#	self.weak_learners[i].train(max_down_samples);
		
		self.aggregated_loss = self.aggregated_loss + self.loss(y, pred);
		print "the average negative log likelihood is {}, at iteration {}".format(self.aggregated_loss / (self.iter*1), self.iter);
		self.iter = self.iter + 1;
		
	def fit(self, X, Y, n_epoch = 1):
		for e in xrange(n_epoch):
			for i in xrange(len(X)):
				self.online_train(X[i], Y[i], max_down_samples = 10000);
			
					
	def predict(self, X):
		y = [];
		if X.ndim == 1: 
			X = X.reshape([1,-1]); #at least 2-d
		for i in xrange(0,X.shape[0]):
			x = X[i];
			pred = self._linear_combine_wl(x);
			prob_vector = self.compute_prob_vector(pred);
			y.append(np.argmax(prob_vector));
		
		return y;
		
					
if __name__ == '__main__':

	'''
	filename = "/home/wensun/Desktop/UCI_datasets/binary_class/a9a_scaled_dataset.p";
	[train_X, train_y, test_X, test_y] = cPickle.load(open(filename, 'rb'));
	num_learner = 50; 
	step_size = 1e-1;
	regression_tree = DecisionTreeRegressor(max_depth = 8);
	#initlize the regression_tree:
	regression_tree.fit(train_X[0:20], train_y[0:20]);
	
	binary_boost = Online_Gradient_Boost_binary_class(dx = train_X.shape[1], learner = regression_tree, 
									num_wl = num_learner, lr = step_size);
	binary_boost.fit(train_X, train_y);	
	'''
	##load mnist dataset:
	f = gzip.open('mnist.pkl.gz', 'rb')
	mnist = cPickle.load(f);
	train_X, train_y = mnist[0];
	valid_X, valid_y = mnist[1];
	test_X,  test_y  = mnist[2];
	
	num_class = len(np.unique(train_y));
	#transform train_y & valid_y to one-hot representation:
	train_y_one_hot = np.zeros((train_y.shape[0], num_class));
	valid_y_one_hot = np.zeros((valid_y.shape[0], num_class));
	for i in xrange(0, train_y.shape[0]):
		train_y_one_hot[i, train_y[i]] = 1.;
	for i in xrange(0, valid_y.shape[0]):
		valid_y_one_hot[i, valid_y[i]] = 1.;

	#construct regression tree:
	num_learner = 1500; 
	lr = 1e-1;
	FNN = FeedForward_NN_regression(dx = train_X.shape[1], dy = num_class, 
		nhs = [100], relu = 1, ridge = 1e-15);
	#regression_tree = DecisionTreeRegressor(max_depth = 3);
	#initlize the regression_tree:
	#regression_tree.fit(train_X[0:20], train_y_one_hot[0:20]);
	ob = Online_Gradient_Boost_multi_class(dx = train_X.shape[1], k = num_class, 
		learner = FNN, num_wl = num_learner, lr = lr);					  
	ob.fit(train_X, train_y_one_hot, n_epoch = 5);
              

     
        
        
        
     
    