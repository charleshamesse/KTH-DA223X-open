import pandas as pd
import numpy as np
import tensorflow as tf
import edward as ed
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from edward.models import Normal


class ETEModel():
    def __init__(self, config):
        print('Instanciating model..')
        self.D = config['D']
        self.trait_mean = config['trait_mean']
        self.trait_stddev = config['trait_stddev']
        self.obs_noise = config['obs_noise']
        self.data_split = config['data_split']

    def read_data(self):
        print('Reading data..')
        self.predictors = pd.read_csv('P.csv', index_col=0)
        self.scores = pd.read_csv('S.csv', index_col=0)

    def _build_indices(self):
        # Fill with 0s (inference doesn't take nans, it takes 0-filled matrices and masks)
        self.predictors_zeros = np.array(self.predictors.fillna(0).as_matrix())
        self.scores_zeros = np.array(self.scores.fillna(0).as_matrix())
        
        # Shapes
        self.N = self.predictors.shape[0] # Patients
        self.M = self.predictors.shape[1] # Predictors
        self.P = self.scores.shape[1] # Scores

        # Indices
        I_predictors_notnull =  1 - np.isnan(self.predictors)
        I_scores_notnull =  1 - np.isnan(self.scores)
        I_notnull = np.concatenate((I_predictors_notnull, I_scores_notnull), axis=1)

        # P: randomly pick in all values
        I_P_all = np.ones((self.N,self.M))
        I_P_train = np.random.choice(a=[False, True], size=(self.N, self.M), p=[self.data_split, 1-self.data_split])
        I_P_test = I_P_all - I_P_train

        # S: randomly pick in patients
        I_S_all = np.ones((self.N, self.P))
        I_S_train = np.ones(self.N) # Vector for patient mask
        I_S_train[:round(self.N*self.data_split)] = 0
        random.shuffle(I_S_train) 
        I_S_train = np.expand_dims(I_S_train, 1) # To apply patient mask vector on the N,P matrix
        I_S_train = I_S_all * I_S_train
        I_S_test = I_S_all - I_S_train
        
        # All multiplied by the notnull indices
        self.I_train = np.concatenate([I_P_train, I_S_train], axis=1) * I_notnull
        self.I_test = np.concatenate([I_P_test, I_S_test], axis=1) * I_notnull

    def neural_network(self, X):
        h = tf.nn.tanh(tf.matmul(X, self.W_0) + self.b_0)
        h = tf.nn.tanh(tf.matmul(h, self.W_1) + self.b_1)
        return h

    def build(self):
        print("Building model..")
        self._build_indices()

        # Data imputation
        self.M_I = tf.placeholder(tf.float32, [self.N, self.M+self.P])
        self.M_P_observed = tf.placeholder(tf.float32, [self.N, self.M])

        self.M_U = Normal(loc=tf.ones([self.D, self.N]) * self.trait_mean,
            scale=self.trait_stddev * tf.ones([self.D, self.N]))
        self.M_V = Normal(loc=tf.ones([self.D, self.M]),
            scale=self.trait_stddev * tf.ones([self.D, self.M])) # Note: Removed symmetry breaking
        self.M_P = Normal(
            loc=(tf.matmul(self.M_U, self.M_V, transpose_a=True))* self.M_I[:,:self.M],
            scale = self.obs_noise * tf.ones([self.N, self.M]) #0.0001 .. + 1.5 * 
        )
        self.M_P_with_observations = self.M_P_observed * self.M_I[:,:self.M] + self.M_P * (1 - self.M_I[:,:self.M])
        

        # Outcome prediction
        L1 = self.P
        L2 = self.P
        S_scale_multiplier = 1
        W_scale_multiplier = 1
        self.W_0 = Normal(loc=tf.zeros([self.M, L1]), scale=W_scale_multiplier * tf.ones([self.M, L1]) * 2 / (self.M + L1))
        self.b_0 = Normal(loc=tf.zeros(L1), scale=W_scale_multiplier * tf.ones(L1) * 1 /  L1)
        self.W_1 = Normal(loc=tf.zeros([L1, L2]), scale=W_scale_multiplier * tf.ones([L1, L2])  * 2 / (L1 + L2))
        self.b_1 = Normal(loc=tf.zeros(L2), scale=W_scale_multiplier * tf.ones(L2) * 1 / L2)
        self.M_S = Normal(
            loc=self.neural_network(self.M_P_with_observations) * self.M_I[:,self.M:], 
            scale= S_scale_multiplier * self.obs_noise * self.M_I[:,self.M:] + self.obs_noise * (1-self.M_I[:,self.M:])
        )

        # Variational model for both components
        self.M_qU = Normal(loc=tf.Variable(tf.random_normal([self.D, self.N])),
                        scale=tf.nn.softplus(tf.Variable(tf.random_normal([self.D, self.N]))))
        self.M_qV = Normal(loc=tf.Variable(tf.random_normal([self.D, self.M])),
                        scale=tf.nn.softplus(tf.Variable(tf.random_normal([self.D, self.M]))))

        with tf.variable_scope("qW_0"):
            loc = tf.Variable(tf.random_normal([self.M, L1]))
            scale = tf.nn.softplus(tf.get_variable("scale", [self.M, L1]))
            self.qW_0 = Normal(loc=loc, scale=scale)
        with tf.variable_scope("qb_0"):
            loc = tf.Variable(tf.random_normal([L1]))
            scale = tf.nn.softplus(tf.get_variable("scale", [L1]))
            self.qb_0 = Normal(loc=loc, scale=scale)
        with tf.variable_scope("qW_1"):
            loc = tf.Variable(tf.random_normal([L1, L2]))
            scale = tf.nn.softplus(tf.get_variable("scale", [L1, L2]))
            self.qW_1 = Normal(loc=loc, scale=scale)
        with tf.variable_scope("qb_1"):
            loc = tf.Variable(tf.random_normal([L2]))
            scale = tf.nn.softplus(tf.get_variable("scale", [L2]))
            self.qb_1 = Normal(loc=loc, scale=scale)

    def infer(self):
        print("Running inference..")
        inference = ed.KLqp({
            self.M_U: self.M_qU,
            self.M_V: self.M_qV,
            self.W_0: self.qW_0, self.b_0: self.qb_0,
            self.W_1: self.qW_1, self.b_1: self.qb_1,
        }, data={
            self.M_P: self.predictors_zeros,
            self.M_P_observed: self.predictors_zeros,
            self.M_S: self.scores_zeros,
            self.M_I: self.I_train
        })

        inference.run(n_samples=5, n_iter=2000)

        # Posterior predictive distributions
        self.M_qP = ed.copy(self.M_P, {
            self.M_U: self.M_qU,
            self.M_V: self.M_qV,
            self.M_I: tf.ones((self.N, self.M+self.P))
        })
        
        self.M_qS = ed.copy(self.M_S, {
            self.M_P_observed: self.predictors_zeros.astype(np.float32),
            self.M_U: self.M_qU,
            self.M_V: self.M_qV,
            self.M_P: self.M_qP,
            self.W_0: self.qW_0, self.b_0: self.qb_0,
            self.W_1: self.qW_1, self.b_1: self.qb_1,
            self.M_I: tf.ones((self.N, self.M+self.P)),
        })

    def eval(self):
        print("Evaluating..")
        # Make predictions
        sess = ed.get_session()

        print("Predictors MAE:")
        predictions_predictors = self.make_predictions(sess, self.M_qP, 300)
        mae_test = self.compute_mae(sess, predictions_predictors, self.predictors_zeros, (self.I_test[:,:self.M]).astype(bool), 'mae')
        mae_train = self.compute_mae(sess, predictions_predictors, self.predictors_zeros, (self.I_train[:,:self.M]).astype(bool), 'mae')
        print("Test:\t", mae_test,"\nTrain:\t", mae_train)
                
        print("Scores MAE:")
        predictions_scores = self.make_predictions(sess, self.M_qS, 300)
        mae_test = self.compute_mae(sess, predictions_scores, self.scores_zeros, (self.I_test[:,self.M:]).astype(bool), 'mae')
        mae_train = self.compute_mae(sess, predictions_scores, self.scores_zeros, (self.I_train[:,self.M:]).astype(bool), 'mae')
        print("Test:\t", mae_test,"\nTrain:\t", mae_train)

        predictors_df = pd.DataFrame(predictions_predictors.eval(session = sess))
        predictors_df.to_csv('P_hat.csv')
        scores_df = pd.DataFrame(predictions_scores.eval(session = sess))
        scores_df.to_csv('S_hat.csv')

    def make_predictions(self, sess, posterior, n_samples):
        pred = [sess.run(posterior) for _ in range(n_samples)]
        pred = tf.cast(tf.add_n(pred), pred[0].dtype) / tf.cast(n_samples, pred[0].dtype)
        return pred

    def compute_mae(self, sess, predictions, targets, indices, metric):
        # Apply mask
        predictions = tf.boolean_mask(tf.cast(predictions, tf.float32), indices.astype(bool))
        targets = tf.boolean_mask(tf.cast(targets, tf.float32), indices.astype(bool))

        # Compute error
        evaluations = []
        if metric == 'mse':
            evaluations += [tf.reduce_mean(tf.square(predictions - targets))]
        elif metric == 'mae':
            evaluations += [tf.reduce_mean(tf.abs(predictions - targets))]

        if len(evaluations) == 1:
            return sess.run(evaluations[0])
        else:
            return sess.run(evaluations)