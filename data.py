import numpy as np
import pandas as pd

def generate_data(config):
    print('Generating data..')

    # Config
    D = config['D']
    N = config['N']
    M = config['M']
    P = config['P']
    missing_prob = config['missing_prob']
    trait_mean = config['trait_mean']
    trait_stddev = config['trait_stddev']
    obs_noise = config['obs_noise']

    # Patient traits
    U = np.random.normal(trait_mean, trait_stddev, (D, N))
    
    # Measurement traits (fix upper square)
    V = np.random.normal(trait_mean, trait_stddev, (D, M))
    V[:D,:D] = np.eye(D)

    # Generate predictors
    predictors = np.matmul(np.transpose(U), V) + np.random.normal(0, obs_noise, (N, M))# NxM
    
    # Generate scores by applying a linear transformation
    W = np.random.normal(0, 0.5, (M, P))
    b = np.random.normal(0, 0.5, P)
    scores = np.matmul(predictors, W) + b + np.random.normal(0, obs_noise, (N, P)) 

    # Normalize
    predictors = (predictors - np.amin(predictors, axis=0)) / (np.amax(predictors, axis=0) - np.amin(predictors, axis=0))
    scores = (scores - np.amin(scores, axis=0)) / (np.amax(scores, axis=0) - np.amin(scores, axis=0))
    
    # Apply nans
    I_P_nan = np.random.choice(a=[False, True], size=(N, M), p=[1-missing_prob, missing_prob])
    I_S_nan = np.random.choice(a=[False, True], size=(N, P), p=[1-missing_prob, missing_prob])
    predictors[I_P_nan] = np.nan
    scores[I_S_nan] = np.nan

    # Save
    predictors_df = pd.DataFrame(predictors)
    predictors_df.to_csv('P.csv')
    scores_df = pd.DataFrame(scores)
    scores_df.to_csv('S.csv')
    
    return U, V, predictors, scores