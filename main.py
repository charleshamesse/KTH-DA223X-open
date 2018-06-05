import numpy as np
import pandas as pd
from data import generate_data
from model import ETEModel


if __name__ == '__main__':
    config = {
        'N': 100,
        'M': 50,
        'P': 10,
        'D': 4,
        'missing_prob': 0.3,
        'trait_mean': 0.5,
        'trait_stddev': 0.5,
        'obs_noise': 0.1,
        'data_split': 0.2,
    }

    # Generate data
    U, V, predictors, scores = generate_data(config)

    # Build model
    model = ETEModel(config)
    model.read_data()
    model.build()
    model.infer()
    model.eval()
    
    '''
    # Check the predictions, old version
    P = pd.read_csv('P.csv', index_col=0).as_matrix()
    S = pd.read_csv('S.csv', index_col=0).as_matrix()
    P_hat = pd.read_csv('P_hat.csv', index_col=0).as_matrix()
    S_hat = pd.read_csv('S_hat.csv', index_col=0).as_matrix()

    P_err = 0
    P_cnt = 0
    S_err = 0
    S_cnt = 0
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if not np.isnan(P[i,j]):
                P_err += np.absolute(P[i,j] - P_hat[i,j])
                P_cnt += 1
        for j in range(S.shape[1]):
            if not np.isnan(S[i,j]):
                S_err += np.absolute(S[i,j] - S_hat[i,j])
                S_cnt += 1
    print('Predictors training+testing MAE:\t', P_err/P_cnt)
    print('Scores training+testing MAE:\t', S_err/S_cnt) 
    '''