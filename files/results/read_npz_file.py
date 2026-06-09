import numpy as np
data = np.load('results/qsann_fed_log.npz')
rounds = data['rounds']
mse    = data['mse']
mae    = data['mae']
comm   = data['comm']