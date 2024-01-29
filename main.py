import numpy as np
import matplotlib.pyplot as plt
from models import SupportVectorRegression

# Usage Example
x = np.arange(1, 5, 0.5)
y = x ** 2 + np.random.rand(len(x))
threshold = 0.25 * (y[-1] - y[0]) + y[-1]

# Usage for ploynomial kernel with my customized scoring function
options = {
    'kernel': 'poly',
    'degree': 2,
    'gamma': 'scale',
}
model = SupportVectorRegression().customized_extrapolation(x, y, threshold, options)

# model is yeilding result so we need to iterate on it 
# it uses a portion of the x and y each time
# you can skip all but the last one which uses all training data
for iteration, result in enumerate(model):
    
    x_prediction = result['x_prediction']
    y_prediction = result['y_prediction']

    if x_prediction is None: continue
    
    plt.plot(x, y, '--', label='All Data', color='black', lw=0.25)
    plt.plot(x[:iteration + 1], y[:iteration + 1], '-o', label='Training Data', color='darkgreen')
    plt.plot(x_prediction, y_prediction, label='Extrapolated Curve', alpha=0.9)
    plt.axhline(y=threshold, label='Failure Threshold', ls='--', color='darkorange')
    plt.legend()
    plt.show()