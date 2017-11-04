NEURAL NETWORKS ALGORITHMS
1.	USING SCIKIT LEARN : 
<br /> MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08, hidden_layer_sizes=(13, 13, 13), learning_rate='constant', learning_rate_init=0.001, max_iter=500, momentum=0.9, nesterovs_momentum=True, power_t=0.5, random_state=None, shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False, warm_start=False)
 You can change the activation function to be:
I.	‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x 
II.	 ‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
III.	 ‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x). 
IV.	‘relu’, the rectified linear unit function, returns f(x) = max(0, x) 
 
You can change the solver to be: Stochastic Gradient Descent(SGD), Adam, or L-BFGS D. Play around with other parameters to check how the classifier behaves.
2.	USING HARD CODE: 
STOCHASTIC GRADIENT DESCENT IS USED . 
FEATURE NORMALISATION IS USED,
 ACTIVATION FUNCTION- LOGISTIC SIGMOID FUNCTION
