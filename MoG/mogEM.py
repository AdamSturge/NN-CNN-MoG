from kmeans import *
import sys
import matplotlib.pyplot as plt
import math
from util import unblockshaped
plt.ion()

if sys.version_info.major == 3:
    raw_input = input


def mogEM(x, K, iters, randConst=1, minVary=0, useKMeans = False):
    """
    Fits a Mixture of K Diagonal Gaussians on x.

    Inputs:
      x: data with one data vector in each column.
      K: Number of Gaussians.
      iters: Number of EM iterations.
      randConst: scalar to control the initial mixing coefficients
      minVary: minimum variance of each Gaussian.

    Returns:
      p: probabilities of clusters (or mixing coefficients).
      mu: mean of the clusters, one in each column.
      vary: variances for the cth cluster, one in each column.
      logLikelihood: log-likelihood of data after every iteration.
    """
    N, T = x.shape

    # Initialize the parameters
    p = randConst + np.random.rand(K, 1)
    p = p / np.sum(p)   # mixing coefficients
    mn = np.mean(x, axis=1).reshape(-1, 1)
    vr = np.var(x, axis=1).reshape(-1, 1)

    # Question 4.3: change the initializaiton with Kmeans here
    #--------------------  Add your code here --------------------
    if useKMeans:
        mu = KMeans(x,K,5)
    else :
        mu = mn + np.random.randn(N, K) * (np.sqrt(vr) / randConst)
    
    #------------------- Answers ---------------------

    #------------------------------------------------------------
    vary = vr * np.ones((1, K)) * 2
    vary = (vary >= minVary) * vary + (vary < minVary) * minVary

    logLikelihood = np.zeros((iters, 1))

    # Do iters iterations of EM
    for i in range(iters):
        # Do the E step
        respTot = np.zeros((K, 1))
        respX = np.zeros((N, K))
        respDist = np.zeros((N, K))
        ivary = 1 / vary
        logNorm = np.log(p) - 0.5 * N * np.log(2 * np.pi) - \
            0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1)
        logPcAndx = np.zeros((K, T))
        for k in range(K):
            dis = (x - mu[:, k].reshape(-1, 1))**2
            logPcAndx[k, :] = logNorm[k] - 0.5 * \
                np.sum(ivary[:, k].reshape(-1, 1) * dis, axis=0)

        mx = np.max(logPcAndx, axis=0).reshape(1, -1)
        PcAndx = np.exp(logPcAndx - mx)
        Px = np.sum(PcAndx, axis=0).reshape(1, -1)
        PcGivenx = PcAndx / Px
        logLikelihood[i] = np.sum(np.log(Px) + mx)

        print('Iter %d logLikelihood %.5f' % (i + 1, logLikelihood[i]))

        # Plot log likelihood of data
        plt.figure(0)
        plt.clf()
        plt.plot(np.arange(i), logLikelihood[:i], 'r-')
        plt.title('Log-likelihood of data versus # iterations of EM')
        plt.xlabel('Iterations of EM')
        plt.ylabel('Log-likelihood')
        plt.draw()

        # Do the M step
        # update mixing coefficients
        respTot = np.mean(PcGivenx, axis=1).reshape(-1, 1)
        p = respTot

        # update mean
        respX = np.zeros((N, K))
        for k in range(K):
            respX[:, k] = np.mean(x * PcGivenx[k, :].reshape(1, -1), axis=1)

        mu = respX / respTot.T

        # update variance
        respDist = np.zeros((N, K))
        for k in range(K):
            respDist[:, k] = np.mean(
                (x - mu[:, k].reshape(-1, 1))**2 * PcGivenx[k, :].reshape(1, -1), axis=1)

        vary = respDist / respTot.T
        vary = (vary >= minVary) * vary + (vary < minVary) * minVary

    return p, mu, vary, logLikelihood


def mogLogLikelihood(p, mu, vary, x):
    """ Computes log-likelihood of data under the specified MoG model

    Inputs:
      x: data with one data vector in each column.
      p: probabilities of clusters.
      mu: mean of the clusters, one in each column.
      vary: variances for the cth cluster, one in each column.

    Returns:
      logLikelihood: log-likelihood of data after every iteration.
    """
    K = p.shape[0]
    N, T = x.shape
    ivary = 1 / vary
    logLikelihood = np.zeros(T)
    for t in range(T):
        # Compute log P(c)p(x|c) and then log p(x)
        logPcAndx = np.log(p) - 0.5 * N * np.log(2 * np.pi) \
            - 0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1) \
            - 0.5 * \
            np.sum(ivary * (x[:, t].reshape(-1, 1) - mu)
                   ** 2, axis=0).reshape(-1, 1)

        mx = np.max(logPcAndx, axis=0)
        logLikelihood[t] = np.log(np.sum(np.exp(logPcAndx - mx))) + mx

    return logLikelihood


def q2():
    # Question 4.2 and 4.3
    K = 7
    iters = 20
    minVary = 0.01
    randConst = 100

    # load data
    inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData(
        '../toronto_face.npz')

    # Train a MoG model with 7 components on all training data, i.e., inputs_train,
    # with both original initialization and kmeans initialization.
    #------------------- Add your code here ---------------------
    p, mu, vary, logLikelihood = mogEM(inputs_train,K,iters,randConst,minVary,True)
    
    N,M = mu.shape
    N2 = int(math.sqrt(N))
    mu = np.reshape(mu,(N2,N2,M))
    mu = np.transpose(mu,[2,0,1])
    mu = unblockshaped(mu,7*N2,N2)
    
    N,M = vary.shape
    N2 = int(math.sqrt(N))
    vary = np.reshape(vary,(N2,N2,M))
    vary = np.transpose(vary,[2,0,1])
    vary = unblockshaped(vary,7*N2,N2)
    
    A = np.concatenate((mu,vary),axis=1)
    
    plt.matshow(A, fignum=2, cmap=plt.cm.gray)
    plt.savefig(filename="mu_and_sigma.png")

    N = len(p)
    x = range(1,N+1)
    width = 1/1.5
    plt.figure()
    plt.bar(x, p, width, color="blue",align='center')
    plt.show()


    #------------------- Answers ---------------------


def q4():
    # Question 4.4
    iters = 20
    minVary = 0.01
    randConst = 100.0

    numComponents = np.array([7, 14, 21, 28, 35])
    T = numComponents.shape[0]
    errorTrain = np.zeros(T)
    errorTest = np.zeros(T)
    errorValidation = np.zeros(T)

    # extract data of class 1-Anger, 4-Happy
    dataQ4 = LoadDataQ4('../toronto_face.npz')
    # images
    x_train_anger = dataQ4['x_train_anger']
    x_train_happy = dataQ4['x_train_happy']
    x_train = np.concatenate([x_train_anger, x_train_happy], axis=1)
    x_valid = np.concatenate(
        [dataQ4['x_valid_anger'], dataQ4['x_valid_happy']], axis=1)
    x_test = np.concatenate(
        [dataQ4['x_test_anger'], dataQ4['x_test_happy']], axis=1)

    # label
    y_train = np.concatenate(
        [dataQ4['y_train_anger'], dataQ4['y_train_happy']])
    y_valid = np.concatenate(
        [dataQ4['y_valid_anger'], dataQ4['y_valid_happy']])
    y_test = np.concatenate([dataQ4['y_test_anger'], dataQ4['y_test_happy']])
    # Hints: this is p(d), use it based on Bayes Theorem
    num_anger_train = x_train_anger.shape[1]
    num_happy_train = x_train_happy.shape[1]
    log_likelihood_class = np.log(
        [num_anger_train, num_happy_train]) - np.log(num_anger_train + num_happy_train)
    
    for t in range(T):
        K = numComponents[t]

        # Train a MoG model with K components
        # Hints: using (x_train_anger, x_train_happy) train 2 MoGs
        #-------------------- Add your code here ------------------------------
        pA, muA, varyA, logLikelihoodA = mogEM(x_train_anger,K,iters,randConst,minVary,True)
        pH, muH, varyH, logLikelihoodH = mogEM(x_train_happy,K,iters,randConst,minVary,True)
        #------------------- Answers ---------------------

        # Compute the probability P(d|x), classify examples, and compute error rate
        # Hints: using (x_train, y_train), (x_valid, y_valid), (x_test, y_test)
        # to compute error rates, you may want to use mogLogLikelihood function
        #-------------------- Add your code here ------------------------------
        N,M = x_train.shape
        x_train_prob_A = mogLogLikelihood(pA, muA, varyA, x_train) + log_likelihood_class[0]
        x_train_prob_H = mogLogLikelihood(pH, muH, varyH, x_train) + log_likelihood_class[1]
        x_train_prob = np.column_stack((x_train_prob_A,x_train_prob_H))
        classifcation = np.argmax(x_train_prob,axis=1)
        errorTrain[t] = np.sum(np.array(list(map(lambda x : abs(x),y_train-classifcation)))) / M
        
        N,M = x_valid.shape
        x_valid_prob_A = mogLogLikelihood(pA, muA, varyA, x_valid) + log_likelihood_class[0]
        x_valid_prob_H = mogLogLikelihood(pH, muH, varyH, x_valid) + log_likelihood_class[1]
        x_valid_prob = np.column_stack((x_valid_prob_A,x_valid_prob_H))
        classifcation = np.argmax(x_valid_prob,axis=1)
        errorValidation[t] = np.sum(np.array(list(map(lambda x : abs(x),y_valid-classifcation)))) / M
        
        N,M = x_test.shape
        x_test_prob_A = mogLogLikelihood(pA, muA, varyA, x_test) + log_likelihood_class[0]
        x_test_prob_H = mogLogLikelihood(pH, muH, varyH, x_test) + log_likelihood_class[1]
        x_test_prob = np.column_stack((x_test_prob_A,x_test_prob_H))
        classifcation = np.argmax(x_test_prob,axis=1)
        errorTest[t] = np.sum(np.array(list(map(lambda x : abs(x),y_test-classifcation)))) / M
        #------------------- Answers ---------------------

    # Plot the error rate
    plt.figure(0)
    plt.clf()
    #-------------------- Add your code here --------------------------------
    print(errorTest)
    #------------------- Answers ---------------------
    # to be removed before release
    plt.plot(numComponents, errorTrain, 'r', label='Training')
    plt.plot(numComponents, errorValidation, 'g', label='Validation')
    plt.plot(numComponents, errorTest, 'b', label='Testing')
    plt.xlabel('Number of Mixture Components')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.draw()
    plt.pause(0.0001)
    

if __name__ == '__main__':
    #-------------------------------------------------------------------------
    # Note: Question 4.2 and 4.3 both need to call function q2
    # you need to comment function q4 below
    #q2()

    #-------------------------------------------------------------------------
    # Note: Question 4.4 both need to call function q4
    # you need to comment function q2 above
    q4()

    input('Press Enter to continue.')
