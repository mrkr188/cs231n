import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  num_classes = W.shape[1]
  num_train = X.shape[0]

  
    
  for i in xrange(num_train):

      # loss
      scores = X[i].dot(W)
    
      # shift values of scores for numeric reasons to prevent overflows
      # http://cs231n.github.io/linear-classify/#softmax
      scores -= scores.max()
        
      exp_sum = np.sum(np.exp(scores))
      correct_exp = np.exp(scores[y[i]])
      loss += - np.log( correct_exp / exp_sum)

      # gradient: w_j = 1/num_train * \sum_i[x_i * (p(y_i = j)-Ind{y_i = j} )]
      # gradient for correct class
      dW[:, y[i]] += (-1) * (exp_sum - correct_exp) / exp_sum * X[i]
      for j in xrange(num_classes):
          # pass correct class gradient
          if j == y[i]:
              continue
          # for incorrect classes
          dW[:, j] += np.exp(scores[j]) / exp_sum * X[i]

  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  # scores is N by C matrix containing class scores for each training example
  scores = X.dot(W)
  # normalize scores
  scores -= scores.max()
  
  scores = np.exp(scores)
  # denominator of softmax function, aka sum of class socres
  den_sums = np.sum(scores, axis=1)
  # numinator of softmax function, aka correct class scores
  nums = scores[range(num_train), y]

  # loss term  
  loss = -np.sum(np.log(nums / den_sums))/num_train

  # add regularization to loss
  loss += reg * np.sum(W * W)

  # gradient: w_j = 1/num_train * \sum_i[x_i * (p(y_i = j)-Ind{y_i = j} )]
  exp_scores = np.exp(scores)/np.sum(np.exp(scores), axis=1).reshape(num_train, 1)
  s = np.divide(scores, den_sums.reshape(num_train, 1))
  s[np.arange(num_train), y] = - (den_sums - nums) / den_sums
  # s[np.arange(num_train), y] -= 1   # this is simpler form of the equation above

  # see cs229-linalg notes - http://cs229.stanford.edu/section/cs229-linalg.pdf
  # section 2.3, matrix matrix products
  # by writing the matrix product in that form, we are doing essentially the same thing as in naive implementation code
  dW = X.T.dot(s)
  dW /= num_train


  dW += 2 * reg * W 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

