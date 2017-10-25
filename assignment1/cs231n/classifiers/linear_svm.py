import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        
        # check gradient equation for loss at http://cs231n.github.io/optimization-1/
        # in the section "Computing the gradient analytically with Calculus"

        # for a single training example xi, which has lable yi (in this code, the for loop with i, each i is a training example)
        # gradients only exist when margin > 0
        # dW_j and dW_yi has same dimention as xi
        
        # dW_yi is -xi times number of times margin > 0, when we loop through classes (in this code, the for loop with j)
        # to put it simply, dW_yi = (number of classes that have margin > 0) * xi
        # so here we just add -xi to dW_yi in the for loop when margin > 0, by the end of loop we add -xi exactly how many times margin > 0 
        
        # print(X[i, :].shape, dW[:, y[i]].shape)
        
        dW[:, y[i]] -= X[i, :]
        
        # dW_j when j != yi is xi when margin > 0
        # dW_j = xi, if margin > 0 for the class j
        dW[:, j] += X[i, :]
        
        # NOTE: gradients accumulate over each training example. so we use -= and += in these equations

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  # finally the gradient for regularization term 
  dW = dW/num_train + 2*reg*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]

  scores = np.dot(X, W)
    
  correct_scores = scores[np.arange(num_train), y]

  # print(correct_scores.shape)
  # print(scores.shape)

  margins = scores - correct_scores[:, np.newaxis] + 1  # delta = 1
  margins[np.arange(num_train), y] = 0  # set all marigins for correct class to 0

  # do the max operation
  margins[margins < 0] = 0
  
  # margins = np.maximum(np.zeros((num_classes,num_train)), margins)
    
  # Compute loss 
  loss = np.sum(margins)
  loss /= num_train

  # Add regularization
  loss += reg * np.sum(W * W)
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
    
  
  binary = margins
  binary[margins > 0] = 1

  col_sum = np.sum(binary, axis=1)
  binary[np.arange(num_train), y] = -col_sum[np.arange(num_train)]

  # see cs229-linalg notes - http://cs229.stanford.edu/section/cs229-linalg.pdf
  # section 2.3, matrix matrix products
  # by writing the matrix product in that form, we are doing essentially the same thing as in naive implementation code
  dW = np.dot(X.T, binary)
    

  # gradient for regularization term 
  dW = dW/num_train + 2*reg*W
  
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
