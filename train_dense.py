'''
Deep Learning Programming Assignment 2
--------------------------------------
Name:
Roll No.:

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import numpy as np
import tensorflow as tf

def train(trainX, trainY,testX,testY):
    BatchSize=100
    HiddenLayer=120
    TrainX=np.reshape(trainX,(60000,784))
    TestX=np.reshape(testX,(10000,784))
    Labels=np.zeros((60000,10))
    for i in range(trainY.shape[0]):
        Labels[i,trainY[i]]=1
    Labels_test=np.zeros((10000,10))
    for i in range(testY.shape[0]):
        Labels_test[i,testY[i]]=1
    W1_numpy=np.random.randn(784,HiddenLayer)*(2/BatchSize)**0.5
    W2_numpy=np.random.randn(HiddenLayer,10)*(2/BatchSize)**0.5

    W1=tf.Variable(W1_numpy.astype(np.float32),tf.float32)
    b1=tf.Variable(0.1,[HiddenLayer])
    X=tf.placeholder(tf.float32,[None,784])
    Y_true=tf.placeholder(tf.float32,[None,10])
    h1=tf.sigmoid(tf.matmul(X,W1)+b1)/BatchSize
    W2=tf.Variable(W2_numpy.astype(np.float32),tf.float32)
    b2=tf.Variable(0.1,[10])
    Y_fc=tf.matmul(h1,W2)+b2
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_true, logits=Y_fc))
    train_step=tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(Y_fc,1), tf.argmax(Y_true,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    

    sess=tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        for j in range(0,TrainX.shape[0],BatchSize):
            batch_x=TrainX[j:j+BatchSize,:]
            batch_y=Labels[j:j+BatchSize,:]
            #print (j)
            sess.run(train_step,feed_dict={X:batch_x,Y_true:batch_y})
        print ( "epoch "+ str(i+1)+" finished.")
    w11,w22=sess.run([W1,W2],feed_dict={X:batch_x,Y_true:batch_y})

    #print (sess.run(correct_prediction,feed_dict={X:TrainX,Y_true:Labels,W1:w11,W2:w22}))



    print ("Accuracy on training data is: ",)
    Acc=sess.run(accuracy,feed_dict={X:TrainX,Y_true:Labels,W1:w11,W2:w22})
    print (Acc*100)
    print ("Accuracy on test data is: ",)
    Acc=sess.run(accuracy,feed_dict={X:TestX,Y_true:Labels_test,W1:w11,W2:w22})
    print (Acc*100)




    '''
    Complete this function.
    '''


def test(testX):
    '''
    Complete this function.
    This function must read the weight files and
    return the predicted labels.
    The returned object must be a 1-dimensional numpy array of
    length equal to the number of examples. The i-th element
    of the array should contain the label of the i-th test
    example.
    '''

    return np.zeros(testX.shape[0])
