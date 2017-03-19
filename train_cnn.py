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

def weight_initial(shape):
    weights=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(weights)
def bias_initial(shape):
    bias=tf.constant(0.1,shape=shape)
    return tf.Variable(bias)
def conv2d(x,w):
    return tf.nn.conv2d(input=x,filter=w,strides=[1,1,1,1],padding="SAME")
def Max_pool(x):
    return tf.nn.max_pool(value=x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

def train(trainX, trainY,testX,testY):
    Labels=np.zeros((60000,10),dtype=np.float32)
    for i in range(trainY.shape[0]):
        Labels[i,trainY[i]]=1.0
    Labels_test=np.zeros((10000,10),dtype=np.float32)
    for i in range(testY.shape[0]):
        Labels_test[i,testY[i]]=1.0
    print (Labels[:10])
    print (trainY[:10])
    x=tf.placeholder(tf.float32,[None,28,28,1])
    x_image=tf.reshape(x,[-1,28,28,1])
    y_true=tf.placeholder(tf.float32,[None,10])
    w1_conv=weight_initial([5,5,1,32])
    b1_conv=bias_initial([32])
    first_conv_out=tf.nn.relu(conv2d(x_image,w1_conv)+b1_conv)
    h_pool1=Max_pool(first_conv_out)
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    #temp=sess.run(first_conv_out,feed_dict={x:trainX[:7].astype(np.float32),y_true:Labels[:7]})
    #Temp=np.array(temp)
    #print (Temp.shape)
    w_conv2=weight_initial([5,5,32,64])
    b_conv2=bias_initial([64])
    second_conv_out=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
    
    h_pool2=Max_pool(second_conv_out)
    w_fc_1=weight_initial([7*7*64,1024]) #1000 neurons in first fully connected layer
    b_fc_1=bias_initial([1024])
    new_h_pool2=tf.reshape(h_pool2,[-1,7*7*64])
    h_fc_1=tf.nn.relu(tf.matmul(new_h_pool2,w_fc_1) +b_fc_1)
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc_1, keep_prob)
    w_fc_2=weight_initial([1024,10])   #1000 neurons in first fully connected layer
    b_fc_2=bias_initial([10])
    y_conv=tf.matmul(h_fc1_drop,w_fc_2) + b_fc_2
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_conv))
    train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_true,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.global_variables_initializer())
    for i in range(2):
        for j in range(0,trainX.shape[0],50):
            batch_x=trainX[j:j+50,:]
            batch_y=Labels[j:j+50,:]
            sess.run(train_step,feed_dict={x:batch_x,y_true:batch_y,keep_prob:0.5})
        print ( "epoch "+ str(i+1)+" finished.")

    #print ("Accuracy on training data is: ",)
    #Acc=sess.run(accuracy,feed_dict={x:trainX,y_true:Labels})
    #print (Acc)
    print ("Accuracy on test data is: ",)
    List=[]
    for j in range(0,testX.shape[0],100):
        batch_x=testX[j:j+100,:]
        batch_y=Labels_test[j:j+100,:]
        Acc,w1,b1,w2,b2,w3,b3,w4,b4=sess.run([accuracy,w1_conv,b1_conv,w_conv2,b_conv2,w_fc_1,b_fc_1,w_fc_2,b_fc_2],feed_dict={x:batch_x,y_true:batch_y,keep_prob:1.0})
        List.append(Acc)
    print ("Accuracy on test data is : ",sum(List)/len(List))
    w1=np.array(w1)
    b1=np.array(b1)
    w2=np.array(w2)
    b2=np.array(b2)
    w3=np.array(w3)
    b3=np.array(b3)
    w4=np.array(w4)
    w4=np.array(b4)
    np.save("weights/w1.npy",w1)
    np.save("weights/b1.npy",b1)
    np.save("weights/w2.npy",w2)
    np.save("weights/b2.npy",b2)
    np.save("weights/w3.npy",w3)
    np.save("weights/b3.npy",b3)
    np.save("weights/w4.npy",w4)
    np.save("weights/b4.npy",b4)



    




















    '''
    Complete this function.
    '''


# def test(testX,testY):
#     '''

#     Complete this function.
#     This function must read the weight files and
#     return the predicted labels.
#     The returned object must be a 1-dimensional numpy array of
#     length equal to the number of examples. The i-th element
#     of the array should contain the label of the i-th test
#     example.
#     '''

#     Labels_test=np.zeros((10000,10),dtype=np.float32)
#     for i in range(testY.shape[0]):
#         Labels_test[i,testY[i]]=1.0
#     w1=np.load("w1.npy")
#     b1=np.load("b1.npy")
#     w2=np.load("w2.npy")
#     b2=np.load("b2.npy")
#     w3=np.load("w3.npy")
#     b3=np.load("b3.npy")
#     w4=np.load("w4.npy")
#     b4=np.load("b4.npy")

#     x=tf.placeholder(tf.float32,[None,28,28,1])
#     x_image=tf.reshape(x,[-1,28,28,1])
#     y_true=tf.placeholder(tf.float32,[None,10])
#     w1_conv=tf.constant(w1,tf.float32)
#     b1_conv=tf.constant(b1,tf.float32)
#     first_conv_out=tf.nn.relu(conv2d(x_image,w1_conv)+b1_conv)
#     h_pool1=Max_pool(first_conv_out)


#     #temp=sess.run(first_conv_out,feed_dict={x:trainX[:7].astype(np.float32),y_true:Labels[:7]})
#     #Temp=np.array(temp)
#     #print (Temp.shape)
#     w_conv2=tf.constant(w2,tf.float32)
#     b_conv2=tf.constant(b2,tf.float32)
#     second_conv_out=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
    
#     h_pool2=Max_pool(second_conv_out)
#     w_fc_1=tf.constant(w3,tf.float32)    #1000 neurons in first fully connected layer
#     b_fc_1=tf.constant(b3,tf.float32)
#     print (tf.shape(b_fc_1))
#     new_h_pool2=tf.reshape(h_pool2,[-1,7*7*64])
#     h_fc_1=tf.matmul(new_h_pool2,w_fc_1) +b_fc_1
#     keep_prob = tf.placeholder(tf.float32)
#     h_fc1_drop = tf.nn.dropout(h_fc_1, keep_prob)
#     w_fc_2=tf.constant(w4,tf.float32)    #1000 neurons in first fully connected layer
#     b_fc_2=tf.constant(b4,tf.float32)
#     print (tf.shape(b_fc_2))
#     y_conv=tf.matmul(h_fc1_drop,w_fc_2) + b_fc_2
#     cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_conv))
#     train_step=tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
#     correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_true,1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     sess=tf.Session()
#     sess.run(tf.global_variables_initializer())
#     List=[]
#     for j in range(0,testX.shape[0],100):
#         batch_x=testX[j:j+100,:]
#         batch_y=Labels_test[j:j+100,:]
#         Acc=sess.run(accuracy,feed_dict={x:batch_x,y_true:batch_y,keep_prob:1.0})
#         List.append(Acc)
#     print ("Accuracy on test data is : ",sum(List)/len(List))


