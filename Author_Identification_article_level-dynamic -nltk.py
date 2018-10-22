import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import nltk
from tensorflow.contrib import rnn
import time
import random
import pickle as pkl
from sklearn.preprocessing import OneHotEncoder

start_time = time.time()
# evaluate time taken for training
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"
    
def add_padding(batch, padded_batch):
    padded_batch = batch
    max_length=0
    for article_idx in range(len(batch)):
        if max_length<len(batch[article_idx]):
            max_length = len(batch[article_idx])
    for article_idx in range(len(batch)):
        for dummy in range(max_length-len(batch[article_idx])):
            padded_batch[article_idx].append(np.ones(input_columns, dtype="float").tolist())
    return padded_batch
            
## intitializing glove
glove_vectors_file = "/glove.6B.50d.txt"

glove_wordmap = {}
with open(glove_vectors_file, "r", encoding="utf8") as glove:
    for line in glove:
        name, vector = tuple(line.split(" ", 1))
        glove_wordmap[name] = np.fromstring(vector, sep=" ")
        
authors = ["AaronPressman","AlanCrosby","AlexanderSmith","BenjaminKangLim","BernardHickey","BradDorfman","DarrenSchuettler","DavidLawder","EdnaFernandes","EricAuchard","FumikoFujisaki","GrahamEarnshaw","HeatherScoffield","JaneMacartney","JanLopatka","JimGilchrist","JoeOrtiz","JohnMastrini","JonathanBirt","JoWinterbottom","KarlPenhaul","KeithWeir","KevinDrawbaugh","KevinMorrison","KirstinRidley","KouroshKarimkhany","LydiaZajc","LynneO'Donnell","LynnleyBrowning","MarcelMichelson","MarkBendeich","MartinWolk","MatthewBunce","MichaelConnor","MureDickie","NickLouth","PatriciaCommins","PeterHumphrey","PierreTran","RobinSidel","RogerFillion","SamuelPerry","SarahDavison","ScottHillis","SimonCowell","TanEeLyn","TheresePoletti","TimFarrand","ToddNissen","WilliamKazer"]
authors_count = 50

folders = ['C50train', 'C50test']

## creating article level data preprocessing
X = []
Y = []
dataset_path = '/C50'
for folder_idx in range(len(folders)):
    author_path = dataset_path + folders[folder_idx] + '\\'
    for author_idx in range(authors_count):
        author_article_path = author_path + authors[author_idx] + '\\'
        for file in os.listdir(author_article_path):
            article_sentences = []
            with open(author_article_path + '\\' + file) as article:
                content = article.readlines()
            content = [x.strip() for x in content]
            for sent_idx in range(len(content)):
                avg_sent = np.zeros(50)
                tokens = nltk.word_tokenize(content[sent_idx])
                token_count = 0
                for token_idx in range(len(tokens)):
                    #word = re.sub('[^A-Za-z0-9]+', '', words[word_idx]).lower()
                    token = tokens[token_idx].lower()
                    if token in glove_wordmap:
                        token_count = token_count + 1
                        avg_sent = np.add(avg_sent, glove_wordmap[token])
                if token_count!=0:
                    avg_sent = np.divide(avg_sent, token_count)
                    article_sentences.append(avg_sent.tolist())
            article.close()
            X.append(article_sentences)
            Y.append(float(author_idx))

## one hot encoding of y, ie authors
onehotencoder = OneHotEncoder(categorical_features=[0]) # for creating dummy variable columns
y_onehot=onehotencoder.fit_transform(np.reshape(Y, (-1, 1))).toarray()

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y_onehot, test_size = 0.2, random_state = 0, shuffle=True)
y_test = y_test.tolist()
y_train = y_train.tolist()


#L = sorted(zip(x_train,y_train), key=lambda x: len(x[0]))
#x_train, y_train = zip(*L)
#
#L = sorted(zip(x_test,y_test), key=lambda x: len(x[0]))
#x_test, y_test = zip(*L)

x_train_batch = []
y_train_batch = []
x_test_batch = []
y_test_batch = []
batch_size=10
def batch(data, batch, size):
    counter=0
    new_batch = []
    for idx in range(len(data)):
        if counter == size:
            batch.append(new_batch)
            new_batch = []
            counter = 0
        new_batch.append(data[idx])
        counter = counter + 1
    if counter!=0:
        batch.append(new_batch)
    return batch
  
x_train_batch = batch(x_train, x_train_batch, batch_size)
y_train_batch = batch(y_train, y_train_batch, batch_size)
x_test_batch = batch(x_test, x_test_batch, 1)
y_test_batch = batch(y_test, y_test_batch, 1)

#Parameters
learning_rate = 0.1
training_iters = 100
input_columns = 50
n_output = authors_count #number of authors

#number of units in LSTM cell
n_hidden = 100



tf.reset_default_graph()
#tf graph input
x = tf.placeholder(tf.float32, [None,None,input_columns])
y = tf.placeholder(tf.float32, [None,None])

#LSTM output node weights and biases
weights = {
        'out' : tf.Variable(tf.random_normal([n_hidden, n_output]))
        }

biases = {
        'out' : tf.Variable(tf.random_normal([n_output]))
        }


## LSTM blocks definition
def LSTM(x, weights, biases):
    
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=0.3)
    
    outputs, state = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
    outputs_sum = tf.reduce_mean(outputs, 1)
    return tf.matmul(outputs_sum, weights['out']) + biases['out'], outputs, state, outputs_sum

##Calling LSTM to predict the values
pred, outputs, state, outputs_sum = LSTM(x, weights, biases)
print("pred: "+ str(tf.shape(pred)))

#Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))+learning_rate*tf.nn.l2_loss(biases['out']) + learning_rate*tf.nn.l2_loss(weights['out'])
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

#Model Evaluation
correct_pred = tf.equal(tf.arg_max(pred,1), tf.arg_max(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


#Initializing the variables
init = tf.global_variables_initializer()

#path to store graph
log_path = "/graph"
saver = tf.train.Saver()
training_accuracy = 0
#Launch the graph
with tf.Session() as session:
    session.run(init)
    step = 0; acc_total = 0; loss_total = 0; itr=0
    train_acc_trend = []
    
    
    while step<training_iters:

        x_dummy_whole_batch = []
        for batch_idx in range(len(x_train_batch)):
            padded_batch=[]
            padded_batch = add_padding(x_train_batch[batch_idx], padded_batch)
            test = []
            test.extend(padded_batch)
            x_dummy_whole_batch.append(test)

            _,acc,loss, prediction = session.run([optimizer, accuracy, cost, pred], feed_dict= {x: padded_batch, y: y_train_batch[batch_idx]})
            loss_total += loss
            acc_total += acc
            train_acc_trend.append(100*acc_total/(itr+1))
            
            itr = itr + 1
        print("Training average accuracy= " + "{:.6f}".format(100*acc_total/itr)+"; Iteration= " + str(itr) +"; Avg loss= " + "{:.6f}".format(loss_total/(itr)))
        step = step + 1
    training_accuracy = 100*acc_total/(itr+1)
    saver.save(session, log_path+'ArticleLevel')
    ## Pickle Dump to Accuracy Trend at root directory 
    pkl.dump(train_acc_trend, open("Train Accuracy Trend","wb"))

#Testing!
    print("###########################Testing##############################")
    test_acc = 0; tot_test_acc = 0; test_acc_trend = [];test_itr_count=0
    for test_itr in range(len(x_test_batch)):
        padded_batch=[]
        padded_batch = add_padding(x_test_batch[test_itr], padded_batch)
        
        test_acc, test_pred = session.run([accuracy, pred], feed_dict= {x: padded_batch, y: y_test_batch[test_itr]})
        tot_test_acc += test_acc
        test_acc_trend.append(100*tot_test_acc/(test_itr+1))
        
        test_itr = test_itr + 1
        test_itr_count = test_itr
        print("Test average accuracy= " + "{:.2f}".format(100*tot_test_acc/(test_itr_count)))   
    ## Pickle Dump again 
    pkl.dump(test_acc_trend, open("Test Accuracy Trend","wb"))
session.close()
print("Time elapsed= ", elapsed(time.time() - start_time) )