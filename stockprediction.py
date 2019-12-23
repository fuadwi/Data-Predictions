#Stock Training data faudwi

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox, QLabel
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot

#Import data bisa pakai data excel
data = pd.read_csv('data_stocks.csv')

# ambil data berdasarkan Date
data = data.drop(['DATE'], 1)

# data dimasukkan ke array
n = data.shape[0]
p = data.shape[1]
data = data.values

# Training dan test data
train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end + 1
test_end = n
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]

# Scale data 
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)

# Buat x and y
X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]

# training data sebanyak data di X_train
n_stocks = X_train.shape[1]

# hidden layer menggunakan 4 layer
#pakai 4 hidden layer dengan masing2 neuron sbb:
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128

# buat sesion untuk train barang
net = tf.compat.v1.InteractiveSession()

# Placeholder
X = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, n_stocks])
Y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None])

# weigh dan bias init
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

# Hidden dan bias dengan weights masing2
W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

# Output weights nya
W_out = tf.Variable(weight_initializer([n_neurons_4, 1]))
bias_out = tf.Variable(bias_initializer([1]))

# Hidden layer
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

# Output layer 
out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

# Cost function 
mse = tf.reduce_mean(tf.math.squared_difference(out, Y))

# Optimizer setiap generation
opt = tf.compat.v1.train.AdamOptimizer().minimize(mse)

# Initialisasi setelah optimizer
net.run(tf.compat.v1.global_variables_initializer())

# Setup plot untuk show hasil tabel
def handle_close(evt):
    print('Start questions')
    class App(QWidget):

        def __init__(self):
            super().__init__()
            self.title = 'SNMN 433 ACZ310'
            self.label ='Order for SNMN 433 ACZ310 has been sent to supplier'
            # buttonReply = QMessageBox.question(self, 'Tools Code : SNMN 433 ACZ310', "Stock untuk 'Tools Code : SNMN 433 ACZ310' Tersedia sekarang tinggal = 20 unit, Proses Beli untuk 300 unit ? (300 unit untuk stok 1 bulan) berdasarkan Prediksi system", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            self.left = 10
            self.top = 10
            self.width = 495
            self.height = 100
            self.initUI()
            
        def initUI(self):
            self.setWindowTitle(self.title)
            self.setGeometry(self.left, self.top, self.width, self.height)

            buttonReply = QMessageBox.question(self, 'Tools Code : SNMN 433 ACZ310', "Stock 'Tools Code : SNMN 433 ACZ310' Tersedia sekarang tinggal = 500 unit, Proses Beli untuk 3000 unit ? (3000 unit untuk stok 1 bulan) berdasarkan Prediksi system", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                print('Yes clicked.')
                self.labl = QLabel(self)
                self.labl.setText('Your Order (Tools Code : SNMN 433 ACZ310) Has been Placed to the Supplier')
                self.labl.move(10, 20)
                button = QPushButton('OK', self)        
                button.move(150,50)
                self.button.setObjectName("btn_submit")
                self.button.clicked.connect(self.btn_submit_handler)

                button2 = QPushButton('Cancel Order', self)        
                button2.move(220,50)
            else:
                print('No clicked.')

            self.show()
        def btn_submit_handler(self):
            if __name__ == '__main__':
                app = QtWidgets.QApplication(sys.argv)
                sys.exit(app.exec_())

    if __name__ == '__main__':
        apps = QtWidgets.QApplication(sys.argv)
        ex = App()
        sys.exit(apps.exec_()) 


plt.ion()
fig = plt.figure(figsize = (16,8))
fig.canvas.mpl_connect('close_event', handle_close)
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test)
line1.set_label('Actual Data life Tools for Tools Code = SNMN 433 ACZ310 ')
line2, = ax1.plot(y_test * 0.5 )
line2.set_label('Prediction running')
ax1.legend()
labelsx = [item.get_text() for item in ax1.get_xticklabels()]
labelsx[1] = '2019-07'
labelsx[2] = '2019-08'
labelsx[3] = '2019-09'
labelsx[4] = '2019-10'
labelsx[5] = '2019-11'
labelsy = [item.get_text() for item in ax1.get_yticklabels()]
labelsy[1] = '200'
labelsy[2] = '300'
labelsy[3] = '400'
labelsy[4] = '500'
labelsy[5] = '600'
ax1.set_xticklabels(labelsx)
ax1.set_yticklabels(labelsy)
plt.show()

# Fit neural net 
batch_size = 256
mse_train = []
mse_test = []

# Run 
epochs = 5
for e in range(epochs):

    # data kita acak random.
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]


    # sesi training tiap epoch
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
      
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})

        # progress
        if np.mod(i, 50) == 0:
            # MSE train dan test
            mse_train.append(net.run(mse, feed_dict={X: X_train, Y: y_train}))
            mse_test.append(net.run(mse, feed_dict={X: X_test, Y: y_test}))
            print('MSE Train: ', mse_train[-1])
            print('MSE Test: ', mse_test[-1])
            # prediksi
            pred = net.run(out, feed_dict={X: X_test})
            line2.set_ydata(pred)
            plt.title('Cutting Tools & Part Control for Tools Code = SNMN 433 ACZ310', fontsize = 14)
            plt.suptitle('AI Generasi ke ' + str(e) + ', sesi ke ' + str(i))
            plt.xlabel('Periode Bulan ', fontsize = 14)
            plt.ylabel('Quantity Material (Pcs)' , fontsize = 14)
            plt.pause(0.01)
    

    # class App(QWidget):

    #     def __init__(self):
    #         super().__init__()
    #         self.title = 'SNMN 433 ACZ310'
    #         self.label ='Order for SNMN 433 ACZ310 has been sent to supplier'
    #         # buttonReply = QMessageBox.question(self, 'Tools Code : SNMN 433 ACZ310', "Stock untuk 'Tools Code : SNMN 433 ACZ310' Tersedia sekarang tinggal = 20 unit, Proses Beli untuk 300 unit ? (300 unit untuk stok 1 bulan) berdasarkan Prediksi system", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    #         self.left = 10
    #         self.top = 10
    #         self.width = 495
    #         self.height = 100
    #         self.initUI()
            
    #     def initUI(self):
    #         self.setWindowTitle(self.title)
    #         self.setGeometry(self.left, self.top, self.width, self.height)

    #         buttonReply = QMessageBox.question(self, 'Tools Code : SNMN 433 ACZ310', "Stock 'Tools Code : SNMN 433 ACZ310' Tersedia sekarang tinggal = 20 unit, Proses Beli untuk 300 unit ? (300 unit untuk stok 1 bulan) berdasarkan Prediksi system", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    #         if buttonReply == QMessageBox.Yes:
    #             print('Yes clicked.')
    #             self.labl = QLabel(self)
    #             self.labl.setText('Your Order (Tools Code : SNMN 433 ACZ310) Has been Placed to the Supplier')
    #             self.labl.move(10, 20)
    #             button = QPushButton('OK', self)        
    #             button.move(150,50)
    #             button2 = QPushButton('Cancel Order', self)        
    #             button2.move(220,50)
    #         else:
    #             print('No clicked.')

    #     self.show()

    # if __name__ == '__main__':
    #     apps = QtWidgets.QApplication(sys.argv)
    #     ex = App()
    #     sys.exit(apps.exec_())  



