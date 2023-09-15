#librariile folosite
import csv, warnings, copy
import numpy as np
from skimage import io
from sklearn import preprocessing
from sklearn.metrics import recall_score, f1_score, precision_score
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import rotate
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, LSTM, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
warnings.filterwarnings('ignore')


class CNN:
    def __init__(self, path):
        def readImages(csvname, files, has_labels=True):
            #citirea datelor din fisiere
            #simultan, normalizez pozele prin impartirea la 255
            with open(path + csvname, "r") as f:
                csvreader = csv.reader(f)
                images = []
                labels = []
                names = []
                for nr, row in enumerate(csvreader):
                    if nr != 0:
                        try:
                            img = Image.open(path + files + row[0])
                            imgcpy = np.asarray(img)
                            img.close()

                            images.append(imgcpy)
                            if has_labels:
                                labels.append(int(row[1]))
                            else:
                                names.append(row[0])
                        except FileNotFoundError or ValueError:
                            pass

                images = np.stack(images, axis=0)
                if has_labels:
                    print(images.shape)
                    labels = np.stack(labels, axis=0)
                    return [copy.deepcopy(images)/255, copy.deepcopy(labels)]
                else:
                    print(images.shape)
                    return [copy.deepcopy(images)/255, copy.deepcopy(names)]

        [self.train_images, self.train_labels] = readImages("train.csv", "train_images/")
        rfliped_images = [rotate(np.fliplr(img), angle=30, reshape=False) for img in self.train_images]
        self.train_images = np.concatenate((self.train_images, rfliped_images), axis = 0)
        self.train_labels = np.concatenate((self.train_labels, self.train_labels), axis = 0)
        print(self.train_images.shape, self.train_labels.shape)
        [self.val_images, self.val_labels] = readImages("val.csv", "val_images/")
        [self.test_images, self.test_names] = readImages("test.csv", "test_images/", False)
        

    def printImg(self, i):
        img = self.train_images[i, :]  # prima imagine
        img = np.reshape(img, (64, 64, 3))
        plt.imshow(img.astype(np.uint8), cmap='gray')
        plt.show()
        print(self.train_labels[0])

    #metoda de acuratete
    def nnAccuracy(self, true_labels, predictions):
        return np.mean(true_labels == predictions)

    def model(self):
        #ne construim un CNN cu layerele descrise in documentatie
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True)
             
        nn_model = Sequential()
        nn_model.add(Conv2D(32, (3, 3), strides=(1, 1), padding="valid", activation="relu", input_shape=(64, 64, 3)))
        nn_model.add(MaxPool2D((2, 2)))
        nn_model.add(Conv2D(64, 3, activation="relu"))
        nn_model.add(MaxPool2D((2, 2)))
        nn_model.add(Dropout(0.5))
        nn_model.add(Conv2D(128, 3, activation="relu"))
        nn_model.add(MaxPool2D((2, 2)))
        nn_model.add(Dropout(0.5))
        nn_model.add(Flatten())
        nn_model.add(Dense(128, activation = "relu"))
        nn_model.add(Dense(128, activation = "relu"))
        nn_model.add(Dropout(0.5))
        nn_model.add(Dense(96, activation = "softmax"))
        nn_model.compile(loss=loss, optimizer="adam", metrics=['accuracy'])
        print(nn_model.summary())
        
        # pentru datele de test
        # aici se antreneaza modelul pe datele de train si validare, pentru a mari numarul de date

        # nn_model.fit(np.concatenate((self.train_images, self.val_images), axis=0), 
        #             np.concatenate((self.train_labels, self.val_labels), axis=0),
        #             epochs=200, batch_size=256,
        #             validation_data=(self.val_images, self.val_labels))
        # predictions = nn_model.predict(self.test_images)
        # predicted_classes = np.argmax(predictions, axis=1)
        # self.predictions_test = predicted_classes
        
        # pentru datele de validare
        # normalizam, antrenam modelul, obtinem predinctiile si calculam metricile de acuratete
        # in history obtinem evolutia modelului pe parcursul epocilor

        history = nn_model.fit(self.train_images, 
                                self.train_labels, 
                                batch_size=256,
                     epochs=200,
                    validation_data=(self.val_images, self.val_labels))
                    
        predictions = nn_model.predict(self.val_images)
        predicted_classes = np.argmax(predictions, axis=1)
        print('Predicted classes:', predicted_classes)
        model_accuracy_svm = self.nnAccuracy(self.val_labels, predicted_classes)
        self.predictions_val = predicted_classes
        self.acc = model_accuracy_svm
        recall = recall_score(self.val_labels, predicted_classes, average='micro')
        f1 = f1_score(self.val_labels, predicted_classes, average='micro')
        precision = precision_score(self.val_labels, predicted_classes, average='micro')
        print("Accuracy ", self.acc)
        print("Recall ", recall)
        print("F1 ", f1)
        print("Precision ", precision)
        print("Neural Network model accuracy: ", model_accuracy_svm * 100)
        
        #afisarea graficelor pentru acuratete si loss
        self.graph(history, 'accuracy')
        self.graph(history, 'loss')
        
    def graph(self, history, name):
        plt.plot(history.history[name])
        plt.plot(history.history['val_'+name])
        plt.title('CNN ' + name)
        plt.ylabel(name)
        plt.xlabel('epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    def confunsion_matrix(self, name):
        #construirea matricei de confuzie goale
        confusionMatrix = np.zeros((96, 96))

        for i in range(len(self.predictions_val)):
            confusionMatrix[self.val_labels[i], self.predictions_val[i]] += 1
        print(confusionMatrix)
        plt.suptitle("Accuracy = " + str(self.acc))
        plt.xticks(range(0, 97, 5))
        plt.yticks(range(0, 97, 5))
        plt.imshow(confusionMatrix)
        plt.colorbar()
        plt.savefig(name + '.png')
        plt.show()

    def printTestCSV(self, path):
        #compunerea solutiei finale intr-un csv
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Image","Class"])
            for i, pred in enumerate(self.predictions_test):
                writer.writerow([self.test_names[i], str(pred)])

    def printSol(self, path, filename):
        self.model()
        #folosit cu datele de validare
        self.confunsion_matrix(filename)
        #pentru crearea solutiei
        #self.printTestCSV("/kaggle/working/mysolution.csv")


if __name__ == '__main__':
    mypath = "/kaggle/input/unibuc-dhc-2023/"
    solution = CNN(mypath)
    solution.printSol(mypath, "nn")
    print("done")