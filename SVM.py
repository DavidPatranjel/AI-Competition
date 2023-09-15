#librariile folosite
import csv
import numpy as np
from skimage import io
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import recall_score, f1_score, precision_score
import matplotlib.pyplot as plt
from PIL import Image
import copy
import warnings
warnings.filterwarnings('ignore')


class SVM:
    def __init__(self, path):
        def readImages(csvname, files, has_labels=True):
            #citirea datelor din fisiere
            with open(path + csvname, "r") as f:
                csvreader = csv.reader(f)
                images = []
                labels = []
                names = []
                for nr, row in enumerate(csvreader):
                    if nr != 0:
                        try:
                            img = Image.open(path + files + row[0])
                            #aducem pozele de la shape 64x64x3 la 12288
                            imgcpy = np.asarray(img).flatten()
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
                    labels = np.stack(labels, axis=0)
                    print(images.shape, labels.shape)
                    return [copy.deepcopy(images), copy.deepcopy(labels)]
                else:
                    print(images.shape)
                    return [copy.deepcopy(images), copy.deepcopy(names)]

        [self.train_images, self.train_labels] = readImages("train.csv", "train_images/")
        [self.val_images, self.val_labels] = readImages("val.csv", "val_images/")
        [self.test_images, self.test_names] = readImages("test.csv", "test_images/", False)

    def printImg(self, i):
        img = self.train_images[i, :]  # prima imagine
        img = np.reshape(img, (64, 64, 3))
        plt.imshow(img.astype(np.uint8), cmap='gray')
        plt.show()
        print(self.train_labels[0])

    #metoda scrisa la laborator pentru a normaliza datele
    def normalize_data(self, train_data, test_data, type=None):
        scaler = None
        if type == 'standard':
            scaler = preprocessing.StandardScaler()
        elif type == 'l1':
            scaler = preprocessing.Normalizer(norm='l1')
        elif type == 'l2':
            scaler = preprocessing.Normalizer(norm='l2')

        if scaler is None:
            return [train_data, test_data]

        scaler.fit(train_data)
        train_data_norm = scaler.transform(train_data)
        test_data_norm = scaler.transform(test_data)
        return [train_data_norm, test_data_norm]


    #metoda de acuratete
    def svmAccuracy(self, true_labels, predictions):
        return np.mean(true_labels == predictions)


    def model(self, c, k):
        #SVM C=100 GAMMA=1 KERNEL=POLY NORM=L1
        if k is None:
            svm_model = svm.SVC(C=c, gamma = 1)
        else:
            svm_model = svm.SVC(C=c, kernel=k, gamma = 1)

        #pentru datele de test
        #aici se antreneaza modelul pe datele de train si validare, pentru a mari numarul de date
        
        #[self.train_images, self.test_images] = self.normalize_data(np.concatenate((self.train_images, self.val_images), axis=0), self.test_images, type="l2")
        #svm_model.fit(self.train_images, np.concatenate((self.train_labels, self.val_labels), axis=0))
        #self.predictions_test = svm_model.predict(self.test_images)

        #pentru datele de validare
        #normalizam, antrenam modelul, obtinem predinctiile si calculam metricile de acuratete
        [self.train_images, self.val_images] = self.normalize_data(self.train_images, self.val_images, type="l2")
        svm_model.fit(self.train_images, self.train_labels)
        predictions = svm_model.predict(self.val_images)
        model_accuracy_svm = self.svmAccuracy(self.val_labels, predictions)
        self.predictions_val = predictions
        self.acc = model_accuracy_svm
        recall = recall_score(self.val_labels, predictions, average='micro')
        f1 = f1_score(self.val_labels, predictions, average='micro')
        precision = precision_score(self.val_labels, predictions, average='micro')
        print("Accuracy ", self.acc)
        print("Recall ", recall)
        print("F1 ", f1)
        print("Precision ", precision)
        print("SVM model accuracy: ", model_accuracy_svm * 100)

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

    def printSol(self, path, c, kernel= None):
        #folosit cu datele de validare
        self.model(c, kernel)
        if kernel is None:
            self.confunsion_matrix("newsvm"+str(c)+"nokern")
        else:
            self.confunsion_matrix("newsvm"+str(c)+"kern"+kernel)
    
        #pentru crearea solutiei
        #self.printTestCSV("/kaggle/working/mysolution.csv")

if __name__ == '__main__':
    mypath = "/kaggle/input/unibuc-dhc-2023/"
    solution = SVM(mypath)
    solution.printSol(mypath, 100, "poly")
    print("done")
    