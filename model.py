from keras.models import load_model
from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

from numpy import expand_dims

class Model:

    def __init__(self):
        self.model = load_model('keras-facenet/model/facenet_keras.h5')
        self.model2 = SVC(kernel='linear', probability=True)
        self.out_encoder = LabelEncoder()

    def get_model(self):
        model = self.model
        return model

    def get_embedding(self, face_pixels):
        
        face_pixels = face_pixels.astype('float32')
        
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        
        samples = expand_dims(face_pixels, axis=0)
       
        yhat = self.model.predict(samples)
        return yhat[0]

    def fit(self, trainX, trainy):
        self.out_encoder.fit(trainy)
        trainy = self.out_encoder.transform(trainy)
        self.model2.fit(trainX, trainy)

    def predict(self, trainX, testX):
        self.yhat_train = self.model2.predict(trainX)
        self.yhat_test = self.model2.predict(testX)

    def score_prediction(self, trainy, testy):
        self.score_train = accuracy_score(trainy, self.yhat_train)
        self.score_test = accuracy_score(testy, self.yhat_test)
        print('Accuracy: train=%.3f, test=%.3f' % (self.score_train*100, self.score_test*100))

    def predict_face(self,face_embedding):
        samples = expand_dims(face_embedding[0], axis=0)
        yhat_class = self.model2.predict(samples)
        yhat_prob = self.model2.predict_proba(samples)
        class_index = yhat_class[0]
        class_probability = yhat_prob[0,class_index] * 100
        predict_names = self.out_encoder.inverse_transform(yhat_class)
        return predict_names[0], class_probability