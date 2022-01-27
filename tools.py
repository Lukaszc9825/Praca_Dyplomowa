from PIL import Image
from numpy import asarray
from os import listdir
from os.path import isdir
from sys import path
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
from numpy import load
from numpy import expand_dims
from mtcnn.mtcnn import MTCNN
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
import json

from model import Model


def extract_face(filename, required_size=(160, 160)):
	
	image = Image.open(filename)
	
	image = image.convert('RGB')
	
	pixels = asarray(image)
	
	detector = MTCNN()
	
	results = detector.detect_faces(pixels)
	
	x1, y1, width, height = results[0]['box']
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	
	face = pixels[y1:y2, x1:x2]
	
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

def resize(face, required_size=(160, 160)):
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

def load_faces(directory):
    faces = list()
    for filename in listdir(directory):            
        path = directory + filename       
        face = extract_face(path)
        faces.append(face)
        print(filename)
    return faces


def load_dataset(directory):
    X, y = list(), list()
    for subdir in listdir(directory):
        
        path = directory + subdir + '/'
        print(path)
        if not isdir(path):
            continue
        
        faces = load_faces(path)
        labels = [subdir for _ in range(len(faces))]
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)

def save_dataset(train_dir, test_dir):
    
    trainX, trainy = load_dataset(train_dir)
    print(trainX.shape, trainy.shape)
    
    testX, testy = load_dataset(test_dir)

    quantity = len(listdir(train_dir))
    data = read_config()
    data['number_of_people'] = quantity
    data['filename'] = f'{quantity}-people-dataset.npz'

    override_config(data)

    savez_compressed(f"datasets/{data['filename']}", trainX, trainy, testX, testy)


def load_dataset_compressed():
    conf = read_config()
    data = load(f"datasets/{conf['filename']}")
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    return trainX, trainy, testX, testy


def save_embeddings(Model, trainX, trainy, testX, testy):

    data = read_config()
    data['embeddings'] = f"{data['number_of_people']}-people-embeddings.npz"
    override_config(data)

    newTrainX = list()
    for face_pixels in trainX:
        embedding = Model.get_embedding(face_pixels)
        newTrainX.append(embedding)
    newTrainX = asarray(newTrainX)
    print(newTrainX.shape)

    newTestX = list()
    for face_pixels in testX:
        embedding = Model.get_embedding(face_pixels)
        newTestX.append(embedding)
    newTestX = asarray(newTestX)
    print(newTestX.shape)

    savez_compressed(f"datasets/{data['embeddings']}", newTrainX, trainy, newTestX, testy)

def load_embeddings():
    conf = read_config()
    data = load(f"datasets/{conf['embeddings']}")
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    return trainX, trainy, testX, testy


def normalize_l2(trainX, testX):
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    testX = in_encoder.transform(testX)
    return trainX, testX

def label_encoder(trainy, testy):
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    testy = out_encoder.transform(testy)
    return trainy, testy


def read_config():
    with open("config.json") as json_data_file:
        data = json.load(json_data_file)
    return data

def override_config(data):
    with open("config.json", "w") as outfile:
        json.dump(data, outfile)