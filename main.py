from model import Model
from os import listdir
from tools import *
import cv2
from PIL import Image

train_dir = 'data/train/'
test_dir = 'data/test/'

model = Model()
new = False

in_encoder = Normalizer(norm='l2')

data = read_config()

if not data['filename'] in listdir('datasets'):
    save_dataset(train_dir, test_dir)

elif len(listdir(train_dir)) != data['number_of_people']:
    print('New people in dataset')
    save_dataset(train_dir, test_dir)
    new = True

else:
    print(f"{data['filename']} already exists")


trainX, trainy, testX, testy = load_dataset_compressed()


if not data['embeddings'] in listdir('datasets'):
    save_embeddings(model, trainX, trainy, testX, testy)

elif not new:
    save_embeddings(model, trainX, trainy, testX, testy)

else:
    print(f"{data['embeddings']} already exists")

trainX, trainy, testX, testy = load_embeddings()

trainX, testX = normalize_l2(trainX, testX)

model.fit(trainX, trainy)

face_cascade =  cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

# capture = cv2.VideoCapture('http://192.168.1.45:8080/video')
# capture = cv2.VideoCapture(0)
capture = cv2.VideoCapture('http://192.168.1.56:4747/video')

color = (0, 255, 0)
stroke = 2

while True:

    ret, frame = capture.read()
    bboxes = face_cascade.detectMultiScale(frame)

    for box in bboxes:

        x,y,w,h = box
        face = frame[y:y+h, x:x+w]
        face_embedding = model.get_embedding(resize(face))
        face_embedding = in_encoder.transform(expand_dims(face_embedding, axis=0))
        predict_names, class_probability = model.predict_face(face_embedding)
        print('Predicted: %s (%.3f)' % (predict_names, class_probability))
        cv2.rectangle(frame, (x, y), (x+w+20, y+h+20), color, stroke)
        if class_probability > 65:
            cv2.putText(frame, f"{predict_names} {str(round(class_probability,2))}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)
        else:
            cv2.putText(frame, 'Unnknow person', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)
    cv2.imshow('Praca dyplomowa', frame)
    
    if cv2.waitKey(20) & 0xFF == ord('q'):

        capture.release()
        cv2.destroyAllWindows()