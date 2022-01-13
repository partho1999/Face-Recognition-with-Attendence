'''########### Making single predictions ###########'''
import numpy as np
from keras.preprocessing import image
import keras
from keras.preprocessing.image import ImageDataGenerator
# import face_recognition_using_CNN

ImagePath='Datasets/Test/partho/47.jpg'
test_image=image.load_img(ImagePath,target_size=(64, 64))
test_image=image.img_to_array(test_image)
 
test_image=np.expand_dims(test_image,axis=0)

classifier = keras.models.load_model("my_model.h5")

result=classifier.predict(test_image,verbose=0)
#print(training_set.class_indices)

TrainingImagePath='Datasets/Train/'

train_datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)

training_set = train_datagen.flow_from_directory(
        TrainingImagePath,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

TrainClasses=training_set.class_indices

ResultMap={}
for faceValue,faceName in zip(TrainClasses.values(),TrainClasses.keys()):
    ResultMap[faceValue]=faceName
 
print('####'*10)
print('Prediction is: ',ResultMap[np.argmax(result)])
