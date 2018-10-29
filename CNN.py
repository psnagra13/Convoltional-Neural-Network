from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

## Building CNN
classifier = Sequential()

# Step 1 COnvolution
classifier.add( Convolution2D(32 , 3, 3 , input_shape=(64,64,3), activation='relu' ) )
              # 32: number of filter
              # 3, 3 : size of filer ie. 3x3
# Step 2 Pooling        
classifier.add ( MaxPooling2D( pool_size=(2,2)  ))

# STep 3 Flattening
classifier.add( Flatten( ) )

# Step 4 Full COnnection
classifier.add( Dense( output_dim= 128 , activation='relu') )
classifier.add( Dense( output_dim= 1 , activation='sigmoid') )

# COMPILING THE CNN
classifier.compile( optimizer= 'adam', loss= 'binary_crossentropy', metrics = ['accuracy'])



## FITTING THE CNN to IMAGES

# https://keras.io/preprocessing/image/
# Image Preprocessing - Image Augmentation
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                                                'dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory(
                                        'dataset/test_set',
                                        target_size=(64,64),
                                        batch_size=32,
                                        class_mode='binary')

classifier.fit_generator(
                    training_set,
                    steps_per_epoch=8000,
                    epochs=10,
                    validation_data=test_set,
                    validation_steps=2000)






















 
