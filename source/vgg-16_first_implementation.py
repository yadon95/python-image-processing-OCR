from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array

# On crée un modèle Sequentiel que nous allons enrichir avec les couches de test_VGG16
test_VGG16 = Sequential()

# Les couches s'ajoutenent sois en paramètre d'entrée du constructeur sois avec test_VGG16.add()

# keras.layers contiens les classes correspondant aux couches suivantes :
#   convolution (Conv2D),
#   pooling (MaxPooling2D),
#   fully-connected (Dense),
#   ReLU (Activation)

# Implémentation des couches

# Premier bloc 3x3 conv 64, 3x3 conv 64, pool/2
test_VGG16.add(Conv2D(64, (3, 3), input_shape=(224, 224, 3), padding='same', activation='relu'))  # Convolution + ReLU
test_VGG16.add(Conv2D(64, (3, 3), padding='same', activation='relu'))  # Convolution + ReLU
test_VGG16.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # Pooling

# Second bloc
test_VGG16.add(Conv2D(128, (3, 3), padding='same', activation='relu'))  # Couche 3
test_VGG16.add(Conv2D(128, (3, 3), padding='same', activation='relu'))  # Couche 4
test_VGG16.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Troisième bloc
test_VGG16.add(Conv2D(256, (3, 3), padding='same', activation='relu'))  # Couche 5
test_VGG16.add(Conv2D(256, (3, 3), padding='same', activation='relu'))  # Couche 6
test_VGG16.add(Conv2D(256, (3, 3), padding='same', activation='relu'))  # Couche 7
test_VGG16.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Quatrième bloc
test_VGG16.add(Conv2D(512, (3, 3), padding='same', activation='relu'))  # Couche 8
test_VGG16.add(Conv2D(512, (3, 3), padding='same', activation='relu'))  # Couche 9
test_VGG16.add(Conv2D(512, (3, 3), padding='same', activation='relu'))  # Couche 10
test_VGG16.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Cinquième bloc
test_VGG16.add(Conv2D(512, (3, 3), padding='same', activation='relu'))  # Couche 11
test_VGG16.add(Conv2D(512, (3, 3), padding='same', activation='relu'))  # Couche 12
test_VGG16.add(Conv2D(512, (3, 3), padding='same', activation='relu'))  # Couche 13
test_VGG16.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Implémentation des trois couches fully-connected
test_VGG16.add(Flatten())  # Conversion des matrices 3D en vecteur 2D
test_VGG16.add(Dense(4096, activation='relu'))  # Couche 14 : Fully-connected + Relu
test_VGG16.add(Dense(4096, activation='relu'))
test_VGG16.add(Dense(1000, activation='softmax'))

# Utilisation de VGG16 pré entrainé sur ImageNet
model = VGG16()  # VGG16 implémenté pae keras

# Note : ImageNet peut classifier jusqu'à 1000 catégories différentes

img = load_img("..\\resources\\face.jpg", target_size=(224, 224))  # Chargement image, rescaling pour VGG16 (224,224)px
img = img_to_array(img)  # Convertion de l'image en tableau numpy
img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))  # Crée une collection d'images
img = preprocess_input(img)  # Prétaire l'image comme le veut VGG16

y = model.predict(img)  # Prédir la classe de l'image (parmi les 1000 classes d'ImageNet)
print('Top 3 :', decode_predictions(y, top=3)[0])
