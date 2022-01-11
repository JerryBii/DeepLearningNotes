# This file shows how to create a DATASET
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
from tqdm import tqdm
import pickle

DATADIR = "./PetImages"
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 50
# for category in CATEGORIES:
#     path = os.path.join(DATADIR, category)
#     for img in os.listdir(path):
#         # For just cats and dogs we do not need to colour
#         img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
#         print(img_array)
#         IMG_SIZE = 50
#         new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
#         plt.imshow(new_array, cmap="gray")
#         plt.show()
#         break

training_data = []


# NOTE: we want to do is make sure our data is balanced. If you do not balance, the model will initially learn that
# the best thing to do is predict only one class, whichever is the most common. Then, it will often get stuck here.
def create_training_data():

    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in tqdm(os.listdir(path)):
            try:
                # For just cats and dogs we do not need to colour
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass


create_training_data()
print(len(training_data))


random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])

# feature set
X = []

# labels
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

# features can't be passed as lists, convert to numpy array, ALWAYS DO THIS
X = np.array(X).reshape(-1,  # any number of features
                        IMG_SIZE, IMG_SIZE,  # shape of data
                        1  # greyscale, CHANGE TO 3 FOR RGB
                        )

# Save the dataset for later, save to .pickle file
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

# Open dataset
pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

print(X[1])
