# -*- coding: utf-8 -*-
"""
BMI,Height,Weight,Age,Gender_Prediction .ipynb

@author: Ameyaveer Singh
"""

import pandas as pd
data_folder = "C:/Users/Ameyaveer Singh/OneDrive/Desktop/Age, Gender, BMI Prediction model/indian face images"
from glob import glob
all_files = glob(data_folder+"/*")


all_images = sorted([img for img in all_files if img.endswith((".png",".jpg", ".jpeg", ".JPG"))])

print("Total {} images ".format(len(all_images)))

from pathlib import Path

def get_index_of_digit(string):
    import re
    match = re.search("\d", Path(string).stem)
    return match.start(0)

id_path = [(Path(image).stem[:(get_index_of_digit(Path(image).stem))], image) for image in all_images]

label_file = "C:/Users/Ameyaveer Singh/OneDrive/Desktop/Age, Gender, BMI Prediction model/BMI, age, gender data.csv"
image_df = pd.DataFrame(id_path, columns=['UID', 'path'])
profile_df = pd.read_csv(label_file)
profile_df

data_df = image_df.merge(profile_df)
data_df

import seaborn as sns
sns.distplot(data_df['Age'])

sns.displot(data_df['Gender'])
# 1 = male
# 2 = Female

!pip install face_recognition

import face_recognition
import numpy as np
def get_face_encoding(image_path):
    print(image_path)
    picture_of_me = face_recognition.load_image_file(image_path)
    my_face_encoding = face_recognition.face_encodings(picture_of_me)
    if not my_face_encoding:
        print("no face found !!!")
        return np.zeros(128).tolist()
    return my_face_encoding[0].tolist()
all_faces = []
for images in data_df.path:
    face_enc = get_face_encoding(images)
    all_faces.append(face_enc)

X = np.array(all_faces)

y_age = data_df.Age.values ## all labels
y_gender = data_df.Gender.values
y_height = data_df.height.values ## all labels
y_weight = data_df.weight.values
y_BMI = data_df.BMI.values

from sklearn.model_selection import train_test_split
X_train, X_test, y_age_train, y_age_test, y_gender_train, y_gender_test, y_height_train, y_height_test, y_weight_train, y_weight_test ,y_BMI_train, y_BMI_test = train_test_split(X, y_age,y_gender,y_height,y_weight,y_BMI, random_state=1)

def report_goodness(model,X_test,y_test,predictor_log=True):
    # Make predictions using the testing set
    y_pred = model.predict(X_test)
    y_true = y_test
    if predictor_log:
        y_true = np.log(y_test)
    # The coefficients
    # The mean squared error
    print("Mean squared error: %.2f"      % mean_squared_error(y_true, y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_true, y_pred))

    errors = abs(y_pred - y_true)
    mape = 100 * np.mean(errors / y_true)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

model_age = KernelRidge(kernel='rbf', gamma=0.21,alpha=0.0017)

model_age = model_age.fit(X_train,np.log(y_age_train))

report_goodness(model_age,X_test,y_age_test)

model_gender = KernelRidge(kernel='rbf', gamma=0.21,alpha=0.0017)

model_gender = model_gender.fit(X_train,np.log(y_gender_train))

report_goodness(model_gender,X_test,y_gender_test)

model_height = KernelRidge(kernel='rbf', gamma=0.21,alpha=0.0017)

model_height = model_height.fit(X_train,np.log(y_height_train))

report_goodness(model_height,X_test,y_height_test)

model_weight = KernelRidge(kernel='rbf', gamma=0.21,alpha=0.0017)

model_weight = model_weight.fit(X_train,np.log(y_weight_train))

report_goodness(model_weight,X_test,y_weight_test)

model_BMI = KernelRidge(kernel='rbf', gamma=0.21,alpha=0.0017)
model_BMI = model_BMI.fit(X_train,np.log(y_BMI_train))

report_goodness(model_BMI,X_test,y_BMI_test)

!pip install joblib
import joblib
age_model = 'age_predictor'
gender_model = 'gender_predictor'
joblib.dump(model_age, age_model)
joblib.dump(model_gender, gender_model)
height_model = 'weight_predictor.model'
weight_model = 'height_predictor.model'
bmi_model = 'bmi_predictor.model'
joblib.dump(model_height, height_model)
joblib.dump(model_weight, weight_model)
joblib.dump(model_BMI, bmi_model)

age_model = joblib.load(age_model)
gender_model = joblib.load(gender_model)
height_model = joblib.load(height_model)
weight_model = joblib.load(weight_model)
bmi_model = joblib.load(bmi_model)



def predict_age_gender(test_image, age_model, gender_model, height_model, weight_model, bmi_model):
    test_array = np.expand_dims(np.array(get_face_encoding(test_image)), axis=0)
    age = float(np.exp(age_model.predict(test_array))[0])
    gender = float(np.exp(gender_model.predict(test_array))[0])
    height = float(np.exp(height_model.predict(test_array))[0])
    weight = float(np.exp(weight_model.predict(test_array))[0])
    bmi = float(np.exp(bmi_model.predict(test_array))[0])
    return {'Age': age, 'Gender': gender, 'height': height, 'weight': weight, 'bmi': bmi}

from IPython.display import Image

test_image = '/content/WhatsApp Image 2023-05-25 at 7.01.00 PM.jpeg'
Image(test_image)

predict_age_gender(test_image,age_model,gender_model,height_model,weight_model,bmi_model)