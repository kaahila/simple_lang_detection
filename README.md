# simple_lang_detection
An simple lang detection Notebook with Multinomial Bias and TfidfVectorizer in sklearn

Currently best reached score is: *0.9935111111111111*

Currently supportet Langs are: *de, en, es, fr, hi, it, nl, pl, pt, sw*

To use it u can download the models
````python
import os
import pickle
import numpy as np

def get_max_proba(model, value):
    return np.max(model.predict_proba(value)[0])

def detect_language(text, probability=False):
    """
    This function takes a string as input and returns a string
    depending on the language of the string.
    """
    # Load the model from disk
    model_folder = './models/'
    tfidf_file_name = 'tfidf_lang_detect.sav'
    model_file_name = 'model_mnb_lang_detect.sav'
    tfidf_file_path = model_folder + tfidf_file_name
    model_file_path = model_folder + model_file_name

    loaded_model = pickle.load(open(model_file_path, 'rb'))
    loaded_tfidf = pickle.load(open(tfidf_file_path, 'rb'))

    # Make prediction using loaded model
    text_transformed = loaded_tfidf.transform([text])
    prediction = loaded_model.predict(text_transformed)

    # Get probability
    if probability:
        probability = get_max_proba(loaded_model, text_transformed)

    # Return prediction
    return prediction[0], probability
````

### An confusion matrix for this model:
![confusion_matrix](https://user-images.githubusercontent.com/73160695/221941110-afac9c29-9ce4-46b8-8d1c-5990c15ecc77.png)

#### For mor detail take an look at the jupyter notebook in this repo

It was just an little fun thing and not realy testet. Don´t blame me if something don´t work.
