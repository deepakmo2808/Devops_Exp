from flask import url_for,render_template,redirect,request,Blueprint
from flask_restful import Resource

#Algorithm imports for text_head
import selenium
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.expected_conditions import presence_of_element_located


# Algorithm imports for text
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import pickle

# Algorithm imports for image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.models import load_model
from PIL import Image, ImageChops, ImageEnhance
import os
import itertools
import base64



#Algorithm = Blueprint('Algorithm',__name__)

# Text classification Restful api

class Algorithm():
    def __init__(self,text):
        self.Text=text
        self.Output=None

        Text_data= self.Text.lower()
        Text_data= word_tokenize(Text_data)
        tag_map = defaultdict(lambda : wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV

        
        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(Text_data):
            # Below condition is to check for Stop words and consider only alphabets
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(str(word_Final))
        # The final processed set of words for each iteration will be stored in 'text_final'

        Tfidf_vect = pickle.load(open(os.path.join(os.path.abspath(os.path.dirname(__file__)),'vect_model.pickel'), "rb"))
        Final_words=pd.Series([[str(i) for i in Final_words]])
        Final_words.iloc[0]=str(Final_words.iloc[0])
        Test_X_Tfidf = Tfidf_vect.transform(Final_words)
        filename = os.path.join(os.path.abspath(os.path.dirname(__file__)),'finalized_model.sav')
        loaded_model = pickle.load(open(filename, 'rb'))

        result = loaded_model.predict(np.array(Test_X_Tfidf).tolist()).tolist()
        if result[0]==1:
            result="True"
        else:
            result="False"
        
        self.Output=result

class Text_Algo(Resource):
    def __init__(self):
        self.Text=None
        self.Output=None

    def post(self):
        json_data=request.get_json()
        if not json_data:
            return {'Answer': 'Nothing was provided'}, 400

        Text_data_original=json_data["text"]
        data=Algorithm(Text_data_original)
        ###########################
        # Text_data= Text_data_original.lower()
        # Text_data= word_tokenize(Text_data)
        # tag_map = defaultdict(lambda : wn.NOUN)
        # tag_map['J'] = wn.ADJ
        # tag_map['V'] = wn.VERB
        # tag_map['R'] = wn.ADV

        
        # # Declaring Empty List to store the words that follow the rules for this step
        # Final_words = []
        # # Initializing WordNetLemmatizer()
        # word_Lemmatized = WordNetLemmatizer()
        # # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        # for word, tag in pos_tag(Text_data):
        #     # Below condition is to check for Stop words and consider only alphabets
        #     if word not in stopwords.words('english') and word.isalpha():
        #         word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
        #         Final_words.append(str(word_Final))
        # # The final processed set of words for each iteration will be stored in 'text_final'

        # Tfidf_vect = pickle.load(open(os.path.join(os.path.abspath(os.path.dirname(__file__)),'vect_model.pickel'), "rb"))
        # Final_words=pd.Series([[str(i) for i in Final_words]])
        # Final_words.iloc[0]=str(Final_words.iloc[0])
        # Test_X_Tfidf = Tfidf_vect.transform(Final_words)
        # filename = os.path.join(os.path.abspath(os.path.dirname(__file__)),'finalized_model.sav')
        # loaded_model = pickle.load(open(filename, 'rb'))

        # result = loaded_model.predict(np.array(Test_X_Tfidf).tolist()).tolist()
        # if result[0]==1:
        #     result="True"
        # else:
        #     result="False"
        
        self.Text=Text_data_original
        self.Output=data.Output
        return {'Text':Text_data_original,'Output':data.Output}, 200
    
    def get(self):

        return {'Algorithm': 'Working'}

   
class Image_Algo(Resource):
    def __init__(self):
        self.image_size = (128, 128)

    def convert_to_ela_image(self,path, quality):
        temp_filename = 'temp_file_name.jpg'
       # ela_filename = 'temp_ela.png'
        
        image = Image.open(path).convert('RGB')
        image.save(temp_filename, 'JPEG', quality = quality)
        temp_image = Image.open(temp_filename)
        
        ela_image = ImageChops.difference(image, temp_image)
        
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff
        
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        
        return ela_image

    def ocr_algo(self,image):
        # code for text+image

        return 0


    def prepare_image(self,image_path):
        return np.array(self.convert_to_ela_image(image_path, 90).resize(self.image_size)).flatten() / 255.0

    def get(self):
        return {'Algorithm': 'Working'}

    def post(self):
        json_data=request.get_json()
        if not json_data:
            return {'Answer': 'Nothing was provided'}, 400

        image_code=json_data["image_code"]
        image_64_decode = base64.b64decode(image_code)
        image_result = open('image.jpg', 'wb') # create a writable image and write the decoding result
        image_result.write(image_64_decode)
        class_names = ['fake', 'real']

        model = load_model(os.path.join(os.path.abspath(os.path.dirname(__file__)),'model_casia_run1.h5'))
        real_image_path = os.path.abspath(os.path.dirname(__file__))+"/../../image.jpg"
        image = self.prepare_image(real_image_path)
        image = np.reshape(image,(-1, 128, 128, 3))
        y_pred = model.predict(image)
        y_pred_class = np.argmax(y_pred, axis = 1)[0]
        print(f'Class: {class_names[y_pred_class]} Confidence: {np.amax(y_pred) * 100:0.2f}')

        if (class_names[y_pred_class]== "fake"):
            self.ocr_algo(real_image_path)

        return {'Image':'Check Done','Answer':f'{class_names[y_pred_class]} Confidence {np.amax(y_pred) * 100:0.2f}%'}, 200




class Text_Algo_head(Resource):
    def __init__(self):
        self.Text=None
        self.Output=None

    def post(self):
        json_data=request.get_json()
        if not json_data:
            return {'Answer': 'Nothing was provided'}, 400

        Text_data_original=json_data["text"]
        Text_data_manipulated=Text_data_original.replace(" ","-")
        URL = "https://www.snopes.com/"+Text_data_manipulated+"/"
        r = requests.get(URL) 
        soup = BeautifulSoup(r.content, 'html.parser')

        VALID_TAGS=['p']
        li=[]
        for tag in soup.findAll('p'):
            if tag.name not in VALID_TAGS:
                None
            else:
                li.append(tag.text)
        print(''.join(li[:-3]))
        Text_data_manipulated=''.join(li[:-3])
        notfind='No one said finding the truth would be easy.But don’t give up! The answers could be right here:Read the latest fact checks, original reporting, and curated news from the Snopes editorial team.Check out the most popular fact checks and reporting trending on Snopes.com right now. Send us your questions, your comments, your dubious social media memes.'
        if Text_data_manipulated == notfind:
            driver = webdriver.PhantomJS()

            driver.get("https://snopes.com/")
            driver.set_window_size(1120, 550)
            driver.find_element_by_id('site-search').send_keys(Text_data_original)
            driver.find_element_by_class_name('btn-light').click()
            div=driver.find_elements_by_class_name('ais-hits--item')
            for i in div:
                URL = i.find_element_by_css_selector('a').get_attribute('href')
                r = requests.get(URL) 
                soup = BeautifulSoup(r.content, 'html.parser')

                VALID_TAGS=['p']
                li=[]
                for tag in soup.findAll('p'):
                    if tag.name not in VALID_TAGS:
                        None
                    else:
                        li.append(tag.text)

                print("Here")
                Text_data_manipulated=''.join(li[:-3])
        data=Algorithm(Text_data_manipulated)
        self.Text=Text_data_manipulated
        self.Output=data.Output
        return {'Text':Text_data_manipulated,'Output':data.Output}, 200