import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import \
    tqdm  # a nice pretty percentage bar for tasks.
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, render_template, url_for, request
import sqlite3
import cv2
import shutil
from twilio.rest import Client
account_sid="AC1dde1"
auth_token="8a8a9e551"
client=Client(account_sid, auth_token)

import numpy as np


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('userlog.html')

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
        try:
            dirPath = "static/images"
            fileList = os.listdir(dirPath)
            for fileName in fileList:
                os.remove(dirPath + "/" + fileName)
            fileName=request.form['filename']
            dst = "static/images"
            

            shutil.copy("test/"+fileName, dst)
            
            verify_dir = 'static/images'
            IMG_SIZE = 50
            LR = 1e-3
            MODEL_NAME = 'Cottonleaf-{}-{}.model'.format(LR, '2conv-basic')
        ##    MODEL_NAME='keras_model.h5'
            def process_verify_data():
                verifying_data = []
                for img in os.listdir(verify_dir):
                    path = os.path.join(verify_dir, img)
                    img_num = img.split('.')[0]
                    img = cv2.imread(path, cv2.IMREAD_COLOR)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    verifying_data.append([np.array(img), img_num])
                    np.save('verify_data.npy', verifying_data)
                return verifying_data

            verify_data = process_verify_data()
            #verify_data = np.load('verify_data.npy')

            
            tf.compat.v1.reset_default_graph()
            #tf.reset_default_graph()

            convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

            convnet = conv_2d(convnet, 32, 3, activation='relu')
            convnet = max_pool_2d(convnet, 3)

            convnet = conv_2d(convnet, 64, 3, activation='relu')
            convnet = max_pool_2d(convnet, 3)

            convnet = conv_2d(convnet, 128, 3, activation='relu')
            convnet = max_pool_2d(convnet, 3)

            convnet = conv_2d(convnet, 32, 3, activation='relu')
            convnet = max_pool_2d(convnet, 3)

            convnet = conv_2d(convnet, 64, 3, activation='relu')
            convnet = max_pool_2d(convnet, 3)

            convnet = fully_connected(convnet, 1024, activation='relu')
            convnet = dropout(convnet, 0.8)

            convnet = fully_connected(convnet, 7, activation='softmax')
            convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

            model = tflearn.DNN(convnet, tensorboard_dir='log')

            if os.path.exists('{}.meta'.format(MODEL_NAME)):
                model.load(MODEL_NAME)
                print('model loaded!')


            fig = plt.figure()
            diseasename=" "
            rem=" "
            rem1=" "
            str_label=" "
            accuracy=""
            for num, data in enumerate(verify_data):

                img_num = data[1]
                img_data = data[0]

                y = fig.add_subplot(3, 4, num + 1)
                orig = img_data
                data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
                # model_out = model.predict([data])[0]
                model_out = model.predict([data])[0]
                print(model_out)
                print('model {}'.format(np.argmax(model_out)))

                if np.argmax(model_out) == 0:
                    str_label = 'Aphids_cotton_leaf'
                    
                elif np.argmax(model_out) == 1:
                    str_label = 'Army_worm_cotton_leaf'
                    
                elif np.argmax(model_out) == 2:
                    str_label = 'Bacterial_Blight'
                    
                elif np.argmax(model_out) == 3:
                    str_label = 'Healthy_leaf'
                    
                elif np.argmax(model_out) == 4:
                    str_label = 'Powdery_Mildew'

                elif np.argmax(model_out) == 5:
                    str_label = 'Target_spot'

                elif np.argmax(model_out) == 6:
                    str_label = 'Fussarium_wilt'
                
                if str_label != "Healthy_leaf":
                        #from Serialcode import Send
                        Send()
                         
                if str_label == 'Aphids_cotton_leaf':
                    diseasename = "Aphids_cotton_leaf "
                    print("The predicted image of the Aphids_cotton_leaf_Disease is with a accuracy of {} %".format(model_out[0]*100))
                    accuracy="The predicted image of the Aphids_cotton_leaf_Disease is with a accuracy of {}%".format(model_out[0]*100)
                    rem = "The remedies for Aphids_cotton_leaf are:\n\n "
                    rem1 = [" Discard or destroy any affected plants",  
                    "Do not compost them.", 
                    "Rotate your cotton plants yearly to prevent re-infection next year.", 
                    "Use copper Fungicides"]
                    client.api.account.messages.create(
                                            to="+91-768",
                                            from_="+134",
                                            body="Aphids_cotton_leaf_Disease")
                    
                    
                elif str_label == 'Army_worm_cotton_leaf':
                    diseasename = "Army_worm_cotton_leaf"
                    print("The predicted image of the Army_worm_cotton_leaf_Disease is with a accuracy of {} %".format(model_out[1]*100))
                    accuracy="The predicted image of the Army_worm_cotton_leaf_Disease is with a accuracy of {}%".format(model_out[1]*100)
                    rem = "The remedies for Army_worm_cotton_leaf_Disease are: "
                    rem1 = [" Monitor the field, handpick diseased plants and bury them.",
                    "Use sticky yellow plastic traps.", 
                    "Spray insecticides such as Emamectin benzoate 5 SG @ 0.4 g/l,Spinosad 45 SC @ 0.3 ml/l,Chlorantraniliprole 18.5 SC @ 0.4 ml/I.organophosphates", 
                    "carbametes during the seedliing stage."]
                    client.api.account.messages.create(
                                            to="+91-748",
                                            from_="+1344",
                                            body="Army_worm_cotton_leaf_Disease")
                    
                    
                elif str_label == 'Bacterial_Blight':
                    diseasename = "Bacterial_Blight"
                    print("The predicted image of the Bacterial_Blight is with a accuracy of {} %".format(model_out[2]*100))
                    accuracy="The predicted image of the Bacterial_Blight is with a accuracy of {}%".format(model_out[2]*100)
                    rem = "The remedies for Bacterial_Blight are: "
                    rem1 = [" Monitor the field, handpick diseased plants and bury them.",
                    "Use sticky yellow plastic traps.", 
                    "Spray insecticides such as organophosphates",
                    "carbametes during the seedliing stage.",
                    "Seed treatment with authorized antibiotics and seed dressing with copper oxychloride"]
                    client.api.account.messages.create(
                                            to="+91-748",
                                            from_="+13",
                                            body="Bacterial_Blight")
                    
                elif str_label == 'Healthy_leaf':
                    status= 'Healthy_leaf'
                    print("The predicted image of the Healthy_leaf is with a accuracy of {} %".format(model_out[3]*100))
                    accuracy="The predicted image of the Healthy_leafis with a accuracy of {}%".format(model_out[3]*100)
                    client.api.account.messages.create(
                                            to="+91-74",
                                            from_="+1364",
                                            body="Healthy_leaf")
                    
                    
                elif str_label == 'Powdery_Mildew':
                    diseasename = "Powdery_Mildew"
                    print("The predicted image of the Powdery_Mildew is with a accuracy of {} %".format(model_out[4]*100))
                    accuracy="The predicted image of the Powdery_Mildew is with a accuracy of {}%".format(model_out[4]*100)
                    rem = "The remedies for Powdery_Mildew are: "
                    rem1 = [" Bonide Sulfur Plant Fungicide and BONIDE Copper Fungicide Dust can be sprayed",
                    "Treat organically with copper spray."]
                    client.api.account.messages.create(
                                            to="+91-78",
                                            from_="+1",
                                            body="Powdery_Mildew")
                    

                elif str_label == 'Target_spot':
                    diseasename = "Target_spot"
                    print("The predicted image of the Target_spot is with a accuracy of {} %".format(model_out[5]*100))
                    accuracy="The predicted image of the Target_spot is with a accuracy of {}%".format(model_out[5]*100)
                    rem = "The remedies for Target_spot are: "
                    rem1 = [" Monitor the field, remove and destroy infected leaves.",
                    "Treat organically with copper spray.",
                    "Use chemical fungicides,the best of which for cotton is chlorothalonil."]
                    client.api.account.messages.create(
                                            to="+91-",
                                            from_="+1364",
                                            body="Target_spot")
                    
                elif str_label == 'Fussarium_wilt':
                    diseasename = "Fussarium_wilt"
                    print("The predicted image of the Fussarium_wilt is with a accuracy of {} %".format(model_out[5]*100))
                    accuracy="The predicted image of the Fussarium_wilt is with a accuracy of {}%".format(model_out[5]*100)
                    rem = "The remedies for Fussarium_wilt are: "
                    rem1 = [" Mycostop is a biological fungicide that will safely protect crops against wilt caused by Fusarium."]
                    client.api.account.messages.create(
                                            to="+91-748",
                                            from_="+13",
                                            body="Fussarium_wilt")

            return render_template('userlog.html', status=str_label,accuracy=accuracy, disease=diseasename, remedie=rem, remedie1=rem1, ImageDisplay="http://127.0.0.1:5000/static/images/"+fileName)
        except Exception as error:
            print('{}'.format(error))
            return render_template('userlog.html', msg="This is not the image of leaf")
    return render_template('index.html')

@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
