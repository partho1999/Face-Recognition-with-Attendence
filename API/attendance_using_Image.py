import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import time
import csv
import pandas as pd



def make_attendance():
    path = 'ImagsAttendence/'
    images = []
    classNames = []
    myList = os.listdir(path)
    #print(myList)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
        #print(classNames)
 
    def findEncodings(images):
        encodeList = []


        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList

    def markAttendance(name):
    
        df = pd.read_csv("Attendance.csv")
        time_now = datetime.now()
        tStr = time_now.strftime('%H:%M:%S')
        dStr = time_now.strftime('%Y-%m-%d')

        if len(df[(df['Name']==name) & (df['Date']==dStr)])==0:
            print('Length 0')
            df = df.append({'Name': name,'Date':dStr,'InTime':tStr,'Last_In/Out':tStr}, ignore_index=True)
        else:
            row = df[(df['Name']==name) & (df['Date']==dStr)].index.values[0]
            df.loc[row, 'Last_In/Out']=tStr
    
        df['Date']=pd.to_datetime(df['Date'])
        df.sort_values(by=['Date'], inplace=True, ascending=False)
        df.to_csv('Attendance.csv',index=False)
    
        
    encodeListKnown = findEncodings(images)
    print('Encoding Complete')

    while True:

        img = cv2.imread('VideoImage/image.jpg', cv2.COLOR_BGR2RGB)
        imgS = cv2.resize(img,(0,0),None,0.25,0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
        if len(facesCurFrame)<1:
            print('No face detected!')
            return (f'No face detected!')
    
        for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
            print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name)  
                y1,x2,y2,x1 = faceLoc
                y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                #write_in_csv(name)
                markAttendance(name)
                return (f'Attendance of {name} recorded!')
            else:
                name = 'Unknown'
                print(name)
                y1,x2,y2,x1 = faceLoc
                y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                return ('Unrecognized! Please, try again..')
        break