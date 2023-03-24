'''
DETECCION DE CONDUCTORES CANSADOS
JOSE LUIS HERNANDEZ CAMACHO
UNIVERSIDAD DE GUANAJUATO CAMPUS IRAPUATO-SALAMANCA
DICIS

'''
'''
INSTALACION DE LIBRERIAS
para exportar la lista de requerimientos 
pip freeze > requirements.txt

'''

import cv2 #openCV 2
import os # sistema operativo
from keras.models import load_model # keras para modelo de IA
import numpy as np # numpy para operaciones matematicas
from pygame import mixer # pygame para reproducir sonidos
import time


mixer.init() # inicializamos el reproductor
sound = mixer.Sound('alarm.wav') # selecciona el audio de alarma

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml') # clasificador para cara frontal
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml') # clasificador para ojo izqueirdo
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml') #clasificador para ojo derecho



lbl=['Close','Open'] # lista opcion abierto o cerrado

model = load_model('models/cnncat2.h5') # carga el modelo de IA entrenado para deteccion
path = os.getcwd()  #retorna el directorio de trabajo actual
cap = cv2.VideoCapture(0) # inicia la captura de video por la camara del ordenador
font = cv2.FONT_HERSHEY_COMPLEX_SMALL # tipo de letra fuente
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]

while(True):
    ret, frame = cap.read() # lee los datos capturados por la camara
    height,width = frame.shape[:2]  # toma las dimensiones de la captura

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # pasa a gris la imagen capturada por la camara
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25)) # deteccion de cara
    left_eye = leye.detectMultiScale(gray) # deteccion de ojo izquierdo
    right_eye =  reye.detectMultiScale(gray) #deteccion de ojo derecho

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED ) #rectangulo con opencv

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 ) #por cada rostro detectado dibuja un rectangulo
    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w] #toma el ojo derecho de la imagen original
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY) #pasa el ojo derecho a gris
        r_eye = cv2.resize(r_eye,(24,24)) # redimenziona el ojo derecho
        r_eye= r_eye/255 # normaliza la imagen para que sus valores oscilen entre 0 y 1 (crea una mascara)
        r_eye=  r_eye.reshape(24,24,-1) # redimenziona la imagen
        r_eye = np.expand_dims(r_eye,axis=0) # expande la dimension del array

        #rpred = model.predict_classes(r_eye)
        rpred = model.predict(r_eye) # TOMA PREDICCION CON EL MODELO PARA EL OJO DERECHO
        rpred = np.argmax(rpred, axis=1) #TOMA EL VALOR MAXIMO EJE 1

        if(rpred[0]==1): # LOGICA PARA SABER SI EL OJO ESTA ABIERTO O CERRADO
            lbl='Open' 
        if(rpred[0]==0):
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)

        #lpred = model.predict_classes(l_eye)
        lpred = model.predict(l_eye)
        lpred = np.argmax(lpred,axis=0)

        if(lpred[0]==1):
            lbl='Open'   
        if(lpred[0]==0):
            lbl='Closed'
        break

    if(rpred[0]==0 and lpred[0]==0): # SI AMBOS OJOS ESTAN CERRADOS SUMA UNO A LA SCORE
        score=score+1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else: # SI NO ESTAN ABIERTOS Y RESTA UNO A LA SCORE
        score=score-1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
        
    if(score<0): # COLOCAR SCORE EN PANTALLA
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>15): # SI LA SCORE SUPERA 15 SIGNIFICA QUE SE ESTA DURMIENDO, SE ACTIVA LA ALARMA
        cv2.imwrite(os.path.join(path,'image.jpg'),frame) #GUARDA IMAGEN DE LA PERSONA DURMIENDOSE
        try: # REPRODUCIR SONIDO EN TRY POR SI FALLA
            sound.play()
            
        except:  # isplaying = False
            pass
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
    cv2.imshow('frame',frame) # MUESTRA LA IMAGEN EN PANTALLA
    if cv2.waitKey(1) & 0xFF == ord('q'): # DETENER LA DETECCION CON LA TECLA Q
        break
cap.release()
cv2.destroyAllWindows() # FIN DEL PROGRAMA
