from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml') #görüntüdeki yüz ve gözleri algılamak için kullanılan bir metot
classifier =load_model(r'Emotion_little_vgg.h5') #eğitelen modeli alıyoruz

class_labels = ['Kizgin','Mutlu','Normal','Uzgun','Saskin'] #labeller ı aldık

cap = cv2.VideoCapture(0) #kamerayı açtım


while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    labels = []
    #ben=""
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  #gri tona çevirdi gelen görüntüyü
    faces = face_classifier.detectMultiScale(gray,1.3,5) 
#minNeighbors(5)=kaçtane komşu çerçevenin tutması gerektigini belirtiyoruz=sayı arttıkça daha verimli sonuç alınır,ama tespit ettigi nesne saysı azalır
#scaleFactor(1.3) Her görüntü ölçeğinde görüntü boyutunun ne kadar küçültüleceğini belirten parametre.modelle eşleşen bir boyut şansını arttırırız.
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) #x,y=başlangıç noktası kordinat,w=genişlik,h=uzunluk
        #(hangi resme çizilecek,(başlangıç noktası),(sag alt köşesi nerde bitecek(x,y)),(renk),kalınlık)
        roi_gray = gray[y:y+h,x:x+w] #dikdörtgen içindeki görüntünün piksel değerleini alır(siyah-beyaz)
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        #resize = tekrardan boyutlandırma=(orjinal piksel dizisi,yeni matris boyut,
        #piksel sayısı ile oynama ve piksellerin renk tonlarının komşuları ile etkileşip uyumlu hale gelme)
    # rect,face,image = face_detector(frame)


     #keras model tahmini için uygun hale getiriyouz bulunan yüz ü
        if np.sum([roi_gray])!=0: #toplamı 0 a eşit değilse
            roi = roi_gray.astype('float')/255.0 #normalizasyon yapma, değer leri floata çeviriyor   
            roi = img_to_array(roi) #yeniden boyutlandırma 
            roi = np.expand_dims(roi,axis=0) #yeni bir konum ekliyerek(satır,stun) diziyi genişletir.

        # make a prediction on the ROI, then lookup the class
        #ROI hakkında bir tahmin yapın, ardından sınıfı arayın

            preds = classifier.predict(roi)[0] #1. veriyi tahmin ettik
            label=class_labels[preds.argmax()] #tahmin ettigimiz verinin sonucunun ne oldugunu bulduk
                                   
                
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            #(resim,label yazımız,(x,y pozisyon),yazı tipi,yazı büyüklüğü,yazı rengi,yazı kalınlığı)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    cv2.imshow('Emotion Detector',frame) #çerçeve ismi
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


























