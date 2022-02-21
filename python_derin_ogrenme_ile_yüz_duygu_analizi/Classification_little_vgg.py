from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
import os

num_classes = 5
img_rows,img_cols = 48,48
batch_size = 32

train_data_dir = 'train'
validation_data_dir = 'validation'

train_datagen = ImageDataGenerator( # veriler üzerinde oynama yapmak için = ezberlemeyi engellemek için
					rescale=1./255,   #yeniden ölçeklendirmek
					rotation_range=30,  #dönüş aralığı
					shear_range=0.3,    #kesme aralığı
					zoom_range=0.3,     #yakınlaştırma_aralığı
					width_shift_range=0.4,  #genişlik kaydırma aralığı
					height_shift_range=0.4,  #yükseklik kaydırma aralığı
					horizontal_flip=True,    #yatay çevirme
					fill_mode='nearest')   #doldurma modu


validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(  #dizinden akış , verileri alma ve nasıl alacağımızı belirleme
					train_data_dir,  # datasetinin adresini verir
					color_mode='grayscale',  #verilerin rengini belirtme ,gri tonlama
					target_size=(img_rows,img_cols),  # resimleri boyutlandırmak için
					batch_size=batch_size,     # tek seferde sisteme virilecek data sayısı
					class_mode='categorical', # görüntüleri sınıflandırmak için = 2 den fazla sınıf varsa
					shuffle=True)  #Verilen görüntünün sırasını karıştırmak istiyorsanız True, yoksa False olarak ayarlayın.

validation_generator = validation_datagen.flow_from_directory(
							validation_data_dir,
							color_mode='grayscale',
							target_size=(img_rows,img_cols),
							batch_size=batch_size,
							class_mode='categorical',
							shuffle=True)


model = Sequential() #sinir ağı modelini oluştur , sıralı katman dizlimi

# Block-1

model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))#
# 32 = filtre sayısı,(3,3) = filtrelerin boyutu, padding='same' = resmin etrafına çerçeve oluşturur 0 değerleridnen oluşan
# kernel_initializer = w,b nin ilk değerlerinin belirnmesi için metot., input_shape =giriş katmanı oldugunu belirtiyor(renk belirtiyor)
model.add(Activation('relu')) #elu = - değerleri 0 a yaklaştırır,pozitif değerleri oldugu gibi çıkartır
model.add(BatchNormalization()) #ortalama çıktıyı 0'a yakın ve çıktı standart sapmasını 1'e yakın tutan bir dönüşüm uygular.

model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2))) #resmi küçültme ve önemli noktalara odaklanam (2-2 dolaşır)(1 kayar=default)
model.add(Dropout(0.2)) # %20 yi devre dışı bırak = sürekli aynı nöronlarn kullanımı engelleme = ezberlemeyi engelleme

# Block-2 

model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-3

model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-4 

model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-5  #Full Connected

model.add(Flatten())  #flaten yaptıgı yerde same almamış ,burdan sonra filtre almıyor =48 * 48 = 2304,1
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block-6

model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block-7

model.add(Dense(num_classes,kernel_initializer='he_normal')) #nöron sayısayı = özellik sayısı(son katman da)
model.add(Activation('softmax')) #2 den fazla veri oldugundan kullanılan aktivasyon

print(model.summary()) #Modelin özetlenmiş bir gösterimini döndürür. 

from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
 #   eğitilmiş sinir ağını kayıtlı tutan memot =(metotun özellikleri değişiyor) ,#.hp = çok boyutlu veri dizilerini tutar   
checkpoint = ModelCheckpoint('denemes.h5',  # eğitimi kaydedeğim yer , dosya ismi ,dosya türü
                             monitor='val_loss',#kayıp ölçümünü görüntülemek için((konsolda val_loss değerini her adımın altında yazr)(farklı özelliklerde alınabiliyor) 
                             mode='min',#Val_acc için max olmalı,val_loss için minimum olmalı,
                             save_best_only=True, #((mode = bunu çalıştırmak için onay) (mode = min oldugu için los değerine bakar)
#son çalıştırmanın sonucu kaydedilen sonuçtan daha iyi deyilse ,kaydedilen veri değişmez ama daha iyiyse kayıt yeni eğitimdir. 
                             verbose=1) #konsolda eğitim ilerlemesinin sonuçarının görüntüsü =(tiremi(0,1,2)),2 se epoch sadece gibi))

# epoch sayısından önce sistem artık kendini geliştirmeyi bitirdiyse sistem artık eğitimi durdursun
earlystop = EarlyStopping(monitor='val_loss',  #kayıp ölçümünü görüntülemek için(konsolda val_loss değerini her adımın altında yazar) 
                          min_delta=0, #son çalıştırmanın sonucu kaydedilen sonuçtan daha iyi deyilse alma (durdurma)
                          patience=3,  # durma işlemi için geçmesi gereken minimum epoch sayısı
                          verbose=1,    # ayrıntı modu = konsol için
                          restore_best_weights=True  # eğitim esnasında erken durdurulma yaşandıysa en iyi epoc sonuçlarını almak için =true,en son adımı almak için false
                          )

reduce_lr = ReduceLROnPlateau(monitor='val_loss', # hızlı yükselme veya düşmeyi engellemk için ögren mi oranı katsayısını düşürürüz.
                              factor=0.2,  #ögrenme oranının azaltılacıgı oranı
                              patience=3, #iyleşme olmadan geçmesini bekliyeceğimiz adım sayısı =3 adım iyleşme olmassa oranla oyna
                              verbose=1,  #konsolda = 0: sessiz, 1: mesajları güncelle.
                              min_delta=0.0001)  #inilebilecek minimum ögrenme oranı değeri

callbacks = [earlystop,checkpoint,reduce_lr] # eğitim konturol metotlarını bir diziye topadık

#modeli eğitim için yapılandırı (eğitir)
model.compile(loss='categorical_crossentropy', #çok sınıflı sınıflandırma işlemlerinde kullanılan bir kayıp fonksinudur
              optimizer = Adam(lr=0.001), # modelde güncellemenin(iyleştirme) nasıl yapılağı.
              metrics=['accuracy'])  # Eğitim ve test sırasında model tarafından değerlendirilecek ölçümlerin listesi. 

nb_train_samples = 24176  #eğitim örnek sayısı
nb_validation_samples = 3006  #valiation örnek sayısı
epochs=25

history=model.fit_generator( #eğitim için kullanabileceğimiz kütüphane
                train_generator,  #verileri alma için oluşturdugumuz metot
                steps_per_epoch=nb_train_samples//batch_size, #her epochdaki veri sayısı
                epochs=epochs,  #epock sayısı
                callbacks=callbacks,  #yukarda oluşturdugumuz fonksiyon
                validation_data=validation_generator, #yukarıda belirledik = dogrulama verileri 
                validation_steps=nb_validation_samples//batch_size) #(dogrulama adımları)

#batc size = her epohda verilerin parçalanarak alınmasını sağlıyor.






















































