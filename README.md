# **Şerit Takip / Lane Detection**
Kamera görüntülerinden yol üzerindeki şerit çizgilerini tespit edip vurgular.

It detects and highlights lane markings on the road from camera images.

## Açıklama / Description

Bu yazılım OpenCV kullanarak yol şeritlerini tespit eder ayrıca hata durumunda kendini düzeltebilir. Hem webcam üzerinden gerçek zamanlı (OBS sanal kamera gibi) hem de video dosyaları üzerinde çalışır.

This software detects road lanes using OpenCV and can correct itself in case of errors. It works both in real-time via webcam (like OBS virtual camera) and on video files.

## Özellikler / Features

Gerçek zamanlı şerit tespiti / Real-time lane detection

Video dosyaları üzerinde şerit takibi / Lane tracking on video files

OpenCV tabanlı birden fazla algoritma / Multiple OpenCV-based algorithms

Kenar algılama, kötü yollar yüzünden hata yapınca kendini düzeltme / Edge detection, self-correction when making mistakes due to poor roads

## Kurulum için: / Installation:


```bash

pip install opencv-python numpy

```
## Kullanım / Usage
#### Webcam:
Eğer sanal kamera üzerinden kullanmak veya normal kamera ile kullanmak isterseniz öncelikle "*serit_takip_sanal_webcam.py*" ve "*perspective_setup.py*" içindeki "**webcam = 1**"değerini ana kameranız için **0**, sanal kameranız için **1** veya **2** veya **3** yazmalısınız.
Daha sonra "*serit_takip_sanal_webcam.py*" dosyasını açıp perspektifi yola göre ayarladıktan sonra "**s**" tuşuna basmalısınız. Bu çok daha doğru sonuçlar almamızı sağlıyor.

If you want to use it with a virtual camera or a normal camera, you must first change the value "**webcam = 1**" in "*serit_takip_sanal_webcam.py*" and "*perspective_setup.py*" to **0** for your main camera, and **1**, **2**, or **3** for your virtual camera.
Then, open the "*serit_takip_sanal_webcam.py*" file, adjust the perspective according to the path, and press the "**s**" key. This allows us to obtain much more accurate results.

#### Video:
Eğer bir video işleyip bunu kaydetmek istiyorsanız dosyaların bulunduğu dizine istediğiniz dosyayı "**video.mp4**" olarak adlandırdıktan sonra "*serit_takip.py*" dosyasını açıp perspektif ayarladıktan sonra otomatik olarak çalışacaktır. Sonunda "**output.mp4**" adında bir video çıktısı oluşur.

If you want to process a video and save it, rename the desired file to "**video.mp4**" in the directory where the files are located, then open the "*serit_takip.py*" file and set the perspective. It will run automatically. Finally, a video output named "**output.mp4**" will be created.

### Örnekler / Samples:

![perspective](https://github.com/user-attachments/assets/04f8d9f2-1173-4bc3-828b-836922be57f7)



https://github.com/user-attachments/assets/7aac98f3-75af-406c-853f-17e687781d16

https://github.com/user-attachments/assets/a238ad9d-a239-40ac-a424-22dde38063d3

