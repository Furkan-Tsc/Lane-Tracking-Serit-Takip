import cv2
import numpy as np
import argparse

webcam = 1

parser = argparse.ArgumentParser()
parser.add_argument("--webcam", action="store_true")
parser.add_argument("--video", type=str, default="video.mp4")
args = parser.parse_args()


INITIAL_POINTS = [
    [0.44, 0.63],
    [0.56, 0.63],
    [0.15, 0.95],
    [0.85, 0.95]
]

# Diğer Ayarlar
NOKTA_RENK = (0, 0, 255)  # Kırmızı (BGR formatında)
CİZGİ_RENK = (0, 255, 0)  # Yeşil
NOKTA_YARICAP = 10

# Değişkenler
points_src = []
selected_point_index = -1

def mouse_callback(event, x, y, flags, param):
    # Fare olaylarını yöneten fonksiyon
    global selected_point_index, points_src, frame_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        # Tıklanan yerin herhangi bir noktaya yakın olup olmadığını kontrol et
        for i, point in enumerate(points_src):
            if np.linalg.norm(np.array([x, y]) - point) < NOKTA_YARICAP * 2:
                selected_point_index = i
                break

    elif event == cv2.EVENT_MOUSEMOVE:
        # Eğer bir nokta seçiliyse ve sürükleniyorsa konumunu güncelle
        if selected_point_index != -1:
            points_src[selected_point_index] = [x, y]

    elif event == cv2.EVENT_LBUTTONUP:
        # Fare tuşu bırakıldığında seçimi sıfırla
        selected_point_index = -1

def draw_interactive_ui(image, points):
    # Noktaları ve aralarındaki çizgileri çizen fonksiyon
    # Orijinal görüntüyü kopyala ki üzerine çizim yapıldığında bozulmasın
    img_with_ui = image.copy()
    
    # Noktalar arasındaki yamuğu çiz
    cv2.polylines(img_with_ui, [np.array(points, dtype=np.int32)], isClosed=True, color=CİZGİ_RENK, thickness=2)
    
    # Her bir noktayı daire olarak çiz
    for point in points:
        cv2.circle(img_with_ui, (int(point[0]), int(point[1])), NOKTA_YARICAP, NOKTA_RENK, -1)
        
    return img_with_ui

def create_perspective_tuner():
    # Ana arayüzü oluşturan ve yöneten fonksiyon
    global points_src

    if args.webcam:
        cap = cv2.VideoCapture(webcam)
    else:
        cap = cv2.VideoCapture(args.video)

        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("Kaynak açılamadı!")
        return

    # Sadece ilk kareyi oku
    ret, frame = cap.read()
    if not ret:
        print("HATA: Videodan kare okunamadı.")
        cap.release()
        return
        
    cap.release()

    h, w = frame.shape[:2]
    
    # Başlangıç noktalarını piksel koordinatlarına çevir
    points_src = np.float32([[w * p[0], h * p[1]] for p in INITIAL_POINTS])

    # Pencere oluştur ve fare olaylarını bağla
    window_name_original = "Orijinal Goruntu - Noktalari Ayarlayin"
    window_name_warped = "Kus Bakisi Onizleme"
    cv2.namedWindow(window_name_original)
    cv2.setMouseCallback(window_name_original, mouse_callback)

    print("\n--- Perspektif Ayar Arayüzü ---")
    print("Noktaları fare ile sürükleyerek ayarlayın.")
    print("İşiniz bittiğinde 's' tuşuna basarak koordinatları alın.")
    print("Çıkmak için 'q' tuşuna basın.")
    
    while True:
        ui_frame = draw_interactive_ui(frame, points_src)

        cv2.putText(ui_frame, "Cikmak icin 'q' | Kaydetmek icin 's'", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Perspektif dönüşümünü hesapla ve uygula
        # Hedef noktalar sağdan/soldan %25 boşluklu bir dikdörtgen
        points_dst = np.float32([
            [w * 0.25, 0],
            [w * 0.75, 0],
            [w * 0.25, h],
            [w * 0.75, h]
        ])
        
        M = cv2.getPerspectiveTransform(points_src, points_dst)
        warped_preview = cv2.warpPerspective(frame, M, (w, h), flags=cv2.INTER_LINEAR)

        cv2.imshow(window_name_original, ui_frame)
        cv2.imshow(window_name_warped, warped_preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print("\n--- Koordinatlar Kaydedildi! ---")
            
            # Kordinatları kullanılacak şekilde çevir
            relative_points = points_src / np.array([w, h], dtype=np.float32)
            
            # Kopyalamaya hazır formatta yazdır
            print("    src = np.float32([")
            print(f"        [{relative_points[0][0]:.2f}*w, {relative_points[0][1]:.2f}*h], # Sol üst")
            print(f"        [{relative_points[1][0]:.2f}*w, {relative_points[1][1]:.2f}*h], # Sağ üst")
            print(f"        [{relative_points[2][0]:.2f}*w, {relative_points[2][1]:.2f}*h], # Sol alt")
            print(f"        [{relative_points[3][0]:.2f}*w, {relative_points[3][1]:.2f}*h]  # Sağ alt")
            print("    ])")

            # Webcam ise o dosyaya entegre et perspektifi değilse diğer dosyaya entegre et
            if args.webcam:
                dosya_yolu = "serit_takip_sanal_webcam.py"
                with open(dosya_yolu, "r", encoding="utf-8") as f:
                    satirlar = f.readlines()

                satirlar[14] = "    src = np.float32([\n"
                satirlar[15] = f"        [{relative_points[0][0]:.2f}*w, {relative_points[0][1]:.2f}*h], # Sol üst\n"
                satirlar[16] = f"        [{relative_points[1][0]:.2f}*w, {relative_points[1][1]:.2f}*h], # Sağ üst\n"
                satirlar[17] = f"        [{relative_points[2][0]:.2f}*w, {relative_points[2][1]:.2f}*h], # Sol alt\n"
                satirlar[18] = f"        [{relative_points[3][0]:.2f}*w, {relative_points[3][1]:.2f}*h]  # Sağ alt\n"
                satirlar[19] = "    ])\n"

                with open(dosya_yolu, "w", encoding="utf-8") as f:
                    f.writelines(satirlar)
                    
            else:
                dosya_yolu = "serit_takip.py"
                with open(dosya_yolu, "r", encoding="utf-8") as f:
                    satirlar = f.readlines()

                satirlar[12] = "    src = np.float32([\n"
                satirlar[13] = f"        [{relative_points[0][0]:.2f}*w, {relative_points[0][1]:.2f}*h], # Sol üst\n"
                satirlar[14] = f"        [{relative_points[1][0]:.2f}*w, {relative_points[1][1]:.2f}*h], # Sağ üst\n"
                satirlar[15] = f"        [{relative_points[2][0]:.2f}*w, {relative_points[2][1]:.2f}*h], # Sol alt\n"
                satirlar[16] = f"        [{relative_points[3][0]:.2f}*w, {relative_points[3][1]:.2f}*h]  # Sağ alt\n"
                satirlar[17] = "    ])\n"

                with open(dosya_yolu, "w", encoding="utf-8") as f:
                    f.writelines(satirlar)
                
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    create_perspective_tuner()
