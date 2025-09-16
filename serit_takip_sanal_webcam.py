import time
import cv2
import numpy as np
from collections import deque
import warnings
import subprocess

webcam = 1

subprocess.run(["python", "perspective_setup.py", "--webcam"])

# --- Perspektif ---
def get_perspective_transform(img_size):
    w, h = img_size
    src = np.float32([
        [0.44*w, 0.68*h], # Sol üst
        [0.57*w, 0.68*h], # Sağ üst
        [0.24*w, 0.98*h], # Sol alt
        [0.79*w, 0.96*h]  # Sağ alt
    ])
    dst = np.float32([
        [w * 0.25, 0],
        [w * 0.75, 0],
        [w * 0.25, h],
        [w * 0.75, h]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv

def warp(img, M):
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

# --- Binary Eşikleme ---
def abs_sobel_thresh(img, orient='x', thresh=(20, 100)):
    # Griye çevirme ve Sobel filtresi uygulama
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_64F, int(orient=='x'), int(orient=='y'))
    abs_sobel = np.absolute(sobel)
    scaled = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary = np.zeros_like(scaled)
    binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    return binary

def color_threshold(img, sthresh=(100,255), rthresh=(200,255)):
    # hls ve kırmızı kanal üzerinden renk eşikleme
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s = hls[:,:,2]
    s_binary = np.zeros_like(s)
    s_binary[(s>=sthresh[0]) & (s<=sthresh[1])] = 1

    r = img[:,:,0]
    r_binary = np.zeros_like(r)
    r_binary[(r>=rthresh[0]) & (r<=rthresh[1])] = 1

    combined = np.zeros_like(s)
    combined[(s_binary==1)|(r_binary==1)] = 1
    return combined

def combined_binary(img):
    gradx = abs_sobel_thresh(img, 'x', (20,100))
    grady = abs_sobel_thresh(img, 'y', (20,100))
    color_bin = color_threshold(img)
    combined = np.zeros_like(color_bin)
    combined[((gradx==1)&(grady==1)) | (color_bin==1)] = 1
    return combined

# --- Fit Geçerlilik Kontrolü ---
def is_fit_valid(left_fit, right_fit, ploty,
                 min_dist=200, max_dist=1000, max_std=200):
    left_x = np.polyval(left_fit, ploty)
    right_x = np.polyval(right_fit, ploty)
    width = np.abs(right_x - left_x)
    # Ortalama şerit genişliği ve sapma kontrolü
    return (min_dist < np.mean(width) < max_dist) and (np.std(width) < max_std)

def get_last_valid_fit(fits_deque):
    #  Yol algılamada sorun olduğunda en son geçerli fit'i döndürür
    for fit in reversed(fits_deque):
        if fit is not None:
            return fit
    return None

# --- Sliding Window + Search-Around-Poly ---
def find_lane_pixels(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    midpoint = histogram.shape[0]//2
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    margin = 60
    minpix = 50
    window_height = int(binary_warped.shape[0]//nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
    left_current, right_current = left_base, right_base
    left_inds, right_inds = [], []

    for window in range(nwindows):
        y_low = binary_warped.shape[0] - (window+1)*window_height
        y_high = binary_warped.shape[0] - window*window_height
        x_l_low = left_current - margin
        x_l_high = left_current + margin
        x_r_low = right_current - margin
        x_r_high = right_current + margin

        good_left = ((nonzeroy>=y_low)&(nonzeroy<y_high)&
                     (nonzerox>=x_l_low)&(nonzerox<x_l_high)).nonzero()[0]
        good_right= ((nonzeroy>=y_low)&(nonzeroy<y_high)&
                     (nonzerox>=x_r_low)&(nonzerox<x_r_high)).nonzero()[0]

        left_inds.append(good_left)
        right_inds.append(good_right)

        if len(good_left)>minpix:
            left_current = int(np.mean(nonzerox[good_left]))
        if len(good_right)>minpix:
            right_current = int(np.mean(nonzerox[good_right]))

    left_inds = np.concatenate(left_inds)
    right_inds= np.concatenate(right_inds)
    leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]
    rightx, righty= nonzerox[right_inds], nonzeroy[right_inds]

    return leftx, lefty, rightx, righty

def search_around_poly(binary_warped, left_fit, right_fit, margin=60):
    # Önceki fit etrafında arama yapma
    nonzero = binary_warped.nonzero()
    nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])

    left_inds = ((nonzerox > (np.polyval(left_fit, nonzeroy)-margin)) &
                 (nonzerox < (np.polyval(left_fit, nonzeroy)+margin)))
    right_inds= ((nonzerox > (np.polyval(right_fit, nonzeroy)-margin)) &
                 (nonzerox < (np.polyval(right_fit, nonzeroy)+margin)))

    leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]
    rightx, righty= nonzerox[right_inds], nonzeroy[right_inds]
    return leftx, lefty, rightx, righty

def fit_polynomial_from_points(leftx, lefty, rightx, righty):
    left_fit, right_fit = None, None
    if len(leftx)>0 and len(np.unique(lefty))>1:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=Warning)
            left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx)>0 and len(np.unique(righty))>1:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=Warning)
            right_fit= np.polyfit(righty, rightx, 2)
    return left_fit, right_fit

# --- Şerit ve Bölge Çizimi ---

def draw_lane(img, binary_warped, left_fit, right_fit, Minv):
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    color_warp = np.zeros_like(img).astype(np.uint8)

    # Eğer bir tarafta fit eksikse diğerinin kopyasını alır// Şimdilik güvenli değil
    if left_fit is None and right_fit is not None:
        left_fit = right_fit.copy(); left_fit[2] -= 600
    elif right_fit is None and left_fit is not None:
        right_fit= left_fit.copy(); right_fit[2] += 600

    if left_fit is not None and right_fit is not None:
        if not is_fit_valid(left_fit, right_fit, ploty):
            # Geçersizse direk geri dön
            return img
        # Poligon
        left_pts = np.array([np.vstack([np.polyval(left_fit, ploty), ploty]).T])
        right_pts= np.array([np.flipud(np.vstack([np.polyval(right_fit, ploty), ploty]).T)])
        pts = np.hstack((left_pts, right_pts))
        cv2.fillPoly(color_warp, np.int32([pts]), (0,255,0)) #Yeşil bölgeyi çizer/doldurur
        # Kırmızı ve mavi kenar çizgileri
        for fit, color in zip([left_fit, right_fit], [(255,0,0),(0,0,255)]):
            line_pts = np.array([np.vstack([np.polyval(fit, ploty), ploty]).T]).astype(np.int32)
            cv2.polylines(color_warp, line_pts, False, color, 10)

    newwarp = warp(color_warp, Minv)
    return cv2.addWeighted(img, 1, newwarp, 0.3, 0) #Görüntüye eklenirkenki şeffaflığı

# --- Ana Şerit Tespit Sınıfı ---

QUEUE_LENGTH = 10 # Stabil fit için kuyruk
MAX_BAD_FITS  = 5

class LaneDetector:
    def __init__(self):
        self.left_fits = deque(maxlen=QUEUE_LENGTH)
        self.right_fits= deque(maxlen=QUEUE_LENGTH)
        self.bad_count = 0 # Art arda başarısız fit belirli sayıya ulaşınca geçmiş temizlenir
        self.M, self.Minv = None, None
        self.prev_time = time.time()

    def process(self, image):
        # FPS hesaplama
        now = time.time()
        fps = 1/(now-self.prev_time) if now!=self.prev_time else 0.0
        self.prev_time = now

        und = image
        if self.M is None:
            self.M, self.Minv = get_perspective_transform((und.shape[1], und.shape[0]))

        binary = combined_binary(und)
        warped = warp(binary, self.M)
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])

        # Eğer geçmişte son bulunan fit geçerli ise hızlıca search_around_poly kullanılır. Değilse daha yavaş olan find_lane_pixels sliding window kullanılır.
        
        if self.left_fits and self.right_fits and is_fit_valid(self.left_fits[-1], self.right_fits[-1], ploty):
            lx, ly, rx, ry = search_around_poly(warped, self.left_fits[-1], self.right_fits[-1])
        else:
            lx, ly, rx, ry = find_lane_pixels(warped)

        lf, rf = fit_polynomial_from_points(lx, ly, rx, ry)

        if lf is not None and rf is not None and is_fit_valid(lf, rf, ploty):
            self.left_fits.append(lf); self.right_fits.append(rf)
            lf = np.mean(self.left_fits, axis=0)
            rf = np.mean(self.right_fits, axis=0)
            self.bad_count = 0
        else:
            # yeniden dene
            print("UYARI: Fit geçersiz, yeniden tahmin...")
            lx, ly, rx, ry = find_lane_pixels(warped)
            lf, rf = fit_polynomial_from_points(lx, ly, rx, ry)
            # Yeni fitler geçerliyse sıraya eklenir sonra geçmişin ortalaması alınarak stabil hale getirilir
            # Geçersiz ise önce yeniden find_lane_pixels denenir yine başarısızsa get_last_valid_fit ile son geçerli fit kullanılır bad_count artırılır ve bad_count MAX_BAD_FITS'e ulaşırsa geçmiş temizlenir
            if lf is not None and rf is not None and is_fit_valid(lf, rf, ploty):
                self.left_fits.append(lf); self.right_fits.append(rf)
                lf = np.mean(self.left_fits, axis=0)
                rf = np.mean(self.right_fits, axis=0)
                self.bad_count = 0
            else:
                print("UYARI: Hala geçersiz, eski fit kullanılıyor.")
                lf = get_last_valid_fit(self.left_fits)
                rf = get_last_valid_fit(self.right_fits)
                self.bad_count += 1

        if self.bad_count >= MAX_BAD_FITS:
            print("UYARI: Çok fazla hatalı fit, temizleniyor.")
            self.left_fits.clear(); self.right_fits.clear()
            self.bad_count = 0

        result = draw_lane(und, warped, lf, rf, self.Minv)
        cv2.putText(result, f"FPS: {fps:.1f}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        return result

# --- Ana Döngü ---
def process_webcam():
    detector = LaneDetector()
    cap = cv2.VideoCapture(webcam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Kamera açılamadı!")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Görüntü alınamadı!")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lane_frame = detector.process(rgb)
            display = cv2.cvtColor(lane_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Serit Takip", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    process_webcam()
