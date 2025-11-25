import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import hough_ellipse
from skimage.feature import canny
from matplotlib.patches import Ellipse
import time
import os

# --- 1. Fungsi Pembantu ---

def hitung_bpd_hc_dari_elips(a, b):
    """
    Menghitung BPD (Diameter Minor) dan HC (Lingkar Kepala) dari parameter elips.
    BPD: 2 * radius minor (diameter yang lebih kecil).
    HC: Diestimasi menggunakan rumus klinis sederhana (HC ≈ π * (a + b)).
    """
    # BPD = Diameter Minor (radius minor)
    BPD_pixel = 2 * min(a, b) 
    # HC (Head Circumference) = π * (sumbu mayor + sumbu minor)
    HC_pixel = np.pi * (a + b)
    return BPD_pixel, HC_pixel

# --- 2. Fungsi Utama Proses Citra (Hough Ellipse) ---

def proses_usg_hough_ellipse_final(path_gambar, PIXEL_TO_MM_RATIO=5.0):
    
    start_time = time.time()
    
    if not os.path.exists(path_gambar):
        print(f"❌ ERROR: File tidak ditemukan di {path_gambar}. Periksa path file Anda.")
        return

    gambar_rgb_asli = cv2.imread(path_gambar)
    if gambar_rgb_asli is None:
        print("❌ ERROR: Gagal memuat gambar.")
        return
    
    # --- A. OPTIMASI KECEPATAN: DOWNSIZING AGRESIF ---
    scale_factor = 2.0
    w_new = int(gambar_rgb_asli.shape[1] // scale_factor)
    h_new = int(gambar_rgb_asli.shape[0] // scale_factor)
    
    gambar_rgb = cv2.resize(gambar_rgb_asli, (w_new, h_new), interpolation=cv2.INTER_AREA)
    gambar_gray = cv2.cvtColor(gambar_rgb, cv2.COLOR_BGR2GRAY)
    
    # --- B. PRE-PROCESSING KUAT: SEGMENTASI UNTUK MENGATASI NOISE ---
    
    # 1. Median Blur (Lebih baik untuk noise speckle)
    gambar_blur = cv2.medianBlur(gambar_gray, 5) 
    
    # 2. Adaptive Thresholding (Isolasi kontur tulang tengkorak yang cerah)
    thresh = cv2.adaptiveThreshold(gambar_blur, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # 3. Deteksi Tepi (Canny) pada citra yang sudah disegmentasi
    edges = canny(thresh, sigma=0.5, low_threshold=0, high_threshold=50)
    
    # --- C. HOUGH ELLIPSE DENGAN PARAMETER OPTIMAL ---
    
    h, w = gambar_gray.shape
    r_min = max(int(min(h, w) * 0.15), 30) 
    r_max = int(min(h, w) * 0.45) 

    print(f"⏳ Mencari elips di rentang radius [{r_min}, {r_max}] menggunakan Hough Ellipse...")

    result = hough_ellipse(edges, 
                           threshold=10,    # Ambang batas lebih rendah karena tepi sudah bersih
                           accuracy=60,     # OPTIMASI KECEPATAN EKSTREM
                           min_size=r_min, 
                           max_size=r_max) 

    if not result.size:
        print("⚠️ GAGAL: Elips kepala janin tidak terdeteksi. Waktu komputasi yang lama mungkin terjadi.")
        return

    # --- D. EKSTRAKSI DAN SKALA BALIK ---
    
    result.sort(order='accumulator')
    (acc, yc_scaled, xc_scaled, a_scaled, b_scaled, orientation) = result[-1]
    
    # Skala Balik ke Ukuran Asli
    xc_original = xc_scaled * scale_factor
    yc_original = yc_scaled * scale_factor
    a_original = a_scaled * scale_factor
    b_original = b_scaled * scale_factor
    
    # --- E. PERHITUNGAN BPD, HC, dan GA ---
    
    BPD_pixel, HC_pixel = hitung_bpd_hc_dari_elips(a_original, b_original)
    
    # Konversi ke Satuan Klinis
    BPD_mm = BPD_pixel / PIXEL_TO_MM_RATIO
    HC_cm = (HC_pixel / PIXEL_TO_MM_RATIO) / 10 
    
    # Rumus Estimasi GA
    GA_minggu = (BPD_mm * 0.95) - 4.2 
    
    end_time = time.time()
    time_elapsed = end_time - start_time
    
    # --- F. VISUALISASI HASIL ---
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cv2.cvtColor(gambar_rgb_asli, cv2.COLOR_BGR2RGB))
    
    # 1. Gambar Elips Terdeteksi (mirip referensi Anda)
    elips_patch = Ellipse((xc_original, yc_original), 
                          width=2*a_original, 
                          height=2*b_original, 
                          angle=np.degrees(orientation), 
                          edgecolor='yellow', facecolor='none', lw=2, linestyle='--')
    ax.add_patch(elips_patch)
    
    # 2. Gambar Garis BPD (Diameter Minor)
    r_bpd = min(a_original, b_original)
    # Tentukan sudut BPD (tegak lurus terhadap sumbu mayor)
    angle_bpd_rad = np.radians(orientation) if a_original > b_original else np.radians(orientation) + np.pi/2
    x1 = int(xc_original - r_bpd * np.cos(angle_bpd_rad))
    y1 = int(yc_original - r_bpd * np.sin(angle_bpd_rad))
    x2 = int(xc_original + r_bpd * np.cos(angle_bpd_rad))
    y2 = int(yc_original + r_bpd * np.sin(angle_bpd_rad))
    ax.plot([x1, x2], [y1, y2], 'r--', lw=2, label=f'BPD ({BPD_pixel:.0f} px)')
    
    # 3. Judul Utama
    ax.set_title(f"Deteksi BPD & HC menggunakan Hough Ellipse ({time_elapsed:.2f} detik)", fontsize=14, pad=10)
    
    # 4. Teks Keterangan Hasil (Ditempatkan di bawah plot)
    hasil_teks = (
        f"HASIL PERHITUNGAN:\n"
        f"1. BPD: {BPD_pixel:.0f} px ({BPD_mm:.2f} mm)\n"
        f"2. HC: {HC_pixel:.0f} px ({HC_cm:.2f} cm)\n"
        f"3. Est. GA: {GA_minggu:.1f} minggu"
    )
    
    ax.text(x=0.5, y=-0.15, 
            s=hasil_teks, 
            transform=ax.transAxes, 
            fontsize=12, 
            ha='center', 
            color='black',
            bbox=dict(facecolor='lightgreen', alpha=0.6, boxstyle='round,pad=0.5'))
    
    # 5. Tampilkan Plot
    ax.legend()
    ax.axis('off')
    plt.show()

# --- 3. PENGGUNAAN PROGRAM ---

# ⚠️ GANTI PATH INI dengan lokasi file gambar USG potongan aksial Anda.
path_gambar_usg = "gambarjanin.png" 
PIXEL_TO_MM_RATIO = 5.0 # Asumsi Kalibrasi

try:
    proses_usg_hough_ellipse_final(path_gambar_usg, PIXEL_TO_MM_RATIO)
except Exception as e:
    print(f"❌ Terjadi kesalahan saat menjalankan program: {e}")