# Proje: Gerçek Zamanlı Mozaik Tespiti ve Dijital Çıktı

Bu proje, bir web kamerası kullanarak dört adet ArUco marker'ını (hepsi aynı ID'ye sahip) tespit etmek için C++ ve OpenCV kullanır. Bu marker'lar, bir mozaik şablonunun köşelerini tanımlar.

Program bu köşeleri bulduğunda, iki ayrı pencere oluşturur:
1.  **Kuşbakışı Görünüm:** Kameranın gördüğü mozaik alanının canlı, düzleştirilmiş (kuşbakışı) görünümü.
2.  **Dijital Çıktı:** Canlı görüntüdeki renkleri algılayıp, bu renkleri orijinal `mosaic.jpg` şablonu üzerine işleyerek stilize bir "dijital kopya" oluşturan görünüm.

## 📦 Kullanılan Teknolojiler
* **Dil:** C++
* **Kütüphane:** OpenCV 4 (ve `opencv_contrib` içindeki `aruco` modülü)
* **Paket Yöneticisi:** vcpkg (Kütüphaneleri derlemek için)
* **Yapı Sistemi:** CMake (Projeyi oluşturmak için)
* **IDE:** Visual Studio 2022

---

## 🛠️ Kurulum Adımları (Tüm Sistem)

Bu adımlar, projeyi sıfırdan bir bilgisayara kurmak için gereken tüm işlemleri içerir.

### Bölüm 1: Ortamın Hazırlanması (Her Bilgisayarda Bir Kez Yapılır)

Önce `vcpkg` ve `OpenCV`'yi kurmamız gerekiyor.

1.  **Ön Koşullar:**
    * **Visual Studio 2022:** "C++ ile Masaüstü Geliştirme" (Desktop development with C++) iş yükü.
    * **VS Bileşeni:** Visual Studio Installer'dan "Tekil Bileşenler" (Individual components) sekmesinden **"Windows için C++ CMake araçları"** (C++ CMake tools for Windows) paketinin de kurulu olduğundan emin olun.
    * **Git:** [git-scm.com](https://git-scm.com/downloads) adresinden indirin (vcpkg'yi indirmek için gereklidir).

2.  **vcpkg'yi İndir:**
    * **PowerShell'i Yönetici olarak** açın.
    * `vcpkg`'yi kurmak için bir konum seçin (Örn: `C:\` kök dizini).
        ```powershell
        cd C:\
        git clone [https://github.com/Microsoft/vcpkg.git](https://github.com/Microsoft/vcpkg.git)
        cd vcpkg
        ```
    * `vcpkg`'yi çalıştırılabilir hale getirin:
        ```powershell
        .\bootstrap-vcpkg.bat
        ```

3.  **OpenCV + ArUco'yu Kur:**
    * Hala Yönetici PowerShell'de (`C:\vcpkg` içinde) olduğunuzdan emin olun.
    * `aruco` modülünü içeren `contrib` paketini kurun. **Bu işlem çok uzun sürebilir (30dk - 1+ saat).**
        ```powershell
        .\vcpkg install opencv4[contrib]:x64-windows
        ```

4.  **vcpkg'yi Entegre Et:**
    * `vcpkg`'nin `CMake` tarafından bulunabilmesi için global entegrasyonu kurun.
        ```powershell
        .\vcpkg integrate install
        ```
    * Ekranda "Applied user-wide integration..." mesajını görmelisiniz.

---

### Bölüm 2: Projenin Oluşturulması (Build)

Ortam hazır olduğuna göre, bu projeyi derleyebiliriz.

1.  **Dosyaları Hazırla:**
    * Tüm proje dosyalarını (`main.cpp`, `MosaicDetector.h`, `MosaicDetector.cpp`, `mosaic.jpg` ve `CMakeLists.txt`) bir klasöre koyun (Örn: `C:\Users\arima\Desktop\MosaicCMake`).

2.  **CMake ile Visual Studio Çözümünü (Solution) Oluştur:**
    * **Yönetici PowerShell**'i açın.
    * Proje klasörünüze gidin:
        ```powershell
        cd C:\Users\arima\Desktop\MosaicCMake
        ```
    * Derleme dosyaları için bir `build` klasörü oluşturun ve içine girin:
        ```powershell
        mkdir build
        cd build
        ```
    * CMake'e, `vcpkg`'yi kullanarak Visual Studio 2022 projesini oluşturmasını söyleyin:
        ```powershell
        cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
        ```
    * (Not: `C:/vcpkg` yolu, vcpkg'yi kurduğunuz yerle aynı olmalıdır).

---

## 🏃‍♂️ Projeyi Çalıştırma

Yukarıdaki komut başarılı olduysa, `build` klasörünüzün içinde `MosaicFillingDetection.sln` adında bir Visual Studio çözümü oluşacaktır.

1.  `build` klasöründeki `MosaicFillingDetection.sln` dosyasını Visual Studio ile açın.

2.  **!! ÇOK ÖNEMLİ (Başlangıç Projesi Ayarı) !!**
    * Proje ilk açıldığında, "Başlangıç Projesi" `ALL_BUILD` olarak ayarlanmıştır.
    * Sağdaki **Solution Explorer** penceresinde, **`MosaicFillingDetectionApp`** projesini (adı kalın değildir) bulun.
    * **`MosaicFillingDetectionApp`**'e sağ tıklayın ve **"Set as Startup Project" (Başlangıç Projesi Olarak Ayarla)** seçeneğine tıklayın.
    * Artık `MosaicFillingDetectionApp`'in adı **kalın (bold)** fontla yazılacaktır.

3.  **!! ÇOK ÖNEMLİ (Runtime Dosyası) !!**
    * Programın çalışması için `mosaic.jpg` resmini bulması gerekir.
    * `mosaic.jpg` dosyasını ana proje klasöründen (`C:\Users\arima\Desktop\MosaicCMake`) **kopyalayın**.
    * Bu kopyayı `build\Debug` klasörünün içine (`MosaicFillingDetectionApp.exe` dosyasının yanına) **yapıştırın**.
    * *(İsteğe bağlı)* Hata almamak için bir kopyasını `build` klasörüne de yapıştırın.

4.  **Çalıştır:**
    * Visual Studio'da platformun "Debug" ve "x64" olarak ayarlandığından emin olun.
    * Yeşil "Oynat" (Local Windows Debugger) tuşuna basarak programı derleyin ve çalıştırın.