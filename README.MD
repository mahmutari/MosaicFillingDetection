# Proje: Gerçek Zamanlı Mozaik Tespiti ve Dijital Çıktı

Bu proje, bir web kamerası kullanarak dört adet ArUco marker'ını (hepsi aynı ID'ye sahip) tespit etmek için C++ ve OpenCV kullanır. Bu marker'lar, bir mozaik şablonunun köşelerini tanımlar.

Program bu köşeleri bulduğunda, iki ayrı pencere oluşturur:
1.  **Kuşbakışı Görünüm:** Kameranın gördüğü mozaik alanının canlı, düzleştirilmiş (kuşbakışı) görünümü.
2.  **Dijital Çıktı:** Canlı görüntüdeki renkleri algılayıp, bu renkleri orijinal `mosaic.jpg` şablonu üzerine işleyerek stilize bir "dijital kopya" oluşturan görünüm.

## 📦 Kullanılan Teknolojiler
* **Dil:** C++
* **Kütüphane:** OpenCV 4 (ve `opencv_contrib` içindeki `aruco` modülü)
* **Paket Yöneticisi:** vcpkg
* **Yapı Sistemi:** CMake
* **IDE:** Visual Studio 2022

## 📁 Proje Yapısı
Proje, standart bir CMake yapısını takip eder:
* `include/`: Tüm `.h` (header) dosyaları burada yer alır.
* `src/`: Tüm `.cpp` (kaynak) dosyaları burada yer alır.
* `build/`: Derleme dosyalarının ve Visual Studio projesinin oluşturulduğu klasördür. (**Git'e dahil edilmez.**)
* `CMakeLists.txt`: Projenin nasıl derleneceğini tanımlayan ana yapı dosyasıdır.
* `mosaic.jpg`: Programın dijital çıktı için kullandığı şablon resimdir.

---

## 🛠️ Kurulum Adımları (Sıfırdan Başlayarak)

Bu adımlar, projeyi sıfırdan bir Windows bilgisayara kurmak için gereken tüm işlemleri içerir.

### Bölüm 1: Ortamın Hazırlanması (Her Bilgisayarda Bir Kez Yapılır)

Önce `vcpkg` ve `OpenCV`'yi kurmamız gerekiyor.

1.  **Ön Koşullar:**
    * **Visual Studio 2022:** "C++ ile Masaüstü Geliştirme" (Desktop development with C++) iş yükü.
    * **VS Bileşeni:** Visual Studio Installer'dan "Tekil Bileşenler" (Individual components) sekmesinden **"Windows için C++ CMake araçları"** (C++ CMake tools for Windows) paketinin de kurulu olduğundan emin olun.
    * **Git:** [git-scm.com](https://git-scm.com/downloads) adresinden indirin (vcpkg ve projeyi indirmek için gereklidir).

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
    * `aruco` modülünü içeren `contrib` paketini kurun. **Bu işlem internet hızınıza ve bilgisayar gücünüze bağlı olarak 30 dakika ile 1 saat arası sürebilir.**
        ```powershell
        .\vcpkg install opencv4[contrib]:x64-windows
        ```
    * İşlem bittiğinde `vcpkg` penceresini kapatabilirsiniz.

---

### Bölüm 2: Projenin Klonlanması ve Derlenmesi

Ortam hazır olduğuna göre, projeyi GitHub'dan indirip derleyebiliriz.

1.  **Projeyi Klonla (İndir):**
    * Komut Satırını (veya PowerShell) açın.
    * Projeyi kurmak istediğiniz klasöre gidin (Örn: `Desktop`).
        ```bash
        git clone [https://github.com/KULLANICI_ADINIZ/REPO_ADINIZ.git](https://github.com/KULLANICI_ADINIZ/REPO_ADINIZ.git)
        cd REPO_ADINIZ
        ```
    * (Not: `KULLANICI_ADINIZ/REPO_ADINIZ` kısmını kendi GitHub adresinizle değiştirin.)

2.  **CMake ile Visual Studio Projesini Oluştur:**
    * Projenin ana klasöründeyken (içinde `CMakeLists.txt` olan yer) bir `build` klasörü oluşturun ve içine girin:
        ```bash
        mkdir build
        cd build
        ```
    * Şimdi CMake'e, `vcpkg`'yi kullanarak Visual Studio 2022 projesini oluşturmasını söyleyin:
        ```bash
        cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
        ```
    * (Not: `C:/vcpkg` yolu, vcpkg'yi kurduğunuz yerle aynı olmalıdır).
    * Komutun başarıyla tamamlandığını (`-- Configuring done`, `-- Generating done`) görmelisiniz.

---

## 🏃‍♂️ Projeyi Çalıştırma

Yukarıdaki komut başarılı olduysa, `build` klasörünüzün içinde `MosaicCMake.sln` adında bir Visual Studio çözümü oluşacaktır.

1.  `build` klasöründeki `MosaicCMake.sln` dosyasını Visual Studio ile açın.

2.  **Başlangıç Projesini Ayarla (ÇOK ÖNEMLİ):**
    * Proje ilk açıldığında, "Başlangıç Projesi" `ALL_BUILD` olarak ayarlanmıştır.
    * Sağdaki **Solution Explorer** penceresinde, **`MosaicCMake`** projesini (adı kalın değildir) bulun.
    * **`MosaicCMake`**'e sağ tıklayın ve **"Set as Startup Project" (Başlangıç Projesi Olarak Ayarla)** seçeneğine tıklayın.
    * Artık `MosaicCMake`'in adı **kalın (bold)** fontla yazılacaktır.

3.  **Çalışma Dizinini Ayarla (ÇOK ÖNEMLİ):**
    * Bu, programın `mosaic.jpg` dosyasını bulabilmesi için gereklidir.
    * **`MosaicCMake`** projesine (yeni kalın olan) tekrar sağ tıklayın ve **"Properties" (Özellikler)** deyin.
    * Açılan pencerede sol menüden **"Configuration Properties" -> "Debugging" (Hata Ayıklama)** sekmesini seçin.
    * Sağdaki **"Working Directory" (Çalışma Dizini)** satırını bulun.
    * Kutusuna `$(ProjectDir)/../..` yazın ve **"Apply" (Uygula)** / **"OK"** deyin.

4.  **Çalıştır:**
    * Visual Studio'da platformun "Debug" ve "x64" olarak ayarlandığından emin olun.
    * Üstteki yeşil "Oynat" (Local Windows Debugger) tuşuna basarak programı derleyin ve çalıştırın.

---

## 💡 Projeye Katkı (Yeni Dosya Ekleme)

Bu proje CMake ile yönetilmektedir. Projeye yeni `.cpp` veya `.h` dosyaları eklemek için Visual Studio'nun "Add -> New Item" menüsünü **kesinlikle kullanmayın**. Bu, projenin yapısını bozacaktır.

Doğru yöntem aşağıdaki adımları izlemektir:

1.  **Dosyaları Manuel Oluşturun:**
    * Yeni `.h` dosyanızı `include/` klasörünün içine manuel olarak oluşturun.
    * Yeni `.cpp` dosyanızı `src/` klasörünün içine manuel olarak oluşturun.

2.  **`CMakeLists.txt` Dosyasını Güncelleyin:**
    * Projenin ana dizinindeki `CMakeLists.txt` dosyasını açın.
    * Oluşturduğunuz `.h` dosyasının yolunu `set(HEADERS ...)` bloğundaki listeye ekleyin.
    * Oluşturduğunuz `.cpp` dosyasının yolunu `set(SOURCES ...)` bloğundaki listeye ekleyin.

    **Örnek:** `Utils.h` ve `Utils.cpp` adında yeni dosyalar ekliyorsanız:

    ```cmake
    set(HEADERS
        include/ColorDetector.h
        # ... diğer .h dosyaları ...
        include/Utils.h  # <- YENİ EKLENEN SATIR
    )

    set(SOURCES
        src/main.cpp
        # ... diğer .cpp dosyaları ...
        src/Utils.cpp  # <- YENİ EKLENEN SATIR
    )
    ```

3.  **Projeyi Yeniden Oluşturun (Re-generate):**
    * `CMakeLists.txt` dosyasını kaydettiğinizde, Visual Studio genellikle üst tarafta "CMake projesini yeniden oluştur" (Generate) diyen sarı bir çubuk gösterir. Ona tıklayın.
    * Eğer bu çubuk çıkmazsa, projeyi kapatın, `build` klasörünün içindeki **`CMakeCache.txt`** dosyasını silin ve projeyi yeniden açın.

Yeni dosyalarınız otomatik olarak Visual Studio'daki "Header Files" ve "Source Files" filtrelerinin altına eklenecektir.

## ⚠️ Muhtemel Sorunlar ve Çözümleri

* **SORUN:** `cmake ..` komutu `OpenCV`'yi bulamıyor.
    * **ÇÖZÜM:** `vcpkg`'nin `opencv4[contrib]:x64-windows` paketini başarıyla kurduğunuzdan emin olun. `-DCMAKE_TOOLCHAIN_FILE=` kısmındaki `vcpkg` yolunuzun doğru olup olmadığını kontrol edin.

* **SORUN:** Program çalışıyor ama `[mosaic.jpg] not found` hatası veriyor.
    * **ÇÖZÜM:** "Projeyi Çalıştırma" bölümündeki 3. adımı (Çalışma Dizinini Ayarlama) atlamışsınız demektir. Orayı kontrol edin.

* **SORUN:** Visual Studio'da `.h` dosyaları görünmüyor veya `#include` hataları alıyorum (proje derlenmesine rağmen).
    * **ÇÖZÜM:** Bu, CMake önbellek (cache) sorunudur.
        1.  Visual Studio'yu kapatın.
        2.  `build` klasörünün içindeki **HER ŞEYİ SİLİN**.
        3.  "Bölüm 2: Projenin Klonlanması ve Derlenmesi" adımındaki `cmake ..` komutunu yeniden çalıştırın.
        4.  Yeni `.sln` dosyasını açın.