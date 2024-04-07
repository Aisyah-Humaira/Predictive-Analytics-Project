# Laporan Proyek Machine Learning - Aisyah Humaira

## Domain Proyek
Diabetes merupakan salah satu penyakit kronis yang paling umum dikenal di seluruh dunia dan dapat menimbulkan risiko kesehatan yang serius jika tidak dikelola dengan tepat. Menurut data yang diperoleh sekitar tahun 2019, prevalensi diabetes global diperkirakan mencapai 9,3%, atau sekitar 463 juta orang di seluruh dunia. Proyeksi untuk masa depan menunjukkan tren peningkatan, dengan estimasi mencapai 10,2% atau sekitar 578 juta orang pada tahun 2030, dan meningkat lagi menjadi 10,9% atau sekitar 700 juta orang pada tahun 2045. Selain itu, didapatkan bahwa separuh dari individu yang hidup dengan diabetes, tepatnya sekitar 50,1%, tidak menyadari bahwa mereka sebenarnya menderita penyakit ini. Hal ini menunjukkan betapa pentingnya upaya deteksi dini dan kesadaran akan gejala diabetes.

Untuk mengukur kadar gula darah dan memantau kondisi diabetes, glucometer darah adalah alat yang umum digunakan untuk mengukur glukosa dalam darah dalam periode waktu tertentu. Namun, untuk pemantauan yang lebih kontinu dan akurat, digunakan alat pemantauan glukosa kontinu. Alat ini dirancang untuk menjadi minim invasif, sehingga pasien dapat dengan mudah memantau dan mengukur kadar glukosa mereka tanpa harus melakukan pengambilan sampel darah secara berulang-ulang.

Dampak diabetes yang tidak terkontrol dapat sangat merusak bagi kesehatan tubuh secara keseluruhan. Banyak organ tubuh, termasuk ginjal, mata, jantung, dan sistem saraf, dapat mengalami kerusakan serius akibat komplikasi diabetes yang tidak terdeteksi atau dikelola dengan baik. Oleh karena itu, sangat penting untuk melakukan diagnosis diabetes dengan cepat dan akurat guna mencegah komplikasi serius dan memastikan kualitas hidup yang lebih baik bagi penderita diabetes.

Referensi: [Egyptian Informatics Journal](https://www.sciencedirect.com/science/article/pii/S1110866524000045) 

## Business Understanding

### Problem Statements
- Bagaimana cara mendapatkan model machine learning untuk memprediksi penyakit diabetes?
- Model development apa yang memberikan hasil prediksi paling akurat untuk memprediksi penyakit diabetes?

### Goals
- Mendapatkan model machine learning yang dapat digunakan untuk memprediksi penyakit diabetes.
- Mengetahui model development yang memberikan hasil prediksi paling akurat untuk memprediksi penyakit diabetes.

## Data Understanding
Dataset yang digunakan dalam proyek ini diperoleh dari situs Kaggle mengenai [Simple Feature To Detect Diabetes](https://www.kaggle.com/datasets/simaanjali/diabetes-simple-diagnosis). Dataset ini terdiri dari 88380 baris dan 9 kolom. 

Berdasarkan informasi dari Kaggle, variabel-variabel pada dataset adalah sebagai berikut:
- *Age*: Mewakili usia pasien dalam tahun. Usia dapat menjadi faktor risiko untuk diabetes, karena risiko diabetes meningkat seiring bertambahnya usia.
- *Gender*: Menunjukkan jenis kelamin pasien, yang dapat menjadi faktor dalam prediksi diabetes. Beberapa studi menyarankan bahwa wanita mungkin memiliki risiko yang berbeda dibandingkan pria dalam mengembangkan diabetes.
- *Body Mass Index (BMI)*: BMI adalah ukuran yang menggunakan tinggi dan berat badan seseorang untuk menentukan apakah mereka berada dalam kategori berat badan normal, kelebihan berat badan, atau obesitas. BMI yang tinggi dikaitkan dengan risiko diabetes yang lebih tinggi.
- *High Blood Pressure (High_BP)*: Indikator apakah seorang pasien menderita hipertensi. Tekanan darah tinggi adalah faktor risiko yang signifikan untuk diabetes tipe 2.
- *Fasting Blood Glucose (FBS)*: Mewakili tingkat glukosa dalam darah setelah puasa semalaman. Tingkat gula darah puasa yang tinggi dapat menunjukkan risiko diabetes atau prediabetes.
- *HbA1c (HbA1c_level)*: Pengukuran rata-rata tingkat gula darah selama 2-3 bulan terakhir. Ini adalah indikator penting untuk diagnosis dan pengelolaan diabetes.
- *Smoking*: Menunjukkan apakah pasien merokok atau tidak. Merokok dapat menjadi faktor risiko tambahan untuk diabetes tipe 2.
- *Diagonisis*: Indikator bahwa seseorang memiliki diabetes.

Selain dari deskripsi variable, didapatkan pula informasi mengenai dataset sebagai berikut
- Terdapat 1 kolom non-numerik dengan tipe object yaitu *Gender*. Kolom ini merupakan categorical features.
- Terdapat 1 kolom numerik dengan tipe data float64 yaitu *HbA1c_level*. Kolom ini merupakan numerical features.
- Terdapat 7 kolom numerik dengan tipe data int64, yaitu: *Unnamed: 0, Age, BMI, High_BP, FBS, Smoking, dan Diagnosis*. Kolom ini merupakan numerical features.

Pada tahap selanjutnya, dilakukan Exploratory Data Analysis (EDA) untuk memahami dan menganalisis karakteristik dari data yang digunakan. EDA bertujuan untuk menemukan pola, mengidentifikasi anomali, serta memeriksa asumsi-asumsi yang ada pada dataset. Terdapat dua metode yang digunakan yaitu metode bersifat univariate yang melibatkan satu variate atau variabel dan multivariate yang melibatkan dua atau lebih variabel.  
1. *Univariate Analysis* melibatkan satu variate atau variabel. Pada proses analisis terhadap fitur kategori didapatkan ![image](https://github.com/Aisyah-Humaira/Dicoding-Proyek-Akhir-Machine-Learning/assets/83213518/3933c4fd-8105-4e04-93d8-869c877f1989)
Selanjutnya, untuk fitur numerik didapatkan ![image](https://github.com/Aisyah-Humaira/Dicoding-Proyek-Akhir-Machine-Learning/assets/83213518/991f9797-6757-4479-8a0e-d06b73870c37)

3. *Multivariate Analysis* melibatkan dua atau lebih variabel. Pada proses ini didapatkan ![image](https://github.com/Aisyah-Humaira/Dicoding-Proyek-Akhir-Machine-Learning/assets/83213518/00add511-e329-48e1-95a8-a5ca7898d393)

## Data Preparation
Pada tahap ini, dilakukan data preparation atau persiapan data yang bertujuan untuk melakukan transformasi pada dataset. Transformasi ini dilakukan agar data memiliki format atau bentuk yang sesuai dan cocok untuk proses pemodelan dalam machine learning. Beberapa langkah yang umum dilakukan dalam data preparation antara lain:
- Encoding Fitur Kategori yang dilakukan dengan menggunakan Teknik * one-hot-encoding*. Teknik ini digunakan untuk mengubah variabel kategorikal menjadi bentuk biner (0 atau 1) sehingga dapat diolah oleh algoritma machine learning. Pengaplikasian Teknik ini dilakukan terhadap fitur Gender
- Pembagian dataset dengan fungsi train_test_split agar dataset menjadi data latih (train) dan data uji (test). Teknik ini digunakan untuk menguji kinerja model pada data yang belum pernah dilihat sebelumnya, sehingga dapat memeriksa apakah model tersebut overfitting atau generalisasi dengan baik pada data baru.
- Standardisasi dapat membantu untuk membuat variabel memiliki skala yang serupa, sehingga algoritma machine learning yang berbasis jarak atau optimasi dapat bekerja dengan lebih efisien dan akurat. Proses transformasi data ini mengubah nilai rata-rata (mean) menjadi 0 dan nilai standar deviasi menjadi 1.


## Modeling
Model development dilakukan sebagai proses pembuatan, pelatihan, dan evaluasi model untuk memprediksi atau mengklasifikasikan data berdasarkan fitur yang ada. Pada Tahapan ini akan mengembangkan model machine learning dengan tiga algoritma yaitu 
1.	K-Nearest Neighbor (KNN): bekerja dengan mengukur jarak antara sebuah sampel tertentu dengan seluruh sampel dalam set pelatihan, lalu memilih k-tetangga terdekat. Algoritma KNN memanfaatkan konsep 'kesamaan fitur' untuk menentukan prediksi nilai dari data baru. Dengan cara ini, setiap data baru akan diberikan nilai berdasarkan seberapa miripnya dengan titik-titik dalam set data pelatihan. 

3.	Random Forest: termasuk dalam kategori model ensemble. Terdiri dari berbagai pohon keputusan (decision tree), Random Forest menggunakan pendekatan pemilihan data dan fitur secara acak. 

5.	Boosting Algorithm : bekerja dengan membuat model awal dari data latih. Kemudian, algoritma ini membuat model berikutnya yang fokus untuk mengkoreksi kesalahan yang dilakukan oleh model sebelumnya. Proses ini berulang dengan penambahan model baru sampai prediksi pada data latih menjadi optimal atau telah mencapai jumlah model maksimum yang telah ditentukan.

## Evaluation
Metrik evaluasi yang untuk menilai keakuratan model regresi dalam memprediksi data numerik adalah [Mean Squared Error (MSE)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4420880). MSE mengukur perbedaan antara prediksi model dengan nilai aktual dari data, lalu mengkuadratkan perbedaan tersebut untuk menghindari nilai selisih yang negatif. Setelah itu, perbedaan kuadrat dari setiap data dijumlahkan dan diambil rata-ratanya untuk mendapatkan nilai MSE.

[mse](https://user-images.githubusercontent.com/88262711/195906174-0257deb8-0fab-4f64-af01-7509cf371c2c.jpeg)

Keterangan:
N = jumlah dataset
yi = nilai sebenarnya
y_pred = nilai prediksi

sebelum menghitung nilai MSE dalam model, perlu dilakukan proses scaling fitur numerik pada data uji terlebih dahulu untuk memastikan bahwa fitur-fitur numerik dalam data test memiliki skala yang serupa dengan data train, yang telah di-scaling sebelumnya. Selanjutnya, dilakukan proses evaluasi ketiga model dengan metrik MSE dan didapatkan hasil ![image](https://github.com/Aisyah-Humaira/Dicoding-Proyek-Akhir-Machine-Learning/assets/83213518/c3e7fffe-dd38-4a27-bb4d-9424e34a7195)

Dari gambar di atas, terlihat bahwa, model Random Forest (RF) memberikan nilai eror yang paling kecil. Sedangkan model dengan algoritma K-Nearest Neighbor (KNN) memiliki eror yang paling besar.
