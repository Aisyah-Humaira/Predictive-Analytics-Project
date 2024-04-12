# Laporan Proyek Machine Learning - Aisyah Humaira

## Domain Proyek
![image](https://github.com/Aisyah-Humaira/Dicoding-Proyek-Akhir-Machine-Learning/assets/83213518/768bb849-7586-4158-8a52-c8f867d405b2)

Diabetes merupakan salah satu penyakit kronis yang paling umum dikenal di seluruh dunia dan dapat menimbulkan risiko kesehatan yang serius jika tidak dikelola dengan tepat. Menurut data yang diperoleh sekitar tahun 2019, prevalensi diabetes global diperkirakan mencapai 9,3%, atau sekitar 463 juta orang di seluruh dunia. Proyeksi untuk masa depan menunjukkan tren peningkatan, dengan estimasi mencapai 10,2% atau sekitar 578 juta orang pada tahun 2030, dan meningkat lagi menjadi 10,9% atau sekitar 700 juta orang pada tahun 2045. Selain itu, didapatkan bahwa separuh dari individu yang hidup dengan diabetes, tepatnya sekitar 50,1%, tidak menyadari bahwa mereka sebenarnya menderita penyakit ini. Hal ini menunjukkan betapa pentingnya upaya deteksi dini dan kesadaran akan gejala diabetes. Dampak diabetes yang tidak terkontrol dapat sangat merusak bagi kesehatan tubuh secara keseluruhan. Banyak organ tubuh, termasuk ginjal, mata, jantung, dan sistem saraf, dapat mengalami kerusakan serius akibat komplikasi diabetes yang tidak terdeteksi atau dikelola dengan baik. Oleh karena itu, sangat penting untuk melakukan diagnosis diabetes dengan cepat dan akurat guna mencegah komplikasi serius dan memastikan kualitas hidup yang lebih baik bagi penderita diabetes [[1]](https://www.sciencedirect.com/science/article/pii/S1110866524000045).

Banyak teknik *machine learning* (ML) yang digunakan di sektor medis untuk mendeteksi dan memprediksi gangguan kesehatan. Salah satu penyakit yang menggunakan teknik ML untuk menemukan pengobatan yang paling efektif adalah diabetes. Teknik-teknik *machine learning* diterapkan di hampir setiap bidang kehidupan untuk mengatasi masalah praktis karena kemampuannya untuk menghasilkan hasil yang konsisten, dapat diandalkan, dan akurat. Dengan demikian, penggunaan ML dalam diagnosa diabetes dapat mempercepat proses identifikasi penyakit, mengurangi risiko kesalahan manusia, dan memastikan pasien mendapatkan pengobatan yang sesuai dengan kondisinya [[2]](https://www.sciencedirect.com/science/article/pii/S2772442523001405).

## Business Understanding

### Problem Statements
- Bagaimana cara mendapatkan model *machine learning* untuk memprediksi penyakit diabetes?
- *Model development* apa yang memberikan hasil dengan error paling kecil untuk memprediksi penyakit diabetes?

### Goals
- Mendapatkan model *machine learning* yang dapat digunakan untuk memprediksi penyakit diabetes.
- Mengetahui *model development* yang memberikan hasil dengan error paling kecil untuk memprediksi penyakit diabetes.

## Data Understanding
Dataset yang digunakan dalam proyek ini diperoleh dari situs Kaggle mengenai [Simple Feature To Detect Diabetes](https://www.kaggle.com/datasets/simaanjali/diabetes-simple-diagnosis). Dataset ini terdiri dari 88380 baris dan 9 kolom. 

Berdasarkan informasi dari Kaggle, variabel-variabel pada dataset adalah sebagai berikut:
- ***Age***: Mewakili usia pasien dalam tahun. Usia dapat menjadi faktor risiko untuk diabetes, karena risiko diabetes meningkat seiring bertambahnya usia.
- ***Gender***: Menunjukkan jenis kelamin pasien, yang dapat menjadi faktor dalam prediksi diabetes. Beberapa studi menyarankan bahwa wanita mungkin memiliki risiko yang berbeda dibandingkan pria dalam mengembangkan diabetes.
- ***Body Mass Index (BMI)***: BMI adalah ukuran yang menggunakan tinggi dan berat badan seseorang untuk menentukan apakah mereka berada dalam kategori berat badan normal, kelebihan berat badan, atau obesitas. BMI yang tinggi dikaitkan dengan risiko diabetes yang lebih tinggi.
- ***High Blood Pressure (High_BP)***: Indikator apakah seorang pasien menderita hipertensi. Tekanan darah tinggi adalah faktor risiko yang signifikan untuk diabetes tipe 2.
- ***Fasting Blood Glucose (FBS)***: Mewakili tingkat glukosa dalam darah setelah puasa semalaman. Tingkat gula darah puasa yang tinggi dapat menunjukkan risiko diabetes atau prediabetes.
- ***HbA1c (HbA1c_level)***: Pengukuran rata-rata tingkat gula darah selama 2-3 bulan terakhir. Ini adalah indikator penting untuk diagnosis dan pengelolaan diabetes.
- ***Smoking***: Menunjukkan apakah pasien merokok atau tidak. Merokok dapat menjadi faktor risiko tambahan untuk diabetes tipe 2.
- ***Diagonisis***: Indikator bahwa seseorang memiliki diabetes.

Selain dari deskripsi variable, didapatkan pula informasi mengenai dataset sebagai berikut
- Terdapat 1 kolom non-numerik dengan tipe object yaitu *Gender*. Kolom ini merupakan *categorical features*.
- Terdapat 1 kolom numerik dengan tipe data float64 yaitu *HbA1c_level*. Kolom ini merupakan *numerical features*.
- Terdapat 7 kolom numerik dengan tipe data int64, yaitu: *Unnamed: 0, Age, BMI, High_BP, FBS, Smoking, dan Diagnosis*. Kolom ini merupakan *numerical features*.

Pada tahap selanjutnya, dilakukan *Exploratory Data Analysis* (EDA) untuk memahami dan menganalisis karakteristik dari data yang digunakan. EDA bertujuan untuk menemukan pola, mengidentifikasi anomali, serta memeriksa asumsi-asumsi yang ada pada dataset. Terdapat dua metode yang digunakan yaitu metode bersifat *univariate*  dan *multivariate*.

1. *Univariate Analysis* melibatkan satu *variate* atau variabel. Pada proses analisis terhadap fitur kategori didapatkan yaitu
   
![grafik_gender](https://github.com/Aisyah-Humaira/Dicoding-Proyek-Akhir-Machine-Learning/assets/83213518/16b8144c-e1dd-44de-a677-353d791c0492)

Gambar 1. Grafik Gender


Pada gambar 1, terdapat 3 fitur *gender* yaitu *Female, Male,* dan *Other*. Dari data persentase dapat disimpulkan bahwa gender yang paling banyak adalah *Female* dengan persante sebesar 58.10%, kemudian *gender Male* sebesar 41.88%, dan terakhir *other* hanya berkisar 0.02%.

Selanjutnya, untuk fitur numerik didapatkan 

![image](https://github.com/Aisyah-Humaira/Dicoding-Proyek-Akhir-Machine-Learning/assets/83213518/991f9797-6757-4479-8a0e-d06b73870c37)

Gambar 2. Histogram Fitur Numerik

Pada gambar 2, terlihat pada fitur *Age* sampel paling banyak pada umur 80. Lalu pada fitur *BMI*, paling banyak sampel memiliki berat badan pada kisaran 30 kg. Kemudian pada sampel *High_BP*, terlihat jika kebanyak sampel memiliki tekanan darah tinggi. Selanjutnya pada fitur *FBS*, dapat dilihat lebih banyak sampel yang memiliki tingkat glukosa dalam darah lebih dari 150. Kemudian untuk fitur *HbA1c_level*, rata-rata tingkat gula darah selama 2-3 bulan terakhir paling banyak pada kisaran 6. Lalu pada fitur *Smoking*, lebih banyak sampel yang merokok daripada tidak. Terakhir pada fitur Diagnosis, terlihat jika lebih banyak sampel yang terdiagnosis diabetes.

2. *Multivariate Analysis* melibatkan dua atau lebih variabel. Pada proses ini didapatkan hasil yaitu
   
![image](https://github.com/Aisyah-Humaira/Dicoding-Proyek-Akhir-Machine-Learning/assets/83213518/00add511-e329-48e1-95a8-a5ca7898d393)

Gambar 3. Korelasi Antar Fitur

Dari gambar 3 bisa dilihat evaluasi skor korelasi antar fitur. jika melihat korelasi terhadap diagnosis maka fitur yang memiliki skor tertinggi adalah *FBS* atau *Fasting Blood Glucose* yang memeperlihatkan tingkat glukosa dalam darah setelah puasa semalaman dimana tingkat gula darah puasa yang tinggi dapat menunjukkan risiko diabetes atau prediabetes. Selain itu, terdapat fitur *HbA1c_level* yang memiliki skor tertinggi, dimana indikator ini merupakan rata-rata tingkat gula darah selama 2-3 bulan terakhir.

## Data Preparation
Pada tahap ini, dilakukan persiapan data yang bertujuan untuk melakukan transformasi pada dataset. Transformasi ini dilakukan agar data memiliki format atau bentuk yang sesuai dan cocok untuk proses pemodelan dalam machine learning. Beberapa langkah yang umum dilakukan dalam data preparation antara lain:

- Encoding Fitur Kategori yang dilakukan dengan menggunakan Teknik *one-hot-encoding*. Teknik ini digunakan untuk mengubah variabel kategorikal menjadi bentuk biner (0 atau 1) sehingga dapat diolah oleh algoritma *machine learning*. Pengaplikasian Teknik ini dilakukan terhadap fitur *Gender*
- Pembagian dataset dengan fungsi *train_test_split* agar dataset menjadi data latih (*train*) dan data uji (*test*) dengan proporsi yang umum digunakan sebesar 80:20. Teknik ini digunakan untuk menguji kinerja model pada data yang belum pernah dilihat sebelumnya, sehingga dapat memeriksa apakah model tersebut *overfitting* atau generalisasi dengan baik pada data baru. 
- Standardisasi dapat membantu untuk membuat variabel memiliki skala yang serupa, sehingga algoritma *machine learning* yang berbasis jarak atau optimasi dapat bekerja dengan lebih efisien dan akurat. Proses transformasi data ini mengubah nilai rata-rata (*mean*) menjadi 0 dan nilai standar deviasi menjadi 1. Penggunaan standardisasi, dapat memastikan bahwa semua fitur memiliki skala yang serupa, yang dapat meningkatkan kinerja dan meningkatkan interpretasi model. Pada teknik ini menggunakan 

$$z = \frac{x - μ}{σ}$$

Keterangan:

x = setiap nilai dalam fitur numerik

μ = Rata-rata dari sampel *training*

σ = Standar deviasi dari sampel *training*

dari perhitungan diatas didapatkan hasil sebagai berikut

Tabel 1. Hasil Standardisasi Data
|| Age | BMI | High_BP | FBS | HbA1c_level | Smoking |
| --- | --- | --- | ------- | --- | ----------- | ------- |
| **18362** | 1.700510 | 0.280882 | 3.294713 | 0.491845 | 0.602952 | -0.674932 |
| **7629** | -0.722260	 | 2.018572 | -0.303517 | 0.032739 | 0.142361 | -0.674932 |
| **24394** | 0.668589 | 2.887417 | -0.303517 | 0.395191 | 0.418716 | -0.674932 |
| **14573** | 0.354526 | 0.715304 | -0.303517 | 1.482546 | -0.686703 | 1.481630 |
| **6118** | 0.623723 | -0.008733 | -0.303517 | -0.208895 | 3.182264 | -0.674932 |


Selanjutnya dilakukan pengecek nilai mean yang diubah menjadi 0 dan nilai standar deviasi menjadi 1 setelah proses standarisasi dimana hasilnya sebagai berikut

Tabel 2. Hasil *Descriptive Statistics* Setelah Standarisasi
| | Age | BMI | High_BP | FBS | HbA1c_level | Smoking |
| --- | --- | --- | ------- | --- | ----------- | ------- |
| **count** | 70704.0000 | 70704.0000 | 70704.0000 | 70704.0000 | 70704.0000 | 70704.0000 |
| **mean** | -0.0000	| 0.0000 | -0.0000 | -0.0000 | 0.0000 | -0.0000 |
| **std** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **min** | -1.8888 | -2.4705 | -0.3035 | -1.4171 | -1.8842	 | -0.6749 |
| **25%** | -0.8120	 | -0.5880	 | -0.3035 | -0.9338	 | -0.6867 | -0.6749 |
| **50%** | 0.0405 | -0.0087 | -0.3035 | 0.0327 | 0.2345	| -0.6749 |
| **75%** | 0.8032 | 0.4257 | -0.3035 | 0.4918 | 0.6030 | 1.4816 |
| **max** | 1.7005 | 9.2589 | 3.2947 | 3.8989 | 5.9458 | 1.4816 |

## Modeling
*Model development* dilakukan sebagai proses pembuatan, pelatihan, dan evaluasi model untuk memprediksi atau mengklasifikasikan data berdasarkan fitur yang ada. Pada Tahapan ini akan mengembangkan model *machine learning* dengan tiga algoritma yaitu *K-Nearest Neighbor, Random Forest, dan Boosting Algorithm*

### [K-Nearest Neighbor (KNN)](https://lp2m.uma.ac.id/2023/02/16/algoritma-k-nearest-neighbors-knn-pengertian-dan-penerapan/)
***K-Nearest Neighbor*** merupakan salah satu algoritma dasar dalam *machine learning* yang digunakan untuk regresi dan klasifikasi. Dalam algoritma KNN, diasumsikan bahwa data yang serupa cenderung berada dalam jarak yang dekat atau bertetangga, sehingga data-data dengan karakteristik yang mirip akan berada di lokasi yang berdekatan. Tujuan dari KNN adalah untuk menemukan tetangga terdekat dari titik kueri yang diberikan, dengan demikian kita dapat menentukan label kelas untuk titik tersebut berdasarkan mayoritas label kelas dari tetangga terdekatnya. KNN hanya memerlukan dua parameter utama, yaitu nilai k dan metrik jarak, yang jumlahnya relatif lebih sedikit dibandingkan dengan kebanyakan algoritma *machine learning* lainnya.

Tahapan Langkah algoritma metode KNN: 
1. Model KNN untuk regresi diinisialisasi menggunakan *KNeighborsRegressor* dengan jumlah tetangga terdekat (*n_neighbors*) sebanyak 10.
2. Model KNN yang telah diinisialisasi kemudian dilatih dengan menggunakan data pelatihan (*X_train*) dan label pelatihan (*y_train*). Proses pelatihan ini bertujuan agar model dapat memahami pola dan hubungan antara fitur (*X*) dan label (*y*).
3. Performa model dievaluasi menggunakan metrik *Mean Squared Error* (*MSE*)
   
### [Random Forest (RF)](https://kantinit.com/kecerdasan-buatan/random-forest-pengertian-cara-kerja-dan-contoh-penerapannya/) 
***Random Forest*** adalah salah satu metode yang memiliki kemiripan dengan *Decision Tree*. Metode ini merupakan salah satu algoritma yang paling populer karena keakuratannya, kemudahannya, dan fleksibilitasnya. Kemampuannya untuk digunakan dalam klasifikasi dan regresi, ditambah dengan sifat nonlinernya, membuatnya sangat mudah beradaptasi dengan berbagai jenis data dan situasi. Untuk mendapatkan prediksi yang akurat dan konsisten, *random forest* menggunakan metode bagging, yaitu teknik penggabungan beberapa meta algoritma untuk meningkatkan akurasi algoritma *machine learning*. Metode bagging ini mengambil sampel acak dari dataset melalui proses *raw sampling*. Setelah itu, sampel yang diperoleh dari *raw sampling* digunakan kembali dengan penggantian, proses ini dikenal sebagai *bootstrap*, dan menghasilkan sampel *bootstrap*. Setiap model kemudian dilatih secara mandiri hingga menghasilkan prediksi. Hasil akhir ditentukan berdasarkan prediksi mayoritas dari semua model. Secara sederhana, prediksi dari setiap model dikumpulkan, dan kemudian dianalisis untuk menentukan hasil mayoritas. Proses ini disebut agregasi.

Tahapan Langkah algoritma metode *Random Forest*: 
1. Model *Random Forest* untuk regresi (*RandomForestRegressor*) diinisialisasi dengan parameter sebagai berikut:
   - *n_estimators* = 50 (Jumlah pohon keputusan dalam hutan)
   - *max_depth* = 16 (Kedalaman maksimal dari setiap pohon keputusan)
   - *random_state* = 55 (Seed untuk pengacakan agar hasil dapat direproduksi)
   - *n_jobs* = -1 (Menggunakan semua core prosesor yang tersedia untuk pelatihan)
2.  Model *Random Forest* yang telah diinisialisasi kemudian dilatih dengan menggunakan data pelatihan dan label pelatihan.
3.  Performa model dievaluasi menggunakan metrik *Mean Squared Error* (*MSE*)
   
### [Boosting Algorithm](https://aws.amazon.com/id/what-is/boosting/) 
***Boosting Algorithm*** adalah teknik dalam *machine learning* yang digunakan untuk mengurangi kesalahan dalam prediksi data. Teknik ini meningkatkan akurasi dan kinerja model *machine learning* dengan mengubah beberapa model lemah menjadi satu model pembelajaran yang kuat. AdaBoost (*Adaptive Boosting*) adalah salah satu metode *boosting* yang dikembangkan pertama kali. Dalam AdaBoost, setiap data awalnya diberi bobot yang sama. Setelah setiap iterasi atau pembentukan pohon keputusan, bobot dari setiap data akan disesuaikan secara otomatis. Bobot lebih akan diberikan kepada data yang salah diklasifikasikan untuk diperbaiki pada iterasi berikutnya. Proses ini akan diulang hingga kesalahan prediksi, atau selisih antara nilai sebenarnya dan prediksi, berada di bawah tingkat kesalahan yang dapat diterima.

Tahapan Langkah algoritma metode Boosting Algorithm:
1. Model AdaBoostRegressor diinisialisasi dengan parameter sebagai berikut:
   - *learning_rate* = 0.05 (Tingkat pembelajaran yang mengontrol seberapa besar kontribusi dari setiap model lemah dalam gabungan model akhir)
   - *random_state* = 55 (Seed untuk pengacakan agar hasil dapat direproduksi)
3. Model *Random Forest* yang telah diinisialisasi kemudian dilatih dengan menggunakan data pelatihan dan label pelatihan.
4. Performa model dievaluasi menggunakan metrik *Mean Squared Error* (MSE)

## Evaluation
Metrik evaluasi yang untuk menilai keakuratan model regresi dalam memprediksi data numerik adalah *Mean Squared Error* (MSE). MSE mengukur perbedaan antara prediksi model dengan nilai aktual dari data, lalu mengkuadratkan perbedaan tersebut untuk menghindari nilai selisih yang negatif. Setelah itu, perbedaan kuadrat dari setiap data dijumlahkan dan diambil rata-ratanya untuk mendapatkan nilai [MSE](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4420880).

$$MSE = \frac{1}{N} \sum_{k=1}^n \left( y_i - \hat{y} \right)^2$$

Keterangan:

N = jumlah dataset

$\hat{y}$ = nilai sebenarnya

$`y_i`$ = nilai prediksi

sebelum menghitung nilai MSE dalam model, perlu dilakukan proses *scaling* fitur numerik pada data uji terlebih dahulu untuk memastikan bahwa fitur-fitur numerik dalam data test memiliki skala yang serupa dengan data train, yang telah di-scaling sebelumnya. Selanjutnya, dilakukan proses evaluasi ketiga model dengan metrik MSE.

Hasil evaluasi pada data train dan data test adalah sebagai berikut.

Tabel 3. Hasil Evaluasi Data Ketiga Model
|| train | test | 
| --- | --- | --- |
| **KNN** | 0.027474 | 0.031791 | 
| **RF** | 0.015403 | 0.026833 | 
| **Boosting** | 0.031805 | 0.030721 | 

Untuk memudahkan melihat hasil dari tabel 3, dibuat plot metrik tersebut dengan menggunakan *bar chart* sehingga didapatkan hasilnya seperti dibawah ini:

![image](https://github.com/Aisyah-Humaira/Dicoding-Proyek-Akhir-Machine-Learning/assets/83213518/c3e7fffe-dd38-4a27-bb4d-9424e34a7195)

Gambar 3. Grafik Evaluasi Metrik

Dari gambar 3, terlihat bahwa model *Random Forest* (RF) memberikan nilai eror yang paling kecil sedangkan model dengan algoritma *K-Nearest Neighbor* (KNN) memiliki eror yang paling besar. Oleh karena itu, model RF yang dapat dikatakan sebagai model terbaik untuk melakukan prediksi karena memiliki eror yang paling kecil.

Untuk mengujinya, dibuat prediksi menggunakan beberapa model *machine learning* yang telah dilatih sebelumnya terhadap salah satu baris data. Proses ini memberikan gambaran tentang bagaimana masing-masing model memprediksi data yang diberikan.

Tabel 4. Hasil Prediksi Terhadap Model
|| y_true | test | test | test | 
| --- | --- | --- | --- | --- |
| **71760** | 0 | 0.1 | 0.05 | 0.18 |

Terlihat pada tabel 4 bahwa prediksi dengan *Random Forest* (RF) memberikan hasil yang paling mendekati nilai benar (*y_true*).

## Conclusion
Dari projek Machine Learning Terapan yang telah dikerjakan mengenai prediksi penyakit diabetes dapat disimpulkah:

1. Untuk mengembangkan model *machine learning* dalam memprediksi penyakit diabetes, beberapa algoritma yang dapat digunakan antara lain adalah *K-Nearest Neighbor* (KNN), *Random Forest, dan Boosting Algorithm*.
2. Dari ketiga algoritma tersebut, hasil evaluasi menunjukkan bahwa algoritma *Random Forest* menghasilkan eror paling kecil dibandingkan dengan KNN dan *Boosting Algorithm*. Selain itu, *Random Forest* juga menghasilkan prediksi yang paling mendekati nilai sebenarnya.
   
## Referensi
[1] B. A. N.G., "En-RfRsK: An ensemble machine learning technique for prognostication of diabetes mellitus," Egyptian Informatics Journal, vol. 25, 2024. (https://doi.org/10.1016/j.eij.2024.100441)

[2] S. S. Bhat, M. Banu, G. A. Ansari and V. Selvam, "A risk assessment and prediction framework for diabetes mellitus using machine learning algorithms," Healthcare Analytics, vol. 4, 2023. (https://doi.org/10.1016/j.health.2023.100273)
