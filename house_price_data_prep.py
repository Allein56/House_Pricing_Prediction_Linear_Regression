import pandas as pd

df = pd.read_csv('D:\Ma_Data\Kuliah\Semester_5\Data_Mining_with_Python_UNPAR\Kuliah 2_Data Pre-processing\House_pricing_data_prep\house_price_data_2.csv')

df.head
df.columns
df.dtypes
df.size
df.shape


#1. EKSPLORASI DATA
#Menghilangkan kolom id, date, view, zipcode, long, dan lat dari dataframe karena 
#tidak memberikan manfaat pada model algoritma regression.
df = df[['price', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront','condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
       'sqft_living15', 'sqft_lot15']]

#Melihat nilai minimum dan maksimum setiap kolom
min_max_value = {'price':[df.price.min(), df.price.max()], 
                 'bedrooms':[df.bedrooms.min(), df.bedrooms.max()], 
                 'bathrooms':[df.bathrooms.min(), df.bathrooms.max()], 
                 'sqft_living':[df.sqft_living.min(), df.sqft_living.max()], 
                 'sqft_lot':[df.sqft_lot.min(), df.sqft_lot.max()], 
                 'floors':[df.floors.min(), df.floors.max()], 
                 'waterfront':[df.waterfront.min(), df.waterfront.max()], 
                 'condition':[df.condition.min(), df.condition.max()], 
                 'grade':[df.grade.min(), df.grade.max()], 
                 'sqft_above':[df.sqft_above.min(), df.sqft_above.max()], 
                 'sqft_basement':[df.sqft_basement.min(), df.sqft_basement.max()], 
                 'yr_built':[df.yr_built.min(), df.yr_built.max()], 
                 'yr_renovated':[df.yr_renovated.min(), df.yr_renovated.max()], 
                 'sqft_living15':[df.sqft_living15.min(), df.sqft_living15.max()], 
                 'sqft_lot15':[df.sqft_lot.min(), df.sqft_lot.max()]}

#Melihat nilai rata-rata setiap kolom
avg = df.mean()

#Melihat nilai standar deviasi setiap kolom
std = df.std()

#DATA VISUALIZATION
#Untuk memahami distribusi data disetiap kolom untuk mengeliminasi outliers
import matplotlib.pyplot as plt
import numpy as np


#Sebaran data kolom price
count, bin_edges = np.histogram(df['price'], bins=7)
df['price'].plot(
    kind = 'hist', 
    xticks = bin_edges, 
    edgecolor='black'
    )
plt.title('Histogram Harga Rumah', fontsize=20)
plt.ylabel('Frekuensi', fontsize=12)
plt.xlabel('Harga Rumah', fontsize=12)
plt.show()

df['price'].plot(kind = 'box')
plt.title('Boxplot Harga Rumah', fontsize=20)
plt.ylabel('Harga', fontsize=12)
plt.show()


#Sebaran data kolom bedrooms
count, bin_edges = np.histogram(df['bedrooms'], bins=10)
df['bedrooms'].plot(
    kind = 'hist', 
    xticks = bin_edges, 
    edgecolor='black'
    )
plt.title('Histogram Jumlah Kamar', fontsize=20)
plt.ylabel('Frekuensi', fontsize=12)
plt.xlabel('Jml. Kamar', fontsize=12)
plt.show()

df['bedrooms'].plot(kind = 'box')
plt.title('Boxplot Jumlah Kamar', fontsize=20)
plt.ylabel('Jml. Kamar', fontsize=12)
plt.show()


#Sebaran data kolom bathrooms
count, bin_edges = np.histogram(df['bathrooms'], bins=10)
df['bathrooms'].plot(
    kind = 'hist', 
    xticks = bin_edges, 
    edgecolor='black'
    )
plt.title('Histogram Jml. Kamar Mandi', fontsize=20)
plt.ylabel('Frekuensi', fontsize=12)
plt.xlabel('Jml. Kamar Mandi', fontsize=12)
plt.show()

df['bathrooms'].plot(kind = 'box')
plt.title('Boxplot Jml. Kamar Mandi', fontsize=20)
plt.ylabel('Jml.Kamar Mandi', fontsize=12)
plt.show()


#Sebaran data kolom sqft_living
count, bin_edges = np.histogram(df['sqft_living'], bins=10)
df['sqft_living'].plot(
    kind = 'hist', 
    xticks = bin_edges, 
    edgecolor='black'
    )
plt.title('Histogram Luas Rumah', fontsize=20)
plt.ylabel('Frekuensi', fontsize=12)
plt.xlabel('Luas (Kaki Kuadrat)', fontsize=12)
plt.show()

df['sqft_living'].plot(kind = 'box')
plt.title('Boxplot Luas Rumah', fontsize=20)
plt.ylabel('Luas Rumah(Kaki Kuadrat)', fontsize=12)
plt.show()


#Sebaran data kolom sqft_lot
count, bin_edges = np.histogram(df['sqft_lot'], bins=8)
df['sqft_lot'].plot(
    kind = 'hist', 
    xticks = bin_edges, 
    edgecolor='black'
    )
plt.title('Histogram Luas Tanah', fontsize=20)
plt.ylabel('Frekuensi', fontsize=12)
plt.xlabel('Luas Tanah(Kaki kuadrat', fontsize=12)
plt.show()

df['sqft_lot'].plot(kind = 'box')
plt.title('Boxplot Luas Tanah', fontsize=20)
plt.ylabel('Luas Tanah(Kaki Kuadrat)', fontsize=12)
plt.show()


#Sebaran data kolom floors
count, bin_edges = np.histogram(df['floors'], bins=10)
df['floors'].plot(
    kind = 'hist', 
    xticks = bin_edges, 
    edgecolor='black'
    )
plt.title('Histogram Jumlah Lantai', fontsize=20)
plt.ylabel('Frekuensi', fontsize=12)
plt.xlabel('Jumlah Lantai', fontsize=12)
plt.show()

df['floors'].plot(kind = 'box')
plt.title('Boxplot Jumlah Lantai', fontsize=20)
plt.ylabel('Jml. Lantai', fontsize=12)
plt.show()


#Sebaran data waterfront
count, bin_edges = np.histogram(df['waterfront'], bins=2)
df['waterfront'].plot(
    kind = 'hist', 
    xticks = bin_edges, 
    edgecolor='black'
    )
plt.title('Histogram Jumlah Rumah yang dipinggir perairan', fontsize=20)
plt.ylabel('Frekuensi', fontsize=12)
plt.xlabel('Jumlah Rumah', fontsize=12)
plt.show()


#Sebaran data kolom condition
df['condition'].plot(
    kind = 'hist', 
    xticks = [1, 2, 3, 4, 5], 
    edgecolor='black'
    )
plt.title('Histogram Kondisi Rumah', fontsize=20)
plt.ylabel('Frekuensi', fontsize=12)
plt.xlabel('Kondisi Rumah', fontsize=12)
plt.show()


#Sebaran data kolom grade
i = 1
bin_edges = []
while i < 14:
    bin_edges.append(i)
    i += 1
df['grade'].plot(
    kind = 'hist', 
    xticks = bin_edges, 
    edgecolor='black'
    )
plt.title('Histogram Penilaian Rumah', fontsize=20)
plt.ylabel('Frekuensi', fontsize=12)
plt.xlabel('Penilaian Rumah', fontsize=12)
plt.show()


#Sebaran data kolom sqft_above
count, bin_edges = np.histogram(df['sqft_above'], bins=10)
df['sqft_above'].plot(
    kind = 'hist', 
    xticks = bin_edges, 
    edgecolor='black'
    )
plt.title('Histogram Luas Rumah Diatas Basement', fontsize=20)
plt.ylabel('Frekuensi', fontsize=12)
plt.xlabel('Luas Rumah (Kaki Kuadrat)', fontsize=12)
plt.show()

df['sqft_above'].plot(kind = 'box')
plt.title('Boxplot Luas Rumah Diatas Basement', fontsize=20)
plt.ylabel('Luas Rumah(Kaki Kuadrat)', fontsize=12)
plt.show()


#Sebaran data kolom sqft_basement
count, bin_edges = np.histogram(df['sqft_basement'], bins=10)
df['sqft_basement'].plot(
    kind = 'hist', 
    xticks = bin_edges, 
    edgecolor='black'
    )
plt.title('Histogram Luas Basement', fontsize=20)
plt.ylabel('Frekuensi', fontsize=12)
plt.xlabel('Luas Basement', fontsize=12)
plt.show()

df['sqft_basement'].plot(kind = 'box')
plt.title('Boxplot Luas Basement', fontsize=20)
plt.ylabel('Luas Basement(Kaki Kuadrat)', fontsize=12)
plt.show()


#Sebaran data kolom yr_built
count, bin_edges = np.histogram(df['yr_built'], bins=5)
df['yr_built'].plot(
    kind = 'hist', 
    xticks = bin_edges, 
    edgecolor='black'
    )
plt.title('Histogram Tahun Rumah Dibuat', fontsize=20)
plt.ylabel('Frekuensi', fontsize=12)
plt.xlabel('Tahun', fontsize=12)
plt.show()

df['yr_built'].plot(kind = 'box')
plt.title('Boxplot Tahun Rumah Dibuat', fontsize=20)
plt.ylabel('Tahun', fontsize=12)
plt.show()

#Sebaran data kolom yr_renovated
count, bin_edges = np.histogram(df['yr_renovated'], bins=2)
df['yr_renovated'].plot(
    kind = 'hist', 
    xticks = bin_edges, 
    edgecolor='black'
    )
plt.title('Histogram Tahun Rumah direnovasi', fontsize=20)
plt.ylabel('Frekuensi', fontsize=12)
plt.xlabel('Tahun', fontsize=12)
plt.show()


#Sebaran data kolom sqft_living15
count, bin_edges = np.histogram(df['sqft_living15'], bins=10)
df['sqft_living15'].plot(
    kind = 'hist', 
    xticks = bin_edges, 
    edgecolor='black'
    )
plt.title('Histogram Rata-rata Luas 15 Rumah Tetangga', fontsize=20)
plt.ylabel('Frekuensi', fontsize=12)
plt.xlabel('Luas (Kaki Kuadrat)', fontsize=12)
plt.show()

df['sqft_living15'].plot(kind = 'box')
plt.title('Boxplot Rata-rata Luas 15 Rumah Tetangga', fontsize=20)
plt.ylabel('Luas (Kaki Kuadrat)', fontsize=12)
plt.show()


#Sebaran data kolom sqft_lot15
count, bin_edges = np.histogram(df['sqft_lot15'], bins=5)
df['sqft_lot15'].plot(
    kind = 'hist', 
    xticks = bin_edges, 
    edgecolor='black'
    )
plt.title('Histogram Rata-rata Luas Tanah 15 Rumah Tetangga', fontsize=20)
plt.ylabel('Frekuensi',fontsize=12)
plt.xlabel('Luas Tanah', fontsize=12)
plt.show()

df['sqft_lot15'].plot(kind = 'box')
plt.title('Boxplot Rata-rata Luas Tanah 15 Rumah Tetangga', fontsize=20)
plt.ylabel('Luas (Kaki Kuadrat)', fontsize=12)
plt.show()


# 2. DATA CLEANING

#Menghapus baris dengan harga yang tidak masuk akal dan menghilangkan outliers
df = df[(df.price > 1000)]
df = df[(df.price < 1000000)]

#Menghapus baris dengan jml. kamar yang tidak masuk akal dan menghilangkan outliers
df = df[(df.bedrooms > 1)]
df = df[(df.bedrooms < 5)]

#Menghapus baris dengan jml. kamar madni yang tidak masuk akal dan 
#menghilangkan outliers
df = df[(df.bathrooms > 1)]
df = df[(df.bathrooms < 4)]

#Menghapus baris dengan luas rumah yang tidak masuk akal dan menghilangkan outliers
df = df[(df.sqft_living > 100)]
df = df[(df.sqft_living < 3100)]

#Menghapus baris dengan luas tanah yang tidak masuk akal dan menghilangkan outliers
df = df[(df.sqft_lot > 100)]
df = df[(df.sqft_lot < 6000)]

#Menghapus baris dengan luas rumah tanpa basement yang tidak masuk akal dan menghilangkan outliers
df = df[(df.sqft_above > 100)]
df = df[(df.sqft_above < 2150)]

#Menghapus baris dengan luas basement yang tidak masuk akal dan menghilangkan outliers
df = df[(df.sqft_basement > 100)]
df = df[(df.sqft_basement < 1500)]

#Menghapus baris dengan luas rumah tetangga yang tidak masuk akal dan menghilangkan outliers
df = df[(df.sqft_living15 > 100)]
df = df[(df.sqft_living15 < 2300)]

#Menghapus baris dengan luas tanah tetangga yang tidak masuk akal dan menghilangkan outliers
df = df[(df.sqft_lot15 > 100)]
df = df[(df.sqft_lot15 < 6000)]

# HASIL DATA CLEANING

#Sebaran data kolom price
count, bin_edges = np.histogram(df['price'], bins=7)
df['price'].plot(
    kind = 'hist', 
    xticks = bin_edges, 
    edgecolor='black'
    )
plt.title('Histogram Harga Rumah', fontsize=20)
plt.ylabel('Frekuensi', fontsize=12)
plt.xlabel('Harga Rumah', fontsize=12)
plt.show()

df['price'].plot(kind = 'box')
plt.title('Boxplot Harga Rumah', fontsize=20)
plt.ylabel('Harga', fontsize=12)
plt.show()


#Sebaran data kolom bedrooms
count, bin_edges = np.histogram(df['bedrooms'], bins=5)
df['bedrooms'].plot(
    kind = 'hist', 
    xticks = bin_edges, 
    edgecolor='black'
    )
plt.title('Histogram Jumlah Kamar', fontsize=20)
plt.ylabel('Frekuensi', fontsize=12)
plt.xlabel('Jml. Kamar', fontsize=12)
plt.show()

df['bedrooms'].plot(kind = 'box')
plt.title('Boxplot Jumlah Kamar', fontsize=20)
plt.ylabel('Jml. Kamar', fontsize=12)
plt.show()


#Sebaran data kolom bathrooms
count, bin_edges = np.histogram(df['bathrooms'], bins=10)
df['bathrooms'].plot(
    kind = 'hist', 
    xticks = bin_edges, 
    edgecolor='black'
    )
plt.title('Histogram Jml. Kamar Mandi', fontsize=20)
plt.ylabel('Frekuensi', fontsize=12)
plt.xlabel('Jml. Kamar Mandi', fontsize=12)
plt.show()

df['bathrooms'].plot(kind = 'box')
plt.title('Boxplot Jml. Kamar Mandi', fontsize=20)
plt.ylabel('Jml.Kamar Mandi', fontsize=12)
plt.show()


#Sebaran data kolom sqft_living
count, bin_edges = np.histogram(df['sqft_living'], bins=10)
df['sqft_living'].plot(
    kind = 'hist', 
    xticks = bin_edges, 
    edgecolor='black'
    )
plt.title('Histogram Luas Rumah', fontsize=20)
plt.ylabel('Frekuensi', fontsize=12)
plt.xlabel('Luas (Kaki Kuadrat)', fontsize=12)
plt.show()

df['sqft_living'].plot(kind = 'box')
plt.title('Boxplot Luas Rumah', fontsize=20)
plt.ylabel('Luas Rumah(Kaki Kuadrat)', fontsize=12)
plt.show()


#Sebaran data kolom sqft_lot
count, bin_edges = np.histogram(df['sqft_lot'], bins=8)
df['sqft_lot'].plot(
    kind = 'hist', 
    xticks = bin_edges, 
    edgecolor='black'
    )
plt.title('Histogram Luas Tanah', fontsize=20)
plt.ylabel('Frekuensi', fontsize=12)
plt.xlabel('Luas Tanah(Kaki kuadrat', fontsize=12)
plt.show()

df['sqft_lot'].plot(kind = 'box')
plt.title('Boxplot Luas Tanah', fontsize=20)
plt.ylabel('Luas Tanah(Kaki Kuadrat)', fontsize=12)
plt.show()


#Sebaran data kolom floors
count, bin_edges = np.histogram(df['floors'], bins=5)
df['floors'].plot(
    kind = 'hist', 
    xticks = bin_edges, 
    edgecolor='black'
    )
plt.title('Histogram Jumlah Lantai', fontsize=20)
plt.ylabel('Frekuensi', fontsize=12)
plt.xlabel('Jumlah Lantai', fontsize=12)
plt.show()

df['floors'].plot(kind = 'box')
plt.title('Boxplot Jumlah Lantai', fontsize=20)
plt.ylabel('Jml. Lantai', fontsize=12)
plt.show()


#Sebaran data waterfront
count, bin_edges = np.histogram(df['waterfront'], bins=2)
df['waterfront'].plot(
    kind = 'hist', 
    xticks = bin_edges, 
    edgecolor='black'
    )
plt.title('Histogram Jumlah Rumah yang dipinggir perairan', fontsize=20)
plt.ylabel('Frekuensi', fontsize=12)
plt.xlabel('Jumlah Rumah', fontsize=12)
plt.show()

df.drop('waterfront', axis=1)


#Sebaran data kolom condition
df['condition'].plot(
    kind = 'hist', 
    xticks = [1, 2, 3, 4, 5], 
    edgecolor='black'
    )
plt.title('Histogram Kondisi Rumah', fontsize=20)
plt.ylabel('Frekuensi', fontsize=12)
plt.xlabel('Kondisi Rumah', fontsize=12)
plt.show()

# Setelah pembersihan data, data rumah dengan waterfront tidak ada.
# Sehingga tidak memberikan manfaat untuk algoritma model regresi.
df = df.drop('waterfront', axis=1)

#Sebaran data kolom grade
i = 1
bin_edges = []
while i < 14:
    bin_edges.append(i)
    i += 1
df['grade'].plot(
    kind = 'hist', 
    xticks = bin_edges, 
    edgecolor='black'
    )
plt.title('Histogram Penilaian Rumah', fontsize=20)
plt.ylabel('Frekuensi', fontsize=12)
plt.xlabel('Penilaian Rumah', fontsize=12)
plt.show()


#Sebaran data kolom sqft_above
count, bin_edges = np.histogram(df['sqft_above'], bins=10)
df['sqft_above'].plot(
    kind = 'hist', 
    xticks = bin_edges, 
    edgecolor='black'
    )
plt.title('Histogram Luas Rumah Diatas Basement', fontsize=20)
plt.ylabel('Frekuensi', fontsize=12)
plt.xlabel('Luas Rumah (Kaki Kuadrat)', fontsize=12)
plt.show()

df['sqft_above'].plot(kind = 'box')
plt.title('Boxplot Luas Rumah Diatas Basement', fontsize=20)
plt.ylabel('Luas Rumah(Kaki Kuadrat)', fontsize=12)
plt.show()


#Sebaran data kolom sqft_basement
count, bin_edges = np.histogram(df['sqft_basement'], bins=10)
df['sqft_basement'].plot(
    kind = 'hist', 
    xticks = bin_edges, 
    edgecolor='black'
    )
plt.title('Histogram Luas Basement', fontsize=20)
plt.ylabel('Frekuensi', fontsize=12)
plt.xlabel('Luas Basement', fontsize=12)
plt.show()

df['sqft_basement'].plot(kind = 'box')
plt.title('Boxplot Luas Basement', fontsize=20)
plt.ylabel('Luas Basement(Kaki Kuadrat)', fontsize=12)
plt.show()


#Sebaran data kolom yr_built
count, bin_edges = np.histogram(df['yr_built'], bins=5)
df['yr_built'].plot(
    kind = 'hist', 
    xticks = bin_edges, 
    edgecolor='black'
    )
plt.title('Histogram Tahun Rumah Dibuat', fontsize=20)
plt.ylabel('Frekuensi', fontsize=12)
plt.xlabel('Tahun', fontsize=12)
plt.show()

df['yr_built'].plot(kind = 'box')
plt.title('Boxplot Tahun Rumah Dibuat', fontsize=20)
plt.ylabel('Tahun', fontsize=12)
plt.show()

#Sebaran data kolom yr_renovated
count, bin_edges = np.histogram(df['yr_renovated'], bins=2)
df['yr_renovated'].plot(
    kind = 'hist', 
    xticks = bin_edges, 
    edgecolor='black'
    )
plt.title('Histogram Tahun Rumah direnovasi', fontsize=20)
plt.ylabel('Frekuensi', fontsize=12)
plt.xlabel('Tahun', fontsize=12)
plt.show()


#Sebaran data kolom sqft_living15
count, bin_edges = np.histogram(df['sqft_living15'], bins=10)
df['sqft_living15'].plot(
    kind = 'hist', 
    xticks = bin_edges, 
    edgecolor='black'
    )
plt.title('Histogram Rata-rata Luas 15 Rumah Tetangga', fontsize=20)
plt.ylabel('Frekuensi', fontsize=12)
plt.xlabel('Luas (Kaki Kuadrat)', fontsize=12)
plt.show()

df['sqft_living15'].plot(kind = 'box')
plt.title('Boxplot Rata-rata Luas 15 Rumah Tetangga', fontsize=20)
plt.ylabel('Luas (Kaki Kuadrat)', fontsize=12)
plt.show()


#Sebaran data kolom sqft_lot15
count, bin_edges = np.histogram(df['sqft_lot15'], bins=5)
df['sqft_lot15'].plot(
    kind = 'hist', 
    xticks = bin_edges, 
    edgecolor='black'
    )
plt.title('Histogram Rata-rata Luas Tanah 15 Rumah Tetangga', fontsize=20)
plt.ylabel('Frekuensi',fontsize=12)
plt.xlabel('Luas Tanah', fontsize=12)
plt.show()

df['sqft_lot15'].plot(kind = 'box')
plt.title('Boxplot Rata-rata Luas Tanah 15 Rumah Tetangga', fontsize=20)
plt.ylabel('Luas (Kaki Kuadrat)', fontsize=12)
plt.show()

# 3. Memahami Korelasi data prediktor dengan atribut prediksinya

#Visualisasi Bivariate Correlation seluruh kolom
import seaborn as sns

df.corr()
sns.heatmap(df.corr(method='pearson'))

#Semakin muda warna merah, maka semakin erat hubungan linier antara 2 kolom.
#Kolom sqft_living, sqft_above, & sqft_living15 memiliki nilai korelasi yang tinggi


#Visualisasi korelasi kolom sqft_living dengan kolom price
count, bin_edges = np.histogram(df['sqft_living'], bins=10)
df.plot(
    	kind='scatter',
    	x='sqft_living',
    	y='price', 
        xticks=bin_edges
        )
plt.title('Korelasi luas Rumah dengan Harga', fontsize=20)
plt.xlabel('Luas Rumah (Kaki Kuadrat)', fontsize=12)
plt.ylabel('Harga', fontsize=12)
plt.show()


#Visualisasi korelasi kolom sqft_above dengan kolom price
count, bin_edges = np.histogram(df['sqft_above'], bins=10)
df.plot(
    	kind='scatter',
    	x='sqft_above',
    	y='price', 
        xticks=bin_edges
        )
plt.title('Korelasi Luas Rumah Diatas Basement dengan Harga', fontsize=20)
plt.xlabel('Luas (Kaki Kuadrat)', fontsize=12)
plt.ylabel('Harga', fontsize=12)
plt.show()


#Visualisasi korelasi kolom sqft_living15 dengan kolom price
count, bin_edges = np.histogram(df['sqft_living15'], bins=10)
df.plot(
    	kind='scatter',
    	x='sqft_living15',
    	y='price', 
        xticks=bin_edges
        )
plt.title('Korelasi Luas Rumah Tetangga dengan Harga', fontsize=20)
plt.xlabel('Luas (Kaki Kuadrat)', fontsize=12)
plt.ylabel('Harga', fontsize=12)
plt.show()
