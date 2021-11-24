import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('D:\Ma_Data\Kuliah\Semester_5\Data_Mining_with_Python_UNPAR\Kuliah 2_Data Pre-processing\House_pricing_data_prep\house_price_data_2.csv')

# MENYIAPKAN DATA
#Menghilangkan kolom yang tidak bermanfaat bagi model regresi
df = df[['price', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors','condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
       'sqft_living15', 'sqft_lot15']]

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
#Menghapus baris dengan luas rumah tanpa basement yang tidak masuk akal dan 
#menghilangkan outliers
df = df[(df.sqft_above > 100)]
df = df[(df.sqft_above < 2150)]
#Menghapus baris dengan luas basement yang tidak masuk akal dan menghilangkan 
#outliers
df = df[(df.sqft_basement > 100)]
df = df[(df.sqft_basement < 1500)]
#Menghapus baris dengan luas rumah tetangga yang tidak masuk akal dan 
#menghilangkan outliers
df = df[(df.sqft_living15 > 100)]
df = df[(df.sqft_living15 < 2300)]
#Menghapus baris dengan luas tanah tetangga yang tidak masuk akal dan 
#menghilangkan outliers
df = df[(df.sqft_lot15 > 100)]
df = df[(df.sqft_lot15 < 6000)]


#Visualisasi Bivariate Correlation seluruh kolom
import seaborn as sns

corr_matrix = df.corr(method='pearson')
sns.heatmap(df.corr(method='pearson'))


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

#Kolom yang dapat digunakan untuk dijadikan atribut prediktor adalah 
#sqft_living, sqft_living15, sqft_above
predict_atribute = ['sqft_living', 'sqft_living15', 'sqft_above']

# REGRESI LINIER
from sklearn.model_selection import train_test_split 
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

mae = []
mse = []
rmse = []
r_2 = []
regr_equation = []
    
#1. Train/Test split method
for atrbt in predict_atribute:
    # Linear regresi dari tiap atribut prediktor
    X = df[atrbt]
    y = df.price
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=3)
    
    #Melatih model
    smpl_regr = linear_model.LinearRegression()
    X_train = np.asanyarray(X_train).reshape(-1, 1)
    y_train = np.asanyarray(y_train).reshape(-1, 1)
    smpl_regr.fit(X_train, y_train)
        
    #Menguji model
    X_test = np.asanyarray(X_test).reshape(-1, 1)
    y_test = np.asanyarray(y_test).reshape(-1, 1) 
    y_predict = smpl_regr.predict(X_test) #hasil prediksi model dari data
    
    # Memasukan nilai tiap metrik pada tiap atribut prediktor ke dalam list
    mae.append(metrics.mean_absolute_error(y_test, y_predict))
    mse.append(metrics.mean_squared_error(y_test, y_predict))
    rmse.append(np.sqrt(metrics.mean_absolute_error(y_test, y_predict)))
    r_2.append(r2_score(y_test, y_predict))
    if smpl_regr.intercept_ > 0:
        regr_equation.append('y = {}x + {}'.format('%.2f' % smpl_regr.coef_, 
                                                   '%.2f' % smpl_regr.intercept_))
    else:
        regr_equation.append('y = {}x {}'.format('%.2f' % smpl_regr.coef_, 
                                                   '%.2f' % smpl_regr.intercept_))

#Membuat dataframe evaluasi metrics regresi linier
traintest_smpl_metrics = {'MAE':mae, 'MSE':mse, 'RMSE':rmse, 'r^2':r_2, 
                    'Persamaan Regresi':regr_equation}
traintest_smpl_metrics = pd.DataFrame(data=traintest_smpl_metrics, index=predict_atribute)    
traintest_smpl_metrics

#2. K-Fold Cross Validation Method
folds = KFold(n_splits=3)

#Membagi data dengan metode K-fold dengan fold sebanyak 5, tiap fold ke-i 
#dibentuk melalui tahapan iterasi ke-a
mae = []
mse = []
rmse = []
r_2 = []
mae_avg = []
mse_avg = []
rmse_avg = []
r_2_avg = []
#Membagi data dengan metode K-fold dengan fold sebanyak 5, tiap fold ke-i dibentuk melalui tahapan iterasi ke-a
for atrbt in predict_atribute:
    for train_index, test_index in folds.split(df[atrbt], df.price):
        train_x, test_x, train_y, test_y = df[atrbt].loc[df[atrbt].index.intersection(
            train_index
            )], df[atrbt].loc[df[atrbt].index.intersection(
            test_index
            )], df.price.loc[df[atrbt].index.intersection(
            train_index
            )], df.price.loc[df[atrbt].index.intersection(
            test_index
            )]
        
        #melatih model dari hasil pembagian data K-fold iterasi ke-i
        train_x = np.asanyarray(train_x).reshape(-1, 1)
        train_y = np.asanyarray(train_y).reshape(-1, 1)
        smpl_regr.fit(train_x, train_y) 
        smpl_regr.coef_, smpl_regr.intercept_
            
        #Menguji hasil model K-fold iterasi ke-a
        test_x = np.asanyarray(test_x).reshape(-1, 1)
        test_y = np.asanyarray(test_y).reshape(-1, 1) 
        y_predict = smpl_regr.predict(test_x)
        
        #Mencatat nilai evaluasi metrik tiap iterasi
        mae.append(metrics.mean_absolute_error(test_y, y_predict))
        mse.append(metrics.mean_squared_error(test_y, y_predict))
        rmse.append(np.sqrt(metrics.mean_squared_error(test_y, y_predict)))
        r_2.append(r2_score(test_y, y_predict))
        
    mae_avg.append(np.mean(mae))
    mse_avg.append(np.mean(mse))
    rmse_avg.append(np.mean(rmse))
    r_2_avg.append(np.mean(r_2))

kfold_smpl_metrics = {'MAE':mae_avg, 'MSE':mse_avg, 'RMSE':rmse_avg, 'r^2':r_2_avg}
kfold_smpl_metrics = pd.DataFrame(data=kfold_smpl_metrics, index=predict_atribute)
kfold_smpl_metrics