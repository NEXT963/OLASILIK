import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, pearsonr
from sklearn.linear_model import LinearRegression

file_path = r"C:\Users\adnan\Desktop\data (1).csv"

try:
    data = pd.read_csv(file_path)
    print("Veri başarıyla yüklendi!")
    print(data.head())  

    data['Ranking'] = pd.to_numeric(data['Ranking'], errors='coerce')
    data['Miliseconds'] = pd.to_numeric(data['Miliseconds'], errors='coerce')

    data_cleaned = data.dropna(subset=['Ranking', 'Miliseconds'])

    descriptive_stats = data_cleaned.describe()
    print("\nTanımlayıcı İstatistikler:")
    print(descriptive_stats)

    plt.figure(figsize=(8, 6))
    sns.histplot(data_cleaned['Miliseconds'], kde=True, bins=30, color='blue')
    plt.title('Miliseconds Dağılımı')
    plt.xlabel('Miliseconds')
    plt.ylabel('Frekans')
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data_cleaned['Ranking'], y=data_cleaned['Miliseconds'], color='green')
    plt.title('Ranking ve Miliseconds Scatter Plot')
    plt.xlabel('Ranking')
    plt.ylabel('Miliseconds')
    plt.show()

    plt.figure(figsize=(8, 6))
    avg_milliseconds_by_sex = data_cleaned.groupby('Sex')['Miliseconds'].mean()
    avg_milliseconds_by_sex.plot(kind='bar', color=['orange', 'cyan'])
    plt.title('Cinsiyete Göre Ortalama Miliseconds')
    plt.ylabel('Ortalama Miliseconds')
    plt.xlabel('Cinsiyet')
    plt.show()

    male_times = data_cleaned[data_cleaned['Sex'] == 'Male']['Miliseconds']
    female_times = data_cleaned[data_cleaned['Sex'] == 'Female']['Miliseconds']
    t_stat, p_value = ttest_ind(male_times, female_times, nan_policy='omit')
    print("\nT-Test Sonuçları:")
    print(f"t-statistic: {t_stat}, p-value: {p_value}")


    correlation, corr_p_value = pearsonr(data_cleaned['Ranking'], data_cleaned['Miliseconds'])
    print("\nKorelasyon Analizi:")
    print(f"Korelasyon Katsayısı: {correlation}, p-value: {corr_p_value}")


    X = data_cleaned['Ranking'].values.reshape(-1, 1)
    y = data_cleaned['Miliseconds'].values.reshape(-1, 1)
    reg_model = LinearRegression()
    reg_model.fit(X, y)

    slope = reg_model.coef_[0][0]
    intercept = reg_model.intercept_[0]
    print("\nRegresyon Analizi:")
    print(f"Eğim (Slope): {slope}, Kesişim Noktası (Intercept): {intercept}")
    print(f"Regresyon Denklemi: Miliseconds = {slope} * Ranking + {intercept}")

except FileNotFoundError:
    print(f"Dosya bulunamadı: {file_path}")
except Exception as e:
    print(f"Hata oluştu: {e}")
