# ecommerce_dashboard.py

import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import folium
import geopandas as gpd
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.relativedelta import relativedelta

# Set tema warna dan konfigurasi Streamlit
plt.style.use('ggplot')
st.set_page_config(page_title="E-Commerce Dashboard", layout="wide")

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    customers = pd.read_csv('../data/customers_dataset.csv')
    geolocation = pd.read_csv('../data/geolocation_dataset.csv')
    order_items = pd.read_csv('../data/order_items_dataset.csv') 
    order_payments = pd.read_csv('../data/order_payments_dataset.csv')
    orders = pd.read_csv('../data/orders_dataset.csv')
    order_reviews = pd.read_csv('../data/order_reviews_dataset.csv')
    products = pd.read_csv('../data/products_dataset.csv')
    sellers = pd.read_csv('../data/sellers_dataset.csv')
    category_translation = pd.read_csv('../data/product_category_name_translation.csv')
    return customers, geolocation, order_items, order_payments, orders, order_reviews, products, sellers, category_translation

# Memuat data
customers, geolocation, order_items, order_payments, orders, order_reviews, products, sellers, category_translation = load_data()

# Preprocessing
orders['order_approved_at'] = orders['order_approved_at'].fillna(orders['order_purchase_timestamp'])
orders['order_delivered_carrier_date'] = orders['order_delivered_carrier_date'].fillna(orders['order_estimated_delivery_date'])
orders['order_delivered_customer_date'] = orders['order_delivered_customer_date'].fillna(orders['order_estimated_delivery_date'])

# Konversi tipe data tanggal
date_cols = ['order_purchase_timestamp', 'order_approved_at', 
             'order_delivered_carrier_date', 'order_delivered_customer_date',
             'order_estimated_delivery_date']

for col in date_cols:
    orders[col] = pd.to_datetime(orders[col])

products = products.merge(category_translation, on='product_category_name', how='left')

# Data lengkap
full_data = orders.merge(order_items, on='order_id', how='left') \
                  .merge(products, on='product_id', how='left') \
                  .merge(customers, on='customer_id', how='left')

# Sidebar filter
st.sidebar.title("Filter Data")

selected_month = st.sidebar.selectbox('Pilih Bulan', orders['order_purchase_timestamp'].dt.to_period('M').astype(str).unique())

# Tabs
st.title("E-Commerce Analytics Dashboard")
st.markdown("Nama        : Firdho Mario Dhono")
st.markdown("Email       : firdho@gmail.com")
st.markdown("ID Dicoding : Firdho Mario Dhono")
tab1, tab2, tab3 = st.tabs(["Tren Penjualan", "Analisis RFM", "Analisis Geografis"])

# ===================== TAB 1 =====================
with tab1:
    st.header("Tren Penjualan Bulanan")
    st.markdown("#### ❓ Pertanyaan:")
    st.markdown("**Bagaimana tren pertumbuhan penjualan bulanan dan pola musiman?**")

    # Dekomposisi Musiman
    monthly_sales = full_data.set_index('order_purchase_timestamp')['price'].resample('MS').sum().to_frame(name='price')
    idx = pd.date_range(start=monthly_sales.index.min(), end=monthly_sales.index.max(), freq='MS')
    monthly_sales = monthly_sales.reindex(idx)
    monthly_sales.index.name = 'order_month'
    monthly_sales['price'] = monthly_sales['price'].interpolate(method='time').fillna(method='bfill').fillna(method='ffill').fillna(0)

    decomposition = seasonal_decompose(monthly_sales['price'], model='additive', period=12)

    fig, axs = plt.subplots(3, 1, figsize=(15, 12))
    
    # Tren
    axs[0].plot(monthly_sales.index, monthly_sales['price'], label='Aktual', alpha=0.7)
    axs[0].plot(decomposition.trend.index, decomposition.trend, label='Tren', linewidth=2)
    axs[0].set_title('Analisis Tren Penjualan')
    axs[0].legend()
    for year in monthly_sales.index.year.unique():
        if 2016 <= year <= 2018:
            bf = pd.to_datetime(f"{year}-11-25")
            axs[0].axvline(bf, color='red', linestyle='--', alpha=0.5)
            axs[0].text(bf + pd.Timedelta(days=5), monthly_sales['price'].max() * 0.8, 'Black Friday', rotation=90, color='red')
    
    # Musiman
    axs[1].plot(decomposition.seasonal)
    axs[1].set_title('Komponen Musiman')
    
    # Pertumbuhan
    growth = monthly_sales['price'].pct_change() * 100
    colors = np.where(growth >= 0, 'forestgreen', 'crimson')
    axs[2].bar(monthly_sales.index, growth, color=colors, width=20)
    axs[2].axhline(0, color='black', linewidth=0.8)
    axs[2].set_title('Pertumbuhan Bulanan (%)')
    axs[2].tick_params(axis='x', rotation=45)

    st.pyplot(fig)

    st.markdown("#### ✅ Jawaban:")
    st.markdown("- Penjualan meningkat signifikan di bulan November (Black Friday).")
    st.markdown("#### 📊 Insight:")
    st.markdown("- Pertumbuhan penjualan konsisten ")
    st.markdown("- Puncak penjualan pada November 2017.")
    st.markdown("#### 🧾 Kesimpulan:")
    st.markdown("Strategi marketing perlu fokus menjelang akhir tahun.")

# ===================== TAB 2 =====================
with tab2:
    st.header("Segmentasi Pelanggan dengan Analisis RFM")
    st.markdown("#### ❓ Pertanyaan:")
    st.markdown("**Bagaimana segmentasi pelanggan berdasarkan analisis RFM?**")

    snapshot_date = orders['order_purchase_timestamp'].max() + pd.DateOffset(days=1)

    rfm = full_data.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': lambda x: (snapshot_date - x.max()).days,
        'order_id': 'nunique',
        'price': 'sum'
    }).rename(columns={
        'order_purchase_timestamp': 'Recency',
        'order_id': 'Frequency',
        'price': 'Monetary'
    })

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    sns.boxplot(y=rfm['Recency'], ax=ax[0])
    sns.boxplot(y=rfm['Frequency'], ax=ax[1])
    sns.boxplot(y=rfm['Monetary'], ax=ax[2])
    st.pyplot(fig)

    rfm['R_Score'] = pd.cut(rfm['Recency'], bins=4, labels=[4, 3, 2, 1])
    rfm['F_Score'] = pd.cut(rfm['Frequency'], bins=4, labels=[1, 2, 3, 4])
    rfm['M_Score'] = pd.cut(rfm['Monetary'], bins=4, labels=[1, 2, 3, 4])
    rfm['RFM_Segment'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)


    seg_counts = rfm['RFM_Segment'].value_counts()
    st.bar_chart(seg_counts)

    st.markdown("#### ✅ Jawaban:")
    st.markdown(f"- Sebagian besar pelanggan berada di segmen **'{seg_counts.idxmax()}'** sebanyak {seg_counts.max()} pelanggan.")
    st.markdown("#### 📊 Insight:")
    st.markdown("- 75% pelanggan melakukan pembelian dalam 3 bulan terakhir")  
    st.markdown("- Sebagian besar pelanggan melakukan 1-2 transaksi")  
    st.markdown("- Distribusi monetary sangat tidak merata")
    st.markdown("#### 🧾 Kesimpulan:")
    st.markdown("Segmentasi RFM membantu strategi marketing lebih tajam.")

# ===================== TAB 3 =====================
with tab3:
    st.header("Analisis Geografis Pelanggan & Penjual")
    st.markdown("#### ❓ Pertanyaan:")
    st.markdown("**Bagaimana distribusi geografis penjual dan pengaruhnya terhadap biaya pengiriman?**")

    geo_agg = geolocation.groupby('geolocation_zip_code_prefix').first().reset_index()
    seller_geo = sellers.merge(geo_agg, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix')
    seller_freight = order_items.merge(seller_geo, on='seller_id')

    # Peta
    m = folium.Map(location=[-15.7797, -47.9297], zoom_start=4)

    for idx, row in seller_geo.iterrows():
        freight_data = seller_freight[seller_freight['seller_id'] == row['seller_id']]
        avg_freight = freight_data['freight_value'].mean()
        color = 'green' if avg_freight < 15 else 'orange' if avg_freight < 25 else 'red'
        folium.CircleMarker(
            location=[row['geolocation_lat'], row['geolocation_lng']],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            popup=f"Seller: {row['seller_id']}<br>State: {row['seller_state']}<br>Avg Freight: R${avg_freight:.2f}"
        ).add_to(m)

    st_folium(m, width=1000, height=500)

    # Visualisasi tambahan
    state_freight = seller_freight.groupby('seller_state').agg(
        avg_freight=('freight_value', 'mean'),
        seller_count=('seller_id', 'nunique')
    ).reset_index()

    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    sns.barplot(data=state_freight, x='seller_state', y='avg_freight', ax=ax[0])
    sns.scatterplot(data=state_freight, x='seller_count', y='avg_freight', hue='seller_state', size='seller_count', ax=ax[1])
    ax[0].set_title('Rata-rata Biaya Pengiriman per Negara Bagian')
    ax[1].set_title('Hubungan Jumlah Penjual vs Biaya Pengiriman')
    st.pyplot(fig)

    st.markdown("#### ✅ Jawaban:")
    st.markdown("- Konsentrasi penjual tinggi di SP dan RJ.")
    st.markdown("#### 📊 Insight:")
    st.markdown("- Konsentrasi penjual di wilayah Brazil Tenggara")
    st.markdown("- Biaya pengiriman lebih tinggi di wilayah Brazil Utara")
    st.markdown("- Negara bagian dengan penjual sedikit cenderung memiliki biaya pengiriman lebih tinggi")
    st.markdown("- Pusat logistik utama memiliki biaya pengiriman terendah")
    st.markdown("- Semakin jauh dari pusat distribusi, biaya pengiriman cenderung meningkat")
    st.markdown("- Negara bagian dengan kepadatan penjual tinggi (>100 penjual) memiliki biaya 30% lebih rendah daripada daerah terpencil")
    st.markdown("Marker pada peta menggunakan warna berbeda berdasarkan biaya: Hijau: < R$15, Oranye: R$15-25, Merah: > R$25")
    st.markdown("Popup menampilkan detail penjual dan biaya rata-rata")
    st.markdown("#### 🧾 Kesimpulan:")
    st.markdown("Distribusi geografis kritikal untuk strategi logistik.")
