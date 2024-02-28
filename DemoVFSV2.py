#!/usr/bin/env python
# coding: utf-8

# In[24]:


import streamlit as st
import pandas as pd
import numpy as np
import random
import yfinance as yf
import plotly.graph_objects as go
import time
from PIL import Image
import io
from io import BytesIO
import requests
from bs4 import BeautifulSoup
import os


# In[25]:


# Path to the cache file
cache_file_path = "C:\\Users\\Julian Amuedo\\AppData\\Local\\py-yfinance\\tkr-tz.csv"

# Check if the cache file exists
if os.path.exists(cache_file_path):
    # If it exists, delete the file
    os.remove(cache_file_path)
    print("Cache file deleted successfully.")
else:
    print("Cache file does not exist.")


# In[26]:


# Function to display the loading page with the image
def show_loading_page():
    # Load the image from the raw URL
    image_url = "https://github.com/VieraFinancialSolutions/VFS-repositorio/raw/482c8c0e92cf54c7af779d6dc6bb86a00e52c97b/Grayscale%20on%20Transparent.png"
    response = requests.get(image_url)
    
    if response.status_code == 200:
        # Convert the image data to bytes
        img = Image.open(BytesIO(response.content))
        # Display the image
        st.image(img, use_column_width=True)
        # Centered loading message
        st.markdown("<h3 style='text-align: center;'>Por favor, espere mientras la aplicación está cargando...</h3>", unsafe_allow_html=True)
    else:
        st.error("Error loading image. Please try again later.")

# Function to simulate loading (replace this with your actual loading process)
def simulate_loading():
    time.sleep(8)  # Simulating a 5-second loading time

# Function to display the main app content
def show_main_app():
    st.title("Bienvenido a nuestra demo")
    st.write("Será guiado en el proceso para poder utilizar nuestra I.A")
    # Add more content as needed

# Display the loading page
show_loading_page()

# Simulate loading
simulate_loading()

# Clear the loading page
st.empty()

# Display the main app content
show_main_app()


# In[27]:


# Function to perform risk profile survey

def realizar_encuesta():
    st.title("Asesor financiero I.A de Viera Financial Solutions")
    st.write("Encuesta de Perfil de Riesgo del Inversor")
    st.write("Por favor, responde las siguientes preguntas:")

    edad = st.number_input("1. Edad: ", value=30, step=1)
    genero = st.radio("2. Género:", ('Hombre', 'Mujer', 'Otro'))

    st.write("\nSección 1: Nivel de Educación")
    educacion = st.selectbox("3. Nivel de Educación:", ('Escuela primaria completa', 'Secundaria completa', 'Universidad completa'))

    ocupacion = st.text_input("4. Ocupación: ")

    st.write("\nSección 2: Actitud hacia el Riesgo")
    actitud_riesgo = st.slider("4. ¿Cómo describirías tu actitud hacia el riesgo financiero? Siendo 1 'conservador' y 5 'riesgoso'", 1, 5, 3)

    st.write("\nSección 3: Tolerancia al Riesgo")
    tolerancia_riesgo = st.slider("5. ¿Cuán cómodo/a te sientes asumiendo riesgos en tus inversiones? Siendo 1 'nada cómodo' y 5 'muy cómodo'", 1, 5, 3)

    st.write("\nSección 4: Objetivos Financieros y Horizonte de Inversión")
    objetivo_inversion = st.selectbox("6. ¿Cuál es tu objetivo principal al invertir?", ('Preservar el capital', 'Crecimiento moderado', 'Crecimiento a largo plazo', 'Maximizar el rendimiento'))
    horizonte_inversion = st.slider("7. ¿Cuál es tu horizonte de inversión en años?", 1, 5, 3)

    experiencia_inversion = st.radio("8. ¿Has tenido experiencia previa invirtiendo en acciones, bonos u otros instrumentos financieros?", ('Sí', 'No'))

    return {
        "edad": edad,
        "genero": genero,
        "educacion": educacion,
        "ocupacion": ocupacion,
        "actitud_riesgo": actitud_riesgo,
        "tolerancia_riesgo": tolerancia_riesgo,
        "objetivo_inversion": objetivo_inversion,
        "horizonte_inversion": horizonte_inversion,
        "experiencia_inversion": experiencia_inversion
    }

# Function to calculate risk percentage based on survey responses
def calcular_porcentaje_riesgo(respuestas):
    puntaje_total = (
        int(respuestas["actitud_riesgo"]) +
        int(respuestas["tolerancia_riesgo"]) +
        OBJETIVOS_INVERSION[respuestas["objetivo_inversion"]] +  # Use the mapped value
        int(respuestas["horizonte_inversion"])
    )
    
    if puntaje_total <= 8:
        porcentaje_riesgo = 0
    elif puntaje_total <= 12:
        porcentaje_riesgo = 2
    elif puntaje_total <= 16:
        porcentaje_riesgo = 5
    elif puntaje_total <= 20:
        porcentaje_riesgo = 10
    else:
        porcentaje_riesgo = 15

    return porcentaje_riesgo

# Function to classify risk profile
def clasificar_perfil(respuestas):
    puntaje_total = (
        int(respuestas["actitud_riesgo"]) +
        int(respuestas["tolerancia_riesgo"]) +
        OBJETIVOS_INVERSION[respuestas["objetivo_inversion"]] +  # Use the mapped value
        int(respuestas["horizonte_inversion"])
    )

    if puntaje_total <= 8:
        return "Muy Conservador"
    elif puntaje_total <= 12:
        return "Conservador"
    elif puntaje_total <= 16:
        return "Moderado"
    elif puntaje_total <= 20:
        return "Agresivo"
    else:
        return "Muy Agresivo"

# Define a dictionary to map options to numerical values
OBJETIVOS_INVERSION = {
    'Preservar el capital': 1,
    'Crecimiento moderado': 2,
    'Crecimiento a largo plazo': 3,
    'Maximizar el rendimiento': 4
}

# Realizar la encuesta
respuestas = realizar_encuesta()

# Clasificar el perfil de riesgo y calcular el porcentaje de riesgo
perfil_riesgo = clasificar_perfil(respuestas)
porcentaje_riesgo = calcular_porcentaje_riesgo(respuestas)

# Display risk profile result
st.write("\n¡Gracias por completar la encuesta!")
st.write(f"Tu perfil de riesgo es: {perfil_riesgo}")
st.write(f"Nivel de riesgo medido en un intervalo de porcentaje: {porcentaje_riesgo}%")
        


# In[28]:


# Fetch the HTML content from the Wikipedia page
html_data = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies').text

# Parse the HTML content using BeautifulSoup
beautiful_soup = BeautifulSoup(html_data, "html.parser")

# Find all tables on the page
tables = beautiful_soup.find_all('table')

# Create an empty DataFrame to store the extracted data
S_P_500_companies = pd.DataFrame(columns=["Symbol","Security","Sector","Sub-Industry","Headquarters Location","Date first added","CIK","Founded"])

# Iterate through the rows of the first table
for row in tables[0].tbody.find_all("tr"):
    col = row.find_all("td")
    if col:  # Check if col is not empty
        # Extract data from the columns
        Symbol = col[0].text.strip().replace('\n','')
        Security = col[1].text.strip().replace('\n','')
        Sector = col[3].text.strip().replace('\n','')
        Sub_Industry = col[4].text.strip().replace('\n','')
        Headquarters_Location = col[5].text.strip().replace('\n','')
        Date_first = col[6].text.strip().replace('\n','')
        CIK = col[7].text.strip().replace('\n','')
        Founded = ""  # Assuming founded is not always available
        if len(col) > 8:  # Check if there's at least 9 columns
            Founded = col[8].text.strip().replace('\n','')
        # Append row to DataFrame
        row_data = {"Symbol":Symbol, "Security":Security, "Sector":Sector, "Sub-Industry":Sub_Industry,
                    "Headquarters Location":Headquarters_Location,"Date first added":Date_first,"CIK":CIK,"Founded":Founded}
        S_P_500_companies = pd.concat([S_P_500_companies, pd.DataFrame([row_data])], ignore_index=True)

# Display the DataFrame using Streamlit
st.write(S_P_500_companies)

# Get the list of symbols
Symbols = S_P_500_companies['Symbol'].tolist()
st.write("Total de activos enlistados:", len(Symbols))


# In[29]:


# Group companies by sector and obtain the corresponding symbols
tickers_by_sector = S_P_500_companies.groupby("Sector")["Symbol"].apply(list).to_dict()

# Create a list containing all symbols from all sectors
all_sectors_tickers = sum(tickers_by_sector.values(), [])

# Symbols that generated exceptions during the download
excluded_symbols = ['DRE', 'WLTW', 'INFO', 'FB', 'PBCT', 'FBHS', 'NLOK', 'PKI', 'VIAC', 'DISCA',
                    'KSU', 'SIVB', 'BRK.B', 'BLL', 'CERN', 'XLNX', 'DISCK', 'CTXS', 'TWTR',
                    'ANTM', 'FRC', 'NLSN', 'BF.B', 'MCD', 'KR', 'RE', 'CDAY', 'OTIS', 'OGN', 'FISV']

# Update the ticker_all_sectors object, removing excluded symbols
ticker_all_sectors = [ticker for ticker in all_sectors_tickers if ticker not in excluded_symbols]

# Display the dimensions of the updated ticker
st.write("Número de acciones vigentes:", len(ticker_all_sectors))

# Function to generate random symbols
def generate_random_symbols(num_symbols):
    random_symbols = random.sample(ticker_all_sectors, num_symbols)
    return random_symbols

# Download price data for a list of symbols
def download_price_data(symbols, start_date, end_date):
    price_data = yf.download(symbols, start=start_date, end=end_date)['Adj Close']
    return price_data

# Function to calculate portfolio metrics
def calculate_portfolio_metrics(weights, mean_returns, cov_matrix):
    # Calculate annualized portfolio returns
    port_returns = np.sum(weights * mean_returns) * 252

    # Calculate portfolio risk (annualized standard deviation)
    port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

    # Calculate Sharpe ratio
    sharpe_ratio = port_returns / port_risk

    return port_returns, port_risk, sharpe_ratio

# Function to sum portfolio weights
def sum_portfolio_weights(portfolio_weights):
    return portfolio_weights.sum()

# Function to generate random portfolio weights
def generate_weights(num_assets):
    weights = np.random.uniform(size=num_assets)
    return weights / np.sum(weights)

# Function to generate a custom portfolio
def generate_portfolio(target_risk, symbols, mean_returns, cov_matrix):
    num_assets = len(symbols)
    weights = generate_weights(num_assets)

    portfolio_return = np.sum(weights * mean_returns) * 252
    portfolio_risk = np.sqrt(np.dot(np.dot(weights.T, cov_matrix), weights)) * np.sqrt(252)
    adjustment_factor = target_risk / portfolio_risk
    adjusted_weights = weights * adjustment_factor

    portfolio = {symbols[i]: adjusted_weights[i] for i in range(num_assets)}
    return portfolio

def create_custom_portfolio(num_symbols, mean_ret, cov_mat, custom_risk_level):
    tickers = random.sample(ticker_all_sectors, num_symbols)
    random_symbols = random.sample(tickers, num_symbols)
    # Calculate the portfolio with the custom risk level
    portfolio = generate_portfolio(custom_risk_level, random_symbols, mean_ret, cov_mat)
    return portfolio

# Function to calculate portfolio volatility
def calculate_portfolio_volatility(portfolio, cov_matrix):
    weights = np.array(list(portfolio.values()))
    return np.sqrt(np.dot(np.dot(weights.T, cov_matrix), weights)) * np.sqrt(252)

# Function to extract tickers from a portfolio
def extract_tickers(portfolio):
    tickers = list(portfolio.keys())
    return tickers

# Function to fetch historical prices for a list of tickers
def fetch_prices(tickers, start_date, end_date):
    try:
        price_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        return price_data
    except Exception as e:
        st.error(f"Error fetching price data: {str(e)}")
        return None

def generate_and_download_price_data(num_symbols):
    while True:
        # Generate random symbols
        random_symbols = generate_random_symbols(num_symbols)
        # Download price data
        price_data = download_price_data(random_symbols, '2016-01-01', '2024-02-18')
        # Check for missing symbols
        missing_symbols = price_data.columns[price_data.isnull().any()].tolist()
        if not missing_symbols:
            return price_data, random_symbols
        
def plot_portfolio_bar_chart(portfolio, title):
        assets = list(portfolio.keys())
        weights = np.array(list(portfolio.values()))
        normalized_weights = weights / np.sum(weights)
        weights_percentages = normalized_weights * 100

        fig = go.Figure(data=go.Bar(
            x=assets,
            y=weights_percentages,
            text=[f'{weight:.2f}%' for weight in weights_percentages],
            textposition='auto'
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Acciones',
            yaxis_title='Participación porcentual por acción (%)',
        )

        st.plotly_chart(fig)

# Streamlit app
def main():
    st.title("¡Vamos a crear tu cartera personalizada en base a tu perfil!")

    # Generate and download price data for 10 symbols
    price_data_10, random_symbols_10 = generate_and_download_price_data(10)

    # Generate and download price data for 20 symbols
    price_data_20, random_symbols_20 = generate_and_download_price_data(20)

    # Generate and download price data for 30 symbols
    price_data_30, random_symbols_30 = generate_and_download_price_data(30)

    # Calculamos los retornos logarítmicos diarios
    log_ret_10 = np.log(price_data_10 / price_data_10.shift(1))
    log_ret_20 = np.log(price_data_20 / price_data_20.shift(1))
    log_ret_30 = np.log(price_data_30 / price_data_30.shift(1))
    
    print("Price data downloaded successfully")

    # Calculamos los retornos promedio
    mean_ret_10 = log_ret_10.mean()
    mean_ret_20 = log_ret_20.mean()
    mean_ret_30 = log_ret_30.mean()

    # Calculamos la matriz de covarianza
    cov_mat_10 = log_ret_10.cov() * 252
    cov_mat_20 = log_ret_20.cov() * 252
    cov_mat_30 = log_ret_30.cov() * 252

    
    # Generate random weights for portfolio
    np.random.seed(0)
    wts_10 = np.random.uniform(size=len(price_data_10.columns))
    wts_10 /= np.sum(wts_10)

    wts_20 = np.random.uniform(size=len(price_data_20.columns))
    wts_20 /= np.sum(wts_20)

    wts_30 = np.random.uniform(size=len(price_data_30.columns))
    wts_30 /= np.sum(wts_30)

    # Calculate portfolio metrics
    port_returns_10, port_risk_10, sharpe_ratio_10 = calculate_portfolio_metrics(wts_10, mean_ret_10, cov_mat_10)
    port_returns_20, port_risk_20, sharpe_ratio_20 = calculate_portfolio_metrics(wts_20, mean_ret_20, cov_mat_20)
    port_returns_30, port_risk_30, sharpe_ratio_30 = calculate_portfolio_metrics(wts_30, mean_ret_30, cov_mat_30)

    
     # Generate 7000 random portfolios for each case
    num_port = 7000
    all_wts_10 = np.zeros((num_port, len(price_data_10.columns)))
    all_wts_20 = np.zeros((num_port, len(price_data_20.columns)))
    all_wts_30 = np.zeros((num_port, len(price_data_30.columns)))

    port_returns_10 = np.zeros(num_port)
    port_returns_20 = np.zeros(num_port)
    port_returns_30 = np.zeros(num_port)

    port_risk_10 = np.zeros(num_port)
    port_risk_20 = np.zeros(num_port)
    port_risk_30 = np.zeros(num_port)

    sharpe_ratio_10 = np.zeros(num_port)
    sharpe_ratio_20 = np.zeros(num_port)
    sharpe_ratio_30 = np.zeros(num_port)

    for i in range(num_port):
        wts_10 = np.random.uniform(size=len(price_data_10.columns))
        wts_10 /= np.sum(wts_10)
        all_wts_10[i] = wts_10

        wts_20 = np.random.uniform(size=len(price_data_20.columns))
        wts_20 /= np.sum(wts_20)
        all_wts_20[i] = wts_20

        wts_30 = np.random.uniform(size=len(price_data_30.columns))
        wts_30 /= np.sum(wts_30)
        all_wts_30[i] = wts_30

        port_returns_10[i] = np.sum(wts_10 * mean_ret_10) * 252
        port_returns_20[i] = np.sum(wts_20 * mean_ret_20) * 252
        port_returns_30[i] = np.sum(wts_30 * mean_ret_30) * 252

        port_risk_10[i] = np.sqrt(np.dot(np.dot(wts_10.T, cov_mat_10), wts_10)) * np.sqrt(252)
        port_risk_20[i] = np.sqrt(np.dot(np.dot(wts_20.T, cov_mat_20), wts_20)) * np.sqrt(252)
        port_risk_30[i] = np.sqrt(np.dot(np.dot(wts_30.T, cov_mat_30), wts_30)) * np.sqrt(252)


        sharpe_ratio_10[i] = port_returns_10[i] / port_risk_10[i]
        sharpe_ratio_20[i] = port_returns_20[i] / port_risk_20[i]
        sharpe_ratio_30[i] = port_returns_30[i] / port_risk_30[i]
        
        # Create data frames to store portfolio values for each case
    portfolio_values_10 = pd.DataFrame({'Retorno': port_returns_10, 'Riesgo': port_risk_10, 'Ratio sharpe': sharpe_ratio_10})
    portfolio_values_20 = pd.DataFrame({'Retorno': port_returns_20, 'Riesgo': port_risk_20, 'Ratio sharpe': sharpe_ratio_20})
    portfolio_values_30 = pd.DataFrame({'Retorno': port_returns_30, 'Riesgo': port_risk_30, 'Ratio sharpe': sharpe_ratio_30})
    
    print("Length of all_wts_10:", len(all_wts_10))
    print("Length of random_symbols_10:", len(random_symbols_10))
    
    print("Length of all_wts_20:", len(all_wts_20))
    print("Length of random_symbols_20:", len(random_symbols_20))
    
    print("Length of all_wts_30:", len(all_wts_30))
    print("Length of random_symbols_30:", len(random_symbols_30))

    # Convert weights to data frames and change column names
    all_wts_df_10 = pd.DataFrame(all_wts_10, columns=random_symbols_10)
    all_wts_df_20 = pd.DataFrame(all_wts_20, columns=random_symbols_20)
    all_wts_df_30 = pd.DataFrame(all_wts_30, columns=random_symbols_30)
    
    print("Length of all_wts_df_10:", len(all_wts_df_10))
    print("Length of all_wts_df_20:", len(all_wts_df_20))
    print("Length of all_wts_df_30:", len(all_wts_df_30))
    
    # Get the minimum variance portfolio and tangency portfolio for each case
    min_var_10 = portfolio_values_10[portfolio_values_10['Riesgo'] == portfolio_values_10['Riesgo'].min()]
    max_sr_10 = portfolio_values_10[portfolio_values_10['Ratio sharpe'] == portfolio_values_10['Ratio sharpe'].max()]

    min_var_20 = portfolio_values_20[portfolio_values_20['Riesgo'] == portfolio_values_20['Riesgo'].min()]
    max_sr_20 = portfolio_values_20[portfolio_values_20['Ratio sharpe'] == portfolio_values_20['Ratio sharpe'].max()]

    min_var_30 = portfolio_values_30[portfolio_values_30['Riesgo'] == portfolio_values_30['Riesgo'].min()]
    max_sr_30 = portfolio_values_30[portfolio_values_30['Ratio sharpe'] == portfolio_values_30['Ratio sharpe'].max()]

    # Get the weights of the minimum variance portfolio for each case
    weights_min_var_10 = all_wts_df_10.iloc[min_var_10.index[0]].values
    weights_min_var_20 = all_wts_df_20.iloc[min_var_20.index[0]].values
    weights_min_var_30 = all_wts_df_30.iloc[min_var_30.index[0]].values

    # Get the weights of the tangency portfolio for each case
    weights_max_sr_10 = all_wts_df_10.iloc[max_sr_10.index[0]].values
    weights_max_sr_20 = all_wts_df_20.iloc[max_sr_20.index[0]].values
    weights_max_sr_30 = all_wts_df_30.iloc[max_sr_30.index[0]].values

    # Get the weights of the minimum variance portfolio in percentages (for 10 symbols)
    weights_min_var_perc_10 = weights_min_var_10 * 100

    # Get the weights of the tangency portfolio in percentages (for 10 symbols)
    weights_max_sr_perc_10 = weights_max_sr_10 * 100

    # Get the weights of the minimum variance portfolio in percentages (for 20 symbols)
    weights_min_var_perc_20 = weights_min_var_20 * 100

    # Get the weights of the tangency portfolio in percentages (for 20 symbols)
    weights_max_sr_perc_20 = weights_max_sr_20 * 100

    # Get the weights of the minimum variance portfolio in percentages (for 30 symbols)
    weights_min_var_perc_30 = weights_min_var_30 * 100

    # Get the weights of the tangency portfolio in percentages (for 30 symbols)
    weights_max_sr_perc_30 = weights_max_sr_30 * 100

    # Calculate weights for minimum variance and tangency portfolios
    weights_min_var_10 = all_wts_df_10.iloc[min_var_10.index[0]].values
    total_weights_min_var_10 = sum_portfolio_weights(weights_min_var_10)
    adjusted_weights_min_var_10 = weights_min_var_10 / total_weights_min_var_10

    total_weights_max_sr_10 = weights_max_sr_10.sum()
    adjusted_weights_max_sr_10 = weights_max_sr_10 / total_weights_max_sr_10

    # Calculate sum of weights for minimum variance and tangency portfolios
    suma_min_var_10 = sum_portfolio_weights(weights_min_var_10)
    suma_max_sr_10 = sum_portfolio_weights(weights_max_sr_10)

    # Generate a custom portfolio for the 10-asset case
    portafolio_10 = generate_portfolio(0.10, random_symbols_10, mean_ret_10, cov_mat_10)
    volatilidad_portafolio_10 = calculate_portfolio_volatility(portafolio_10, cov_mat_10)
    

    # Calculate weights for minimum variance and tangency portfolios
    weights_min_var_20 = all_wts_df_20.iloc[min_var_20.index[0]].values
    total_weights_min_var_20 = sum_portfolio_weights(weights_min_var_20)
    adjusted_weights_min_var_20 = weights_min_var_20 / total_weights_min_var_20

    total_weights_max_sr_20 = weights_max_sr_20.sum()
    adjusted_weights_max_sr_20 = weights_max_sr_20 / total_weights_max_sr_20

    # Calculate sum of weights for minimum variance and tangency portfolios
    suma_min_var_20 = sum_portfolio_weights(weights_min_var_20)
    suma_max_sr_20 = sum_portfolio_weights(weights_max_sr_20)

    # Generate a custom portfolio for the 20-asset case
    portafolio_20 = generate_portfolio(0.10, random_symbols_20, mean_ret_20, cov_mat_20)
    volatilidad_portafolio_20 = calculate_portfolio_volatility(portafolio_20, cov_mat_20)
    
    # Display statistics for the 30-asset portfolio
    #st.subheader("Portfolio de 30 activos")
    #st.write("Calculando estadísticos para portafolio de 30 activos")

    # Calculate weights for minimum variance and tangency portfolios
    weights_min_var_30 = all_wts_df_30.iloc[min_var_30.index[0]].values
    total_weights_min_var_30 = sum_portfolio_weights(weights_min_var_30)
    adjusted_weights_min_var_30 = weights_min_var_30 / total_weights_min_var_30

    total_weights_max_sr_30 = weights_max_sr_30.sum()
    adjusted_weights_max_sr_30 = weights_max_sr_30 / total_weights_max_sr_30

    # Calculate sum of weights for minimum variance and tangency portfolios
    suma_min_var_30 = sum_portfolio_weights(weights_min_var_30)
    suma_max_sr_30 = sum_portfolio_weights(weights_max_sr_30)

    # Generate a custom portfolio for the 20-asset case
    portafolio_30 = generate_portfolio(0.10, random_symbols_30, mean_ret_30, cov_mat_30)
    volatilidad_portafolio_30 = calculate_portfolio_volatility(portafolio_30, cov_mat_30)
    
    # Display section for custom portfolios
    #st.header("Portafolios personalizados")
    
    # Define the date range
    start_date = '2018-01-01'
    end_date = '2024-01-01'
    
    # Create custom portfolios with the obtained risk percentage
    
    custom_portfolio_10_symbols = create_custom_portfolio(10, mean_ret_10, cov_mat_10, porcentaje_riesgo)
    custom_portfolio_20_symbols = create_custom_portfolio(20, mean_ret_20, cov_mat_20, porcentaje_riesgo)
    custom_portfolio_30_symbols = create_custom_portfolio(30, mean_ret_30, cov_mat_30, porcentaje_riesgo)
            
    # Fetch prices for each customized portfolio
    price_data_10_symbols_personalizado = fetch_prices(extract_tickers(custom_portfolio_10_symbols), start_date, end_date)
    price_data_20_symbols_personalizado = fetch_prices(extract_tickers(custom_portfolio_20_symbols), start_date, end_date)
    price_data_30_symbols_personalizado = fetch_prices(extract_tickers(custom_portfolio_30_symbols), start_date, end_date)

    print("Price data for personalized portfolio downloaded successfully")

    # Display custom portfolios as interactive bar charts
    st.subheader("Portafolios personalizados según nivel de riesgo del cliente")
    plot_portfolio_bar_chart(custom_portfolio_10_symbols, "Portafolio personalizado (10 acciones)")

    
    plot_portfolio_bar_chart(custom_portfolio_20_symbols, "Portafolio personalizado (20 acciones)")

    
    plot_portfolio_bar_chart(custom_portfolio_30_symbols, "Portafolio personalizado (30 acciones)")
    
    print("Plots displayed successfully")
    
    # Extract tickers from the customized portfolios
    tickers_10_symbols_personalizado = extract_tickers(custom_portfolio_10_symbols)
    tickers_20_symbols_personalizado = extract_tickers(custom_portfolio_20_symbols)
    tickers_30_symbols_personalizado = extract_tickers(custom_portfolio_30_symbols)

    # Convert portfolios to pandas DataFrame for better display
    df_10 = pd.DataFrame.from_dict(custom_portfolio_10_symbols, orient='index', columns=['Weight'])
    df_20 = pd.DataFrame.from_dict(custom_portfolio_20_symbols, orient='index', columns=['Weight'])
    df_30 = pd.DataFrame.from_dict(custom_portfolio_30_symbols, orient='index', columns=['Weight'])


    # Calculate the returns of each customized portfolio
    returns_10_symbols_personalizado = (price_data_10_symbols_personalizado.pct_change() * df_10['Weight']).sum(axis=1)
    returns_20_symbols_personalizado = (price_data_20_symbols_personalizado.pct_change() * df_20['Weight']).sum(axis=1)
    returns_30_symbols_personalizado = (price_data_30_symbols_personalizado.pct_change() * df_30['Weight']).sum(axis=1)
    
    returns_10_symbols_personalizado = returns_10_symbols_personalizado * 100
    returns_20_symbols_personalizado = returns_20_symbols_personalizado * 100
    returns_30_symbols_personalizado = returns_30_symbols_personalizado * 100
    

    # Calculate the cumulative return of each customized portfolio
    cumulative_return_10_symbols_personalizado = (returns_10_symbols_personalizado + 1).cumprod()[-1] - 1
    cumulative_return_20_symbols_personalizado = (returns_20_symbols_personalizado + 1).cumprod()[-1] - 1
    cumulative_return_30_symbols_personalizado = (returns_30_symbols_personalizado + 1).cumprod()[-1] - 1

    # Create traces for each portfolio
    
    cumulative_returns_10 = returns_10_symbols_personalizado.cumsum()
    cumulative_returns_20 = returns_20_symbols_personalizado.cumsum()
    cumulative_returns_30 = returns_30_symbols_personalizado.cumsum()
    
    
    trace_10 = go.Scatter(x=cumulative_returns_10.index, y=cumulative_returns_10, mode='lines', name='Portafolio (10 acciones)', line=dict(color='rgb(31, 119, 180)', width=2))
    trace_20 = go.Scatter(x=cumulative_returns_20.index, y=cumulative_returns_20, mode='lines', name='Portafolio (20 acciones)', line=dict(color='rgb(255, 127, 14)', width=2))
    trace_30 = go.Scatter(x=cumulative_returns_30.index, y=cumulative_returns_30, mode='lines', name='Portafolio (30 acciones)', line=dict(color='rgb(44, 160, 44)', width=2))

    # Create data list
    data = [trace_10, trace_20, trace_30]
    
    st.title("Ahora lo importante, ¿cuánto habría rendido tu capital si lo invertías con nosotros en los últimos 5 años?")
    st.subheader("¡Abajo te mostramos eso!")

    # Define layout
    layout = go.Layout(
        title=dict(text='Retorno acumulado por portafolio', x=0.3, y=0.9, font=dict(size=24, color='rgb(255, 255, 255)')),
        xaxis=dict(title='Fecha', tickformat='%Y-%m-%d', showgrid=False, linecolor='rgb(255, 255, 255)', tickfont=dict(color='rgb(255, 255, 255)')),
        yaxis=dict(title='Retorno acumulado en porcentaje %', showgrid=False, linecolor='rgb(255, 255, 255)', tickfont=dict(color='rgb(255, 255, 255)')),
        template='plotly_dark',
    )

    # Create figure
    fig = go.Figure(data=data, layout=layout)

    # Update figure aesthetics
    fig.update_traces(marker=dict(size=5))

    # Display the figure using Streamlit
    st.plotly_chart(fig)

    print("cumulative returns portfolios plots displayed successfully")
        
if __name__ == "__main__":
    main()


# In[ ]:




