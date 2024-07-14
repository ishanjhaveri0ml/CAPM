import datetime
import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
import numpy as np

# Configure the Streamlit page
st.set_page_config(page_title="CAPM",
                   page_icon="chart_with_upwards_trend",
                   layout='wide')

st.title("Capital Asset Pricing Model (CAPM)")


col1, col2 = st.columns([1, 1])
with col1:
    stocks_list = st.multiselect("Choose 4 Stocks", ('TSLA', 'AAPL', 'NFLX', 'MSFT', 'MGM', 'AMZN', 'NVDA', 'GOOGL'))
with col2:
    years = st.number_input("Number of Years", 1, 10)


end_date = datetime.date.today()
start_date = datetime.date(datetime.date.today().year - years, datetime.date.today().month, datetime.date.today().day)


SP500 = web.DataReader('sp500', 'fred', start_date, end_date)
SP500.dropna(inplace=True)  # Drop any rows with missing data
SP500['Date'] = pd.to_datetime(SP500.index)
SP500.set_index('Date', inplace=True)


stocks_df = pd.DataFrame()


for stock in stocks_list:
    data = yf.download(stock, start=start_date, end=end_date)
    data['Date'] = pd.to_datetime(data.index)
    data.set_index('Date', inplace=True)
    stocks_df[stock] = data['Close']

stocks_returns = stocks_df.pct_change().dropna()
sp500_returns = SP500['sp500'].pct_change().dropna()


combined_df = stocks_returns.join(sp500_returns, how='inner')


def calculate_capm(stock_returns, market_returns):
    cov_matrix = np.cov(stock_returns, market_returns)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    rf = 0.01  # Assume a risk-free rate of 1%
    rm = market_returns.mean() * 252  # Annualize market return
    expected_return = rf + beta * (rm - rf)
    return beta, expected_return


results = []
for stock in stocks_list:
    beta, expected_return = calculate_capm(combined_df[stock], combined_df['sp500'])
    results.append({
        "Stock": stock,
        "Beta": beta,
        "Expected Return": expected_return
    })

results_df = pd.DataFrame(results)
st.write("CAPM Results", results_df)

# Display plots
st.subheader("Stock Prices Over Time")
st.line_chart(stocks_df)

st.subheader("Daily Returns")
st.line_chart(stocks_returns)

# Display correlation matrix
st.subheader("Correlation Matrix")
correlation_matrix = stocks_returns.corr()
st.write(correlation_matrix)
