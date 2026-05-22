# app.py

```python
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from scipy.optimize import newton

st.set_page_config(
    page_title="Portfolio Backtesting & Analytics",
    page_icon="📈",
    layout="wide"
)

# =========================
# Utility Functions
# =========================

def validate_ticker(ticker):
    try:
        data = yf.Ticker(ticker)
        hist = data.history(period="5d")
        return not hist.empty
    except:
        return False


def calculate_cagr(start_value, end_value, years):
    if start_value <= 0 or years <= 0:
        return np.nan
    return ((end_value / start_value) ** (1 / years)) - 1


def calculate_hpr(start_value, end_value):
    if start_value == 0:
        return np.nan
    return (end_value - start_value) / start_value


def calculate_annualized_return(total_return, years):
    if years <= 0:
        return np.nan
    return (1 + total_return) ** (1 / years) - 1


def xnpv(rate, cashflows):
    t0 = cashflows[0][0]
    return sum(
        cf / ((1 + rate) ** ((t - t0).days / 365.0))
        for t, cf in cashflows
    )


def xirr(cashflows):
    try:
        return newton(lambda r: xnpv(r, cashflows), 0.1)
    except:
        return np.nan


def calculate_real_return(nominal_return, inflation_rate):
    return ((1 + nominal_return) / (1 + inflation_rate)) - 1


def calculate_twr(portfolio_values):
    returns = portfolio_values.pct_change().dropna()
    return (1 + returns).prod() - 1


def calculate_rolling_returns(series, window=30):
    rolling = series.pct_change(periods=window)
    return rolling.dropna()


def fetch_price_data(tickers, start_date, end_date):
    try:
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False
        )['Close']

        if isinstance(data, pd.Series):
            data = data.to_frame()

        return data.ffill()

    except Exception as e:
        st.error(f"Error fetching market data: {e}")
        return pd.DataFrame()


# =========================
# Sidebar Inputs
# =========================

st.sidebar.title("📊 Portfolio Configuration")

num_positions = st.sidebar.number_input(
    "Number of Stock Positions",
    min_value=1,
    max_value=20,
    value=2
)

positions = []

for i in range(num_positions):
    st.sidebar.subheader(f"Position {i+1}")

    ticker = st.sidebar.text_input(
        f"Ticker Symbol {i+1}",
        value="AAPL" if i == 0 else "MSFT",
        key=f"ticker_{i}"
    ).upper()

    quantity = st.sidebar.number_input(
        f"Quantity {i+1}",
        min_value=1.0,
        value=10.0,
        step=1.0,
        key=f"quantity_{i}"
    )

    purchase_price = st.sidebar.number_input(
        f"Purchase Price {i+1}",
        min_value=0.01,
        value=100.0,
        key=f"purchase_price_{i}"
    )

    purchase_date = st.sidebar.date_input(
        f"Purchase Date {i+1}",
        value=datetime(2022, 1, 1),
        key=f"purchase_date_{i}"
    )

    positions.append({
        "ticker": ticker,
        "quantity": quantity,
        "purchase_price": purchase_price,
        "purchase_date": purchase_date
    })

portfolio_end_date = st.sidebar.date_input(
    "Portfolio End Date",
    value=datetime.today()
)

inflation_rate = st.sidebar.slider(
    "Annual Inflation Rate (%)",
    min_value=0.0,
    max_value=20.0,
    value=6.0,
    step=0.1
) / 100

run_analysis = st.sidebar.button("🚀 Run Portfolio Analysis")

# =========================
# Main Dashboard
# =========================

st.title("📈 Portfolio Backtesting & Performance Analytics")
st.markdown("Analyze stock portfolio performance with advanced financial metrics.")

if run_analysis:

    valid_positions = []

    for pos in positions:
        if validate_ticker(pos['ticker']):
            valid_positions.append(pos)
        else:
            st.warning(f"Invalid ticker skipped: {pos['ticker']}")

    if not valid_positions:
        st.error("No valid stock tickers found.")
        st.stop()

    tickers = list(set([p['ticker'] for p in valid_positions]))

    start_date = min([p['purchase_date'] for p in valid_positions])

    price_data = fetch_price_data(
        tickers,
        start_date,
        portfolio_end_date
    )

    if price_data.empty:
        st.error("Unable to retrieve market data.")
        st.stop()

    portfolio_df = pd.DataFrame(index=price_data.index)
    portfolio_df['Portfolio Value'] = 0

    transaction_summary = []
    cashflows = []

    for pos in valid_positions:
        ticker = pos['ticker']
        qty = pos['quantity']
        buy_price = pos['purchase_price']
        buy_date = pd.to_datetime(pos['purchase_date'])

        if ticker not in price_data.columns:
            continue

        stock_prices = price_data[ticker]
        stock_prices = stock_prices[stock_prices.index >= buy_date]

        stock_value = stock_prices * qty

        aligned_index = portfolio_df.index.intersection(stock_value.index)

        portfolio_df.loc[aligned_index, 'Portfolio Value'] += stock_value.loc[aligned_index]

        initial_investment = qty * buy_price
        current_value = stock_value.iloc[-1]

        years = (portfolio_end_date - pos['purchase_date']).days / 365.25

        absolute_return = calculate_hpr(initial_investment, current_value)
        cagr = calculate_cagr(initial_investment, current_value, years)
        annualized_return = calculate_annualized_return(absolute_return, years)
        real_return = calculate_real_return(cagr, inflation_rate)

        cashflows.append((buy_date, -initial_investment))
        cashflows.append((portfolio_df.index[-1], current_value))

        transaction_summary.append({
            'Ticker': ticker,
            'Quantity': qty,
            'Investment': round(initial_investment, 2),
            'Current Value': round(current_value, 2),
            'Absolute Return %': round(absolute_return * 100, 2),
            'CAGR %': round(cagr * 100, 2),
            'Annualized Return %': round(annualized_return * 100, 2),
            'Real Return %': round(real_return * 100, 2)
        })

    portfolio_df.dropna(inplace=True)

    if portfolio_df.empty:
        st.error("Portfolio calculation failed.")
        st.stop()

    portfolio_df['Daily Return'] = portfolio_df['Portfolio Value'].pct_change()
    portfolio_df['Cumulative Return'] = (
        (1 + portfolio_df['Daily Return']).cumprod() - 1
    )

    # =========================
    # Portfolio Level Metrics
    # =========================

    total_investment = sum([
        p['quantity'] * p['purchase_price']
        for p in valid_positions
    ])

    final_value = portfolio_df['Portfolio Value'].iloc[-1]

    total_years = (
        portfolio_df.index[-1] - portfolio_df.index[0]
    ).days / 365.25

    portfolio_cagr = calculate_cagr(
        total_investment,
        final_value,
        total_years
    )

    portfolio_hpr = calculate_hpr(
        total_investment,
        final_value
    )

    portfolio_xirr = xirr(cashflows)

    portfolio_twr = calculate_twr(
        portfolio_df['Portfolio Value']
    )

    portfolio_mwr = portfolio_xirr

    portfolio_real_return = calculate_real_return(
        portfolio_cagr,
        inflation_rate
    )

    rolling_returns = calculate_rolling_returns(
        portfolio_df['Portfolio Value'],
        window=30
    )

    # =========================
    # KPI Metrics
    # =========================

    st.header("📌 Portfolio Performance Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Portfolio Value",
            f"${final_value:,.2f}"
        )

    with col2:
        st.metric(
            "Absolute Return",
            f"{portfolio_hpr * 100:.2f}%"
        )

    with col3:
        st.metric(
            "CAGR",
            f"{portfolio_cagr * 100:.2f}%"
        )

    with col4:
        st.metric(
            "XIRR",
            f"{portfolio_xirr * 100:.2f}%"
            if not np.isnan(portfolio_xirr)
            else "N/A"
        )

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric(
            "TWR",
            f"{portfolio_twr * 100:.2f}%"
        )

    with col6:
        st.metric(
            "MWR",
            f"{portfolio_mwr * 100:.2f}%"
            if not np.isnan(portfolio_mwr)
            else "N/A"
        )

    with col7:
        st.metric(
            "Real Return",
            f"{portfolio_real_return * 100:.2f}%"
        )

    with col8:
        st.metric(
            "Inflation Rate",
            f"{inflation_rate * 100:.2f}%"
        )

    # =========================
    # Charts
    # =========================

    st.header("📊 Portfolio Value Over Time")

    fig_portfolio = px.line(
        portfolio_df,
        x=portfolio_df.index,
        y='Portfolio Value',
        title='Portfolio Value Trend'
    )

    fig_portfolio.update_layout(
        xaxis_title='Date',
        yaxis_title='Portfolio Value',
        hovermode='x unified'
    )

    st.plotly_chart(fig_portfolio, use_container_width=True)

    st.header("📈 Cumulative Returns")

    fig_returns = px.line(
        portfolio_df,
        x=portfolio_df.index,
        y='Cumulative Return',
        title='Cumulative Portfolio Returns'
    )

    fig_returns.update_layout(
        yaxis_tickformat='.2%'
    )

    st.plotly_chart(fig_returns, use_container_width=True)

    if not rolling_returns.empty:
        st.header("🔄 30-Day Rolling Returns")

        rolling_df = pd.DataFrame({
            'Date': rolling_returns.index,
            'Rolling Return': rolling_returns.values
        })

        fig_rolling = px.line(
            rolling_df,
            x='Date',
            y='Rolling Return',
            title='30-Day Rolling Returns'
        )

        fig_rolling.update_layout(
            yaxis_tickformat='.2%'
        )

        st.plotly_chart(fig_rolling, use_container_width=True)

    # =========================
    # Allocation Chart
    # =========================

    st.header("🧩 Current Portfolio Allocation")

    allocation_data = []

    for pos in valid_positions:
        ticker = pos['ticker']
        qty = pos['quantity']

        latest_price = price_data[ticker].iloc[-1]
        current_val = latest_price * qty

        allocation_data.append({
            'Ticker': ticker,
            'Value': current_val
        })

    allocation_df = pd.DataFrame(allocation_data)

    fig_pie = px.pie(
        allocation_df,
        values='Value',
        names='Ticker',
        title='Portfolio Allocation'
    )

    st.plotly_chart(fig_pie, use_container_width=True)

    # =========================
    # Transaction Table
    # =========================

    st.header("📋 Position Analytics")

    summary_df = pd.DataFrame(transaction_summary)

    st.dataframe(
        summary_df,
        use_container_width=True
    )

    # =========================
    # Download Section
    # =========================

    csv = summary_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label='⬇ Download Position Analytics CSV',
        data=csv,
        file_name='portfolio_analytics.csv',
        mime='text/csv'
    )

else:
    st.info(
        "Configure your portfolio from the sidebar and click 'Run Portfolio Analysis'."
    )

```

# requirements.txt

```txt
streamlit
pandas
numpy
yfinance
plotly
scipy
```

# Run Command

```bash
streamlit run app.py
```

# Production Enhancements Included

* Multiple stock portfolio support
* Real-time market data using yfinance
* Advanced financial metrics
* Interactive Plotly charts
* Rolling return analytics
* Inflation-adjusted real returns
* Downloadable analytics report
* Input validation and error handling
* Responsive dashboard layout
* Clean modular structure for scalability
