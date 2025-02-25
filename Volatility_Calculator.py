import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import numpy as np
import math
import time  # for the 7-second delay

# --------------------------------------------
#  DARK MODE CSS OVERRIDE
# --------------------------------------------
st.set_page_config(page_title="Options Screener", layout="wide")
st.markdown(
    """
    <style>
    /* Force the page background to dark, and text to light */
    body, .css-18e3th9, .css-1gk4psh {
        background-color: #0E1117 !important;
        color: #FFFFFF !important;
    }
    /* Streamlit's main block background color */
    .css-1offfwp {
        background-color: #0E1117 !important;
    }
    /* Table text color */
    .css-1ex1afd tr, .css-1ex1afd td, .css-1ex1afd th {
        color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------
#  UTILITY FUNCTIONS
# --------------------------------------------
def filter_dates(dates):
    today = datetime.today().date()
    cutoff_date = today + timedelta(days=45)
    
    sorted_dates = sorted(datetime.strptime(date, "%Y-%m-%d").date() for date in dates)
    arr = []
    for i, date in enumerate(sorted_dates):
        if date >= cutoff_date:
            arr = [d.strftime("%Y-%m-%d") for d in sorted_dates[:i+1]]
            break
    
    if len(arr) > 0:
        # If the first found date is actually today, skip it.
        if arr[0] == today.strftime("%Y-%m-%d"):
            return arr[1:]
        return arr

    raise ValueError("No date 45 days or more in the future found.")

def yang_zhang(price_data, window=30, trading_periods=252, return_last_only=True):
    log_ho = (price_data['High'] / price_data['Open']).apply(np.log)
    log_lo = (price_data['Low'] / price_data['Open']).apply(np.log)
    log_co = (price_data['Close'] / price_data['Open']).apply(np.log)
    
    log_oc = (price_data['Open'] / price_data['Close'].shift(1)).apply(np.log)
    log_oc_sq = log_oc**2
    
    log_cc = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)
    log_cc_sq = log_cc**2
    
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    
    close_vol = log_cc_sq.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))
    open_vol = log_oc_sq.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))
    window_rs = rs.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))
    
    k = 0.34 / (1.34 + ((window + 1) / (window - 1)))
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(trading_periods)
    
    if return_last_only:
        return result.iloc[-1]
    else:
        return result.dropna()

def build_term_structure(days, ivs):
    days = np.array(days)
    ivs = np.array(ivs)
    
    sort_idx = days.argsort()
    days = days[sort_idx]
    ivs = ivs[sort_idx]
    
    spline = interp1d(days, ivs, kind='linear', fill_value="extrapolate")
    
    def term_spline(dte):
        if dte < days[0]:
            return ivs[0]
        elif dte > days[-1]:
            return ivs[-1]
        else:
            return float(spline(dte))
    
    return term_spline

def get_current_price(ticker_obj):
    todays_data = ticker_obj.history(period='1d')
    return todays_data['Close'][0] if not todays_data.empty else None

def compute_recommendation(ticker):
    """
    Compute all metrics for a single ticker, returning either a result dict or error string.
    Includes earliest-expiration ATM bid/ask + today's volume for liquidity checks.
    """
    try:
        ticker = ticker.strip().upper()
        if not ticker:
            return "No stock symbol provided."
        
        stock = yf.Ticker(ticker)
        
        # Check if options exist
        if not stock.options:
            return f"Error: No options found for stock symbol '{ticker}'."
        
        # Filter expiration dates
        exp_dates = list(stock.options)
        try:
            exp_dates = filter_dates(exp_dates)
        except ValueError:
            return "Error: Not enough option data (no suitable expiration dates)."
        
        # Retrieve option chains
        options_chains = {}
        for exp_date in exp_dates:
            options_chains[exp_date] = stock.option_chain(exp_date)
        
        # Get underlying price
        underlying_price = get_current_price(stock)
        if underlying_price is None:
            return "Error: Unable to retrieve underlying stock price."
        
        # Calculate ATM IV & find first straddle price
        atm_iv = {}
        straddle = None
        
        # We'll store the ATM call/put info from the earliest expiration
        first_call_bid, first_call_ask = None, None
        first_put_bid, first_put_ask = None, None
        # <<< NEW: We'll store today's volume
        first_call_volume = None
        first_put_volume = None
        
        i = 0
        for exp_date, chain in options_chains.items():
            calls = chain.calls
            puts = chain.puts
            if calls.empty or puts.empty:
                continue
            
            # Closest strike to underlying price
            call_diffs = (calls['strike'] - underlying_price).abs()
            call_idx = call_diffs.idxmin()
            call_iv = calls.loc[call_idx, 'impliedVolatility']
            
            put_diffs = (puts['strike'] - underlying_price).abs()
            put_idx = put_diffs.idxmin()
            put_iv = puts.loc[put_idx, 'impliedVolatility']
            
            atm_iv_value = (call_iv + put_iv) / 2.0
            atm_iv[exp_date] = atm_iv_value
            
            # For the first expiration chain, capture straddle & volumes
            if i == 0:
                first_call_bid = calls.loc[call_idx, 'bid']
                first_call_ask = calls.loc[call_idx, 'ask']
                first_call_volume = calls.loc[call_idx, 'volume']  # <<< NEW
                
                first_put_bid = puts.loc[put_idx, 'bid']
                first_put_ask = puts.loc[put_idx, 'ask']
                first_put_volume = puts.loc[put_idx, 'volume']     # <<< NEW
                
                # Calculate straddle mid
                call_mid = (first_call_bid + first_call_ask)/2.0 if (first_call_bid is not None and first_call_ask is not None) else None
                put_mid = (first_put_bid + first_put_ask)/2.0 if (first_put_bid is not None and first_put_ask is not None) else None
                if call_mid is not None and put_mid is not None:
                    straddle = call_mid + put_mid
            
            i += 1
        
        if not atm_iv:
            return "Error: Could not determine ATM IV for any expiration dates."
        
        # Build term structure
        today = datetime.today().date()
        dtes = []
        ivs = []
        for exp_date, iv in atm_iv.items():
            exp_date_obj = datetime.strptime(exp_date, "%Y-%m-%d").date()
            days_to_expiry = (exp_date_obj - today).days
            dtes.append(days_to_expiry)
            ivs.append(iv)
        
        term_spline = build_term_structure(dtes, ivs)
        
        # Compute slope between earliest DTE and 45 DTE
        first_dte = dtes[0]
        if first_dte >= 45:
            ts_slope_0_45 = 0.0
        else:
            ts_slope_0_45 = (term_spline(45) - term_spline(first_dte)) / (45 - first_dte)
        
        # Compute IV30 / RV30
        price_history = stock.history(period='3mo')
        if len(price_history) < 30:
            return "Error: Not enough historical data to compute Yang-Zhang volatility."
        
        iv30 = term_spline(30)
        rv30 = yang_zhang(price_history)
        if rv30 == 0:
            iv30_rv30 = float('inf')
        else:
            iv30_rv30 = iv30 / rv30
        
        # 30-day average volume (equities volume)
        avg_volume = price_history['Volume'].rolling(30).mean().dropna().iloc[-1]
        
        # Expected move from straddle
        if straddle and underlying_price != 0:
            expected_move = round(straddle / underlying_price * 100, 2)
            expected_move_str = f"{expected_move}%"
        else:
            expected_move = None
            expected_move_str = None
        
        return {
            'ticker': ticker,
            'share_price': underlying_price,
            'avg_volume_pass': avg_volume >= 1500000,
            'iv30_rv30_pass': iv30_rv30 >= 1.25,
            'ts_slope_0_45_pass': ts_slope_0_45 <= -0.00406,
            'expected_move': expected_move,
            'expected_move_str': expected_move_str,
            
            # Bid/Ask
            'atm_call_bid': first_call_bid,
            'atm_call_ask': first_call_ask,
            'atm_put_bid': first_put_bid,
            'atm_put_ask': first_put_ask,
            
            # <<< NEW: Volume
            'atm_call_volume': first_call_volume,
            'atm_put_volume': first_put_volume
        }

    except Exception as e:
        return f"Error: {str(e)}"

def main():
    st.title("Options Screener (Rate Limited)")
    st.write("Enter your stock tickers and run the screener. Because of yfinance limitations, "
             "we'll process them one at a time with a small delay.")

    # --------------------------------------------
    # CLEANING SECTION
    # --------------------------------------------
    st.subheader("Ticker Cleaning Utility")
    st.write("Paste text containing tickers in quotes (e.g. `'AMC', 'CART'`) and click **Clean Tickers** to remove quotes.")
    
    raw_tickers_text = st.text_area("Paste tickers with quotes here:", 
                                    value="'AMC', 'CART', 'CAVA', 'CPNG'")
    
    if st.button("Clean Tickers"):
        cleaned_text = raw_tickers_text.replace("'", "").replace('"', "").replace("[", "").replace("]", "")
        st.write("**Cleaned Tickers** (comma-separated):")
        st.code(cleaned_text.strip())

    st.write("---")
    st.subheader("Run the Options Screener")

    # Main tickers input
    tickers_input = st.text_input(
        "Tickers (comma‐separated, no quotes):",
        value="AAPL, TSLA"
    )

    if st.button("Run Screener"):
        tickers_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        if not tickers_list:
            st.error("No valid tickers provided.")
            return
        
        n = len(tickers_list)
        progress_bar = st.progress(0)
        
        valid_results = []
        error_results = []
        
        for i, ticker in enumerate(tickers_list):
            with st.spinner(f"Processing {ticker} ({i+1}/{n})..."):
                res = compute_recommendation(ticker)
                if isinstance(res, dict):
                    valid_results.append(res)
                else:
                    error_results.append(res)

            # Update progress
            progress_val = int(((i + 1) / n) * 100)
            progress_bar.progress(progress_val)
            
            # Sleep 7 seconds IF this isn't the last ticker
            if i < n - 1:
                time.sleep(7)
        
        # Once done, build a table
        if valid_results:
            # Sort valid results by expected move descending (None goes to bottom)
            def sort_key(item):
                return item['expected_move'] if item['expected_move'] is not None else -9999999
            valid_results.sort(key=sort_key, reverse=True)

            df_rows = []
            for r in valid_results:
                ticker = r['ticker']
                share_price = r['share_price']
                avg_vol_bool = r['avg_volume_pass']
                iv30rv30_bool = r['iv30_rv30_pass']
                slope_bool = r['ts_slope_0_45_pass']
                emove_str = r['expected_move_str'] or "N/A"
                
                # Simple recommendation logic
                if avg_vol_bool and iv30rv30_bool and slope_bool:
                    recommendation = "Recommended"
                elif slope_bool and ((avg_vol_bool and not iv30rv30_bool) or 
                                     (iv30rv30_bool and not avg_vol_bool)):
                    recommendation = "Consider"
                else:
                    recommendation = "Avoid"

                atm_call_ba = "N/A"
                atm_put_ba = "N/A"
                if r.get('atm_call_bid') is not None and r.get('atm_call_ask') is not None:
                    atm_call_ba = f"{r['atm_call_bid']} / {r['atm_call_ask']}"
                if r.get('atm_put_bid') is not None and r.get('atm_put_ask') is not None:
                    atm_put_ba = f"{r['atm_put_bid']} / {r['atm_put_ask']}"

                # <<< NEW: Add today's volume for ATM call & put
                atm_call_vol = r['atm_call_volume'] if r.get('atm_call_volume') is not None else "N/A"
                atm_put_vol  = r['atm_put_volume']  if r.get('atm_put_volume')  is not None else "N/A"

                row = {
                    "Ticker": ticker,
                    "Share Price": round(share_price, 2),
                    "Recommendation": recommendation,
                    "avg_volume": "PASS" if avg_vol_bool else "FAIL",
                    "iv30_rv30": "PASS" if iv30rv30_bool else "FAIL",
                    "ts_slope":  "PASS" if slope_bool else "FAIL",
                    "Expected Move": emove_str,
                    "ATM Call B/A": atm_call_ba,
                    "ATM Put B/A": atm_put_ba,
                    "ATM Call Vol": atm_call_vol,  # <<< NEW
                    "ATM Put Vol": atm_put_vol     # <<< NEW
                }
                df_rows.append(row)

            df = pd.DataFrame(df_rows)
            st.subheader("Screen Results")
            st.dataframe(df, use_container_width=True)

            if df["Expected Move"].eq("N/A").any():
                st.warning("Some 'N/A' values may be due to incomplete yfinance data (often outside market hours).")

        if error_results:
            st.subheader("Errors")
            for err in error_results:
                st.error(err)

if __name__ == "__main__":
    main()
