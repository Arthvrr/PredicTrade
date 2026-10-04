from flask import Flask, render_template, request, jsonify
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import requests

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/search', methods=['GET'])
def search_ticker():
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify([])
        
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=5&newsCount=0"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        res = requests.get(url, headers=headers)
        data = res.json()
        quotes = data.get('quotes', [])
        
        results = [
            {"symbol": q.get('symbol'), "shortname": q.get('shortname', q.get('longname', 'Unknown'))}
            for q in quotes if q.get('quoteType') in ['EQUITY', 'ETF']
        ]
        return jsonify(results)
    except Exception as e:
        print(f"Search API Error: {e}")
        return jsonify([])

@app.route('/api/stock', methods=['POST'])
def get_stock_data():
    data = request.json
    ticker = data.get('ticker', '').strip().upper()
    period = data.get('period', '1Y')
    features = data.get('features', ['close'])
    
    if not ticker:
        return jsonify({"error": "Please enter a stock ticker."}), 400
        
    period_settings = {
        '1D': {'period': '1d', 'interval': '5m'},
        '1W': {'period': '5d', 'interval': '1h'},
        '1M': {'period': '1mo', 'interval': '1d'},
        '3M': {'period': '3mo', 'interval': '1d'},
        '6M': {'period': '6mo', 'interval': '1d'},
        '1Y': {'period': '1y', 'interval': '1d'},
        'YTD': {'period': 'ytd', 'interval': '1d'},
        '5Y': {'period': '5y', 'interval': '1wk'},
        'ALL': {'period': 'max', 'interval': '1mo'}
    }
    
    settings = period_settings.get(period, {'period': '1y', 'interval': '1d'})
        
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=settings['period'], interval=settings['interval'])
        
        if hist.empty:
            return jsonify({"error": f"No data found for '{ticker}' on this period."}), 404
            
        info = stock.info
        company_name = info.get('shortName', ticker)
        website = info.get('website', '')
        domain = website.replace('https://', '').replace('http://', '').replace('www.', '').split('/')[0] if website else None
        
        if settings['interval'] in ['5m', '15m', '1h']:
            dates = hist.index.strftime('%Y-%m-%d %H:%M').tolist()
        else:
            dates = hist.index.strftime('%Y-%m-%d').tolist()
            
        prices = hist['Close'].round(2).tolist()
        first_price = prices[0]
        last_price = prices[-1]
        variation_pct = round(((last_price - first_price) / first_price) * 100, 2) if first_price > 0 else 0
        
        # 1. Tendance globale
        time_X = np.arange(len(prices)).reshape(-1, 1)
        trend_model = LinearRegression()
        trend_model.fit(time_X, prices)
        trend_prices = trend_model.predict(time_X)
        next_time_X = np.array([[len(prices)]])
        trend_pred = trend_model.predict(next_time_X)[0]
        trend_prices_list = np.round(trend_prices, 2).tolist()
        trend_prices_list.append(round(trend_pred, 2))

        # 2. Préparation des Features ML
        df = pd.DataFrame({'Close': prices})
        feature_cols = ['Close']
        frontend_volume = None
        frontend_sma5 = None
        frontend_sma10 = None
        
        if 'volume' in features and 'Volume' in hist.columns:
            df['Volume'] = hist['Volume'].values
            feature_cols.append('Volume')
            frontend_volume = hist['Volume'].tolist()
            
        if 'sma' in features:
            df['SMA_5'] = df['Close'].rolling(window=5).mean()
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            feature_cols.extend(['SMA_5', 'SMA_10'])
            frontend_sma5 = [round(x, 2) if pd.notna(x) else None for x in df['SMA_5']]
            frontend_sma10 = [round(x, 2) if pd.notna(x) else None for x in df['SMA_10']]

        df['Target'] = df['Close'].shift(-1)
        train_df = df.dropna()
        
        ml_lr = None
        ml_rf = None

        if len(train_df) > 10:
            X = train_df[feature_cols].values
            y = train_df['Target'].values
            last_known_features = df[feature_cols].iloc[-1].values
            next_X = np.array([last_known_features])
            
            # --- MODÈLE 1 : LINEAR REGRESSION ---
            lr_model = LinearRegression()
            lr_model.fit(X, y)
            lr_pred = lr_model.predict(next_X)[0]
            lr_hist = lr_model.predict(X)
            lr_rmse = np.sqrt(np.mean((y - lr_hist)**2))
            
            offset = len(prices) - len(lr_hist)
            lr_prices = [None] * offset + np.round(lr_hist, 2).tolist()
            lr_prices.append(round(lr_pred, 2))
            
            ml_lr = {
                "prediction": round(lr_pred, 2),
                "range_min": round(lr_pred - lr_rmse, 2),
                "range_max": round(lr_pred + lr_rmse, 2),
                "ml_prices": lr_prices
            }

            # --- MODÈLE 2 : RANDOM FOREST ---
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X, y)
            rf_pred = rf_model.predict(next_X)[0]
            rf_hist = rf_model.predict(X)
            rf_rmse = np.sqrt(np.mean((y - rf_hist)**2))
            
            rf_prices = [None] * offset + np.round(rf_hist, 2).tolist()
            rf_prices.append(round(rf_pred, 2))
            
            # NOUVEAU : Extraction de l'importance des variables (Cerveau du Random Forest)
            importances = rf_model.feature_importances_
            feature_importance_dict = {
                feat: round(float(imp) * 100, 1) 
                for feat, imp in zip(feature_cols, importances)
            }
            # Trier le dictionnaire du plus important au moins important
            feature_importance_dict = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))
            
            ml_rf = {
                "prediction": round(rf_pred, 2),
                "range_min": round(rf_pred - rf_rmse, 2),
                "range_max": round(rf_pred + rf_rmse, 2),
                "ml_prices": rf_prices,
                "importances": feature_importance_dict # Envoi au frontend
            }
            
            dates.append("Tomorrow (Prediction)")
            prices.append(None)
            if frontend_volume: frontend_volume.append(None)
            if frontend_sma5: frontend_sma5.append(None)
            if frontend_sma10: frontend_sma10.append(None)

        return jsonify({
            "ticker": ticker,
            "name": company_name,
            "domain": domain,
            "current_price": last_price,
            "variation": variation_pct,
            "period": period,
            "dates": dates,
            "prices": prices,
            "volume_data": frontend_volume,
            "sma5_data": frontend_sma5,
            "sma10_data": frontend_sma10,
            "trendline": trend_prices_list,
            "ml_lr": ml_lr,
            "ml_rf": ml_rf
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc()) 
        return jsonify({"error": "Error fetching and analyzing data."}), 500

if __name__ == '__main__':
    app.run(debug=True)