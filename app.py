from flask import Flask, render_template, request, jsonify
import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/stock', methods=['POST'])
def get_stock_data():
    data = request.json
    ticker = data.get('ticker', '').strip().upper()
    period = data.get('period', '1Y')
    
    if not ticker:
        return jsonify({"error": "Veuillez entrer un symbole boursier."}), 400
        
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
            return jsonify({"error": f"Aucune donnée trouvée pour '{ticker}' sur cette période."}), 404
            
        company_name = stock.info.get('shortName', ticker)
        
        if settings['interval'] in ['5m', '15m', '1h']:
            dates = hist.index.strftime('%Y-%m-%d %H:%M').tolist()
        else:
            dates = hist.index.strftime('%Y-%m-%d').tolist()
            
        prices = hist['Close'].round(2).tolist()
        
        first_price = prices[0]
        last_price = prices[-1]
        variation_pct = round(((last_price - first_price) / first_price) * 100, 2) if first_price > 0 else 0
        
        # ==========================================
        # MACHINE LEARNING : RÉGRESSION LINÉAIRE
        # ==========================================
        # On crée un DataFrame pour manipuler facilement les colonnes
        df = pd.DataFrame({'Close': prices})
        
        # La cible à prédire est le prix du pas de temps SUIVANT (shift -1)
        df['Target'] = df['Close'].shift(-1)
        
        # On enlève la dernière ligne car on ne connaît pas encore sa cible
        train_df = df.dropna()
        
        if len(train_df) > 5: # Vérifier qu'on a assez de données pour entraîner
            X = train_df[['Close']].values
            y = train_df['Target'].values
            
            # Entraînement du modèle
            model = LinearRegression()
            model.fit(X, y)
            
            # Prédiction pour le prochain point (en utilisant le tout dernier prix)
            next_X = np.array([[last_price]])
            pred = model.predict(next_X)[0]
            
            # Calcul de la marge d'erreur (Root Mean Squared Error)
            predictions_history = model.predict(X)
            rmse = np.sqrt(np.mean((y - predictions_history)**2))
            
            ml_data = {
                "prediction": round(pred, 2),
                "range_min": round(pred - rmse, 2),
                "range_max": round(pred + rmse, 2)
            }
        else:
            ml_data = None # Pas assez de données pour prédire
            
        return jsonify({
            "ticker": ticker,
            "name": company_name,
            "current_price": last_price,
            "variation": variation_pct,
            "period": period,
            "dates": dates,
            "prices": prices,
            "ml": ml_data
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc()) # Utile pour voir l'erreur exacte dans le terminal
        return jsonify({"error": "Erreur lors de l'analyse et de la récupération des données."}), 500

if __name__ == '__main__':
    app.run(debug=True)