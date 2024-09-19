from flask import Flask, jsonify, request, abort
from apscheduler.schedulers.background import BackgroundScheduler
import datetime as dt
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
from functools import wraps
app = Flask(__name__)


API_KEY = 'f15bbf9ec72933d34c2526db5b6d1314d922bfabaf925dc1a9d7b8594bd203cc'

# Variables globales para almacenar los datos procesados y las predicciones
latest_data = None

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # Obtener la API Key de los headers de la solicitud
        key = request.headers.get('x-api-key')
        # Si no está en los headers, intentar obtenerla de los parámetros de consulta
        if not key:
            key = request.args.get('api_key')
        if not key:
            return jsonify({"error": "API key missing"}), 401
        if key != API_KEY:
            return jsonify({"error": "Invalid API key"}), 403
        return f(*args, **kwargs)
    return decorated


def fetch_and_predict():
    global latest_data
    try:
        symbol = 'BTC-USD'
        
        # Fechas de inicio y fin
        end_date = dt.datetime.today()
        start_date = end_date - pd.Timedelta(days=5)
        
        # Descargar los valores de las criptomonedas con intervalo de una hora
        crypto_data_h = yf.download(tickers=symbol, 
                                    start=start_date, 
                                    end=end_date, 
                                    interval='1h').reset_index().drop(["Adj Close","Volume"], axis=1)
        crypto = crypto_data_h.copy()
        crypto.rename(columns={"Datetime":"Date"}, inplace=True)
        
        ## Transformaciones:
        crypto['Date'] = pd.to_datetime(crypto['Date'])
        crypto.sort_values('Date', inplace=True)
        
        # Establecer 'Date' como índice del DataFrame
        crypto.set_index('Date', inplace=True)
        
            # ========================
        # 1. Medias Móviles (SMA y EMA)
        # ========================

        # Media Móvil Simple (SMA)

        crypto['SMA_10'] = crypto['Close'].rolling(window=10).mean()
        crypto['SMA_20'] = crypto['Close'].rolling(window=20).mean()
        crypto['SMA_50'] = crypto['Close'].rolling(window=50).mean()
        #crypto['SMA_100'] = crypto['Close'].rolling(window=100).mean()

        # Media Móvil Exponencial (EMA)
        #crypto['EMA_10'] = crypto['Close'].ewm(span=10, adjust=False).mean()
        crypto['EMA_20'] = crypto['Close'].ewm(span=20, adjust=False).mean()
        crypto['EMA_50'] = crypto['Close'].ewm(span=50, adjust=False).mean()
        crypto['EMA_100'] = crypto['Close'].ewm(span=100, adjust=False).mean()


        # ========================
        # 2. Índice de Fuerza Relativa (RSI)
        # ========================

        delta = crypto['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        rs = avg_gain / avg_loss
        crypto['RSI_14'] = 100 - (100 / (1 + rs))

        delta = crypto['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        avg_gain = gain.rolling(window=5).mean()
        avg_loss = loss.rolling(window=5).mean()

        rs = avg_gain / avg_loss
        crypto['RSI_5'] = 100 - (100 / (1 + rs))

        # ========================
        # 3. MACD (Moving Average Convergence Divergence)
        # ========================

        ema_12 = crypto['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = crypto['Close'].ewm(span=26, adjust=False).mean()

        crypto['MACD'] = ema_12 - ema_26
        crypto['Signal_Line'] = crypto['MACD'].ewm(span=9, adjust=False).mean()

        # ========================
        # 4. Bandas de Bollinger
        # ========================

        crypto['BB_Middle'] = crypto['Close'].rolling(window=20).mean()
        crypto['BB_Std'] = crypto['Close'].rolling(window=20).std()

        crypto['BB_Upper'] = crypto['BB_Middle'] + (crypto['BB_Std'] * 2)
        crypto['BB_Lower'] = crypto['BB_Middle'] - (crypto['BB_Std'] * 2)

        # ========================
        # 5. On-Balance Volume (OBV)
        # ========================

        # ========================
        # 6. Average True Range (ATR)
        # ========================

        crypto['TR1'] = crypto['High'] - crypto['Low']
        crypto['TR2'] = abs(crypto['High'] - crypto['Close'].shift(1))
        crypto['TR3'] = abs(crypto['Low'] - crypto['Close'].shift(1))
        crypto['True_Range'] = crypto[['TR1', 'TR2', 'TR3']].max(axis=1)

        crypto['ATR_14'] = crypto['True_Range'].rolling(window=14).mean()

        # Eliminar columnas intermedias
        crypto.drop(['TR1', 'TR2', 'TR3', 'True_Range'], axis=1, inplace=True)

        crypto['TR1'] = crypto['High'] - crypto['Low']
        crypto['TR2'] = abs(crypto['High'] - crypto['Close'].shift(1))
        crypto['TR3'] = abs(crypto['Low'] - crypto['Close'].shift(1))
        crypto['True_Range'] = crypto[['TR1', 'TR2', 'TR3']].max(axis=1)

        crypto['ATR_5'] = crypto['True_Range'].rolling(window=5).mean()

        # Eliminar columnas intermedias
        crypto.drop(['TR1', 'TR2', 'TR3', 'True_Range'], axis=1, inplace=True)

        # ========================
        # 7. Oscilador Estocástico
        # ========================

        crypto['Lowest_Low'] = crypto['Low'].rolling(window=14).min()
        crypto['Highest_High'] = crypto['High'].rolling(window=14).max()

        crypto['%K'] = 100 * ((crypto['Close'] - crypto['Lowest_Low']) / (crypto['Highest_High'] - crypto['Lowest_Low']))
        crypto['%D'] = crypto['%K'].rolling(window=3).mean()

        # Eliminar columnas intermedias
        crypto.drop(['Lowest_Low', 'Highest_High'], axis=1, inplace=True)

        # ========================
        # 8. Promedio Direccional (ADX)
        # ========================

        # Cálculo de movimientos direccionales
        crypto['UpMove'] = crypto['High'] - crypto['High'].shift(1)
        crypto['DownMove'] = crypto['Low'].shift(1) - crypto['Low']

        crypto['+DM'] = np.where((crypto['UpMove'] > crypto['DownMove']) & (crypto['UpMove'] > 0), crypto['UpMove'], 0)
        crypto['-DM'] = np.where((crypto['DownMove'] > crypto['UpMove']) & (crypto['DownMove'] > 0), crypto['DownMove'], 0)

        # Cálculo del True Range (ya tenemos ATR pero necesitamos TR para ADX)
        crypto['TR'] = crypto[['High', 'Low', 'Close']].max(axis=1) - crypto[['High', 'Low', 'Close']].min(axis=1)

        crypto['TR14'] = crypto['TR'].rolling(window=14).sum()
        crypto['+DM14'] = crypto['+DM'].rolling(window=14).sum()
        crypto['-DM14'] = crypto['-DM'].rolling(window=14).sum()

        crypto['+DI14'] = 100 * (crypto['+DM14'] / crypto['TR14'])
        crypto['-DI14'] = 100 * (crypto['-DM14'] / crypto['TR14'])

        crypto['DX'] = 100 * (abs(crypto['+DI14'] - crypto['-DI14']) / (crypto['+DI14'] + crypto['-DI14']))
        crypto['ADX'] = crypto['DX'].rolling(window=14).mean()

        # Eliminar columnas intermedias
        crypto.drop(['UpMove', 'DownMove', '+DM', '-DM', 'TR', 'TR14', '+DM14', '-DM14', '+DI14', '-DI14', 'DX'], axis=1, inplace=True)

        # ========================
        # 9. Cálculo de Puntos Pivote
        # ========================

        crypto['PP'] = (crypto['High'].shift(1) + crypto['Low'].shift(1) + crypto['Close'].shift(1)) / 3
        crypto['R1'] = (2 * crypto['PP']) - crypto['Low'].shift(1)
        crypto['S1'] = (2 * crypto['PP']) - crypto['High'].shift(1)
        crypto['R2'] = crypto['PP'] + (crypto['High'].shift(1) - crypto['Low'].shift(1))
        crypto['S2'] = crypto['PP'] - (crypto['High'].shift(1) - crypto['Low'].shift(1))
        crypto['R3'] = crypto['High'].shift(1) + 2 * (crypto['PP'] - crypto['Low'].shift(1))
        crypto['S3'] = crypto['Low'].shift(1) - 2 * (crypto['High'].shift(1) - crypto['PP'])

        # Soportes basados en mínimos anteriores
        crypto['Support_Level'] = crypto['Low'].rolling(window=50, min_periods=1).min()

        # Resistencias basadas en máximos anteriores
        crypto['Resistance_Level'] = crypto['High'].rolling(window=50, min_periods=1).max()


        #=========================
        # 10. Maximos y minimos locales
        #=========================
        
        # Eliminar filas con valores NaN resultantes de las ventanas de cálculo
        crypto.dropna(inplace=True)
        
        # Cargar modelos
        models_path = 'models'
        model_max = joblib.load(os.path.join(models_path, 'model_max.joblib'))
        model_min = joblib.load(os.path.join(models_path, 'model_min.joblib'))
        model_conf_max = joblib.load(os.path.join(models_path, 'model_conf_max.joblib'))
        model_conf_min = joblib.load(os.path.join(models_path, 'model_conf_min.joblib'))
        
        # Definir las características para cada modelo
        features_max_min = ['Open', 'High', 'Low', 'Close', 'SMA_10', 'SMA_20', 'SMA_50',
                            'EMA_20','EMA_50', 'EMA_100','RSI_14','RSI_5','ATR_5','ATR_14',
                            'MACD', '%D', 'BB_Upper', 'BB_Lower', 'ADX', 'PP', 'R1', 'S1']
        features_conf_max = features_max_min + ["y_pred_max"]
        features_conf_min = features_max_min + ["y_pred_min"]
        
        # Hacer predicciones
        crypto['y_pred_max'] = model_max.predict(crypto[features_max_min])
        crypto['y_pred_min'] = model_min.predict(crypto[features_max_min])
        
        crypto['conf_max'] = model_conf_max.predict(crypto[features_conf_max])
        crypto['conf_min'] = model_conf_min.predict(crypto[features_conf_min])
        
        # Seleccionar las columnas necesarias para la salida
        output_columns = ['Open', 'Close', 'y_pred_max', 'y_pred_min', 'conf_max', 'conf_min']
        latest_data = crypto[output_columns].reset_index()
        
        print(f"[{dt.datetime.now()}] Datos actualizados y predicciones realizadas correctamente.")
        
    except Exception as e:
        print(f"Error en fetch_and_predict: {e}")

# Configurar el scheduler para que ejecute fetch_and_predict cada hora
      # Seleccionar las columnas necesarias para la salida
        output_columns = ['Open', 'Close', 'y_pred_max', 'y_pred_min', 'conf_max', 'conf_min']
        latest_data = crypto[output_columns].reset_index()
        
        print(f"[{dt.datetime.now()}] Datos actualizados y predicciones realizadas correctamente.")
        
    except Exception as e:
        print(f"Error en fetch_and_predict: {e}")

# Configurar el scheduler para que ejecute fetch_and_predict 3 segundos de cada hora
scheduler = BackgroundScheduler()
scheduler.add_job(
    func=fetch_and_predict,
    trigger='cron',
    minute=0,
    second=1,
    id='fetch_predict_job',
    replace_existing=True
)
scheduler.start()

# Ejecutar una vez al inicio
fetch_and_predict()

# Ruta para obtener los datos más recientes
@app.route('/latest', methods=['GET'])
@require_api_key
def get_latest():
    if latest_data is not None:
        # Obtener la última fila
        latest_row = latest_data.iloc[-1].to_dict()
        return jsonify(latest_row)
    else:
        return jsonify({"error": "Datos no disponibles"}), 503

# Ruta para obtener todos los datos
@app.route('/data', methods=['GET'])
@require_api_key
def get_all_data():
    if latest_data is not None:
        # Convertir el DataFrame a diccionario
        data = latest_data.to_dict(orient='records')
        return jsonify(data)
    else:
        return jsonify({"error": "Datos no disponibles"}), 503

# Manejar el cierre adecuado del scheduler al detener la aplicación
import atexit
atexit.register(lambda: scheduler.shutdown())

@app.route('/jobs', methods=['GET'])
def list_jobs():
    jobs = scheduler.get_jobs()
    job_list = []
    for job in jobs:
        job_info = {
            'id': job.id,
            'next_run_time': job.next_run_time.strftime("%Y-%m-%d %H:%M:%S") if job.next_run_time else None
        }
        job_list.append(job_info)
    return jsonify(job_list)


if __name__ == '__main__':
    app.run(port=5000)
