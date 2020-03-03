from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import calendar
import datetime
headers = ['date', 'low', 'open', 'high', 'close', 'ticks', 'volume']
base_treinamento = pd.read_csv('WINJ20M15.csv', encoding="utf-16", header=None, names=headers, date_parser=pd.to_datetime)
base_teste = pd.read_csv('WINJ20H1.csv', encoding="utf-16", header=None, names=headers, date_parser=pd.to_datetime)
#preco_real_teste = base_teste.iloc[:, 3:4].values

base_treinamento['date'] = pd.to_datetime(base_treinamento['date'])
base_teste['date'] = pd.to_datetime(base_teste['date'])
#
dia_teste = base_teste.date.map(lambda x: x.strftime('%d')) 
mes_teste = base_teste.date.map(lambda x: x.strftime('%m')) 
hora_teste = base_teste.date.map(lambda x: x.strftime('%H')) 
#
dia_treinamento = base_treinamento.date.map(lambda x: x.strftime('%d')) 
mes_treinamento = base_treinamento.date.map(lambda x: x.strftime('%m')) 
hora_treinamento = base_treinamento.date.map(lambda x: x.strftime('%H')) 
#
base_treinamento = base_treinamento.drop('date', axis = 1)
base_treinamento = base_treinamento.drop('ticks', axis = 1)
base_teste = base_teste.drop('date', axis = 1)
base_teste = base_teste.drop('ticks', axis = 1)
base_treinamento['mes'] = mes_treinamento
base_treinamento['dia'] = dia_treinamento
base_treinamento['hora'] = hora_treinamento
base_teste['mes'] = mes_teste
base_teste['dia'] = dia_teste
base_teste['hora'] = hora_teste
#
normalizador = MinMaxScaler(feature_range=(0.1,0.9))
normalizador_previsao = base_treinamento.iloc[:, 3:4].values
normalizador_previsao = np.array(normalizador_previsao)
normalizador_previsao = normalizador.fit_transform(normalizador_previsao)
#
normalizador = MinMaxScaler(feature_range=(0.1,0.9))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)
base_teste_normalizada = normalizador.fit_transform(base_teste)

#
previsores = []
preco_real = []
for i in range(90, 1388):
    previsores.append(base_treinamento_normalizada[i-90:i, 0:6])
    preco_real.append(base_treinamento_normalizada[i, 0])
previsores, preco_real = np.array(previsores), np.array(preco_real)
#

regressor = Sequential()
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores.shape[1], 6)))
regressor.add(Dropout(0.1))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.0))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.0))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.0))

regressor.add(Dense(units = 1, activation = 'sigmoid'))

regressor.compile(optimizer = 'RMSprop', loss = 'mean_squared_error',
                  metrics = ['mean_squared_error'])
#
es = EarlyStopping(monitor = 'loss', min_delta = 1e-6, patience = 10, verbose = 1)
rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.01, patience = 10, verbose = 1)
mcp = ModelCheckpoint(filepath = 'pesos.h5', monitor = 'loss', 
                    save_best_only = True, verbose = 1)
regressor.fit(previsores, preco_real, epochs = 100, batch_size = 32,
              callbacks = [es, rlr, mcp])
#
preco_real_teste = base_teste.iloc[:, 3:4].values
frames = [base_teste, base_treinamento]
base_completa = pd.concat(frames)
#
entradas = base_completa[len(base_completa) - len(base_teste) - 90:].values
entradas = normalizador.transform(entradas)
#
X_teste = []
for i in range(90, 502):
    X_teste.append(entradas[i-90:i, 0:6])
X_teste = np.array(X_teste)
#
previsoes = [] 
previsoes = base_treinamento_normalizada[:, 3:4]
previsoes = regressor.predict(X_teste)
previsoes = np.array(previsoes)
##### previsoes = previsoes.inverse_transform(previsoes)

















