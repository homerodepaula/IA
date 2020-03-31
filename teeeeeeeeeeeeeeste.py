import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import calendar
import datetime
###############################################
headers = ['date', 'low', 'open', 'high', 'close', 'ticks', 'volume']
###############################################
entradas_treinamento = pd.read_csv("WINJ20M30.csv", encoding="utf-16", header=None, names=headers, date_parser=pd.to_datetime)
entradas_teste = pd.read_csv("WINJ20H11.csv", encoding="utf-16", header=None, names=headers, date_parser=pd.to_datetime)
###############################################
entradas_treinamento_open = entradas_treinamento.iloc[:, 2:3].values
entradas_treinamento_close = entradas_treinamento.iloc[:, 4:5].values 
###############################################
entradas_teste_open = entradas_teste.iloc[:, 2:3].values
entradas_teste_close = entradas_teste.iloc[:, 4:5].values 
###############################################
normalizador = MinMaxScaler(feature_range=(0.1,0.9))
###############################################
entradas_treinamento['date'] = pd.to_datetime(entradas_treinamento['date'])
entradas_teste['date'] = pd.to_datetime(entradas_teste['date'])
dia_teste = []
dia_treinamento = []
mes_teste = []
mes_treinamento = []
hora_teste = []
hora_treinamento = []
#
#
dia_teste = entradas_teste.date.map(lambda x: x.strftime('%d')) 
mes_teste = entradas_teste.date.map(lambda x: x.strftime('%m')) 
hora_teste = entradas_teste.date.map(lambda x: x.strftime('%H')) 
#
dia_treinamento = entradas_treinamento.date.map(lambda x: x.strftime('%d')) 
mes_treinamento = entradas_treinamento.date.map(lambda x: x.strftime('%m')) 
hora_treinamento = entradas_treinamento.date.map(lambda x: x.strftime('%H')) 
###############################################
#replace dos dados em entradas_treinamento para data certa
entradas_treinamento = entradas_treinamento.drop('date', axis = 1)
entradas_treinamento = entradas_treinamento.drop('ticks', axis = 1)
entradas_teste = entradas_teste.drop('date', axis = 1)
entradas_teste = entradas_teste.drop('ticks', axis = 1)
entradas_treinamento['mes'] = mes_treinamento
entradas_treinamento['dia'] = dia_treinamento
entradas_treinamento['hora'] = hora_treinamento
entradas_teste['mes'] = mes_teste
entradas_teste['dia'] = dia_teste
entradas_teste['hora'] = hora_teste
entradas_treinamento_normalizada = normalizador.fit_transform(entradas_treinamento)
entradas_teste_normalizada = normalizador.fit_transform(entradas_teste)
normalizador_previsao = []
normalizador_previsao = MinMaxScaler(feature_range = (0.1, 0.9))
normalizador_previsao = normalizador_previsao.fit_transform(entradas_treinamento)
#############################################
previsores = []
preco_real = []
for i in range(50, 750):
    previsores.append(entradas_treinamento_normalizada[i-50:i, 0:7])
    preco_real.append(entradas_treinamento_normalizada[i, 0:4])
previsores, preco_real = np.array(previsores), np.array(preco_real)
#############
regressor = Sequential()
regressor.add(LSTM(units = 30, return_sequences = True, input_shape = (previsores.shape[1], 7)))
regressor.add(Dropout(0.1))
regressor.add(LSTM(units = 25, return_sequences = True))
regressor.add(Dropout(0))
regressor.add(LSTM(units = 20, return_sequences = True))
regressor.add(Dropout(0))
regressor.add(LSTM(units = 10))
regressor.add(Dropout(0))
    
    regressor.add(Dense(units = 1, activation = 'sigmoid'))
    regressor.compile(optimizer = 'RMSprop', loss = 'mean_squared_error',
                      metrics = ['mean_absolute_error'])
    
  #  es = EarlyStopping(monitor = 'loss', min_delta = 1e-10, patience = 10, verbose = 1)
  #  rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.2, patience = 5, verbose = 1)
  #  mcp = ModelCheckpoint(filepath = 'pesos.h5', monitor = 'loss',
  #                        save_best_only = True, verbose = 1)
    
      
    
    regressor.fit(previsores, preco_real, epochs = 200, batch_size = 32)
                #  callbacks = [es, rlr, mcp])
       
    #

frames = [entradas_treinamento, entradas_teste]
base_completa = pd.concat(frames)
#
entradas = base_completa[len (base_completa) - len (entradas_teste) - 50:].values
entradas = normalizador.transform(entradas)
#
x_teste = []
for i in range(50,455):
    x_teste.append(entradas[i-50:i, 0:7])
x_teste = np.array(x_teste)

#

previsoes = regressor.predict(x_teste)
previ = normalizador_previsao.inverse_transform(previsoes)

    
    
    
    
    
    
    
    























