from datetime import datetime
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
import time
import pika
import pytz


class analitica():
    ventana = 15
    pronostico = 5
    file_name = "data_base.csv"
    servidor = "rabbit"  

    def __init__(self) -> None:
        self.load_data()

    def load_data(self):

        if not os.path.isfile(self.file_name):
            self.df = pd.DataFrame(columns=["fecha", "latitud", "longitud", "1", "temperatura", "2","humedad"])
        else:
            self.df = pd.read_csv (self.file_name)

    def update_data(self, msj):
        msj_vetor = msj.split(",")
        date =  datetime.strptime(msj_vetor[0], '%d.%m.%Y %H:%M:%S')
        date = time.mktime(date.timetuple()) - 18000
        date = datetime.utcfromtimestamp(date)
        new_data = {"fecha": date, "latitud": float(msj_vetor[1]), "longitud": float(msj_vetor[2]),"1": "Temperatura", "temperatura": float(msj_vetor[3]), "2": "Humedad", "humedad": float(msj_vetor[4])}
        self.df = self.df.append(new_data, ignore_index=True)
        self.guardar()
        self.publicar("Datos", "{},{},{},{},{}".format(date, msj_vetor[1], msj_vetor[2], msj_vetor[3], msj_vetor[4]))
        self.analitica_descriptiva()
        self.analitica_predictiva()
        

    def print_data(self):
        print(self.df)

    def analitica_descriptiva(self):
        self.operaciones("temperatura")
        self.operaciones("humedad")

    def operaciones(self, sensor):
        df_filtrado = self.df[sensor]
        df_filtrado = df_filtrado.tail(self.ventana)
        self.publicar("max-{}".format(sensor), str(df_filtrado.max(skipna = True)))
        self.publicar("min-{}".format(sensor), str(df_filtrado.min(skipna = True)))
        self.publicar("mean-{}".format(sensor), str(df_filtrado.mean(skipna = True)))
        self.publicar("median-{}".format(sensor), str(df_filtrado.median(skipna = True)))
        self.publicar("std-{}".format(sensor), str(df_filtrado.std(skipna = True)))

    def analitica_predictiva(self):
        self.regresion("temperatura")
        self.regresion("humedad")

    def regresion(self, sensor):
        df_filtrado = self.df
        df_filtrado = df_filtrado.tail(self.ventana)
        df_filtrado['fecha'] = pd.to_datetime(df_filtrado.pop('fecha'), format='%d.%m.%Y %H:%M:%S')
        df_filtrado['segundos'] = [time.mktime(t.timetuple()) for t in df_filtrado['fecha']]
        
        if len(df_filtrado) == 1:
            return
        else:
            ultimo_tiempo = df_filtrado['segundos'].iloc[-1]
            penultimo_tiempo = df_filtrado['segundos'].iloc[-2]
            ultimo_tiempo = ultimo_tiempo.astype(int)
            penultimo_tiempo = penultimo_tiempo.astype(int)
            tiempo = ultimo_tiempo - penultimo_tiempo
        if np.isnan(tiempo):
            return
        range(ultimo_tiempo + tiempo,(self.pronostico + 1) * tiempo + ultimo_tiempo, tiempo)
        nuevos_tiempos = np.array(range(ultimo_tiempo + tiempo,(self.pronostico + 1) * (tiempo) + ultimo_tiempo, tiempo))
        #self.publicar("tiempo", "{},{}".format(tiempo,ultimo_tiempo))
        X = df_filtrado["segundos"].to_numpy().reshape(-1, 1)  
        Y = df_filtrado[sensor].to_numpy().reshape(-1, 1)  
        linear_regressor = LinearRegression()
        linear_regressor.fit(X, Y)
        Y_pred = linear_regressor.predict(nuevos_tiempos.reshape(-1, 1))
        for tiempo, prediccion in zip(nuevos_tiempos, Y_pred):
            time_format = datetime.utcfromtimestamp(tiempo)
            date_time = time_format.strftime('%d.%m.%Y %H:%M:%S')
            self.publicar("prediccion-{}".format(sensor), "{},{}".format(date_time,prediccion[0]))
    @staticmethod
    def publicar(cola, mensaje):
        connexion = pika.BlockingConnection(pika.ConnectionParameters(host='rabbit'))
        canal = connexion.channel()
        # Declarar la cola
        canal.queue_declare(queue=cola, durable=True)
        # Publicar el mensaje
        canal.basic_publish(exchange='', routing_key=cola, body=mensaje)
        # Cerrar conexión
        connexion.close()

    def guardar(self):
        self.df.to_csv(self.file_name, encoding='utf-8')
