import numpy as np
import streamlit as st
import pandas as pd

st.write(''' # Predicción de la calificacion  ''')
st.image("Examen.webp", caption="Predicción de la calificacion de una persona.")

st.header('Datos de el estudiante')

def user_input_features():
  # Entrada
  Horas_estudio = st.number_input('Horas estudiadas:', min_value=1.0, max_value=12.0, value=12.0, step=0.5)
  Horas_sueno = st.number_input('Horas de sueño:', min_value=4.0, max_value=9.0, value=9.0, step=0.5)
  Asistencia = st.number_input('Porcentaje de asistencia (%):', min_value=50.3, max_value=100.0, value=100.0, step=1.0)
  Calif_previa = st.number_input('Calificación previa:', min_value=40.0, max_value=95.0, value=95.0, step=1.0)

  user_input_data = {
      'hours_studied': Horas_estudio,
      'sleep_hours': Horas_sueno,
      'attendance_percent': Asistencia,
      'previous_scores': Calif_previa
  }

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()
prediccion=0
datos =  pd.read_csv('Calificacion.csv', encoding='latin-1')
X = datos.drop(columns='exam_score')
y = datos['exam_score']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=5)
LR = LinearRegression()
LR.fit(X_train,y_train)

b1 = LR.coef_
b0 = LR.intercept_
prediccion = (b0 + 
              b1[0] * df['hours_studied'] + 
              b1[1] * df['sleep_hours'] + 
              b1[2] * df['attendance_percent'] + 
              b1[3] * df['previous_scores'])

st.subheader('Cálculo de la calificación')
st.write('La calificación aproximada es:', round(prediccion[0], 2))
