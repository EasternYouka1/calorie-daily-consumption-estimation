import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

#datos edad peso nivel actividad
X_train = np.array([[25, 70, 3], [30, 80, 1], [22, 60, 2]])  

#resultados caloricos segun datos anteriores
y_train = np.array([[2000], [2200], [1800]])  


modelo = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(3,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  
])


modelo.compile(optimizer='adam', loss='mse')

modelo.fit(X_train, y_train, epochs=350
          )


new_user_data = np.array([[30, 100, 1]])  
predicted_calories = model.predict(new_user_data)
print(f"Calorías diarias recomendadas: {predicted_calories[0][0]} kcal")