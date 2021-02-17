import keras
from keras.optimizers import SGD

def build_model(hidden_layer_size, learning_rate, activation):
    model = keras.models.Sequential() #sequential -> modelo do keras composto por uma pilha de camadas conectadas

    #primeira camada (Flatten) -> apenas faz o preprocessamento das imagens, convertando cada uma em um array de uma dimensão
    model.add(keras.layers.Flatten(input_shape=[28, 28])) 

    #cada camada Dense abaixo vai ter seu próprio vetor de pesos entre os neurons e os inputs
    #segunda camada -> hidden layer com 300 neurons, função de ativação ReLU
    model.add(keras.layers.Dense(hidden_layer_size, activation=activation))

    #output layer -> 10 neurons (numero de saídas) usando softmax (classes exclusivas e classificaçao p mais de duas classes), se fosse binario, usaria sigmoid
    model.add(keras.layers.Dense(10, activation="softmax"))

    custom_optimizer = SGD(lr = learning_rate)

    model.compile(loss="sparse_categorical_crossentropy", optimizer=custom_optimizer, metrics=["accuracy"])

    return model