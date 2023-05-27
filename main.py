import numpy as np
import os
import cv2
import pickle
import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('fivethirtyeight')

# Dense layer
class Layer:

    # Инициализация слоя
    def __init__(self, n_inputs, n_neurons,
                 w_reg_l1=0, w_reg_l2=0,
                 b_reg_l1=0, b_reg_l2=0):

        # Матрица весов и смещение
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        # Настройка регуляризаций L1 и L2
        self.w_reg_l1 = w_reg_l1
        self.w_reg_l2 = w_reg_l2
        self.b_reg_l1 = b_reg_l1
        self.b_reg_l2 = b_reg_l2


    def forward(self, inputs, training):
        # Входные данные и расчет выходных
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases


    def backward(self, dvalues):
        # Градиенты параметров
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)


        ### Градиент регуляризаций
        # L1
        if self.w_reg_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.w_reg_l1 * dL1
        # L2
        if self.w_reg_l2 > 0:
            self.dweights += 2 * self.w_reg_l2 * \
                             self.weights
        # L1
        if self.b_reg_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.b_reg_l1 * dL1
        # L2
        if self.b_reg_l2 > 0:
            self.dbiases += 2 * self.b_reg_l2 * \
                            self.biases

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

    # Получить параметры сети
    def get_parameters(self):
        return self.weights, self.biases

    # Установить другие параметры сети
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases


# Отключение нейронов (чтобы ибежать переобучения)
class Layer_Dropout:

    # Процент нейронов для отключения
    def __init__(self, rate):
        self.rate = 1 - rate

    # Forward pass
    def forward(self, inputs, training):
        self.inputs = inputs
        if not training:
            self.output = inputs.copy()
            return

        self.binary_mask = np.random.binomial(1, self.rate,
                           size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    # Градиент
    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask


# Слой входных данных
class Layer_Input:

    # Forward pass
    def forward(self, inputs, training):
        self.output = inputs


### Функции Активации

class Activation_ReLU:

    # Forward pass
    def forward(self, inputs, training):

        self.inputs = inputs

        # ReLu
        self.output = np.maximum(0, inputs)


    def backward(self, dvalues):
        #Копия, тк оригинальные даные надо менять
        self.dinputs = dvalues.copy()

        # Обнуляем градиент где входные данные отрицательные
        self.dinputs[self.inputs <= 0] = 0


    def predictions(self, outputs):
        return outputs



class Activation_Softmax:

    # Forward pass
    def forward(self, inputs, training):

        self.inputs = inputs

        # Вероятности
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))
        # Нормировка
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)
        self.output = probabilities

    # Градиент
    def backward(self, dvalues):

        self.dinputs = np.empty_like(dvalues)

        # Матрица Якоби
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            # растягиваем в массив (1, *)
            single_output = single_output.reshape(-1, 1)

            # Расчет матрицы Якоби от выходных данных
            jacobian = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            # Расчет градиента
            self.dinputs[index] = np.dot(jacobian,
                                         single_dvalues)

    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)



class Activation_Sigmoid:

    def forward(self, inputs, training):
        self.inputs = inputs
        # Sigmoid
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs):
        return (outputs > 0.5) * 1


class Activation_Linear:

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = inputs

    # Градиент
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

    def predictions(self, outputs):
        return outputs



class Optimizer_SGD:

    # learning rate == 1 -- по стандарту
    def __init__(self, l_r=1., decay=0., momentum=0.):
        self.l_r = l_r
        self.current_l_r = l_r
        self.decay = decay
        self.iters = 0
        self.momentum = momentum

    # Обновление параметров
    def pre_update_params(self):
        if self.decay:
            self.current_l_r = self.l_r * \
                (1. / (1. + self.decay * self.iters))

    def update_params(self, layer):

        # Если используем импульс
        if self.momentum:

            # Если в слое нет массива импульса,
            # создает их с нулями
            if not hasattr(layer, 'w_momentums'):

                layer.w_momentums = np.zeros_like(layer.weights)

                layer.b_momentums = np.zeros_like(layer.biases)

            # Обновление матрицы импульса,
            # изменяя его с учетом предыдущих данных
            # и коэффициентом сохранения
            weight_updates = \
                self.momentum * layer.w_momentums - \
                self.current_l_r * layer.dweights
            layer.w_momentums = weight_updates

            bias_updates = \
                self.momentum * layer.b_momentums - \
                self.current_l_r * layer.dbiases
            layer.b_momentums = bias_updates


        else:
            weight_updates = -self.current_l_r * \
                             layer.dweights
            bias_updates = -self.current_l_r * \
                           layer.dbiases

        # Обновление весов и смещений
        # с использованием импульса
        layer.weights += weight_updates
        layer.biases += bias_updates

    # Вызывается каждый раз,
    # когда параметры меняются
    def post_update_params(self):
        self.iters += 1



class Optimizer_Adagrad:

    def __init__(self, l_r=1., decay=0., eps=1e-7):
        self.l_r = l_r
        self.current_l_r = l_r
        self.decay = decay
        self.iters = 0
        self.eps = eps

    # Вызывается единожды для задание параметров
    def pre_update_params(self):
        if self.decay:
            self.current_l_r = self.l_r * \
                (1. / (1. + self.decay * self.iters))

    # Обновление параметров
    def update_params(self, layer):

        # По аналогии с SGD
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2


        layer.weights += -self.current_l_r * \
                         layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.eps)
        layer.biases += -self.current_l_r * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.eps)

    def post_update_params(self):
        self.iters += 1


class Optimizer_RMSprop:

    def __init__(self, l_r=0.001, decay=0., eps=1e-7,
                 rho=0.9):
        self.l_r = l_r
        self.current_l_r = l_r
        self.decay = decay
        self.iters = 0
        self.eps = eps
        self.rho = rho

    def pre_update_params(self):
        if self.decay:
            self.current_l_r = self.l_r * \
                (1. / (1. + self.decay * self.iters))

    # По аналогии с SGD
    def update_params(self, layer):

        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = self.rho * layer.weight_cache + \
            (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + \
            (1 - self.rho) * layer.dbiases**2


        # SGD обновление параметров + нормализация
        layer.weights += -self.current_l_r * \
                         layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.eps)
        layer.biases += -self.current_l_r * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.eps)

    def post_update_params(self):
        self.iters += 1


class Optimizer_Adam:

    def __init__(self, l_r=0.001, decay=0., eps=1e-7,
                 beta_1=0.9, beta_2=0.999):
        self.l_r = l_r
        self.current_l_r = l_r
        self.decay = decay
        self.iters = 0
        self.eps = eps
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_l_r = self.l_r * \
                (1. / (1. + self.decay * self.iters))

    def update_params(self, layer):

        if not hasattr(layer, 'weight_cache'):
            layer.w_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.b_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)


        # Обновление импульса по градиентам
        layer.w_momentums = self.beta_1 * \
                                 layer.w_momentums + \
                                 (1 - self.beta_1) * layer.dweights
        layer.b_momentums = self.beta_1 * \
                               layer.b_momentums + \
                               (1 - self.beta_1) * layer.dbiases
        # Коррекция импульса
        # self.iteration == 0 на первом проходе
        # и надо начать с 1

        w_momentums_corrected = layer.w_momentums / \
            (1 - self.beta_1 ** (self.iters + 1))
        b_momentums_corrected = layer.b_momentums / \
            (1 - self.beta_1 ** (self.iters + 1))
        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.dbiases**2
        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2 ** (self.iters + 1))
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2 ** (self.iters + 1))

        # SGD Параметры + нормализация
        # с квадратом cache
        layer.weights += -self.current_l_r * \
                         w_momentums_corrected / \
                         (np.sqrt(weight_cache_corrected) +
                             self.eps)
        layer.biases += -self.current_l_r * \
                         b_momentums_corrected / \
                         (np.sqrt(bias_cache_corrected) +
                             self.eps)

    def post_update_params(self):
        self.iters += 1



class Loss:

    # Регуляризация
    def reg_loss(self):

        reg_loss = 0

        # Расчет потерь
        for layer in self.trainable_layers:

            # Расчитывать только когда  > 0
            # L1
            if layer.w_reg_l1 > 0:
                reg_loss += layer.w_reg_l1 * \
                                       np.sum(np.abs(layer.weights))

            # L2
            if layer.w_reg_l2 > 0:
                reg_loss += layer.w_reg_l2 * \
                                       np.sum(layer.weights * \
                                              layer.weights)

            # L1
            if layer.b_reg_l1 > 0:
                reg_loss += layer.b_reg_l1 * \
                                       np.sum(np.abs(layer.biases))

            # L2
            if layer.b_reg_l2 > 0:
                reg_loss += layer.b_reg_l2 * \
                                       np.sum(layer.biases * \
                                              layer.biases)

        return reg_loss

    # Set/remember trainable layers
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers


    # Вычисляет потери данных и регуляризации
    # заданные выходные данные модели и основные значения
    def calculate(self, output, y, *, include_regularization=False):

        # Расчет потерь
        sample_losses = self.forward(output, y)

        # Средние потери
        data_loss = np.mean(sample_losses)

        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if not include_regularization:
            return data_loss

        return data_loss, self.reg_loss()

    def calculate_accumulated(self, *, include_regularization=False):

        # Среднее потерь
        data_loss = self.accumulated_sum / self.accumulated_count

        if not include_regularization:
            return data_loss

        return data_loss, self.reg_loss()

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0


class Loss_CategoricalCrossentropy(Loss):

    def forward(self, y_pred, y_true):

        # Количество образцов
        samples = len(y_pred)


        # Обрезаем данные чтобы предотвратить деление на 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Вероятности для целевых значений
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        # Потери
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):

        samples = len(dvalues)

        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Градиент
        self.dinputs = -y_true / dvalues
        # Нормировка
        self.dinputs = self.dinputs / samples



# Потеря перекрестной энтропии
class Activation_Softmax_Loss_CategoricalCrossentropy():

    def backward(self, dvalues, y_true):

        samples = len(dvalues)


        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        # Градиент
        self.dinputs[range(samples), y_true] -= 1
        # Нормировка
        self.dinputs = self.dinputs / samples


class Loss_BinaryCrossentropy(Loss):

    def forward(self, y_pred, y_true):

        # Обрезаем данные чтобы предотвратить деление на 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Потери по выборке
        sample_losses = -(y_true * np.log(y_pred_clipped) +
                          (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)
        return sample_losses


    def backward(self, dvalues, y_true):

        samples = len(dvalues)
        outputs = len(dvalues[0])

        # Обрезаем данные чтобы предотвратить деление на 0
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # Градиент
        self.dinputs = -(y_true / clipped_dvalues -
                         (1 - y_true) / (1 - clipped_dvalues)) / outputs
        # Нормировка
        self.dinputs = self.dinputs / samples


class Loss_MeanSquaredError(Loss):  # L2 потери


    def forward(self, y_pred, y_true):

        # Расчет
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)

        return sample_losses

    def backward(self, dvalues, y_true):

        samples = len(dvalues)
        outputs = len(dvalues[0])

        # Градиент
        self.dinputs = -2 * (y_true - dvalues) / outputs
        # Нормировка
        self.dinputs = self.dinputs / samples


class Loss_MeanAbsoluteError(Loss):  # L1 потери

    def forward(self, y_pred, y_true):

        # Расчет
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)

        return sample_losses


    def backward(self, dvalues, y_true):

        samples = len(dvalues)

        outputs = len(dvalues[0])

        # Градиент
        self.dinputs = np.sign(y_true - dvalues) / outputs
        # Нормировка
        self.dinputs = self.dinputs / samples


class Accuracy:

    # Вычисление точности
    # заданных прогнозов и основных значений
    def calculate(self, predictions, y):

        # Результаты сравнения
        comparisons = self.compare(predictions, y)

        # Вычисление точности
        accuracy = np.mean(comparisons)

        # Добавление накопленной суммы совпадающих значений
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        return accuracy

    def calculate_accumulated(self):

        # Вычисление
        accuracy = self.accumulated_sum / self.accumulated_count

        return accuracy

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0



class Accuracy_Categorical(Accuracy):

    def __init__(self, *, binary=False):
        self.binary = binary

    def init(self, y):
        pass

    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y


class Accuracy_Regression(Accuracy):

    def __init__(self):
        self.precision = None

    # Вычисляет значение точности
    # на основе переданных базовых значений
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision


# Класс основной модели
class Model:

    def __init__(self):
        self.accuracy_data = []
        self.loss_data = []
        # Список слоев
        self.layers = []
        # Softmax выходные данные
        self.softmax_classifier_output = None
    # Добавляет объект к модели
    def add(self, layer):
        self.layers.append(layer)


    # Настройка потерь, оптимизатора и точности
    def set(self, *, loss=None, optimizer=None, accuracy=None):

        if loss is not None:
            self.loss = loss

        if optimizer is not None:
            self.optimizer = optimizer

        if accuracy is not None:
            self.accuracy = accuracy

    def finalize(self):

        # Создает входной слой
        self.input_layer = Layer_Input()

        # Подсчет объектов
        layer_count = len(self.layers)

        # Инициализация тренируемых слоев
        self.trainable_layers = []

        for i in range(layer_count):

            # Если это первый слой,
            # то предыдущий объект это слой входных данных
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]

            # Все слои помимо первого и последнего
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]



            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            # Если слой содержит атрибут под названием "веса",
            # это обучаемый слой -
            # добавьте его в список обучаемых слоев
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
        if self.loss is not None:
            self.loss.remember_trainable_layers(
                self.trainable_layers
            )

        # Если активация вывода - Softmax,
        # а функция потерь - категориальная перекрестная энтропия
        # создает объект комбинированной активации
        # и функция потерь, содержащая
        # более быстрый расчет градиента
        if isinstance(self.layers[-1], Activation_Softmax) and \
           isinstance(self.loss, Loss_CategoricalCrossentropy):
            self.softmax_classifier_output = \
                Activation_Softmax_Loss_CategoricalCrossentropy()

    # Тренировка модели
    def train(self, X, y, *, epochs=1, batch_size=None,
              print_every=1, validation_data=None):

        self.accuracy.init(y)


        train_steps = 1

        # Расчет шагов
        if batch_size is not None:
            train_steps = len(X) // batch_size
            if train_steps * batch_size < len(X):
                train_steps += 1


        # основной тренировочный цикл
        for epoch in range(1, epochs+1):

            # Номер эпохи
            print(f'epoch: {epoch}')

            # Ресет точности и потерь
            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                # Если количество шагов не задано,
                # то используется весь дада сет за раз
                if batch_size is None:
                    batch_X = X
                    batch_y = y

                # В ином случае делим данные на части
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                output = self.forward(batch_X, training=True)

                # Расчет потерь
                data_loss, reg_loss = \
                    self.loss.calculate(output, batch_y,
                                        include_regularization=True)
                loss = data_loss + reg_loss

                # Вычисление точности и прогноза
                predictions = self.output_layer_activation.predictions(
                                  output)
                accuracy = self.accuracy.calculate(predictions,
                                                   batch_y)

                self.accuracy_data.append(accuracy)
                self.loss_data.append(loss)
                self.backward(output, batch_y)

                # Оптимизация параметров
                self.optimizer.pre_update_params()

                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()


                if not step % print_every or step == train_steps - 1:

                    print(f'step: {step}, ' +
                          f'acc: {accuracy:.3f}, ' +
                          f'loss: {loss:.3f} (' +
                          f'data_loss: {data_loss:.3f}, ' +
                          f'reg_loss: {reg_loss:.3f}), ' +
                          f'lr: {self.optimizer.current_l_r}')

            # Вывод потерь и точности каждую эпоху
            epoch_data_loss, epoch_regularization_loss = \
                self.loss.calculate_accumulated(
                    include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print(f'training, ' +
                  f'acc: {epoch_accuracy:.3f}, ' +
                  f'loss: {epoch_loss:.3f} (' +
                  f'data_loss: {epoch_data_loss:.3f}, ' +
                  f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                  f'lr: {self.optimizer.current_l_r}')

            if validation_data is not None:

                # Оценка модели
                self.evaluate(*validation_data,
                              batch_size=batch_size)

    def evaluate(self, X_val, y_val, *, batch_size=None):

        validation_steps = 1
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        # Ресчет точности и потерь
        self.loss.new_pass()
        self.accuracy.new_pass()



        for step in range(validation_steps):

            # Если количество шагов не задано,
            # то используется весь дадасет за раз
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val

            else:
                batch_X = X_val[
                    step*batch_size:(step+1)*batch_size
                ]
                batch_y = y_val[
                    step*batch_size:(step+1)*batch_size
                ]

            output = self.forward(batch_X, training=False)

            # Вычисление точности и тд тп
            self.loss.calculate(output, batch_y)

            predictions = self.output_layer_activation.predictions(
                              output)
            self.accuracy.calculate(predictions, batch_y)

        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        print(f'validation, ' +
              f'acc: {validation_accuracy:.3f}, ' +
              f'loss: {validation_loss:.3f}')

    # Прогноз
    def predict(self, X, *, batch_size=None):


        prediction_steps = 1

        if batch_size is not None:
            prediction_steps = len(X) // batch_size

            if prediction_steps * batch_size < len(X):
                prediction_steps += 1

        output = []

        for step in range(prediction_steps):

            if batch_size is None:
                batch_X = X

            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]

            batch_output = self.forward(batch_X, training=False)

            output.append(batch_output)

        return np.vstack(output)

    def forward(self, X, training):

        self.input_layer.forward(X, training)

        # Передает выходные данные предыдущего объекта в качестве параметра
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        # "layer" теперь последний в списке
        # возвращает его вывод
        return layer.output


    def backward(self, output, y):

        if self.softmax_classifier_output is not None:
            # Первый обратный вызов метода
            # при комбинированной активации/потере
            # устанавливет dinputs
            self.softmax_classifier_output.backward(output, y)

            # Поскольку не вызывается обратный метод последнего слоя
            self.layers[-1].dinputs = \
                self.softmax_classifier_output.dinputs

            # Вызов обратного метода, проходящего через
            # все объекты, кроме последнего
            # в обратном порядке передаем dinputs в качестве параметра
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return

        self.loss.backward(output, y)

        # Вызов обратного метода, проходящего через все объекты
        # в обратном порядке передаем dinputs в качестве параметра
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    # Извлекает и возвращает параметры обучаемых слоев
    def get_parameters(self):

        # Массив араметров
        parameters = []

        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())

        return parameters


    # Коррекция параметров модели
    def set_parameters(self, parameters):

        for parameter_set, layer in zip(parameters,
                                        self.trainable_layers):
            layer.set_parameters(*parameter_set)

    # Сохранение параметров модели в файле
    def save_parameters(self, path):

        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)

    # Загружает веса и обновляет с их помощью экземпляр модели
    def load_parameters(self, path):

        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))

    # Сохранение модели
    def save(self, path):

        # Копия модели
        model = copy.deepcopy(self)

        # Сброс накопленных значений в объектах потерь и точности
        model.loss.new_pass()
        model.accuracy.new_pass()

        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs',
                             'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)

        # Откройтие файла в режиме двоичной записи и сохранение модели
        with open(path, 'wb') as f:
            pickle.dump(model, f)


    # Загружает и возвращает модель
    @staticmethod
    def load(path):

        with open(path, 'rb') as f:
            model = pickle.load(f)

        return model


# Загрузка Mnist датасета
def load_mnist_dataset(dataset, path):
    # Сканирование всех каталогов и создание списка меток
    labels = os.listdir(os.path.join(path, dataset))

    # Массивы данных и значений
    X = []
    y = []

    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(
                        os.path.join(path, dataset, label, file),
                        cv2.IMREAD_UNCHANGED)

            X.append(image)
            y.append(label)


    return np.array(X), np.array(y).astype('uint8')


# MNIST dataset (train + test)
def create_data_mnist(path):

    # Раздельная загрузка
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    # Возвращает все данные
    return X, y, X_test, y_test

# Создает датасет
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

# Перемешивает датасет для нормального обучения
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

# Нормировка данных к (-1, 1)
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) -
             127.5) / 127.5

fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}


model = Model()

# Добавление слоев
model.add(Layer(X.shape[1], 128))
model.add(Activation_ReLU())
model.add(Layer(128, 128))
model.add(Activation_ReLU())
model.add(Layer(128, 10))
model.add(Activation_Softmax())

model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(decay=1e-3),
    accuracy=Accuracy_Categorical()
)

model.finalize()

class Animate:
    def __init__(self):
        self._steps = range(len(model.accuracy_data))
        self.fig, self.ax = plt.subplots(1, 2, figsize = (10, 3.5))
        self.steps = []
        self.accuracy = []
        self.loss = []

    def _update(self, i):
        self.steps.append(self._steps[i])
        self.accuracy.append(model.accuracy_data[i])
        self.loss.append(model.loss_data[i])
        self.ax[0].clear()
        self.ax[1].clear()
        self.ax[0].plot(self.steps, self.accuracy)
        self.ax[1].plot(self.steps, self.loss)
        self.ax[0].text(0.60, 0.11, f"Точность: {self.accuracy[-1]:.3f}",
                        transform=self.ax[0].transAxes,
                        bbox=dict(facecolor='white', edgecolor='black'), size=12)
        self.ax[1].text(0.6, 0.80, f"Потери: {self.loss[-1]:.3f}",
                        transform=self.ax[1].transAxes,
                        bbox=dict(facecolor='white', edgecolor='black'), size=12)


    def start(self):
        self.anim = animation.FuncAnimation(self.fig, self._update, interval = 2)
        plt.show()


# Тренировка модели
#model.train(X, y, validation_data=(X_test, y_test),
            #epochs=10, batch_size=128, print_every=500)

model.save('fashion_mnist.model')


# Тестовая картинка
image_data = cv2.imread('T-shirt.png', cv2.IMREAD_GRAYSCALE)

image_data = cv2.resize(image_data, (28, 28))
# Инвертируем цвета (если она не как картинки с дата сета)
image_data = 255 - image_data
# Нормируем
image_data = (image_data.reshape(1, -1).astype(np.float32) -
127.5) / 127.5

# Загружаем модель
model = Model.load('fashion_mnist.model')

confidences = model.predict(image_data)
predictions = model.output_layer_activation.predictions(confidences)

# Get label name from label index
prediction = fashion_mnist_labels[predictions[0]]
print(f"prediction {prediction}", ', real : T-shirt.png')


# rand = Animate()
# rand.start()
