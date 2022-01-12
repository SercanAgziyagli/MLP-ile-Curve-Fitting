import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from rbflayer import RBFLayer, InitCentersRandom

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if logs.get("loss") < 1:
            self.model.stop_training = True


num=150

x_dummy = np.linspace(start=5, stop=15, num=num)
y_plot = np.array(np.cos(x_dummy)*x_dummy)
#y_plot = np.array(np.cos(x_dummy)*x_dummy+ (0.1*np.random.normal(size=num)))

x_dummy=x_dummy.reshape(-1,1)



model = tf.keras.Sequential([RBFLayer(30,initializer=InitCentersRandom(x_dummy),betas=1.0, input_shape=(1,)),
                             tf.keras.layers.Dense(units=1),
                             ])

opt = tf.keras.optimizers.Adam(0.01)
model.compile(optimizer=opt, loss='mean_squared_error')

mycallback=MyCallback()
model.fit(x_dummy,y_plot, epochs=5000,callbacks=[mycallback])

# Compute the output
x_test = np.linspace(start=5.01, stop=14.99, num=num)
y_predicted = model.predict(x_test)

# Display the result
plt.title("RBF")
plt.plot(x_dummy,y_plot,"o")
plt.plot(x_dummy,y_predicted)
plt.xlabel("x")
plt.ylabel("y")
plt.legend(["train", "test"])
plt.grid()
plt.show()
