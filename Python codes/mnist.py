from typing import List, Optional

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


# data preparation pipeline
from algorithms import Node


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


(train_data, dev_data), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


def standard_train_data(shard_index: int = 0, num_shards: int = 0):
    t_data = train_data.map(normalize_img)
    if num_shards > 1:
        t_data = t_data.shard(num_shards, shard_index)
    return t_data.shuffle(1024).batch(32).prefetch(20)


def standard_dev_data(shard_index: int = 0, num_shards: int = 0):
    d_data = dev_data.map(normalize_img)
    if num_shards > 1:
        d_data = d_data.shard(num_shards, shard_index)
    return d_data.batch(32).prefetch(20)


def unbalanced_train_data(shard_index: int = 0, num_shards: int = 1):
    t_data = train_data.map(normalize_img)
    if num_shards > 1:
        t_data = t_data.filter(lambda img, label: label % num_shards == shard_index)
    return t_data.shuffle(1024).batch(32).prefetch(20)

# dev_batch_1 = tfds.load('mnist', split='test', as_supervised=True).map(normalize_img)

# standard model to load
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)
])
# adding optimizer, loss
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(name="kbest"),
        # tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True, name="ce")
    ]
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=loss_fn,
    metrics=metrics
)


class NodeMnist(Node):
    def __init__(self, local_index: int = 0, num_nodes: int = 1, sharding: str = "std"):
        self.dev_data = standard_dev_data()
        if sharding == "std":
            self.train_data = iter(standard_train_data(shard_index=local_index, num_shards=num_nodes).repeat())
        else:
            self.train_data = iter(unbalanced_train_data(shard_index=local_index, num_shards=num_nodes).repeat())
        self.model = model.from_config(model.get_config())
        self.model(next(self.train_data)[0])  # building the model
        self.trainable_variable_shapes = [var.shape for var in self.model.trainable_variables]
        self._primal_dim = sum(np.prod(var_shape) for var_shape in self.trainable_variable_shapes)
        self.x_av = np.zeros(self._primal_dim)
        self.t = 0
        self.loss_fn = loss_fn
        self.metrics = [metric.from_config(metric.get_config()) for metric in metrics]

    @property
    def primal_dim(self) -> int:
        return self._primal_dim

    def unflatten_weight(self, weight: np.ndarray):
        counter = 0
        var_weights = []
        for var_shape in self.trainable_variable_shapes:
            delta = np.prod(var_shape)
            var_weights.append(tf.cast(tf.reshape(weight[counter:counter + delta], var_shape), dtype=tf.float32))
            counter += delta
        assert counter == self._primal_dim
        return var_weights

    def model_assign(self, var_weights: List[tf.Tensor]):
        for var_weight, var in zip(var_weights, self.model.trainable_variables):
            var.assign(var_weight)

    def update_average_weight(self, xt: np.ndarray):
        self.t += 1
        self.x_av = ((self.t - 1) * self.x_av + xt) / self.t

    def gradient(self, xt: np.ndarray):
        self.update_average_weight(xt)
        self.model_assign(self.unflatten_weight(xt))
        images, labels = next(self.train_data)
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(
                self.loss_fn(labels, self.model(images, training=True))
            )
        grad = tape.gradient(loss, self.model.trainable_variables)
        return flatten_grad(grad).numpy()

    def evaluate(self, x: Optional[np.ndarray] = None):
        if x is None:
            x = self.x_av
        self.model_assign(self.unflatten_weight(x))
        for metric in self.metrics:
            metric.reset_states()
        for images, labels in self.dev_data:
            res = self.model(images, training=False)
            for metric in self.metrics:
                metric.update_state(labels, res)
        return {metric.name: metric.result() for metric in self.metrics}


def flatten_grad(grad):
    def _flatten(tensor):
        return tf.reshape(tensor, [-1])

    return tf.concat(tf.nest.map_structure(_flatten, grad), 0)


CKPT_MODEL = "artifacts/mnist_model"


def restore_checkpoint(ckpt_model):
    tf.train.Checkpoint(model=model).restore(ckpt_model + "-1")


def test_train():
    # training the model
    model.fit(
        train_data, epochs=6, validation_data=dev_data
    )
    # saving the model
    tf.train.Checkpoint(model=model).save(CKPT_MODEL)


def test_mnist_node():
    node = NodeMnist()
    init_weight = np.zeros(node.primal_dim)
    lr = 1e-2
    for ind in range(1000):
        init_weight -= lr * node.gradient(init_weight)
        if ind % 100 == 0:
            print(node.evaluate(init_weight))


if __name__ == "__main__":
    test_mnist_node()
