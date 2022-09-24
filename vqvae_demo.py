import imp
from operator import mod
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf
from util import *
from data import *
import time

class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        if DEBUG:
            print("quantized input shape ", x.shape)
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])
        if DEBUG:
            print("after reshape x ", x.shape)

        # Quantization. 找到距离自己最近的一个向量
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        # Reshape the quantized values back to the original input shape
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        if DEBUG:
            # print("quantized input shape ", input_shape)
            print("before quantized x shape ", x.shape)
            print("quantized shape ", quantized.shape)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs**2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings**2, axis=0, keepdims=True)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices

def get_encoder(latent_dim=16):
    # encoder_inputs = keras.Input(shape=(28, 28, 1))
    # encoder_inputs = keras.Input(shape=(time_step, input_dim), name='encoder_input')
    encoder_inputs = keras.Input(shape=(1, 900), name='encoder_input')
    run1 = layers.Bidirectional(layers.GRU(rnn_dim, return_sequences=True), name='rnn1')(encoder_inputs)
    # run2 = layers.Bidirectional(layers.GRU(rnn_dim, return_sequences=True), name='rnn2')(run1)
    run2 = layers.Bidirectional(layers.GRU(rnn_dim), name='rnn2')(run1)
    if DEBUG:
        print("encoder input shape ", encoder_inputs.shape)
        print("run1 shape", run1.shape)
    # encoder_outputs = layers.Conv2D(latent_dim, 1, padding="same")(x)
    # z_mean = layers.Dense(z_dim, name='z_mean')(rnn2)
    # z_log_var = layers.Dense(z_dim, name='z_log_var')(rnn2)

    # def sampling(args):
    #     z_mean, z_log_var = args
    #     batch = K.shape(z_mean)[0]
    #     dim = K.int_shape(z_mean)[1]
    #     # by default, random_normal has mean=0 and std=1.0
    #     epsilon = K.random_normal(shape=(batch, dim))
    #     return z_mean + K.exp(0.5 * z_log_var) * epsilon

    # z = Lambda(sampling, output_shape=(z_dim,), name='z')([z_mean, z_log_var])
    return keras.Model(encoder_inputs, run2, name="encoder")


def get_decoder(latent_dim=16):
    latent_inputs = keras.Input(shape=2 * rnn_dim, name='z_sampling')
    if DEBUG:
        print("latent input shape ", latent_inputs.shape)
    repeated_z = layers.RepeatVector(time_step, name='repeated_z_tension')(latent_inputs)
    if DEBUG:
        print("repeated z shape ", repeated_z.shape)
    rnn1_output = layers.GRU(rnn_dim, name='decoder_rnn1', return_sequences=True)(repeated_z)
    # rnn1_output = layers.GRU(rnn_dim, name='decoder_rnn1', return_sequences=True)(latent_inputs)
    rnn2_output = layers.GRU(rnn_dim, name='decoder_rnn2', return_sequences=True)(rnn1_output)
    # decoder_outputs = layers.Conv2DTranspose(1, 3, padding="same")(x)

    tensile_middle_output = layers.TimeDistributed(layers.Dense(tension_middle_dim, activation='elu'),
                                            name='tensile_strain_dense1')(rnn2_output)

    tensile_output = layers.TimeDistributed(layers.Dense(tension_output_dim, activation='elu'),
                                     name='tensile_strain_dense2')(tensile_middle_output)

    diameter_middle_output = layers.TimeDistributed(layers.Dense(tension_middle_dim, activation='elu'),
                                             name='diameter_strain_dense1')(rnn2_output)

    diameter_output = layers.TimeDistributed(layers.Dense(tension_output_dim, activation='elu'),
                                      name='diameter_strain_dense2')(diameter_middle_output)

    melody_rhythm_1 = layers.TimeDistributed(layers.Dense(start_middle_dim, activation='elu'),
                                      name='melody_start_dense1')(rnn2_output)
    melody_rhythm_output = layers.TimeDistributed(layers.Dense(melody_note_start_dim, activation='sigmoid'),
                                           name='melody_start_dense2')(
        melody_rhythm_1)

    melody_pitch_1 = layers.TimeDistributed(layers.Dense(melody_bass_dense_1_dim, activation='elu'),
                                     name='melody_pitch_dense1')(rnn2_output)

    melody_pitch_output = layers.TimeDistributed(layers.Dense(melody_output_dim, activation='softmax'),
                                          name='melody_pitch_dense2')(melody_pitch_1)

    bass_rhythm_1 = layers.TimeDistributed(layers.Dense(start_middle_dim, activation='elu'),
                                    name='bass_start_dense1')(rnn2_output)

    bass_rhythm_output = layers.TimeDistributed(layers.Dense(bass_note_start_dim, activation='sigmoid'),
                                         name='bass_start_dense2')(
        bass_rhythm_1)

    bass_pitch_1 = layers.TimeDistributed(layers.Dense(melody_bass_dense_1_dim, activation='elu'),
                                   name='bass_pitch_dense1')(rnn2_output)
    bass_pitch_output = layers.TimeDistributed(layers.Dense(bass_output_dim, activation='softmax'),
                                        name='bass_pitch_dense2')(bass_pitch_1)

    decoder_output = [melody_pitch_output, melody_rhythm_output, bass_pitch_output, bass_rhythm_output,
                      tensile_output, diameter_output
                      ]
    return keras.Model(latent_inputs, decoder_output, name="decoder")


"""
## Standalone VQ-VAE model
"""


def get_vqvae(latent_dim=16, num_embeddings=64):
    vq_layer = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")
    encoder = get_encoder(latent_dim)
    decoder = get_decoder(latent_dim)
    # inputs = keras.Input(shape=(time_step, input_dim), name='encoder_input')
    inputs = keras.Input(shape=(1, 900), name='encoder_input')
    encoder_outputs = encoder(inputs)
    quantized_latents = vq_layer(encoder_outputs)
    if DEBUG:
        print("qualized shape ", quantized_latents.shape)
        print("encoder output ", encoder_outputs.shape)
    reconstructions = decoder(quantized_latents)
    model = keras.Model(inputs, reconstructions, name="vq_vae")
    # vqvae_trainer = model(latent_dim=16, num_embeddings=128)
    # model.compile(optimizer=keras.optimizers.Adam())
    model.compile(optimizer=keras.optimizers.Adam(),
                loss=['categorical_crossentropy', 'binary_crossentropy',
                        'categorical_crossentropy', 'binary_crossentropy',
                        'mse', 'mse'
                        ],
                metrics=[[keras.metrics.CategoricalAccuracy()],
                            [keras.metrics.BinaryAccuracy()],
                            [keras.metrics.CategoricalAccuracy()],
                            [keras.metrics.BinaryAccuracy()],
                            [keras.metrics.MeanSquaredError()],
                            [keras.metrics.MeanSquaredError()]
                            ]
                )

    return model


# def vqvae(latent_dim=16, num_embeddings=64):
#     vq_layer = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")
#     encoder = get_encoder(latent_dim)
#     decoder = get_decoder(latent_dim)
#     inputs = keras.Input(shape=(time_step, input_dim), name='encoder_input')
#     encoder_outputs = encoder(inputs)
#     quantized_latents = vq_layer(encoder_outputs)
#     reconstructions = decoder(quantized_latents)
#     return keras.Model(inputs, reconstructions, name="vq_vae")

get_vqvae().summary()

"""
Note that the output channels of the encoder should match the `latent_dim` for the vector
quantizer.
"""

"""
## Wrapping up the training loop inside `VQVAETrainer`
"""


class VQVAETrainer(keras.models.Model):
    def __init__(self, latent_dim=32, num_embeddings=128, **kwargs):
        super(VQVAETrainer, self).__init__(**kwargs)
        # self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        self.vqvae = get_vqvae(self.latent_dim, self.num_embeddings)

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

def train_step(model, train_x):
    with tf.GradientTape() as tape:
        # Outputs from the VQ-VAE.
        if DEBUG:
            print("epoch x shape ", train_x.shape)
        reconstructions = model(train_x)

        # Calculate the losses.
        reconstruction_loss = (
            # tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
            tf.reduce_mean((train_x - reconstructions) ** 2)
        )
        total_loss = reconstruction_loss + sum(model.losses)

    # Backpropagation.
    grads = tape.gradient(total_loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Loss tracking.
    # self.total_loss_tracker.update_state(total_loss)
    # self.reconstruction_loss_tracker.update_state(reconstruction_loss)
    # self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

    # Log results.
    return {
        # "loss": self.total_loss_tracker.result(),
        # "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        # "vqvae_loss": self.vq_loss_tracker.result(),
    }


# """
# ## Load and preprocess the MNIST dataset
# """

# (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
# TrackSet_1, TrackSet_2 = DatasetLoader(DATA_PATH)

# x_train = np.expand_dims(x_train, -1)
# x_test = np.expand_dims(x_test, -1)
# x_train_scaled = (x_train / 255.0) - 0.5
# x_test_scaled = (x_test / 255.0) - 0.5

# data_variance = np.var(x_train / 255.0)

# """
# ## Train the VQ-VAE model
# """

# vqvae_trainer = VQVAETrainer(latent_dim=16, num_embeddings=128)
# vqvae_trainer.compile(optimizer=keras.optimizers.Adam())
# vqvae_trainer.compile(optimizer=keras.optimizers.Adam(),
#             loss=['categorical_crossentropy', 'binary_crossentropy',
#                     'categorical_crossentropy', 'binary_crossentropy',
#                     'mse', 'mse'
#                     ],
#             metrics=[[keras.metrics.CategoricalAccuracy()],
#                         [keras.metrics.BinaryAccuracy()],
#                         [keras.metrics.CategoricalAccuracy()],
#                         [keras.metrics.BinaryAccuracy()],
#                         [keras.metrics.MeanSquaredError()],
#                         [keras.metrics.MeanSquaredError()]
#                         ]
#             )

# vqvae_trainer.fit(train_dataset, epochs=30, batch_size=5)

def train(train_dataset, test_dataset, model, save):
    # for epoch in range(1, epochs + 1):
    for epoch in range(1, 2):
        start_time = time.time()
        print("train dataset shape ", len(train_dataset))
        for train_x in train_dataset:
            # print(len(train_x))
            train_x = np.asarray(train_x)[0]
            # print("train_x ", train_x)
            print("train shape", train_x.shape)
            train_step(model, train_x)
        end_time = time.time()

        # loss = tf.keras.metrics.Mean()
        # for test_x in test_dataset:
        #     test_x = np.asarray(test_x)[0]
        #     loss(compute_loss(model, test_x))
        # display.clear_output(wait=False)
        # elbo = -loss.result()
        # print("train shape", train_x.shape)
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'.format(epoch, 
                                                                                    #    elbo, 
                                                                                       end_time - start_time
                                                                                      ))
        # generate_and_save_images(model,
        #                          epoch, 
        #                          test_sample,
        #                          save)

model = get_vqvae()

train(train_dataset, test_dataset, model, 'jazz')

# /home/u21s052015/miniconda3/envs/tension/bin

# """
# ## Reconstruction results on the test set
# """


# def show_subplot(original, reconstructed):
#     plt.subplot(1, 2, 1)
#     plt.imshow(original.squeeze() + 0.5)
#     plt.title("Original")
#     plt.axis("off")

#     plt.subplot(1, 2, 2)
#     plt.imshow(reconstructed.squeeze() + 0.5)
#     plt.title("Reconstructed")
#     plt.axis("off")

#     plt.show()


# trained_vqvae_model = vqvae_trainer.vqvae
# idx = np.random.choice(len(x_test_scaled), 10)
# test_images = x_test_scaled[idx]
# reconstructions_test = trained_vqvae_model.predict(test_images)

# for test_image, reconstructed_image in zip(test_images, reconstructions_test):
#     show_subplot(test_image, reconstructed_image)

# """
# These results look decent. You are encouraged to play with different hyperparameters
# (especially the number of embeddings and the dimensions of the embeddings) and observe how
# they affect the results.
# """

# """
# ## Visualizing the discrete codes
# """

# encoder = vqvae_trainer.vqvae.get_layer("encoder")
# quantizer = vqvae_trainer.vqvae.get_layer("vector_quantizer")

# encoded_outputs = encoder.predict(test_images)
# flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
# codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
# codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])

# for i in range(len(test_images)):
#     plt.subplot(1, 2, 1)
#     plt.imshow(test_images[i].squeeze() + 0.5)
#     plt.title("Original")
#     plt.axis("off")

#     plt.subplot(1, 2, 2)
#     plt.imshow(codebook_indices[i])
#     plt.title("Code")
#     plt.axis("off")
#     plt.show()

# """
# The figure above shows that the discrete codes have been able to capture some
# regularities from the dataset. Now, how do we sample from this codebook to create
# novel images? Since these codes are discrete and we imposed a categorical distribution
# on them, we cannot use them yet to generate anything meaningful until we can generate likely
# sequences of codes that we can give to the decoder. 

# The authors use a PixelCNN to train these codes so that they can be used as powerful priors to
# generate novel examples. PixelCNN was proposed in
# [Conditional Image Generation with PixelCNN Decoders](https://arxiv.org/abs/1606.05328)
# by van der Oord et al. We borrow the implementation from
# [this PixelCNN example](https://keras.io/examples/generative/pixelcnn/). It's an auto-regressive
# generative model where the outputs are conditional on the prior ones. In other words, a PixelCNN
# generates an image on a pixel-by-pixel basis. For the purpose in this example, however, its task
# is to generate code book indices instead of pixels directly. The trained VQ-VAE decoder is used
# to map the indices generated by the PixelCNN back into the pixel space.
# """














# """
# ## PixelCNN hyperparameters
# """

# num_residual_blocks = 2
# num_pixelcnn_layers = 2
# pixelcnn_input_shape = encoded_outputs.shape[1:-1]
# print(f"Input shape of the PixelCNN: {pixelcnn_input_shape}")

# """
# This input shape represents the reduction in the resolution performed by the encoder. With "same" padding,
# this exactly halves the "resolution" of the output shape for each stride-2 convolution layer. So, with these
# two layers, we end up with an encoder output tensor of 7x7 on axes 2 and 3, with the first axis as the batch
# size and the last axis being the code book embedding size. Since the quantization layer in the autoencoder
# maps these 7x7 tensors to indices of the code book, these output layer axis sizes must be matched by the
# PixelCNN as the input shape. The task of the PixelCNN for this architecture is to generate _likely_ 7x7
# arrangements of codebook indices.

# Note that this shape is something to optimize for in larger-sized image domains, along with the code
# book sizes. Since the PixelCNN is autoregressive, it needs to pass over each codebook index sequentially
# in order to generate novel images from the codebook. Each stride-2 (or rather more correctly a 
# stride (2, 2)) convolution layer will divide the image generation time by four. Note, however, that there
# is probably a lower bound on this part: when the number of codes for the image to reconstruct is too small,
# it has insufficient information for the decoder to represent the level of detail in the image, so the
# output quality will suffer. This can be amended at least to some extent by using a larger code book. 
# Since the autoregressive part of the image generation procedure uses codebook indices, there is far less of 
# a performance penalty on using a larger code book as the lookup time for a larger-sized code from a larger
# code book is much smaller in comparison to iterating over a larger sequence of code book indices, although
# the size of the code book does impact on the batch size that can pass through the image generation procedure.
# Finding the sweet spot for this trade-off can require some architecture tweaking and could very well differ
# per dataset.
# """

# """
# ## PixelCNN model

# Majority of this comes from
# [this example](https://keras.io/examples/generative/pixelcnn/).

# ## Notes

# Thanks to [Rein van 't Veer](https://github.com/reinvantveer) for improving this example with
# copy-edits and minor code clean-ups.
# """

# # The first layer is the PixelCNN layer. This layer simply
# # builds on the 2D convolutional layer, but includes masking.
# class PixelConvLayer(layers.Layer):
#     def __init__(self, mask_type, **kwargs):
#         super(PixelConvLayer, self).__init__()
#         self.mask_type = mask_type
#         self.conv = layers.Conv2D(**kwargs)

#     def build(self, input_shape):
#         # Build the conv2d layer to initialize kernel variables
#         self.conv.build(input_shape)
#         # Use the initialized kernel to create the mask
#         kernel_shape = self.conv.kernel.get_shape()
#         self.mask = np.zeros(shape=kernel_shape)
#         self.mask[: kernel_shape[0] // 2, ...] = 1.0
#         self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
#         if self.mask_type == "B":
#             self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

#     def call(self, inputs):
#         self.conv.kernel.assign(self.conv.kernel * self.mask)
#         return self.conv(inputs)


# # Next, we build our residual block layer.
# # This is just a normal residual block, but based on the PixelConvLayer.
# class ResidualBlock(keras.layers.Layer):
#     def __init__(self, filters, **kwargs):
#         super(ResidualBlock, self).__init__(**kwargs)
#         self.conv1 = keras.layers.Conv2D(
#             filters=filters, kernel_size=1, activation="relu"
#         )
#         self.pixel_conv = PixelConvLayer(
#             mask_type="B",
#             filters=filters // 2,
#             kernel_size=3,
#             activation="relu",
#             padding="same",
#         )
#         self.conv2 = keras.layers.Conv2D(
#             filters=filters, kernel_size=1, activation="relu"
#         )

#     def call(self, inputs):
#         x = self.conv1(inputs)
#         x = self.pixel_conv(x)
#         x = self.conv2(x)
#         return keras.layers.add([inputs, x])


# pixelcnn_inputs = keras.Input(shape=pixelcnn_input_shape, dtype=tf.int32)
# ohe = tf.one_hot(pixelcnn_inputs, vqvae_trainer.num_embeddings)
# x = PixelConvLayer(
#     mask_type="A", filters=128, kernel_size=7, activation="relu", padding="same"
# )(ohe)

# for _ in range(num_residual_blocks):
#     x = ResidualBlock(filters=128)(x)

# for _ in range(num_pixelcnn_layers):
#     x = PixelConvLayer(
#         mask_type="B",
#         filters=128,
#         kernel_size=1,
#         strides=1,
#         activation="relu",
#         padding="valid",
#     )(x)

# out = keras.layers.Conv2D(
#     filters=vqvae_trainer.num_embeddings, kernel_size=1, strides=1, padding="valid"
# )(x)

# pixel_cnn = keras.Model(pixelcnn_inputs, out, name="pixel_cnn")
# pixel_cnn.summary()

# """
# ## Prepare data to train the PixelCNN

# We will train the PixelCNN to learn a categorical distribution of the discrete codes.
# First, we will generate code indices using the encoder and vector quantizer we just
# trained. Our training objective will be to minimize the crossentropy loss between these
# indices and the PixelCNN outputs. Here, the number of categories is equal to the number
# of embeddings present in our codebook (128 in our case). The PixelCNN model is
# trained to learn a distribution (as opposed to minimizing the L1/L2 loss), which is where
# it gets its generative capabilities from.
# """

# # Generate the codebook indices.
# encoded_outputs = encoder.predict(x_train_scaled)
# flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
# codebook_indices = quantizer.get_code_indices(flat_enc_outputs)

# codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])
# print(f"Shape of the training data for PixelCNN: {codebook_indices.shape}")

# """
# ## PixelCNN training
# """

# pixel_cnn.compile(
#     optimizer=keras.optimizers.Adam(3e-4),
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=["accuracy"],
# )
# pixel_cnn.fit(
#     x=codebook_indices,
#     y=codebook_indices,
#     batch_size=128,
#     epochs=30,
#     validation_split=0.1,
# )

# """
# We can improve these scores with more training and hyperparameter tuning.
# """

# """
# ## Codebook sampling

# Now that our PixelCNN is trained, we can sample distinct codes from its outputs and pass
# them to our decoder to generate novel images.
# """

# # Create a mini sampler model.
# inputs = layers.Input(shape=pixel_cnn.input_shape[1:])
# outputs = pixel_cnn(inputs, training=False)
# categorical_layer = tfp.layers.DistributionLambda(tfp.distributions.Categorical)
# outputs = categorical_layer(outputs)
# sampler = keras.Model(inputs, outputs)

# """
# We now construct a prior to generate images. Here, we will generate 10 images.
# """

# # Create an empty array of priors.
# batch = 10
# priors = np.zeros(shape=(batch,) + (pixel_cnn.input_shape)[1:])
# batch, rows, cols = priors.shape

# # Iterate over the priors because generation has to be done sequentially pixel by pixel.
# for row in range(rows):
#     for col in range(cols):
#         # Feed the whole array and retrieving the pixel value probabilities for the next
#         # pixel.
#         probs = sampler.predict(priors)
#         # Use the probabilities to pick pixel values and append the values to the priors.
#         priors[:, row, col] = probs[:, row, col]

# print(f"Prior shape: {priors.shape}")

# """
# We can now use our decoder to generate the images.
# """

# # Perform an embedding lookup.
# pretrained_embeddings = quantizer.embeddings
# priors_ohe = tf.one_hot(priors.astype("int32"), vqvae_trainer.num_embeddings).numpy()
# quantized = tf.matmul(
#     priors_ohe.astype("float32"), pretrained_embeddings, transpose_b=True
# )
# quantized = tf.reshape(quantized, (-1, *(encoded_outputs.shape[1:])))

# # Generate novel images.
# decoder = vqvae_trainer.vqvae.get_layer("decoder")
# generated_samples = decoder.predict(quantized)

# for i in range(batch):
#     plt.subplot(1, 2, 1)
#     plt.imshow(priors[i])
#     plt.title("Code")
#     plt.axis("off")

#     plt.subplot(1, 2, 2)
#     plt.imshow(generated_samples[i].squeeze() + 0.5)
#     plt.title("Generated Sample")
#     plt.axis("off")
#     plt.show()

# """
# We can enhance the quality of these generated samples by tweaking the PixelCNN.
# """

# """
# ## Additional notes

# * After the VQ-VAE paper was initially released, the authors developed an exponential
# moving averaging scheme to update the embeddings inside the quantizer. If you're
# interested you can check out
# [this snippet](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py#L124).
# * To further enhance the quality of the generated samples,
# [VQ-VAE-2](https://arxiv.org/abs/1906.00446) was proposed that follows a cascaded
# approach to learn the codebook and to generate the images.
# """
