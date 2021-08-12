setwd("D:\\Projects\\R\\GAN")

library(keras)

latent_dim <- 32
height <- 32
width <- 32
channels <- 3
frame <-
  data.frame(matrix(ncol = 2,
                    nrow = 0))
colnames(frame) <- c("generator_loss", "discriminator_loss", use.names = TRUE)
generator_input <- layer_input(shape = c(latent_dim))
generator_output <- generator_input %>%
  
  # First, transform the input into a 16x16 128-channels feature map
  layer_dense(units = 128 * 16 * 16) %>%
  layer_activation_leaky_relu() %>%
  layer_reshape(target_shape = c(16, 16, 128)) %>%
  
  # Then, add a convolution layer
  layer_conv_2d(filters = 256,
                kernel_size = 5,
                padding = "same") %>%
  layer_activation_leaky_relu() %>%
  
  # Upsample to 32x32
  layer_conv_2d_transpose(
    filters = 256,
    kernel_size = 4,
    strides = 2,
    padding = "same"
  ) %>%
  layer_activation_leaky_relu() %>%
  
  # Few more conv layers
  layer_conv_2d(filters = 256,
                kernel_size = 5,
                padding = "same") %>%
  layer_activation_leaky_relu() %>%
  layer_conv_2d(filters = 256,
                kernel_size = 5,
                padding = "same") %>%
  layer_activation_leaky_relu() %>%
  
  # Produce a 32x32 1-channel feature map
  layer_conv_2d(
    filters = channels,
    kernel_size = 7,
    activation = "tanh",
    padding = "same"
  )
generator <- keras_model(generator_input, generator_output)
summary(generator)

discriminator_input <-
  layer_input(shape = c(height, width, channels))
discriminator_output <- discriminator_input %>%
  layer_conv_2d(filters = 128, kernel_size = 3) %>%
  layer_activation_leaky_relu() %>%
  layer_conv_2d(filters = 128,
                kernel_size = 4,
                strides = 2) %>%
  layer_activation_leaky_relu() %>%
  layer_conv_2d(filters = 128,
                kernel_size = 4,
                strides = 2) %>%
  layer_activation_leaky_relu() %>%
  layer_conv_2d(filters = 128,
                kernel_size = 4,
                strides = 2) %>%
  layer_activation_leaky_relu() %>%
  layer_flatten() %>%
  # One dropout layer - important trick!
  layer_dropout(rate = 0.4) %>%
  # Classification layer
  layer_dense(units = 1, activation = "sigmoid")
discriminator <-
  keras_model(discriminator_input, discriminator_output)
summary(discriminator)

# To stabilize training, we use learning rate decay
# and gradient clipping (by value) in the optimizer.
discriminator_optimizer <- optimizer_rmsprop(lr = 0.0008,
                                             clipvalue = 1.0,
                                             decay = 1e-8)
discriminator %>% compile(optimizer = discriminator_optimizer,
                          loss = "binary_crossentropy")

# Set discriminator weights to non-trainable
# (will only apply to the `gan` model)
freeze_weights(discriminator)
gan_input <- layer_input(shape = c(latent_dim))
gan_output <- discriminator(generator(gan_input))
gan <- keras_model(gan_input, gan_output)
gan_optimizer <- optimizer_rmsprop(lr = 0.0004,
                                   clipvalue = 1.0,
                                   decay = 1e-8)
gan %>% compile(optimizer = gan_optimizer,
                loss = "binary_crossentropy")

# Loads CIFAR10 data
cifar10 <- dataset_cifar10()
c(c(x_train, y_train), c(x_test, y_test)) %<-% cifar10

i <- 5
# Select class i (from 1 through 10)
x_train <- x_train[as.integer(y_train) == i, , ,]
# Normalizes data
x_train <- x_train / 255

iterations <- 10000
batch_size <- 20
save_dir <- paste("gan_images")

dir.create(save_dir)

# Start the training loop
start <- 1

for (step in 1:iterations) {
  # Samples random points in the latent space
  random_latent_vectors <- matrix(rnorm(batch_size * latent_dim),
                                  nrow = batch_size,
                                  ncol = latent_dim)
  
  # Decodes them to fake images
  generated_images <- generator %>% predict(random_latent_vectors)
  
  # Combines them with real images
  stop <- start + batch_size - 1
  real_images <- x_train[start:stop, , ,]
  rows <- nrow(real_images)
  combined_images <-
    array(0, dim = c(rows * 2, dim(real_images)[-1]))
  combined_images[1:rows, , ,] <- generated_images
  combined_images[(rows + 1):(rows * 2), , ,] <- real_images
  
  # Assembles labels discriminating real from fake images
  labels <- rbind(matrix(1, nrow = batch_size, ncol = 1),
                  matrix(0, nrow = batch_size, ncol = 1))
  
  # Adds random noise to the labels -- an important trick!
  labels <- labels + (0.5 * array(runif(prod(dim(
    labels
  ))),
  dim = dim(labels)))
  
  # Trains the discriminator
  d_loss <-
    discriminator %>% train_on_batch(combined_images, labels)
  
  # Samples random points in the latent space
  random_latent_vectors <- matrix(rnorm(batch_size * latent_dim),
                                  nrow = batch_size,
                                  ncol = latent_dim)
  
  # Assembles labels that say "all real images"
  misleading_targets <- array(0, dim = c(batch_size, 1))
  
  # Trains the generator (via the gan model, where the
  # discriminator weights are frozen)
  g_loss <- gan %>% train_on_batch(random_latent_vectors,
                                   misleading_targets)
  
  start <- start + batch_size
  if (start > (nrow(x_train) - batch_size))
    start <- 1
  
  # Saves model weights
  save_model_weights_hdf5(gan, "gan.h5")
  
  # Prints metrics
  cat("discriminator loss:", d_loss, "\n")
  cat("generator loss:", g_loss, "\n")
  
  frame <- rbind(frame, c(g_loss, d_loss))
  
  # Saves one generated image
  image_array_save(generated_images[1, , ,] * 255,
                   path = file.path(save_dir, paste("generated_", i, "_", step, ".png")))
  
  # Saves one real image for comparison
  image_array_save(real_images[1, , ,] * 255,
                   path = file.path(save_dir, paste("real_", i, "_", step, ".png")))
}

# # Plot loss function
# library(ggplot2)
# ggplot(data = frame, aes(x = 1:nrow(frame))) +
#   geom_line(aes(y = X0.692201972007751), color="#ff0000") +
#   geom_line(aes(y = X0.689634203910828), color="#0000ff") +
#   xlab("Epochs") +
#   ylab("Loss value") +
#   ggtitle("Loss Function Plot")