# load the library
library(keras)

# 1. load data and rescaling
mnist <- dataset_mnist()
mnist$train$x <- mnist$train$x/255
mnist$test$x <- mnist$test$x/255

# 2. modeling
# 2.1 specify a model
model <- keras_model_sequential() %>% 
  layer_flatten(input_shape = c(28, 28)) %>% 
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dropout(0.2) %>% 
  layer_dense(10, activation = "softmax")

# 2.2 show the model summary
summary(model)

# 2.3 compile the model
model %>% 
  compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = "adam",
    metrics = "accuracy"
  )

# 2.4 fit the model
model %>% 
  fit(
    x = mnist$train$x, y = mnist$train$y,
    epochs = 5,
    validation_split = 0.3,
    verbose = 2
  )

# 3. predict using the model
predictions <- predict(model, mnist$test$x)
head(predictions, 2)

# 4. evaluate the model
model %>% 
  evaluate(mnist$test$x, mnist$test$y, verbose = 0)

