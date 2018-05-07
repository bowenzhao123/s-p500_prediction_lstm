
library(keras)
library(tensorflow)
library(ggplot2)
library(purrr)
library(dplyr)

# file combine
df1 <- read.csv("train.csv",header=T)
df2 <- read.csv("S&P500.csv",header=T)
df2<-df2[dim(df2)[1]:1,]
df3 <- data.frame(open = df1$Open, high = df2$High, low = df2$Low, vol = df1$Volume/1000000,price = df1$Close)

# Data Preprocessing
standard_scaler <- function(target){
  train_1 <- dim(target)[1]
  train_2 <- dim(target)[2]
  train_3 <- dim(target)[3]
  target <- array(target,c(dim(target)[1],dim(target)[2]*dim(target)[3]))
  target <- scale(target,center = T,scale=T)
  target <- array(target,c(train_1,train_2,train_3))
  return(target)
}

get_series <- function(index,data,sequence_length){
  return(data[index:(index+sequence_length),])
}

preprocess_data <- function(stock,seq_len){
  amount_of_features <- dim(stock)[2]
  data <- as.matrix(stock)
  sequence_length <- seq_len
  result <- t(sapply(1:(dim(data)[1]-sequence_length),get_series,data=data,sequence_length=sequence_length))
  result <- array(result,c(dim(result)[1],dim(result)[2]/amount_of_features,amount_of_features))
  row <- as.integer(.9*dim(result)[1])
  train <- result[1:row,,]
  train <- standard_scaler(train)
  result <- standard_scaler(result)
  X_train <<- train[,1:(dim(train)[2]-1),]
  y_train <<- train[,dim(train)[2],dim(train)[3]]
  X_test <<- result[(row-50):dim(result)[1],1:(dim(result)[2]-1),]
  y_test <<- result[(row-50):dim(result)[1],dim(result)[2],dim(result)[3]]
}

# Model training
model <- keras_model_sequential()
model %>%
  layer_lstm(64,return_sequences = T, input_shape = list(NULL,5),dropout = .2,recurrent_dropout = .2,go_backwards=T) %>%
  layer_lstm(64,return_sequences = F, dropout = .2,recurrent_dropout = .2) %>%
  layer_dense(10) %>%
  layer_dense(1,activation="linear")
model %>% compile(loss="mse", optimizer="adam", metrics=c('accuracy'))

# Model fitting
preprocess_data(df3[1:1000,],20)
model %>% fit(X_train,y_train,epochs = 240, batch_size=200,validation_split = .1)
model %>% predict(X_train) -> predictions

# train_result
train_result <- data.frame(x=1:length(y_train),pred = predictions,real = y_train)
ggplot(train_result) + geom_line(aes(x=x,y=pred),color = 'red',size = .2) + geom_line(aes(x=x,y=real),color = 'blue',size = .2)

# test_result
model %>% predict(X_test) -> predictions_test
test_result <- data.frame(x=1:length(y_test),pred = predictions_test,real = y_test)
ggplot(test_result) + geom_line(aes(x=x,y=pred),color = 'red',size = .2) + geom_line(aes(x=x,y=real),color = 'blue',size = .2) + ylim(c(-2,3))








