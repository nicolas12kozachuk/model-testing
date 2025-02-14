library(pROC)
library(caret)
library(randomForest)
library(MLmetrics)
library(nnet)
library(keras)
library(tensorflow)
library(dplyr)
iyerData <- read.csv("datasets/iyer.txt", sep="\t", header=FALSE)
colnames(iyerData) <- c('ID', 'Class','x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12')
iyerData <- subset(iyerData, Class != -1)
iyerData <- iyerData[,-1] # Don't need ID, repetitive
iyerData$x1 <- as.numeric(iyerData$x1)
iyerData$x2 <- as.numeric(iyerData$x2)
iyerData$x3 <- as.numeric(iyerData$x3)
iyerData$x4 <- as.numeric(iyerData$x4)
iyerData$x5 <- as.numeric(iyerData$x5)
iyerData$x6 <- as.numeric(iyerData$x6)
iyerData$x7 <- as.numeric(iyerData$x7)
iyerData$x8 <- as.numeric(iyerData$x8)
iyerData$x9 <- as.numeric(iyerData$x9)
iyerData$x10 <- as.numeric(iyerData$x10)
iyerData$x11 <- as.numeric(iyerData$x11)
iyerData$x12 <- as.numeric(iyerData$x12)
iyerData$Class <- factor(iyerData$Class, ordered=T)
choData <- read.csv("datasets/cho.txt", sep="\t", header=FALSE)
colnames(choData) <- c('ID', 'Class','x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16')
choData$x1 <- as.numeric(choData$x1)
choData$x2 <- as.numeric(choData$x2)
choData$x3 <- as.numeric(choData$x3)
choData$x4 <- as.numeric(choData$x4)
choData$x5 <- as.numeric(choData$x5)
choData$x6 <- as.numeric(choData$x6)
choData$x7 <- as.numeric(choData$x7)
choData$x8 <- as.numeric(choData$x8)
choData$x9 <- as.numeric(choData$x9)
choData$x10 <- as.numeric(choData$x10)
choData$x11 <- as.numeric(choData$x11)
choData$x12 <- as.numeric(choData$x12)
choData$x3 <- as.numeric(choData$x13)
choData$x14 <- as.numeric(choData$x14)
choData$x15 <- as.numeric(choData$x15)
choData$x16 <- as.numeric(choData$x16)
choData <- choData[,-1] # Don't need ID, repetitive
choData$Class <- factor(choData$Class, ordered=T)
yaleData.train1 <- read.csv("datasets/YaleB32x32/StTrainFile1.txt", sep=" ", header=FALSE)
colnames(yaleData.train1)[1025] <- 'Class'
yaleData.train1$Class <- as.factor(yaleData.train1$Class)

yaleData.train2 <- read.csv("datasets/YaleB32x32/StTrainFile2.txt", sep=" ", header=FALSE)
colnames(yaleData.train2)[1025] <- 'Class'
yaleData.train2$Class <- as.factor(yaleData.train2$Class)

yaleData.train3 <- read.csv("datasets/YaleB32x32/StTrainFile3.txt", sep=" ", header=FALSE)
colnames(yaleData.train3)[1025] <- 'Class'
yaleData.train3$Class <- as.factor(yaleData.train3$Class)

yaleData.test1 <- read.csv("datasets/YaleB32x32/StTestFile1.txt", sep=" ", header=FALSE)
colnames(yaleData.test1)[1025] <- 'Class'
yaleData.test1$Class <- as.factor(yaleData.test1$Class)

yaleData.test2 <- read.csv("datasets/YaleB32x32/StTestFile2.txt", sep=" ", header=FALSE)
colnames(yaleData.test2)[1025] <- 'Class'
yaleData.test2$Class <- as.factor(yaleData.test2$Class)

yaleData.test3 <- read.csv("datasets/YaleB32x32/StTestFile3.txt", sep=" ", header=FALSE)
colnames(yaleData.test3)[1025] <- 'Class'
yaleData.test3$Class <- as.factor(yaleData.test3$Class)
randomForestPredictions <- function(train_data, test_data, ntrees = 500) {
  rf_model <- randomForest(Class ~ ., data = train_data, ntree = ntrees, importance = TRUE)
  predictions <- predict(rf_model, newdata = test_data, type='class')
  predictions <- as.factor(predictions)
  return(predictions)
}
logisticalRegressionPredictions <- function(train_data, test_data) {
  l_model <- nnet(Class ~ ., data = train_data, size = 10, maxit = 1000, linout = TRUE)
  predictions <- predict(l_model, newdata = test_data, type='class')
  predictions <- as.factor(predictions)
  return(predictions)
}
backend <- keras::backend()
f1_score <- function(y_true, y_pred) {
  tp <- backend$sum(backend$cast(y_true * y_pred, 'float'))
  tn <- backend$sum(backend$cast((1 - y_true) * (1 - y_pred), 'float'))
  fp <- backend$sum(backend$cast((1 - y_true) * y_pred, 'float'))
  fn <- backend$sum(backend$cast(y_true * (1 - y_pred), 'float'))
  precision <- tp / (tp + fp + backend$epsilon())
  recall <- tp / (tp + fn + backend$epsilon())
  f1 <- 2 * precision * recall / (precision + recall + backend$epsilon())
  return(f1)
}
f1_score_helper <- function(predictions, labels) {
  labels <- as.factor(labels)
  # Calculate the number of classes
  num_classes <- length(levels(labels))

  # Initialize empty vectors for precision, recall, and f1 scores
  precision <- rep(0, num_classes)
  recall <- rep(0, num_classes)
  f1 <- rep(0, num_classes)

  # Calculate precision, recall, and f1 scores for each class
  for (i in 1:num_classes) {
    tp <- sum(predictions == i & labels == i)
    fp <- sum(predictions == i & labels != i)
    fn <- sum(predictions != i & labels == i)

    if ((tp + fp) > 0) {
      precision[i] <- tp / (tp + fp)
    }

    if ((tp + fn) > 0) {
      recall[i] <- tp / (tp + fn)
    }

    if ((precision[i] + recall[i]) > 0) {
      f1[i] <- 2 * precision[i] * recall[i] / (precision[i] + recall[i])
    }
  }

  # Take the weighted average of f1 scores to obtain the overall f1 score
  weights <- table(labels) / length(labels)
  overall_f1 <- sum(f1 * weights)

  return(overall_f1)
}
cnnPredictions <- function(train_data, train_classes, test_data, test_classes, input_height, input_width, color, num_classes) {
  model <- keras_model_sequential() %>%
    layer_conv_2d(filters = num_classes, kernel_size = c(3, 3), activation = "relu", input_shape = c(input_height, input_width, color)) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = num_classes, kernel_size = c(3, 3), activation = "relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_dropout(rate = 0.25) %>%
    layer_flatten() %>%
    layer_dense(units = 128, activation = "relu") %>%
    layer_dropout(rate = 0.5) %>%
    layer_dense(units = num_classes, activation = "softmax")
  
  # Compile the model
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = "adam",
    metrics = c("accuracy",tensorflow::tf$metrics$AUC(curve = "ROC"), f1_score)
  )

  n_train = dim(train_data)[1]
  n_test = dim(test_data)[1]
  train_data_reshaped <- array_reshape(train_data, c(n_train,32,32,1))
  test_data_reshaped <- array_reshape(test_data, c(n_test,32,32,1))
  train_labels <- to_categorical(train_classes-1, num_classes = num_classes)
  test_labels <- to_categorical(test_classes-1, num_classes = num_classes)
  
  # Fit the model to the training data
  model %>% fit(
    train_data_reshaped, train_labels,
    batch_size = 32,
    epochs = 10,
    validation_split = 0.1
  )

  # Generate predictions for the test data

  predictions <- model %>% predict(test_data_reshaped)
  pr <- model %>% evaluate(test_data_reshaped, test_labels)
  names(pr)[3] <- "auc"
  names(pr)[4] <- "f1"
  return (pr)
}
cnnPredictions1D<- function(train_data, train_classes, test_data, test_classes, num_classes) {
  # Define the model architecture
  model <- keras_model_sequential() %>%
    layer_conv_1d(filters = num_classes, kernel_size = 3, activation = "relu", input_shape = c(dim(train_data)[2], 1)) %>%
    layer_max_pooling_1d(pool_size = 2) %>%
    layer_conv_1d(filters = num_classes, kernel_size = 3, activation = "relu") %>%
    layer_max_pooling_1d(pool_size = 2) %>%
    layer_dropout(rate = 0.25) %>%
    layer_flatten() %>%
    layer_dense(units = 128, activation = "relu") %>%
    layer_dropout(rate = 0.5) %>%
    layer_dense(units = num_classes, activation = "softmax")
  
  # Compile the model
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = "adam",
    metrics = c("accuracy",tensorflow::tf$metrics$AUC(curve = "ROC"), f1_score)
  )

  # Reshape the data to match the input shape of the model
  train_data_reshaped <- array_reshape(train_data, c(dim(train_data)[1], dim(train_data)[2]))
  test_data_reshaped <- array_reshape(test_data, c(dim(test_data)[1], dim(test_data)[2]))

  # Convert the class labels to categorical format
  train_labels <- to_categorical(as.numeric(train_classes) - 1, num_classes = num_classes)
  test_labels <- to_categorical(as.numeric(test_classes) - 1, num_classes = num_classes)

  # Fit the model to the training data
  model %>% fit(
    train_data_reshaped, train_labels,
    batch_size = 32,
    epochs = 10,
    validation_split = 0.1
  )

  # Generate predictions for the test data
  predictions <- model %>% predict(test_data_reshaped)

  # Evaluate the model performance
  pr <- model %>% evaluate(test_data_reshaped, test_labels)
  names(pr)[3] <- "auc"
  names(pr)[4] <- "f1"
  return(pr)
}
analyzeResults <- function(predictions, test_data) {
  levels(predictions) <- levels(test_data$Class)
  accuracy <- mean(predictions == test_data$Class)
  
  f1 <- f1_score_helper(predictions, test_data$Class)

  suppressMessages(
    roc.multi <- multiclass.roc(test_data$Class, as.numeric(predictions))
  )
  res <- list("accuracy" = accuracy, "f1" = f1, "auc" = auc(roc.multi))
  return(res)
}
runTestsKFold <- function(data, n_folds, model_type, n_classes) {
  folds <- createFolds(data$Class, k = n_folds, list = TRUE, returnTrain = FALSE)
  if (!(model_type == 'rf') && !(model_type == 'lr') && !(model_type == 'cnn')) {
    print(paste0('Error, invalid model_type: ', model_type))
    return()
  }
  # Create empty vectors to store the results
  accuracy_results <- rep(0, n_folds)
  f1_results <- rep(0, n_folds)
  auc_results <- rep(0, n_folds)
  for (i in 1:n_folds) {
    # Extract the current fold for testing
    test_indices <- folds[[i]]
    test_data <- data[test_indices, ]
    test_data$Class <- factor(test_data$Class, ordered=F)
    
    # Use the remaining folds for training
    train_data <- data[-test_indices, ]
    if (model_type == 'rf') {
      predictions = randomForestPredictions(train_data, test_data)
      res = analyzeResults(predictions, test_data)
      accuracy_results[i] = res$accuracy
      f1_results[i] = res$f1
      auc_results[i] = res$auc
    } else if (model_type == 'lr') {
      predictions = logisticalRegressionPredictions(train_data, test_data)
      res = analyzeResults(predictions, test_data)
      accuracy_results[i] = res$accuracy
      f1_results[i] = res$f1
      auc_results[i] = res$auc
    } else if (model_type == 'cnn') {
      train_classes <- train_data$Class
      test_classes <- test_data$Class
      train_data$Class <- NULL
      test_data$Class <- NULL
      num_classes <- length(levels(train_classes))
      res = cnnPredictions1D(train_data, train_classes, test_data, test_classes, num_classes)
      accuracy_results[i] = res["accuracy"]
      f1_results[i] = res["f1"]
      auc_results[i] = res["auc"]
      #predictions = logisticalRegressionPredictions(train_data, test_data)
    }
  }
  accuracy = mean(accuracy_results)
  f1 = mean(f1_results)
  auc = mean(auc_results)
  #print(sd(accuracy_results))
  #print(sd(f1_results))
  #print(sd(auc_results))
  return(res)
}
printResults <- function(name, res) {
  print(paste0(name, ' Results'))
  print(paste0(" Accuracy: ", res["accuracy"]))
  print(paste0(" F1: ", res["f1"]))
  print(paste0(" AUC: ", res["auc"]))
}
pcaDimReduction <- function(data, n_components=32) {
  pca_model <- prcomp(data, center = TRUE, scale. = FALSE, retx = TRUE)
  transformed_data <- as.data.frame(pca_model$x[, 1:n_components])
  return(transformed_data)
}
yaleData.train1PCA <- pcaDimReduction(yaleData.train1[1:1024])
yaleData.test1PCA <- pcaDimReduction(yaleData.test1[1:1024])
yaleData.train1PCA$Class <- yaleData.train1$Class
yaleData.test1PCA$Class <- yaleData.test1$Class
yaleData.train1PCA$Class <- as.factor(yaleData.train1PCA$Class)
yaleData.test1PCA$Class <- as.factor(yaleData.test1PCA$Class)

yaleData.train2PCA <- pcaDimReduction(yaleData.train2[1:1024])
yaleData.test2PCA <- pcaDimReduction(yaleData.test2[1:1024])
yaleData.train2PCA$Class <- yaleData.train2$Class
yaleData.test2PCA$Class <- yaleData.test2$Class
yaleData.train2PCA$Class <- as.factor(yaleData.train2PCA$Class)
yaleData.test2PCA$Class <- as.factor(yaleData.test2PCA$Class)

yaleData.train3PCA <- pcaDimReduction(yaleData.train3[1:1024])
yaleData.test3PCA <- pcaDimReduction(yaleData.test3[1:1024])
yaleData.train3PCA$Class <- yaleData.train3$Class
yaleData.test3PCA$Class <- yaleData.test3$Class
yaleData.train3PCA$Class <- as.factor(yaleData.train3PCA$Class)
yaleData.test3PCA$Class <- as.factor(yaleData.test3PCA$Class)
iyer.randomForest <- runTestsKFold(data=iyerData, n_folds=3, model_type='rf')
cho.randomForest <- runTestsKFold(data=choData, n_folds=3, model_type='rf')
ntrees <- 5
yaleData.randomForestPred1 <- randomForestPredictions(yaleData.train1, yaleData.test1, ntrees=ntrees)
yaleData.randomForestPred2 <- randomForestPredictions(yaleData.train2, yaleData.test2, ntrees=ntrees)
yaleData.randomForestPred3 <- randomForestPredictions(yaleData.train3, yaleData.test3, ntrees=ntrees)
yaleData.randomForestRes1 <- analyzeResults(yaleData.randomForestPred1, yaleData.test1)
yaleData.randomForestRes2 <- analyzeResults(yaleData.randomForestPred2, yaleData.test2)
yaleData.randomForestRes3 <- analyzeResults(yaleData.randomForestPred3, yaleData.test3)
printResults('Iyer Random Forests', iyer.randomForest)
printResults('Cho Random Forests', cho.randomForest)
printResults('Yale Random Forests 1', yaleData.randomForestRes1)
printResults('Yale Random Forests 2', yaleData.randomForestRes2)
printResults('Yale Random Forests 3', yaleData.randomForestRes3)
iyerData.logisticRegression <- runTestsKFold(data=iyerData, n_folds=3, model_type='lr')
choData.logisticRegression <- runTestsKFold(data=choData, n_folds=3, model_type='lr')
yaleData.logisticRegressionPred1 <- logisticalRegressionPredictions(yaleData.train1PCA, yaleData.test1PCA)
yaleData.logisticRegressionPred2 <- logisticalRegressionPredictions(yaleData.train2PCA, yaleData.test2PCA)
yaleData.logisticRegressionPred3 <- logisticalRegressionPredictions(yaleData.train3PCA, yaleData.test3PCA)

yaleData.logisticRegressionRes1 <- analyzeResults(yaleData.logisticRegressionPred1, yaleData.test1PCA)
yaleData.logisticRegressionRes2 <- analyzeResults(yaleData.logisticRegressionPred2, yaleData.test2PCA)
yaleData.logisticRegressionRes3 <- analyzeResults(yaleData.logisticRegressionPred3, yaleData.test3PCA)
printResults('Iyer Logistical Regression', iyerData.logisticRegression)
printResults('Cho Logistical Regression', choData.logisticRegression)
printResults('Yale Logistical Regression 1', yaleData.logisticRegressionRes1)
printResults('Yale Logistical Regression 2', yaleData.logisticRegressionRes2)
printResults('Yale Logistical Regression 3', yaleData.logisticRegressionRes3)
yaleData.train1$Class <- as.numeric(yaleData.train1$Class)
yaleData.test1$Class <- as.numeric(yaleData.test1$Class)
yaleData.train2$Class <- as.numeric(yaleData.train2$Class)
yaleData.test2$Class <- as.numeric(yaleData.test2$Class)
yaleData.train3$Class <- as.numeric(yaleData.train3$Class)
yaleData.test3$Class <- as.numeric(yaleData.test3$Class)
yaleData.cnn1 <- cnnPredictions(yaleData.train1[1:1024], yaleData.train1$Class, yaleData.test1[1:1024], yaleData.test1$Class, input_height=32, input_width=32, color=1, num_classes=38)
yaleData.cnn2 <- cnnPredictions(yaleData.train2[1:1024], yaleData.train2$Class, yaleData.test2[1:1024], yaleData.test2$Class, input_height=32, input_width=32, color=1, num_classes=38)
yaleData.cnn3 <- cnnPredictions(yaleData.train3[1:1024], yaleData.train3$Class, yaleData.test3[1:1024], yaleData.test3$Class, input_height=32, input_width=32, color=1, num_classes=38)
iyerData.cnn <- runTestsKFold(iyerData, n_folds=3, model_type='cnn')
choData.cnn <- runTestsKFold(choData, n_folds=3, model_type='cnn')
printResults('Iyer Data CNN', iyerData.cnn)
printResults('Cho Data CNN', choData.cnn)
printResults('Yale CNN Set 1', yaleData.cnn1)
printResults('Yale CNN Set 2', yaleData.cnn2)
printResults('Yale CNN Set 3', yaleData.cnn3)
