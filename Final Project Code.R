#546 final project
library(caret)
library(tidyverse)
library(dplyr)
library(glmnet)
library(emulator)
library(data.table)
library(MLmetrics)
library(class)

train<-cbind(X_train, y_train)
set.seed(2022)
index<-createDataPartition(train$y_train, p=0.8, list=F, times=1)
train_df<-train[index, ]
test_df<-train[-index, ]

#Logistic regression with lasso penalty
fit.lasso <- cv.glmnet(x=data.matrix(train_df[,-188]), train_df[,188], 
                       alpha=1, nlambda=100, family="binomial")
plot(fit.lasso)
log(fit.lasso$lambda.min)
lasso.coef <- fit.lasso$glmnet.fit$beta[,which(fit.lasso$lambda == fit.lasso$lambda.min)]

coefficients<-coef(fit.lasso, s=fit.lasso$lambda.min)

prediction_test<-predict(fit.lasso, newx=as.matrix(test_df[-188]), type="response")
prediction_test<-as.data.frame(prediction_test)
prediction_test$s2.test<-ifelse(prediction_test$lambda.1se>=0.5, 1, 0)
mean(prediction_test$s2.test==test_df$y_train)

prediction_train<-predict(fit.lasso, newx=as.matrix(train_df[-188]), type="response")
prediction_train<-as.data.frame(prediction_train)
prediction_train$s2.test<-ifelse(prediction_train$lambda.1se>=0.5, 1, 0)
mean(prediction_train$s2.test==train_df$y_train)


var<-rownames(coefficients)[coefficients[,1]!=0]
var<-as.data.frame(var)
var<-var[-1,]
var<-as.data.frame(var)
X_train<-X_train[,colnames(X_train)%in%var$var]
train<-cbind(X_train, y_train)
set.seed(2022)
index<-createDataPartition(train$y_train, p=0.8, list=F, times=1)
train_df<-train[index, ]
test_df<-train[-index, ]

#KNN
ctrl<-trainControl(method="cv", number=10)
knn_mod<-train(y_train~.,data=train_df, method="knn", metric="Accuracy", trControl=ctrl)
summary(knn_mod)
knn_mod$finalModel

fit.knn<- knn(train = train_df[,1:125],
              test=test_df[,1:125],
              cl=train_df$y_train, k = 1)

mean(fit.knn == test_df$y_train)

fit.knn<- knn(train = train_df[,1:125],
              test=train_df[,1:125],
              cl=train_df$y_train, k = 1)

mean(fit.knn == train_df$y_train)


klist = c(1:20)
### create empty lists/vectors to store the results
confusion_train = vector("list",length(klist))
confusion_test = confusion_train
plots_knn = confusion_train
accuracy_train = rep(0,length(klist))
accuracy_test = accuracy_train

for (i in 1:length(klist)){
  train_pred_knn <- knn(train = train_df[,1:125],test=train_df[,1:125],cl=train_df$y_train,k=klist[i])
  accuracy_train[i] = mean(train_df$y_train == train_pred_knn)
  
  test_pred_knn <- knn(train=train_df[,1:125],test=test_df[,1:125],cl=train_df$y_train,k=klist[i])
  accuracy_test[i] = mean(test_df$y_train == test_pred_knn)
}

### plot accuracy against k
accuracy = data.frame(cbind(rep(klist,2),c(accuracy_train,accuracy_test)))
colnames(accuracy) = c("k","accuracy")
accuracy$Set = c(rep("training",20),rep("test",20))

ggplot(accuracy, aes(x=k, y=accuracy, color=Set)) + geom_point() + geom_line() + 
  theme_bw() +ylab("Prediction Accuracy")



#Tree
library(tree)
tree.med<-tree(y_train~.,train_df)

set.seed(2022)
cv.med=cv.tree(tree.med, FUN=prune.misclass)
plot(cv.med$size,cv.med$dev/200,type="b", xlab="Number of Terminal Nodes (Leaves)", ylab="Misclassification Error Rate")

prune.med<-prune.tree(tree.med,best=11)

prune.label.test = predict(prune.med, type = "class", newdata = train_df)
mean(prune.label.test == train_df$y_train)

prune.label.test = predict(prune.med, type = "class", newdata = test_df)
mean(prune.label.test == test_df$y_train)


#Random Forest
library(randomForest)
set.seed(2022)
ctrl<-trainControl(method="cv", number=10)
rf_mod<-train(y_train~.,data=train_df, method="rf", metric="Accuracy", trControl=ctrl)
summary(rf_mod)
rf_mod$finalModel

rf.cv<-randomForest(y_train~.,data=train_df, ntree=500, mtry = 25)
rf.label.test = predict(rf.cv, type = "class", newdata = test_df[,-126])
mean(rf.label.test == test_df$y_train)

rf.label.train = predict(rf.cv, type = "class", newdata = train_df[,-126])
mean(rf.label.train == train_df$y_train)

rf.label.predict = predict(rf.cv, type = "class", newdata = X_test)
rf.label.predict<-as.data.frame(rf.label.predict)

set.seed(2022)
klist = seq(from=20, to=100, by=5)

### create empty lists/vectors to store the results
accuracy_test = rep(0,length(klist))

for (i in 1:length(klist)){
  rf <- randomForest(y_train~.,data=train_df, ntree=500, mtry=klist[i])
  
  predict.rf.test<-predict(rf, type = "class", newdata = test_df[,-126])
  accuracy_test[i] = mean(test_df$y_train == predict.rf.test)
}


### plot accuracy against mtry
accuracy = data.frame(klist, accuracy_test)
colnames(accuracy) = c("mtry","accuracy")

ggplot(accuracy, aes(x=mtry, y=accuracy)) + geom_point() + geom_line() + 
  theme_bw() + xlab("Number of Variables Sampled at Each Split") + ylab("Test Prediction Accuracy")

write_csv(rf.label.predict, "final prediction.csv")
