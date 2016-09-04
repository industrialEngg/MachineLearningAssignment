library(caret)
library(AppliedPredictiveModeling)
set.seed(12331)

setwd("C:/Users/Balagopal/Documents/ML")
pml1<-read.csv(header=TRUE,sep=",", "pml-training2.csv")

# Split Data in Training and Validation Sets
pml1<-pml1[,c(-1,-2,-3,-4,-5,-6,-7)] # Remove columns pertaining to person name

intrain<-createDataPartition(pml1$classe, list=FALSE,p=0.8)
pmlval<-pml1[-intrain,] ## create validation set

pmltemp<-pml1[intrain,] 

pmltrain<-pmltemp
dim(pmltrain) # Print dimension of the data 

# Dimension Reduction using Principal Component Analysis#
pcatest<-prcomp(pmltrain[,-53]) #Principal component analysis was initially perfomed to understand the proportion of variance explained by each PC
summary(pcatest)
plot(pcatest)
pcamodel<-preProcess(pmltrain[,-53], method="pca", pcaComp=52) # Create PCA Model
trainpcadata<-predict(pcamodel, pmltrain[,-53]) # Create principal components for training data
valpcadata<-predict(pcamodel, pmlval[,-53]) # Create principal components for training data


# Parameter Selection and Tuning of SVM
fitcontrol<-trainControl(method="repeatedcv", number = 3, repeats =3) # Setting control parameters for the SVM training; 3 Fold, repeated cross validation

svm.tune<-train(x=trainpcadata, y=pmltrain$classe, method = "svmRadial",tuneLength =9, trControl=fitcontrol)#best parameters can be identified by svm.tune$best 
grid<-expand.grid(sigma = c(0.01308048), C= c(32)) # select best tuning parameters for the SVM
svm.tune2<-train(x=trainpcadata, y=pmltrain$classe, method = "svmRadial",tuneGrid=grid, trControl=fitcontrol)
svmlinear.tune<-train(x=trainpcadata, y=pmltrain$classe, method = "svmLinear",tuneLength =9, trControl=fitcontrol) # Use the Linear Kernel for the SVM vector

