setwd("C:/Users/Doug/Desktop/Caterpillar/Data")

memory.limit(size = 30000)

#loading necessary packages and files
library(tidyverse)
library(dummy)
library(FNN)
library(magrittr)
library(caret)
library(Metrics)
library(e1071)
library(factoextra)
library(randomForest)

source('score.R')

# Reading in full dataset
df <- read.csv("MERGED_BASE_DATA_RBIND.csv", stringsAsFactors = FALSE, na.strings = "")
# Code to put distance in, which actually makes model predictions worse

#dist <- read.csv("distance.csv", stringsAsFactors = FALSE, na.strings = c(""), strip.white = T)

#df = df[order(df1$SER_COMB, df1$LT_ORDRCPT_DLRIN_DLROUT),]
#dist = dist[order(dist$SER_COMB, dist$LT_ORDRCPT_DLRIN_DLROUT),]

#df = cbind(df, DISTANCE = dist$DISTANCE)

# Lists product lines
unique(df$PRODUCT_LINE)

# Place Desired Product Line here
product_line <- ("MEDIUM EXCAVATORS (320-335)")

# Subsetting data for product line
product <- df %>%
  filter(PRODUCT_LINE == product_line )%>%
  mutate(month = lubridate::month(ORD_RCPT_DT, label =TRUE))%>%
  dplyr::select(SRC_FAC_CD,
                ORD_DLR_CD,
                SLS_MDL,
                DIFF_ORD_STAT,
                month,
                INVENTORY,
                LANE,
                MKT_REGN_ABR_NM,
                MKT_DST_ABR_NM,
                Dealer,
                STRT_DT_FLAG,
                ORD_FCTRY_RTS_DT_FLAG,
                DLR_IN_DT_FLAG,
                FSHIP_DT_FLAG,
                ORD_SHP_AFT_DT_FLAG,
                ORD_SHP_BEF_DT_FLAG,
                DLR_OUT_DT_FLAG,
                ORD_DLR_COUNTRY,
                LT_ORDRCPT_DLRIN_DLROUT)
                #DISTANCE)


# Creating Dummy Variables
cats <- categories(product[,c("SRC_FAC_CD", 
                                 "ORD_DLR_CD", 
                                 "SLS_MDL",
                                 "DIFF_ORD_STAT",
                                 "INVENTORY",
                                 "LANE",
                                 "month",
                                 "MKT_REGN_ABR_NM",
                                 "MKT_DST_ABR_NM",
                                 "ORD_DLR_COUNTRY")], p = "all")

dummies <- dummy(product[,c("SRC_FAC_CD", 
                               "ORD_DLR_CD", 
                               "SLS_MDL",
                               "DIFF_ORD_STAT",
                               "INVENTORY",
                               "LANE",
                               "month",
                               "MKT_REGN_ABR_NM",
                               "MKT_DST_ABR_NM",
                               "ORD_DLR_COUNTRY")],
                 object=cats,
                 int=TRUE)


cats2 <- data.frame(product, dummies)

# removing unneeded features
cats2 <- cats2 %>% 
  dplyr::select(-SRC_FAC_CD, -ORD_DLR_CD, -DIFF_ORD_STAT, -SLS_MDL, -INVENTORY,
                -LANE, -MKT_REGN_ABR_NM, -MKT_DST_ABR_NM, -ORD_DLR_COUNTRY, -month)

# Filtering outliers on 90% confidence 
lower = quantile(cats2$LT_ORDRCPT_DLRIN_DLROUT, probs = seq(0, 1, by= 0.05))[2] ;lower
upper = quantile(cats2$LT_ORDRCPT_DLRIN_DLROUT, probs = seq(0, 1, by= 0.05))[20] ;upper
lower = ifelse(lower < 5, 5, lower)
upper = ifelse(upper > 300, 300, upper)
# Med Excavators: 90% of data falls between 8 and 199

cats2 <- filter(cats2, LT_ORDRCPT_DLRIN_DLROUT >= lower)
cats2 <- filter(cats2, LT_ORDRCPT_DLRIN_DLROUT <= upper)


# removing incomplete observations
cats2 <- na.omit(cats2)
lt <- cats2$LT_ORDRCPT_DLRIN_DLROUT
#dist <- cats2$DISTANCE

qplot(cats2$LT_ORDRCPT_DLRIN_DLROUT,
      geom="histogram",
      # binwidth = 2,  
      main = "Lead Time Distribution", 
      xlab = "Days",
      fill = I('seagreen4'),
      col = I('black'))

#################### PRINCIPAL COMPONENTS ANALYSIS ###########

cats2$LT_ORDRCPT_DLRIN_DLROUT <- NULL
#cats2$DISTANCE <- NULL

prin_comp <- prcomp(cats2)

# Finding eigenvalues of PCs
eig.val <- get_eigenvalue(prin_comp)
eig.val[,3] 

cats2 <- data.frame(LT_ORDRCPT_DLRIN_DLROUT = lt, prin_comp$x)
cats2 <- cats2[,1:14]
#13 pcs is best

#Partitioning data to 80% train and 20% test
#set.seed(1234)
train.ind <- rbinom(nrow(cats2), size=1, prob=0.80)
training.data <- cats2[train.ind==1,]
testing.data  <- cats2[train.ind==0,]

y_train <- training.data[, "LT_ORDRCPT_DLRIN_DLROUT"]
y_test <- testing.data[, "LT_ORDRCPT_DLRIN_DLROUT"]

####################### MODELING ################################

####################### RANDOM FOREST ###########################

fit <- randomForest(LT_ORDRCPT_DLRIN_DLROUT ~ .,data = training.data, ntree=500)
summary(fit)
te_rf_yhat <-  predict(fit, testing.data)
tr_rf_yhat <-  predict(fit, training.data)

# rf scoring
score(y_test, te_rf_yhat)
rmse(y_train, tr_rf_yhat)

save.image('save1')
##################### SUPPORT VECTOR MACHINE ##################### 

svm <- svm(LT_ORDRCPT_DLRIN_DLROUT ~ ., data = training.data)

tr_svm_yhat <- predict(svm, training.data)
te_svm_yhat <- predict(svm, testing.data)

save.image('save2')
################### KNN REGRESSION ##############################

knnreg_tr <- knn.reg(train = training.data[-1], y = y_train, k = 5)
knnreg_te <- knn.reg(train = training.data[-1], test = testing.data[-1], y = y_train, k = 5)

############################## GRADIENT BOOSTING METHOD################### 

set.seed(1234)
fitControl <- trainControl(method = 'cv', number = 6, summaryFunction=defaultSummary)
Grid <- expand.grid( n.trees = seq(50,2000,50), interaction.depth = c(30), shrinkage = c(0.1), n.minobsinnode=10)
formula <- y_train ~ .
fit.gbm <- train(LT_ORDRCPT_DLRIN_DLROUT ~ ., data = training.data, method = 'gbm', trControl = fitControl,
                 tuneGrid = Grid, metric='RMSE', maximize = FALSE)

par(cex.lab=2)
plot(fit.gbm, main =  "Bagged Trees: Error vs Number of Trees")

tr_gbm_yhat <- predict(fit.gbm, training.data)
te_gbm_yhat <-  predict(fit.gbm, testing.data)

gbm_results = data.frame(y = y_test, yhat = te_gbm_yhat)    
gbm_results1 = data.frame(y = y_train, yhat = tr_gbm_yhat)

save.image('save3')
load('save3')

####################### NEURAL NET ########################

ctrl = trainControl(method="cv", number=5, classProbs = F, summaryFunction = defaultSummary, allowParallel=T)

nnet1 <- train(LT_ORDRCPT_DLRIN_DLROUT/300 ~ .,
               data = training.data,     
               method = "nnet",    
               trControl = ctrl,    
               tuneLength = 15,
               maxit = 100,
               metric = "RMSE")

nnet1$finalModel$tuneValue

yhat_nn_tr <- predict(nnet1, newdata=training.data)
yhat_nn_te <- predict(nnet1, newdata=testing.data)

yhat_nn_tr = yhat_nn_tr * 300
yhat_nn_te = yhat_nn_te * 300

score(y_test, yhat_nn_te)
rmse(y_train, yhat_nn_tr)

####################### MODEL SCORING ####################

# rf scoring
score(y_test, te_rf_yhat)
rmse(y_train, tr_rf_yhat)

# svm scoring
score(y_test, te_svm_yhat)
rmse(y_test, te_svm_yhat)

# knn scoring
score(y_test, knnreg_te$pred)
rmse(y_train, knnreg_tr$pred)

# gbm scoring
score(y_test, te_gbm_yhat)
rmse(y_train, tr_gbm_yhat)

######################## Ensemble Modeling #########################

ensemble_tr <- data.frame(y = y_train, knn_yhat = knnreg_tr$pred, svm_yhat = tr_svm_yhat,
                          gbm_yhat = tr_gbm_yhat, rf_yhat = tr_rf_yhat)

ensemble_te <- data.frame(y = y_test, knn_yhat = knnreg_te$pred, svm_yhat = te_svm_yhat,
                          gbm_yhat = te_gbm_yhat, rf_yhat = te_rf_yhat)

######################### RF Ensemble ###############################

E_rf_tr <- ensemble_tr[-5]
E_rf_te <- ensemble_te[-5]

rf <- randomForest(y ~ ., data = E_rf_tr, ntree=500)

rf_ensemble_tr <-  predict(rf, E_rf_tr)
rf_ensemble_te <-  predict(rf, E_rf_te)

score(y_test, rf_ensemble_te)
rmse(y_train, rf_ensemble_tr)

############################ knn ensemble #####################

E_knn_tr <- ensemble_tr[-2]
E_knn_te <- ensemble_te[-2]

knnE_tr <- knn.reg(train = E_knn_tr[-1], y = y_train, k = 5)
knnE_te <- knn.reg(train = E_knn_tr[-1], test = E_knn_te[-1], y = y_train, k = 5)

score(y_test, knnE_te$pred)
rmse(y_train, knnE_tr$pred)


########################## svm ensemble #######################
# best rmse of the 3

E_svm_tr <- ensemble_tr[-3]
E_svm_te <- ensemble_te[-3]

svmE <- svm(y ~ ., data = E_svm_tr)

tr_svmE_yhat <- predict(svmE, E_svm_tr)
te_svmE_yhat <- predict(svmE, E_svm_te)

score(y_test, te_svmE_yhat)
rmse(y_train, tr_svmE_yhat)

load('save4')
save.image('save4')

