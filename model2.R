#load libraries
library(caret) 		
library(rpart.plot)
library(tidyverse)
library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)
library(corrplot)

#load data
excel_sheets("AB_Data.xlsx")
product = read_excel("AB_Data.xlsx",sheet = 1)
custs = read_excel("AB_Data.xlsx",sheet = 2)
sales = read_excel("AB_Data.xlsx",sheet = 3)
teams = read_excel("AB_Data.xlsx",sheet = 4)


######To load from database
#require(odbcConnect)
#db <- odbcConnect("DatabaseName")
# <- sqlQuery("select * from table", stringsAsFactors=FALSE)
#f <- read.csv("~/table.csv", stringsAsFactors=FALSE)
###########------------------------

#Data EDA
colSums(is.na(product))
colSums(is.na(sales))
colSums(is.na(custs))
colSums(is.na(teams))

summary(sales$qty) #min is 1 maxis 18
summary(product$unit_price) # min is 2.9 max is 19
summary(custs$cust_id)
colnames(product)[1] = "sku_id"
colnames(sales)[1] = "sales_id"

#data join
# Data join prod and sales
ps_data = merge(x = sales, y = product, by = "sku_id", all.x=T, all.y=F)
glimpse(ps_data)

#merge all the 3 prod,sales and custs
psc_data = merge(x=ps_data, y=custs, by="cust_id",all.x=T, all.y=F)
glimpse(psc_data)

#check for missing values
colSums(is.na(psc_data))

# Modify to date format
psc_data$order_date <- as.Date(psc_data$order_date, "%Y-%m-%d")

#relace with na with 0
na.omit(psc_data)



##########FEATURE ENGINEERING#################
#Create sales,weekday,day
psc = psc_data %>% 
  group_by(sku_id,prod_name) %>% 
  summarize(sale = psc_data$qty * psc_data$unit_price)

psc_data$sales <- psc_data$qty * psc_data$unit_price

#Creating box plots and histograms of numerical variables to understand sales distribution
par(mfrow=c(2,1)) # showing one graph under the other
par(mar=c(2,2,2,2)) # changes margins within the figure

boxplot(psc_data$sales, 
        horizontal = TRUE, 
        col = "red3")

hist(psc_data$sales,
     breaks = 20,
     main = "Distribution of sales")

#It seems like this variable is positively skewed (the majority of the values are placed on the left from the mean and are clustered in the lower end of the distribution)
#and there are many outliers with values higher than 1.5 * IQR above the 3rd quartile 

####BASED on DATE#####
#having variables in date format can be interesting to discover other trends in this dataset.
df=psc_data

# Creating additional variables for a weekday and a month
df$month = format(df$order_date,"%B")
df$weekday = format(df$order_date,"%a")

weekday = factor(df$weekday, levels = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"))

# Let's see on which day of the week customers order their products
x = barplot(table(weekday), main = "On which days do customers create most orders?", 
            col = c(terrain.colors(7)),
            xlab = "Day",
            ylab = "Number of orders")

text(y = table(weekday), 
     x, 
     table(weekday), 
     cex=0.8, 
     pos = 1)

##############
#BINNING of SALES
library(classInt)
bin_data <- df$sales
#EQUAL WIDTH
classIntervals(bin_data,5,style='equal')
#EQUAL FREQUENCY
classIntervals(bin_data,5,style='quantile')

########################
#VISUALIZATIONS 
#######################

#pie chart for other gender variables 
pie(table(df$gender),
    labels = table(df$gender), 
    col = c("paleturquoise1","steelblue1","lightslateblue"),
    main = "Number of orders by customer segment - gender")

legend("bottomright", legend = paste(unique(df$gender)), 
       fill = c("paleturquoise1","steelblue1","lightslateblue"))

# Dotchart for total sales per product name
beer_sales = tapply(df$sales, df$weekday, count)

dotchart(beer_sales, pch = 50, bg = "purple1", cex = 1.3, 
         xlab="Sum of sales",
         main = "What are the total sales \nfor each product?")

#######################MODELING#######################
######################################################

set.seed(10311)
split = sample(1:nrow(df),0.8*nrow(df))
train = df[split,]
test = df[-split,]

dim(train) 
dim(test)

#remove sales from test
drop <- c("sales")
test = df[,!(names(df) %in% drop)]

#visualize train data
library(ggplot2)
# Assign plot to a variable
plot <- ggplot(data = train,mapping = aes(x = sales, y = prod_name)) +
  geom_point(alpha = 0.1, color = 'blue')
# Draw the plot
surveys_plot +
  geom_point()

#Correlation plot before selecting features
test %>% 
  select_if(is.numeric) %>%
  cor() %>%
  corrplot()

#INDEPENDENT VARIABLES
df$race
df$marital
df$gender
#df$st
df$occ
df$edu
df$income
df$team

df[!is.na(df)]


##########convert all to facator#########
features = colnames(df)

for (f in features){
  if( (class(df[[f]]) == "character") || (class(df[[f]]) == "factor"))
  {
    levels = unique(df[[f]])
    df[[f]] = factor(df[[f]], level = levels)
  }
}

#one-hot-encoding categorical features

data = as.data.frame(df)
ohe_feats = c('race','marital','gender','occ','edu','income','team')
dummies = dummyVars(~ race+marital+gender+occ+edu+income+team , data = data)
df_all_ohe <- as.data.frame(predict(dummies, newdata = data))
df_all_combined <- cbind(data[,-c(which(colnames(data) %in% ohe_feats))],df_all_ohe)

library(data.table)
data = as.data.table(df_all_combined)

train = data[data$cust_id %in% df$cust_id,]
y_train <- train[!is.na(sales),sales]
train = train[,sales:=NULL]
train = train[,cust_id:=NULL]
#train_sparse <- as(data.matrix(train),"dgCMatrix")
train_sparse <- data.matrix(train)

test = data[data$cust_id %in% df$cust_id,]
test_ids <- test[,cust_id]
test[,sales:=NULL]
test[,cust_id:=NULL]
#test_sparse <- as(data.matrix(test),"dgCMatrix")
test_sparse <- data.matrix(test)

dtrain <- xgb.DMatrix(data=train_sparse, label=y_train)
dtest <- xgb.DMatrix(data=test_sparse);

gc()

# Params for xgboost
param <- list(booster = "gbtree",
              eval_metric = "rmse", 
              objective = "reg:linear",
              eta = .1,
              gamma = 1,
              max_depth = 4,
              min_child_weight = 1,
              subsample = .7,
              colsample_bytree = .7)

#cross validation to find optimal nrounds
cvFoldsList <- createFolds(1:nrow(train), k = 5)
xgb_cv <- xgb.cv(data = dtrain,
                                 params = param,
                                 nrounds = 500,
                                 maximize = FALSE,
                                 prediction = TRUE,
                                 folds = cvFoldsList,
                                 print.every.n = 5,
                                 early.stop.round = 50); gc()
 
#used  (Mb) gc trigger  (Mb) limit (Mb)  max used  (Mb)
#Ncells  2910914 155.5    5151648 275.2         NA   5151648 275.2
#Vcells 60885157 464.6  103172086 787.2      16384 103167320 787.2


# Find nrounds with the lowest RMSE       
rounds <- xgb_cv$evaluation_log[, test_rmse_mean]
rounds 
which.min(xgb_cv$evaluation_log$test_rmse_mean)
#500

#visualize rmse
ggplot(data=xgb_cv$evaluation_log, aes(x=iter, y=test_rmse_mean)) +
  geom_point(size=0.4, color="sienna") +
  geom_line(size=0.1, alpha=0.1) +
  theme_bw()



#prediction for each customers
mpreds = data.table(id=test_ids)

#for random seed from 1 to 10

for(random.seed.num in 1:10) {
  print(paste("[", random.seed.num , "] training xgboost begin ",sep=""," : ",Sys.time()))
  set.seed(random.seed.num)
  xgb_model <- xgb.train(data = dtrain,
                         params = param,
                         watchlist = list(train = dtrain),
                         nrounds = rounds,
                         verbose = 0,
                         print.every.n = 5)
  #verbose is to print everything so we can add value 50 so that it prints for every 50th iter
  vpreds = predict(xgb_model,dtest) 
  mpreds = cbind(mpreds, vpreds)    
  colnames(mpreds)[random.seed.num+1] = paste("pred_seed_", random.seed.num, sep="")
}

mpreds_2 = mpreds[, id:= NULL]
mpreds_2 = mpreds_2[, y := rowMeans(.SD)]
head(mpreds_2)
#y is sales

#store results in a final_sales 
final_sales = data.table(ID=test_ids, y=mpreds_2$y)

# Lets start with finding what the actual tree looks like
model <- xgb.dump(xgb, with.stats = T)
model[1:10] #This statement prints top 10 nodes of the model

summary(model_xgboost)

# > xgb_model$evaluation_log
# iter train_rmse
# 1:    1  18.262966
# 2:    2  16.581079
# 3:    3  15.068364
# 4:    4  13.605139
# 5:    5  12.292393
# 6:    6  11.100674
# 7:    7  10.378788
# 8:    8   9.378365
# 9:    9   8.817434
# 10:   10   7.977038
# 11:   11   7.280284
# 12:   12   6.588018
# 13:   13   6.231519
# 14:   14   5.930965
# 15:   15   5.375000
# 16:   16   4.877075
# 17:   17   4.419382
# 18:   18   4.232664

# # Get the feature real names
# names <- dimnames(data.matrix(X[,-1]))[[2]]
# names
# 
# importance <- xgb.importance(dimnames(df_all_ohe), model = xgb_model)
# 
# importance <- xgb.importance(dimnames(dtrain), model = xgb_model)
# xgb.plot.importance(importance[1:5,])

