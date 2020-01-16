library('dplyr')
library('plyr')
library(BayesFactor)
library(nlme)
library(effsize)
library(psych)

# load files
df_logit<-read.csv('logit_result_2020.csv')
df_glm<-read.csv('glm_result_2020.csv')
df_sem<-read.csv('sem_result.csv')

df_adam<-read.csv('adam_result.csv')
df_adam_softmax<-read.csv('adam_softmax_result.csv')
df_sdgm<-read.csv('sgdm_result.csv')
df_sdgm_softmax<-read.csv('sgdm_softmax_result.csv')
df_rmsprop<-read.csv('rmsprop_result.csv')
df_rmsprop_softmax<-read.csv('rmsprop_softmax_result.csv')

# convert into percentages
# and record method nums
# adam = 1
df_adam<-df_adam/100
df_adam$method<-1

# adam_softmax = 2
df_adam_softmax<-df_adam_softmax/100
df_adam_softmax$method<-2

# sdgm = 3
df_sdgm<-df_sdgm/100
df_sdgm$method<-3

# sdgm_softmax = 4
df_sdgm_softmax<-df_sdgm_softmax/100
df_sdgm_softmax$method<-4

# rmsprop = 5
df_rmsprop<-df_rmsprop/100
df_rmsprop$method<-5

# rmsprop_softmax = 6
df_rmsprop_softmax<-df_rmsprop_softmax/100
df_rmsprop_softmax$method<-6

# logistic reg = 7
df_logit$method<-7

# mixed logit = 8
df_glm$method<-8

# sem = 9
df_sem$method<-9

# merge all
df<-rbind.fill(df_adam,df_adam_softmax,df_sdgm,df_sdgm_softmax,df_rmsprop,df_rmsprop_softmax,df_logit,
               df_glm,df_sem)

# deep learning?
df$deep<-as.numeric(df$method<7)


# comparing training sets
for (i in 1:6){
  for (j in 7:9){
    print(sprintf("%d vs %d",i,j) )
    current_test<-df[df$method==i|df$method==j,]
    print(t.test(train_accuracy~method,data=current_test))
    print(cohen.d(train_accuracy~method,data=current_test))
    #print(ttestBF(x=current_test[current_test$method==i,]$train_accuracy,
    #                y=current_test[current_test$method==j,]$train_accuracy))
    print(2*log(BayesFactor::extractBF(ttestBF(x=current_test[current_test$method==i,]$train_accuracy,
                                        y=current_test[current_test$method==j,]$train_accuracy))$bf))
    print(sd(current_test[current_test$method==i,]$train_accuracy))
    print(sd(current_test[current_test$method==j,]$train_accuracy))
  }
}

# comparing validation sets
for (i in 1:6){
  for (j in 7:9){
    print(sprintf("%d vs %d",i,j) )
    current_test<-df[df$method==i|df$method==j,]
    print(t.test(validity_accuracy~method,data=current_test))
    print(cohen.d(validity_accuracy~method,data=current_test))
    #print(ttestBF(x=current_test[current_test$method==i,]$validity_accuracy,
    #              y=current_test[current_test$method==j,]$validity_accuracy))
    print(2*log(BayesFactor::extractBF(ttestBF(x=current_test[current_test$method==i,]$validity_accuracy,
                                               y=current_test[current_test$method==j,]$validity_accuracy))$bf))
    print(sd(current_test[current_test$method==i,]$validity_accuracy))
    print(sd(current_test[current_test$method==j,]$validity_accuracy))
  }
}

# trial # column
df$index<-seq.int(df)

# do planned ANOVAs
# training set
model_train<-lme(train_accuracy~deep, random=~1 | index, method="ML",data=df)
anova(model_train)
model_valid<-lme(validity_accuracy~deep, random=~1 | index, method="ML",data=df)
anova(model_valid)
