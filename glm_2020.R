# clear all
# rm(list=ls())

library('dplyr')
require(lme4)
require(compiler)

validity_accuracy = rep(0, times = 1000)
train_accuracy = rep(0,times = 1000)

# 1000 trials
for (trials in 1:1000){

# Load files
data=read.csv("test.csv",header=TRUE)

# remove na
data <- data[!(rowSums(is.na(data))),]

# get 1/3 of the dataset for validation set
length = nrow(data)
valid_length = round(length/3)

valid_data = sample_n(data,valid_length)

# extract train data
datalist = data.frame()
nowcount = 1
for (i in 1:length){
	flag = 0
	for (j in 1:valid_length){
		# somewhere in the validset?
		if (data$ID[i] == valid_data$ID[j]){
			# set flag and end
			flag = 1
			break
		}
	}
	# not in valid dataset? then add
	if (flag == 0){
		data_frame <- data[i,]
		datalist <- rbind(datalist , data_frame)
		nowcount=nowcount + 1
	}
}

train_data <- datalist

# glm
logit <- glmer(pVolsOX ~ Gender + attn + unatt + cont + Int + Vols + Touch + Excel + Diff + (1|ID), family = "binomial", data=train_data,control=glmerControl(optimizer = "bobyqa"), nAGQ=1)

# predict training set
train_prob <- predict(logit, train_data, type="response")
train_acc = 0
train_length = nowcount-1
num_train = 0

for (i in 1:train_length){
	# null value?

	if (!is.na(train_prob[i])){

		pox = 0

		if (as.numeric(train_prob[i]) >= .5){
			pox = 1
		}

		if (pox == train_data$pVolsOX[i]){
			train_acc = train_acc + 1
		}

		num_train = num_train + 1
	}
}

# predict validation dataset
#valid_prob <- predict(logit, newdata=valid_data, type="response")
valid_prob<-coef(logit)$ID[1,2]*valid_data$Gender+coef(logit)$ID[1,3]*valid_data$attn+
  coef(logit)$ID[1,4]*valid_data$Int+coef(logit)$ID[1,5]*valid_data$Vols+
  coef(logit)$ID[1,6]*valid_data$Touch+coef(logit)$ID[1,7]*valid_data$Excel+
  coef(logit)$ID[1,8]*valid_data$Diff+coef(logit)$ID[1,1]
valid_acc = 0
num_valid = 0

for (i in 1:valid_length){
	# null value?

	if (!is.na(valid_prob[i])){

		pox = 0

		if (as.numeric(valid_prob[i]) >= .5){
			pox = 1
		}

		if (pox == valid_data$pVolsOX[i]){
			valid_acc = valid_acc + 1
		}

		num_valid = num_valid + 1
	}
}
# store results
validity_accuracy [trials] = valid_acc / num_valid
train_accuracy [trials] = train_acc / num_train
# current done 
#print(trials)
}

# save results
result <- data.frame(train_accuracy,validity_accuracy)
write.csv(result,file="glm_result_2020.csv")
