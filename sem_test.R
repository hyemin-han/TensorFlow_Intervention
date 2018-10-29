# clear all
rm(list=ls())

library('dplyr')
library(lavaan)
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

# sem
# modification
mod_model1 <- '
	# Touch
	Touch ~ Int 
	# Excel
	Excel ~  Int 
	# Diff
	Diff ~  attn + Vols
	# Final output
	pVolsOX ~ attn + Vols + Excel
	Touch ~~ Excel
	'

modified1 <- sem(mod_model1, data = train_data, ordered = 'pVolsOX')

# predict training set
# train_prob <- sem.predict(modified1, train_data, type="response")
coefs  <- parTable(modified1)
results <- (train_data$Int * coefs[2,14] + coefs[25,14]) * coefs[7,14] + coefs[5,14] * train_data$att + coefs[6,14] * train_data$Vols

train_acc = 0
train_length = nowcount-1
num_train = 0

for (i in 1:train_length){
	# null value?

	if (!is.na(results[i])){

		pox = 0

		if (as.numeric(results [i]) >= coefs[9,14]){
			pox = 1
		}

		if (pox == train_data$pVolsOX[i]){
			train_acc = train_acc + 1
		}

		num_train = num_train + 1
	}
}

# predict validation dataset
# valid_prob <- sem.predict(modified1, valid_data, type="response")
results1 <- (valid_data$Int * coefs[2,14] + coefs[25,14]) * coefs[7,14] + coefs[5,14] * valid_data$att + coefs[6,14] * valid_data$Vols

valid_acc = 0
num_valid = 0

for (i in 1:valid_length){
	# null value?

	if (!is.na(results1[i])){

		pox = 0

		if (as.numeric(results1 [i]) >= coefs[9,14]){
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
write.csv(result,file="sem_result.csv")
