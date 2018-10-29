# Identify the SEM model
library(lavaan)

# load data
data=read.csv("test.csv",header=TRUE)

# remove na
data <- data[!(rowSums(is.na(data))),]

# set the initial model
# Input vars -> emotional responses -> output

model <- '
	# Touch
	Touch ~ Gender + attn + Int + Vols
	# Excel
	Excel ~ Gender + attn + Int + Vols
	# Diff
	Diff ~ Gender + attn + Int + Vols
	# Final output
	pVolsOX ~ Gender + attn + Int + Vols + Touch + Excel + Diff
	'

# run SEM
original <- sem(model, data = data, ordered = 'pVolsOX')
summary(original)

# remove non-significant paths
mod_model <- '
	# Touch
	Touch ~ Int 
	# Excel
	Excel ~  Int 
	# Diff
	Diff ~  attn + Vols
	# Final output
	pVolsOX ~ attn + Vols + Excel
	'

modified <- sem(mod_model, data = data, ordered = 'pVolsOX')
summary(modified,fit.measures=TRUE, modindices=T)

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

modified1 <- sem(mod_model1, data = data, ordered = 'pVolsOX')
summary(modified1,fit.measures=TRUE, modindices=T)
