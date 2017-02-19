## Import necessary modules
import csv
import numpy

## make 

## Return this dataset        
class DataSet:
        x =[]
        y = []

## Declare a blank DataSet class variable for future return
dataset = DataSet

## function _ return shuffled data indices
def shuffle_order(varnum, count):
	import numpy as np
	## make the whole list
	arr = np.arange(varnum)

	## shuffle it
	np.random.shuffle(arr)

	## only return the first =count= indices
	output = arr[0:count]
	return output

	

## function _ read from csv file
def read_from_csv(filename='', varlist='', yvarlist=''):

        ## if, testing then use civic data
        if filename == '':
                filename = 'oxtest1.csv'

        ## if, testing then use civic data for var names
        if varlist == '':
                varlist = ["Gender","attn","unatt","cont","Vols","Touch","Excel","Diff"]
        if yvarlist == '':
                yvarlist = ["pVolsOX"]
        
        ## get the number of columns
        varnum = len(varlist)
        yvarnum = len(yvarlist)

        ## Initialize number variables
        subjectnum = 0
        flags = []

        ## Open CSV file in order to check the data availability for each row
        input_file = csv.DictReader(open(filename))

        ## From the first row to the last row
        for row in input_file:

                ## Set data flag as zero
                flag = 0;

                ## For each variable in each row
                for i in range(varnum):
                        ## Check whether or not a certain column is blank
                        if row[varlist[i]] == '':
                                ## If blank, then mark it
                                flag = 1
                ## Check again for yvarlist
                for i in range(yvarnum):
                        if row[yvarlist[i]] == '':
                                flag = 1

                ## If even one variable is blank in this row, then mark it              
                if flag == 1:
                        flags.append(1);
                        continue

                ## Mark that this row is valid. (Have all valid number variables)
                flags.append(0);

                ## We have one more valid subject data
                subjectnum+=1

        ## Print out the number of valid subjects
        print ("A total of %d subjects' data was imported....\n" % subjectnum)                

        ## Create a variable storing actual data
        dataset.x  = numpy.zeros((subjectnum,varnum))
        dataset.y = numpy.zeros((subjectnum,yvarnum))
        ## And initialize index variables
        i = 0
        ii = 0

        ## Open the CSV file again
	## testing?

	input_file = csv.DictReader(open(filename))

        ## From the first row to the last row
        for row in input_file:
                ## If the current row is not valid (has missing column)
                if flags[ii] == 1:
                        ## Then move to the next row
                        ii += 1
                        continue
                ii += 1
                ## Read all columns for each row
                for j in range (varnum):
                        dataset.x[i][j] = float(row[varlist[j]])

                ## For y variables
                for j in range(yvarnum):
                        dataset.y[i][j] = float(row[yvarlist[j]])
                        
                ## increasing index
                i += 1

        return dataset




