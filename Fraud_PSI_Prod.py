#defining a custome fucntion for producing the range
def scale_range (input, min, max):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input

buckets = 10
raw_breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
breakpoints = scale_range(raw_breakpoints, np.min(preds_DM), np.max(preds_DM))

# preds_DM is a numpy array of the prediction output of test data set, to be load and stored as static numpy array using loadtxt form numpy . Data shared in csv file and one time load


#Building the Train and Prod data basis the scoring parameter into buckets
train_count = np.histogram(preds_DM, breakpoints)[0]
prod_count = np.histogram(XXXXXX, breakpoints)[0]

#Converting into Pandas Dataframe
df = pd.DataFrame({'Bucket': np.arange(1, 11), 'Breakpoint Value':breakpoints[1:], 'Train Count':train_count, 'Prod Count':prod_count})
df['Train Percent'] = df['Train Count'] / len(preds_DM)
df['Prod Percent'] = df['Prod Count'] / len(XXXXXX)

# Where XXXX is the numpy array format of model_output_score column taken from production table

#Exception handling for division by zero error
df['Prod Percent'][df['Prod Percent'] == 0] = 0.001

#PSI Calculation
df['PSI'] = (df['Prod Percent'] - df['Train Percent']) * np.log(df['Prod Percent'] / df['Train Percent'])

np.sum(df['PSI'])

#Interpretation
#PSI < 0.1: no significant population change

#PSI < 0.2: moderate population change

#PSI >= 0.2: significant population change