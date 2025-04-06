# Housing Prices walkthrough

## Problem definition
Here we have a bunch of data

By inspecting data_description.txt we can get the meanings of all the different features we have. 
Examples:
```
MiscVal: $Value of miscellaneous feature

MoSold: Month Sold (MM)

YrSold: Year Sold (YYYY)

SaleType: Type of sale
		
       WD 	Warranty Deed - Conventional
       CWD	Warranty Deed - Cash
       VWD	Warranty Deed - VA Loan
       New	Home just constructed and sold
       COD	Court Officer Deed/Estate
       Con	Contract 15% Down payment regular terms
       ConLw	Contract Low Down payment and low interest
       ConLI	Contract Low Interest
       ConLD	Contract Low Down
       Oth	Other
```

The train set from train.csv has 1460 samples. the test set from test.csv has a similar number of samples.

Our goal is to produce a final output that will look something like:
```
Id,SalePrice
1461,169277.0524984
1462,187758.393988768
1463,183583.683569555
1464,179317.47751083
1465,150730.079976501
```
Where for each given ID we produce the expected sale price.

## Set up

I decided to start simple. 
I inspected the shapes of the sets and just for funsies drew a simple graph to show the distribution of the SalePrice variable that we will be predicting.

Its nice to use pandas because apparently it handles the csv really well by automatically considering the first row to be the header.

so at this point I have the following (check try1.py)

```python
train_df = pd.read_csv("home_data/train.csv")

test_df = pd.read_csv("home_data/test.csv")

  

print("Train shape:", train_df.shape)

print("Test shape:", test_df.shape)

  

sns.histplot(train_df['SalePrice'], kde=True)
plt.title("Distribution of Sale Prices")
plt.xlabel("SalePrice")
plt.ylabel("Frequency")
plt.show()
```

Then we get cooking.

## The build up and the multiple tries

### Try #1

I dropped the SalePrice from the training (because that is the variable we want to predict, not fit).

I decided to go with **Random Forest** which is good for when you have different types of data. It grows trees separately and then averages them (ensamble ml). If we had a huge dataset it would not be computationally optimal, however, since our datasets are rather small that should not be an issue.
One thing to consider about random forest is that it is unlikely to extrapolate well (Extrapolation involves estimating values outside the range of observed data, which can lead to significant uncertainty. Random forests and decision trees generally perform poorly in extrapolation scenarios because **they are confined to the training data**.). If the testing is performed on a dataset with saleprices that are not within the previously seen range then I fear it will underperform.

I also checked the RMSE just out of curiousity and it was very high
`Cross-validated RMSE: 30294.124580816315`. That confused me.

### Try #2

I decided to optimize the parameters. I've heard about GridSearchCV that basically works by defining a grid of hyperparameters and then training and evaluating the model for each hyperparameter combination.

It was kinda time costly but again, it was under a minute so I did not really care. 
The RMSE did not impove, I was disappointed once again.


### Try #3

While I was quite confused at this point I decided to move past it. I remembered that there was something about logs. I decided to log transform the SalePrice. 
```python
y = np.log1p(train_df['SalePrice'])
...
preds = np.expm1(best_model.predict(X_test))
```

Now the RMSE did not look so terrible. This is where I decided to stop and try to submit this to the competition. I got ranked 2099th and that was withing the first 50%. I thought this was okay for now.