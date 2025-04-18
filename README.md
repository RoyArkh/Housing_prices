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



### Try #4

This is when I turned to AI. ChatGPT suggested I "Expand my `GridSearchCV`"
So I did as it suggested 
```python
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}
```

To sum it up: that was a disaster. A code resulted in some error for some reason but it still managed to produce an output somehow which I decided to submit. The score was **lower** than my previous, so I gave up on that idea.

But I did not give up on the pover of AI. DeepSeek fully re-wrote my code and as my (mere human's) knowledge is limited I decided to give it a shot. It did actually improve things quite noticeably. I ran the code without understanding it first just to see whether it is worth analising. It ran, produced a result, I submitted, and that made me skyrocket into the first 1000, more precisely \#782. I decided to study what deepseek has cooked. 

Upon inspection I understood that deepseek decided to do the following:
1. taking test and train and saving test IDs for including into the submission 
2. prepares our target (SalePrice) by returning the natural logarithm, then it drops saleprice from train and id from test -> im not sure why it dropped id
3. combines train and test
4. then unlike our previous implementation it explicitly creates new features
5. then it drops some columns that it deemed "redundant" but im not sure why. will test this later.
6. then it decided to Handle Missing Values but I will look into that more later
7. then it encodes categorical variables and splits it back into test and train
8. then based on importance of weights it uses sickit's SelectFromModel to filter out the most important  features
9. tries 3 different models instead of just one
10. Uses ensamble to average predictions from three models
This all resulted in a pretty good outcome but I decided to experiment a bit more.

Side note:
I was quite surprised by the ensamble idea because I didnt think of that at all before
```
# Try multiple models
models = {
    'RandomForest': RandomForestRegressor(random_state=42, n_estimators=500, max_depth=15),
    'XGBoost': xgb.XGBRegressor(random_state=42, n_estimators=1000, learning_rate=0.01),
    'LightGBM': lgb.LGBMRegressor(random_state=42, n_estimators=1000, learning_rate=0.01)
}

for name, model in models.items():
    score = rmse_cv(model, X_train, y)
    print(f"{name} CV RMSE: {score:.5f}")

# Ensemble predictions
final_predictions = []
for model in models.values():
    model.fit(X_train, y)
    preds = model.predict(X_test)
    final_predictions.append(preds)

# Average predictions
final_pred = np.mean(final_predictions, axis=0)
final_pred = np.expm1(final_pred)
```
### Try 5

In this try my main goal is to check what can be removed from try 4 to make things better still.

From the get go I decided to remove the part where the missing values are handled because I believed that the data was cleaned up beforehand.
I also removed the part where it dropped some columns that it deemed "redundant".
Those two alone have improved my score.

Then I decided to experiment with the ensamble. I decided to remove two models at a time leaving only one. At first I left only LightGBM. That made the score worse so I mistankely thought that removing models will make the ensamble worse. 
I experimented a bunch and apparently out of the three models XGBoost had the best performance. I thought then that I would leave only that. However, when I tested the ensamble of XGBoost and LightGBM together it had the best performance I had managed to achieve so far, therefore the ensamble stayed. 

This time I got ranked 698 and learned a looot more.