# Credit-Card-Fraud-Detection

Project Group: Credit Card Fraud Detection

Group Members: Colby Degan, Ethan Fremder, Benjamin Griffith, Nick Neubecker

### Note: A lot of this code take a long time to run, espeically depending on hardware. For us sometimes it exceed 15 minutes on certain cells of code. We will do our best to have ran versions of everything on the final upload of the notebook on the Git.

## Abstract: 

The goal of this project was to create a model that could accurately predict whether a given purchase was fraudulent or not. We got our dataset from Kaggle and it contained just under one million data points. Specific features include distance from home, distance from last transaction, ratio to median purchase price, repeat retailer, used chip, used pin number, and fraud. There were about 87,000 fraud purchases and 910,000 nonfraud purchases in our dataset. The Python packages used in this project were.

- Statsmodel.api -> logisitical models
- Scikit.learn -> SVM models and kernels, grid search, model fitting, etc.
- Matplotlib.pyplot -> plot creation and labeling
- Seaborn -> special plots such as pair plots and heatmaps

Initially, we wanted to get a brief look into how the variables were correlated with fraud. This was done by creating a correlation heatmap and ran a logistic regression model using seaborn, pyplot, and statmodel.api for the model and plotting. The correlation heatmap showed that the ratio to median purchase price was the highest correlated with fraud. Both online order and distance from home also had a notable correlation with fraud. The logistic regression results were not as helpful as we had hoped they would be as although the aforementioned variables had higher correlation than the rest of the features, none exceeded an psuedo R-squared value of 0.5. The logistic model did not even let us trim away uneeded variables as they all had P values under 0.005 and were considered statistically significant. Given this, there were better models to use to classify multivariant binary data like ours, so we moved away from the logistical model and statmodel.api.

This caused our focus to shift to Support Vector Machines (SVMs), which we considered to be a good fit for our project. However, after using the seaborn pairplot function to see if even a subset of the data was linearly separable, we realized that we needed a method to would allow us to account for two issues. One was that higher dimensions beyond the 7 features from our dataset were needed to model this data and make it separable. Two, we needed to split the data more evenly between fraudulent purchases and non-fraud purchases to account for the heavy bias toward non-fraud present in the dataset and the majority of randomly sampled subsets. Removing the bias from the sampled data was done by masking the data frames into two frames, one with all fraud and one with no fraud. These were then both randomly sampled for 25,000 samples and concatenated together. The higher dimension issue was by switching the SVM kernel from poly to rbf. Leveraging some code learned from class we implemented the sklearn grid search function to find the optimal pair of C and gamma for our SVM model.

The optimal form of the SVM model had around 98% accuracy which varied by at worst 1.5% depending on the sampling of the data. This was true when tested on a testing set of data AND when testing on the entire original dataset. The error also shifted from the earlier SVMs where most of the error was false negatives, not predicting fraud when there was fraud, to false positives, where fraud was predicted on a non-fraudulent transaction. In the test the SVM did on the entire dataset, assuming the model predicted a transaction as fraudulent, there was about a 16% chance that the purchase was not fraudulent. It could be argued that this means the model became biased toward predicting fraud, we argue that for credit like credit cards, letting fraud go is more dangerous than being overly cautious.

When it came to creating a function to allow users to input their own transaction data to predict fraud with. We were able to get the function working; however, there were limitations to it. The main limitation is that the data must be of the same feature size and labeling as the original format, meaning that missing data is not allowed. If there were time, we wanted to try to use PCAs in order to create several SVMs for various cases of missing data and then have the model work with varying reported accuracies for any combination of inputted features. 