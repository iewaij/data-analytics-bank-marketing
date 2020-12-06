# Exploratory Data Analysis
Exploratory Data Analysis is a process to explore the dataset with no assumptions or hypotheses using non-graphical and graphical, univariate and multivariate methods. The objective is to gain intuitive insights, discover distribution characteristics, and find out missing values in the dataset.

<br />

## Exploration of the target variable Y
The first thing we investigate is the target variable Y, which is a Y/N binary variable measuring the campaign outcome and representing whether a client has subscribed to a long-term deposit. 
![Distribution of campain outcome Y.](../figures/2_1_Y_distribution.png)

There are more than 40,000 observations in our dataset, and only 11.3% of them have positive “YES” outcomes, which means that we have a significantly unbalanced dataset. Since our data was collected during the 2008 financial crisis, we pay particular attention to this time factor and visualize the positive Y values across time. 
![Uneven distribution of positive outcome.](../figures/2_2_Uneven_distribution.png)

In the graph above, the thin orange line indicates the outbreak of the financial crisis. We can see a massive surge in positive campaign outcomes afterward, meaning people were actively taking advantage of certain economic factors, such as lower interest rates. We can also see a steady growth of the positive outcome rate since July 2009 from the graph below.
![Positive outcome rate by month.](../figures/2_3_Positive_rate_by_month.png)
![Five economic indicators.](../figures/2_4_Five_econ_indicators.png)

Highly relevant to the crisis are the five economic indicators in our dataset, displayed as above, which show significant predicting power in almost all of our models. In 2008, all of them went down first, but the consumer confidence index (green line) was the leading recovery factor, followed by the consumer price index (orange line). The recovery of the positive outcome rate of our campaign started at the same time. Furthermore, the drop of interest rate captured by the Euribor 3-month rate (red line) significantly correlated with the campaign success.

<br />

## Missing Values
We use the `info()` function to get an overview of our data and notice many missing values, especially for features like Pdays and Poutcome, in which 90% of the rows have missing values, as shown in the bar chart below. These missing values have been an enormous issue for our feature engineering process.
![Percentage of missing values.](../figures/2_5_Missing_value_percentage.png)

<br />

## Feature Explorations
### Age
For client profile features, first we plot the distribution of age relative to Y, as shown below.30-50-year-old people are the majority in our dataset. People in their 30s and people older than 60 are more likely to give positive responses. There are not many outrageous outliers, therefore we keep all data for Age.
![Age distribution histogram.](../figures/2_6_Age_histogram.png)
![Age distribution box-plot.](../figures/2_6_Age_box.png)

<br />

### Job
This next graph shows the outcome Y percentage in each job group, with the orange denoting positive outcomes. On the right is the instance count distribution of each group. There is a large percentage of technicians, blue-collar workers and admins. However, it is the students and retired people that are most likely to say ‘YES’ to the long-term loans.
![Outcome percentage and distribution by Job.](../figures/2_7_Job.png)

<br />

### Education
In terms of Education, although most people in our dataset have above-high-school education, the groups that are most likely to respond positively are the least and the most educated. 
![Outcome percentage and distribution by Education.](../figures/2_8_Education.png)

<br />

### Default
Default is an peculiar feature that captures whether people previously had credit default, and it is a highly sensitive piece of information. As presented in the figure below, the majority of clients declare no default record, however, there are 8600 instances that are “unknown”. Given the privacy nature of this feature, we believe there is some unspeakable stories behind the “unknown” values that might influence their financial decisions. Therefore, we decide to treat the unknown values as an individual category and let it speak for itself.
![Outcome percentage and distribution by Default.](../figures/2_9_Default.png)

<br />

### Contact
For features capturing characteristics of current campaign, Contact is an important binary feature that represents the tool used to contact clients. And we see a stronger positive outcome rate for cellular usage.

![Outcome percentage and distribution by Contact.](../figures/2_10_Contact.png)

<br />

### Month
With regard to the month in which last contact was made, the distribution concentrates in summer, however, these months have the lowest positive outcome rates. In months with less contacts, there seeme to be higher success rate of securing a long-term deposit with clients.

![Outcome percentage and distribution by Month.](../figures/2_11_Month.png)

<br />

### Pdays
Next comes the most challenging feature, Pdays, which represents the number of days passed since last contact with a client. It has almost 40,000 missing values. As demonstrated in the graph below, the "<Na>" category on the top dominates the entire distribution. At the same time, the other 1500 rows that do have values seem to show positive relationships with the outcome. We spent a significant amount of time and effort in deal with this feature, and this process will be discussed in the feature engineering section. 
![Outcome percentage and distribution by Pdays.](../figures/2_12_Pdays.png)

<br />

### Previous
The Previous feature is also very challenging, which measures the number of contacts performed to a client before this campaign. It has 36,000 missing values, and for those observations with actual values their relationships with the outcome is seemingly positive. 
![Outcome percentage and distribution by Previous.](../figures/2_13_Previous.png)

<br />

### Poutcome
This is the feature that reports the outcome of the previous campaign. Over 35,000 observations have missing values, as presented below. However, when combining this featuring with Pdays and Previous, there seem to be some contradictory stories.
![Poutcome distribution.](../figures/2_14_Poutcome.png)

 Missing values in Pdays mean that the clients were not previously contacted and therefore should not have values in Poutcome. But Poutcome has less missing values than Pdays dose, as seen in the second graph below. We print out the 4110 rows where clients have not been contacted but have Poutcome values and see how many times they have been contacted before. The results suggest that maybe these clients have been actually contacted but it was more than 30 days ago, thus the contact date was not recorded. 
![Poutcome distribution.](../figures/2_14_Pdays+Previous.png)

<br />

### Client Data Multivariate Explorations
Furthermore, we expore some multivariate distributions of positive outcome. First, we found both married and divorced retired people respond positively to our campaign, and single and divorced students are even more enthusiastic. It’s also quite interesting that students, retired and illiterate people are more likely to say ‘yes’ to our long-term deposit. Additionally,  divorced illiterate people respond to our campaign extremely well.
![Positive Outcome Percentage by Job and Marital Status.](../figures/2_15_Job+Marital.png)
![Positive Outcome Percentage by Job and Marital Status.](../figures/2_15_Job+Education.png)
![Positive Outcome Percentage by Job and Marital Status.](../figures/2_15_Education+Marital.png)

<br />

### Key Numerical Features
We also made a scatterplots across important quantitative features, with the outcome variable denoted by colour. 
![Scatterplots of key numerical features.](../figures/2_16_Numerial_features.png)

<br />

### Correlation Heatmap
With this heatmap, we can get a better look at the correlations among features. Four out of five economic indicators have strong correlations with each other. We were worried about collinearity and tried many ways to deal with these features, such ads deletion or transformation, but all efforts led to relatively poor model results. Then we realise that they are probably very important features in our dataset, so we keep them for the moment. In addition, Some features show great correlations with the outcome, such as Previous and Poutcome. We try to use PCA on the entire dataset to avoid collinearity, but again, all efforts led to poor model results. Therefore, we decide to keep all features and make changes if needed for specific models.
![Correlation heatmap.](../figures/2_17_Heatmap.png)





<br />

### Age


<br />

### Age