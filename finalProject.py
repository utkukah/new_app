import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer

st.set_option('deprecation.showPyplotGlobalUse', False)

header = st.container()
dataset = st.container()


with header:
    st.title('ADA442 Final Project')
    ('In this project, we build a machine learning model to predict whether a client of a bank will subscribe to a term deposit or not.')

with dataset:
    st.header('Bank Marketing Dataset')
    st.text('The dataset used for this project is the Bank Marketing Data Set, which can be found at https://archive.ics.uci.edu/ml/datasets/Bank+Marketing')

    ba = pd.read_csv('bank-additional.csv', sep=';')
    st.write(ba.head())

    
st.text('Data exploration')
ba = pd.read_csv('bank-additional.csv', sep=';')
st.write(ba.info())
st.write(ba.describe())

st.text('Check for null data')
st.write(ba.isnull().sum())
st.text('no null values')


st.text('Data Cleaning')

st.text('Numerical Attributes : with z-scores greater than 3 (or less than -3) are considered outliers and drop')

st.text('1. age (Integer) : clients age')
ba.boxplot(column='age')
plt.ylabel('Age')
plt.title('Box Plot for Age Column', color='maroon')
plt.show()
st.pyplot()
# we have outliers

# with z-scores greater than 3 (or less than -3) are considered outliers
z_scores_age = (ba['age'] - ba['age'].mean()) / ba['age'].std()
outliers_age = (z_scores_age.abs() > 3)

# Remove outliers from the DataFrame
ba_cleaned1 = ba[~outliers_age]

# 2. 'duration' (Integer) : last contact duration, in seconds
st.text('Important note: this attribute highly affects the output target (e.g., if duration=0 then y=no)')
ba_cleaned2 = ba_cleaned1[ba_cleaned1['duration'] >= 1]

st.text('There is 1 client that duration=0, so on this, y will be definitely "no", thus we removed it')

st.text('Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known')
st.text('Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model')
ba_cleaned2 = ba_cleaned2.drop('duration', axis=1)
st.text('We drop the column from our data set')


st.text('3. "campaign" (Integer) : number of contacts performed during this campaign')
ba_cleaned2.boxplot(column='campaign')
plt.ylabel('Campaign')
plt.title('Box Plot for Campaign Column', color='maroon')
plt.show()
st.pyplot()
# we have outliers

max_campaign = ba_cleaned2['campaign'].max()
# for example the maximum campaign is 35, it is possible that there may be disruptions in contacts with this client.

# with z-scores greater than 3 (or less than -3) are considered outliers
z_scores_campaign = (ba_cleaned2['campaign'] - ba_cleaned2['campaign'].mean()) / ba_cleaned2['campaign'].std()
outliers_campaign = (z_scores_campaign.abs() > 3)

# Remove outliers from the DataFrame
ba_cleaned3 = ba_cleaned2[~outliers_campaign]


st.text('4. "pdays" (Integer) : number of days that passed by after the client was last contacted from a previous campaign')
ba_cleaned3.boxplot(column='pdays')
plt.ylabel('Previous Days')
plt.title('Box Plot for Previous Days Column', color='maroon')
plt.show()
st.pyplot()
st.text('the value 999 means the client was not previously contacted or contacted very long time ago')


st.text('5. "previous" (Integer) : number of contacts performed before this campaign and for this client')
ba_cleaned3.boxplot(column='previous')
plt.ylabel('Previous')
plt.title('Box Plot for Previous Column', color='maroon')
plt.show()
st.pyplot()
 
st.text('Lets check if "pdays" and "previous" confirm each others')
st.text('If pdays=999 we think like that; client was not previously contacted, so then previous must be 0')
filtered_previous = ba_cleaned3[(ba_cleaned3['pdays'] == 999) & (ba_cleaned3['previous'] != 0)]
filtered_previous
st.text('There are 429 rows like that, we do not want to miss them. We will trust the data and assume that these contacts are from very long time ago')
st.text('We will consider these in the "not previously contacted" category and reset their "previous" to 0')
st.text('We identify rows where "pdays" is 999 and "previous" is not 0, the Change "previous" values to 0 for the identified rows')
# Identify rows where 'pdays' is 999 and 'previous' is not 0
nz = (ba_cleaned3['pdays'] == 999) & (ba_cleaned3['previous'] != 0)
# Change 'previous' values to 0 for the identified rows
ba_cleaned3.loc[nz, 'previous'] = 0

# check it
filtered_previous2 = ba_cleaned3[(ba_cleaned3['pdays'] == 999) & (ba_cleaned3['previous'] != 0)]
# If pdays=999 then previous=0


st.text('Socioeconomic factors (float) : emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m and nr.employed')

st.text('6. emp.var.rate : employment variation rate - quarterly indicator')
ba_cleaned3.boxplot(column='emp.var.rate')
plt.ylabel('Employment Variation Rate')
plt.title('Box Plot for Employment Variation Rate Column', color='maroon')
plt.show()
st.pyplot()

grouped_evr = ba_cleaned3.groupby('y')['emp.var.rate'].mean().reset_index()
plt.bar(grouped_evr['y'], grouped_evr['emp.var.rate'], color=['red', 'blue'])
plt.xlabel('NO/YES')
plt.ylabel('Average')
plt.title('Employment Variation Rate', color='maroon')
plt.show()
st.pyplot()


st.text('7. cons.price.idx : consumer price index - monthly indicator')
ba_cleaned3.boxplot(column='cons.price.idx')
plt.ylabel('Consumer Price Index')
plt.title('Box Plot for Consumer Price Index Column', color='maroon')
plt.show()
st.pyplot()


st.text('8. cons.conf.idx : consumer confidence index - monthly indicator')
ba_cleaned3.boxplot(column='cons.conf.idx')
plt.ylabel('Consumer Confidence Index')
plt.title('Box Plot for Consumer Confidence Index Column', color='maroon')
plt.show()
st.pyplot()
# we have outliers

# with z-scores greater than 3 (or less than -3) are considered outliers
z_scores_cci = (ba_cleaned3['cons.conf.idx'] - ba_cleaned3['cons.conf.idx'].mean()) / ba_cleaned3['cons.conf.idx'].std()
outliers_cci = (z_scores_cci.abs() > 3)

# Remove outliers from the DataFrame
ba_cleaned4 = ba_cleaned3[~outliers_cci]


st.text('9. euribor3m : Euribor 3 month rate - daily indicator')
ba_cleaned4.boxplot(column='euribor3m')
plt.ylabel('Euribor 3 Month Rate')
plt.title('Box Plot for Euribor 3 Month Rate Column', color='maroon')
plt.show()
st.pyplot()

grouped_euribor = ba_cleaned4.groupby('y')['euribor3m'].mean().reset_index()
plt.bar(grouped_euribor['y'], grouped_euribor['euribor3m'], color=['red', 'blue'])
plt.xlabel('NO/YES')
plt.ylabel('Average')
plt.title('Euribor 3 Month Rate', color='maroon')
plt.show()
st.pyplot()


st.text('10. nr.employed : number of employees - quarterly indicator')
ba_cleaned4.boxplot(column='nr.employed')
plt.ylabel('Number of Employees')
plt.title('Box Plot for Number of Employees Column', color='maroon')
plt.show()
st.pyplot()

grouped_nre = ba_cleaned4.groupby('y')['nr.employed'].mean().reset_index()
plt.bar(grouped_nre['y'], grouped_nre['nr.employed'], color=['red', 'blue'])
plt.xlabel('NO/YES')
plt.ylabel('Average')
plt.title('Employment Variation Rate', color='maroon')
plt.show()
st.pyplot()

st.text('Categorical Attributes')

# check for the "unknown" values for object
object_columns = ba_cleaned4.select_dtypes(include='object').columns
# Check for rows with 3 or more 'unknown' values across different columns
rows_with_multiple_unknown = ba_cleaned4[ba_cleaned4[object_columns].apply(lambda x: (x == "unknown").sum() >= 3, axis=1)]
st.text('We check for "unknown" values and remove rows with 3 or more "unknown" values')
ba_cleaned5 = ba_cleaned4[~ba_cleaned4.index.isin(rows_with_multiple_unknown.index)]

st.text('Check the percentage of "unknown" values for the object columns for the last data, ba_cleaned5')
for column in object_columns:
    total_rows = len(ba_cleaned5)
    unknown_count = (ba_cleaned5[column] == "unknown").sum()
    unknown_percentage = (unknown_count / total_rows) * 100
    
    if unknown_count > 0:
        print(f"Column '{column}' has {unknown_count} 'unknown' values, which is %{unknown_percentage:.2f} of the total rows.")
st.text('Column "job" has 30 "unknown" values, which is %0.76 of the total rows.')
st.text('Column "marital" has 11 "unknown" values, which is %0.28 of the total rows.')
st.text('Column "education" has 152 "unknown" values, which is %3.83 of the total rows.')
st.text('Column "default" has 747 "unknown" values, which is %18.84 of the total rows.')
st.text('Column "housing" has 79 "unknown" values, which is %1.99 of the total rows.')
st.text('Column "loan" has 79 "unknown" values, which is %1.99 of the total rows.')

st.text('11. "job" : type of job')
# 11. 'job' : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
unique_job_count = ba_cleaned5['job'].nunique()
unique_job_values = ba_cleaned5['job'].unique()
print(f'{unique_job_count} unique values at "job" column')
print(unique_job_values)
st.text('"job" : type of job 12 unique values at "job" column')
st.text('less than %1 of values are "unknown", with a small number of "unknown" and large number of unique values, it wont have a major influence')
st.text('we will use the most common and popular approach is to model the missing value in a categorical column as a new category called Unknown')

st.text('12. "marital" : marital status')
# 12. 'marital' : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
st.text('%0.28 of its is "unknown"')
unique_marital_count = ba_cleaned5['marital'].nunique()
unique_marital_values = ba_cleaned5['marital'].unique()
print(f'{unique_marital_count} unique values at "marital" column')
print(unique_marital_values)
st.text('4 unique values at "marital" column')
st.text('less than %1 of values are "unknown", but differently than "job" there are 3 unique variables at "marital" column except "unknown"')
st.text('thus we will impute values by applying the known distribution to the "unknown" values')

# write a function that takes a variable and returns a randomly selected, value if the variable is 'unknown'
def impute_values(variable, values, prob):
    if variable == 'unknown':
        return np.random.choice(values, p=prob)
    else: 
        return variable 

# drop 'unknown'
values = list(set(ba_cleaned5['marital'].values))
values.remove('unknown')

# Calculate the weight of each variable
prob = ba_cleaned5[ba_cleaned5['marital'] != 'unknown']['marital'].value_counts(normalize=True)
prob = [i/sum(prob) for i in prob]

# impute the values for 'unknown'
imputed_values = np.random.choice(values, size=len(ba_cleaned5[ba_cleaned5['marital'] == 'unknown']), p=prob)

# Update with the imputed values
ba_cleaned5.loc[ba_cleaned5['marital'] == 'unknown', 'marital'] = imputed_values

st.text('13. "education" : (categorical)')
# 13. 'education' : (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
st.text('%3.83 of its is "unknown"')
unique_education_count = ba_cleaned5['education'].nunique()
unique_education_values = ba_cleaned5['education'].unique()
print(f'{unique_education_count} unique values at "education" column')
print(unique_education_values)
st.text('8 unique values at "education" column')
st.text('we will impute values by applying the known distribution to the "unknown" values')

# drop 'unknown'
values = list(set(ba_cleaned5['education'].values))
values.remove('unknown')

# Calculate the weight of each variable
prob = ba_cleaned5[ba_cleaned5['education'] != 'unknown']['education'].value_counts(normalize=True)
prob = [i/sum(prob) for i in prob]

# impute the values for 'unknown'
imputed_values = np.random.choice(values, size=len(ba_cleaned5[ba_cleaned5['education'] == 'unknown']), p=prob)

# Update with the imputed values
ba_cleaned5.loc[ba_cleaned5['education'] == 'unknown', 'education'] = imputed_values


st.text('14. "default" : has credit in default? (yes, no, unknown)')
st.text('%18.84 of its is "unknown"')
unique_default_count = ba_cleaned5['default'].nunique()
unique_default_values = ba_cleaned5['default'].unique()
print(f'{unique_default_count} unique values at "default" column')
print(unique_default_values)
st.text('3 unique values at "default" column')
st.text('this is just a yes/no variable, but also includes %18.84 "unknown" and it would be a large value to impute or drop.')


st.text('15. "housing" : has housing loan? (yes, no, unknown)')
# %1.99 of its is 'unknown'
unique_housing_count = ba_cleaned5['housing'].nunique()
unique_housing_values = ba_cleaned5['housing'].unique()
print(f'{unique_housing_count} unique values at "housing" column')
print(unique_housing_values)
st.text('3 unique values at "housing" column')
st.text('this is again a yes/no variable, but this time includes %1.99 "unknown"')
st.text('thus we will impute values by applying the known distribution to the "unknown" values')

# drop 'unknown'
values = list(set(ba_cleaned5['housing'].values))
values.remove('unknown')

# Calculate the weight of each variable
prob = ba_cleaned5[ba_cleaned5['housing'] != 'unknown']['housing'].value_counts(normalize=True)
prob = [i/sum(prob) for i in prob]

# impute the values for 'unknown'
imputed_values = np.random.choice(values, size=len(ba_cleaned5[ba_cleaned5['housing'] == 'unknown']), p=prob)

# Update with the imputed values
ba_cleaned5.loc[ba_cleaned5['housing'] == 'unknown', 'housing'] = imputed_values


st.text('16. "loan" : has personal loan? (yes, no, unknown)')
st.text('%1.99 of its is "unknown"')
unique_loan_count = ba_cleaned5['loan'].nunique()
unique_loan_values = ba_cleaned5['loan'].unique()
print(f'{unique_loan_count} unique values at "loan" column')
print(unique_loan_values)
st.text('3 unique values at "loan" column')
st.text('this is again a yes/no variable and includes %1.99 "unknown"')
st.text('thus we will impute values by applying the known distribution to the "unknown" values')

# drop 'unknown'
values = list(set(ba_cleaned5['loan'].values))
values.remove('unknown')

# Calculate the weight of each variable
prob = ba_cleaned5[ba_cleaned5['loan'] != 'unknown']['loan'].value_counts(normalize=True)
prob = [i/sum(prob) for i in prob]

# impute the values for 'unknown'
imputed_values = np.random.choice(values, size=len(ba_cleaned5[ba_cleaned5['loan'] == 'unknown']), p=prob)

# Update with the imputed values
ba_cleaned5.loc[ba_cleaned5['loan'] == 'unknown', 'loan'] = imputed_values


st.text('17. "contact" : communication type (categorical: cellular,telephone)')
unique_contact_count = ba_cleaned5['contact'].nunique()
unique_contact_values = ba_cleaned5['contact'].unique()
print(f'{unique_contact_count} unique values at "contact" column')
print(unique_contact_values)
st.text('2 unique values at "contact" column without any "unknown"')


st.text('18. "month" : last contact month of year (without any unknown)')
st.text('no missing values, no any unknown')

st.text('19. "day_of_week" : last contact day of the week (without any unknown')
st.text('no missing values, no any unknown')

st.text('20. "poutcome" : outcome of the previous marketing campaign (categorical: failure,nonexistent,success)')
unique_poutcome_count = ba_cleaned5['poutcome'].nunique()
unique_poutcome_values = ba_cleaned5['poutcome'].unique()
print(f'{unique_poutcome_count} unique values at "poutcome" column')
print(unique_poutcome_values)
st.text('3 unique values at "poutcome" column without any unknown')


st.text('21. "y" : has the client subscribed a term deposit?')
st.text('target variable, no missing values, no any unknown')

# select columns of numeric types (continuous columns)
unit_columns = ba_cleaned5.select_dtypes(include=['number'])

st.text('we can search about the correlations with headmaps')
correlation = abs(round(unit_columns.corr(), 2))
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(correlation, annot=True, square=True, cmap="coolwarm")
plt.title('Correlation Heatmap', fontweight='bold', fontsize=15, color='maroon')
plt.show()
st.pyplot()
st.text('emp.var.rate highly correlated with nr.employed and euribor3m')
st.text('nr.employed and euribor3m correlated with each other')
st.text('we will drop these columns because;')
st.text('the correlated features, the model better correctly predicts the false examples while maintaining its performance of the true samples, which is a great performance gain for modeling.')
columns_to_drop = ['emp.var.rate', 'nr.employed', 'euribor3m']
ba_cleaned5 = ba_cleaned5.drop(columns=columns_to_drop, axis=1)


# our cleaned data
ba_cleaned_last = ba_cleaned5


st.text('check for correlations and multicollinearity at continuous columns')
unit_columns2 = ba_cleaned_last.select_dtypes(include=['number'])
pd.plotting.scatter_matrix(unit_columns2)
plt.tight_layout()
plt.show()
st.pyplot()
# we searched for relations between continuous variables; non of them have any linear relationship.

st.text('Lets we check the relations between variables and "yes/no" answer')

# 1. 'age' :
grouped_age = ba_cleaned_last.groupby('y')['age'].mean().reset_index()
plt.bar(grouped_age['y'], grouped_age['age'], color=['red', 'blue'])
plt.xlabel('NO/YES')
plt.ylabel('Average')
plt.title('Age', color='maroon')
plt.show()
st.pyplot()
# average age of 'yes' and 'no' are very similar

st.text('there are lots of different values of age so we can make scatter plots of it with the others')
unit_columns_wa = list(unit_columns2.columns.values)
unit_columns_wa.remove('age')
custom_palette = {'yes': 'blue', 'no': 'red'}
for v in unit_columns_wa:
    sns.scatterplot(x=ba_cleaned_last['age'], y=ba_cleaned_last[v], hue='y', data=ba_cleaned_last, palette=custom_palette)
    plt.title(f'Age vs {v} with "yes" values', color='maroon')
    plt.show()
    st.pyplot()


# 2. 'duration' : (we droped it)


# 3. 'campaign' :
grouped_campaign = ba_cleaned_last.groupby('y')['campaign'].mean().reset_index()
plt.bar(grouped_campaign['y'], grouped_campaign['campaign'], color=['red', 'blue'])
plt.xlabel('NO/YES')
plt.ylabel('Average')
plt.title('Campaign', color='maroon')
plt.show()
st.pyplot()


# 4. 'pdays' :
st.text('the value 999 means the client was not previously contacted or contacted very long time ago')
filtered_pdays = ba_cleaned_last[ba_cleaned_last['pdays'] != 999]
filtered_pdays
st.text('152 of 3999 rows are smaller than 999, so these are previously contacted clients')

max_pdays = filtered_pdays['pdays'].max()
min_pdays = filtered_pdays['pdays'].min()
# values are between 0 and 21 days

grouped_pdays = filtered_pdays.groupby('y')['pdays'].mean().reset_index()
plt.bar(grouped_pdays['y'], grouped_pdays['pdays'], color=['red', 'blue'])
plt.xlabel('NO/YES')
plt.ylabel('Average')
plt.title('Pdays', color='maroon')
plt.show()
st.pyplot()
st.text('we can see that the average of yes and no are too close, so we will only evaluate previously contacted or not')
st.text('pday=999 or not')
st.text('transform data as; if pday=999, then client not contacted before(0). if pdays<999, then client contacted before(1).')
def transform_pdays(value):
    if value == 999:
        return 0
    elif value < 999:
        return 1

ba_cleaned_last['pdays'] = ba_cleaned_last['pdays'].apply(transform_pdays)

grouped_pdays = ba_cleaned_last.groupby('y')['pdays'].mean().reset_index()
plt.bar(grouped_pdays['y'], grouped_pdays['pdays'], color=['red', 'blue'])
plt.xlabel('NO/YES')
plt.ylabel('Average')
plt.title('Pdays', color='maroon')
plt.show()
st.pyplot()


# 5. 'previous' :
grouped_previous = ba_cleaned_last.groupby('y')['previous'].mean().reset_index()
plt.bar(grouped_previous['y'], grouped_previous['previous'], color=['red', 'blue'])
plt.xlabel('NO/YES')
plt.ylabel('Average')
plt.title('Previous', color='maroon')
plt.show()
st.pyplot()


# 6. emp.var.rate : (we droped it)


# 7. cons.price.idx :
grouped_cpi = ba_cleaned_last.groupby('y')['cons.price.idx'].mean().reset_index()
plt.bar(grouped_cpi['y'], grouped_cpi['cons.price.idx'], color=['red', 'blue'])
plt.xlabel('NO/YES')
plt.ylabel('Average')
plt.title('Consumer Price Index', color='maroon')
plt.show()
st.pyplot()


# 8. cons.conf.idx :
grouped_cci = ba_cleaned_last.groupby('y')['cons.conf.idx'].mean().reset_index()
plt.bar(grouped_cci['y'], grouped_cci['cons.conf.idx'], color=['red', 'blue'])
plt.xlabel('NO/YES')
plt.ylabel('Average')
plt.title('Consumer Confidence Index', color='maroon')
plt.show()
st.pyplot()


# 9. euribor3m : (we droped it)


# 10. nr.employed : (we droped it)


# 11. 'job' :
# count and percentage of them
job_counts = ba_cleaned_last['job'].value_counts()
job_percentage = (job_counts / job_counts.sum()) * 100
colors = ['lightcoral', 'skyblue', 'lightgreen', 'yellow', 'purple', 'orange', 'pink', 'brown', 'teal', 'cyan', 'lavender', 'maroon']
ax = job_counts.plot(kind='bar', color=colors, edgecolor='black')
plt.title('Job Distribution', color='maroon')
for i, v in enumerate(job_percentage):
    ax.text(i, v + 1, f'%{v:.2f}', color='black', ha='center', va='bottom')
plt.tight_layout()
plt.show()
st.pyplot()
 
# Create a cross-tabulation between 'job' and 'y' columns and see the percentage of 'yes' for them
job_y_cross_tab = pd.crosstab(ba_cleaned_last['job'], ba_cleaned_last['y'])
percentage_distribution_job = job_y_cross_tab.div(job_y_cross_tab.sum(axis=1), axis=0) * 100
ax = sns.countplot(data=ba_cleaned_last, x='job', hue='y', palette=custom_palette)
plt.title('Distribution of "YES/NO" for each Job and %yes', color='maroon')
plt.legend(title='yes/no')
for i, (yes, no) in enumerate(zip(job_y_cross_tab['yes'], job_y_cross_tab['no'])):
    total = yes + no
    percentage_yes = (yes / total) * 100
    ax.text(i, yes + 1, f'%{percentage_yes:.2f}', color='black', ha='center', va='bottom')
plt.tight_layout()
plt.show()
st.pyplot()


# 12. 'marital' :
# count and percentage of them
marital_counts = ba_cleaned_last['marital'].value_counts()
marital_percentage = (marital_counts / marital_counts.sum()) * 100
ax = marital_counts.plot(kind='bar', color=colors, edgecolor='black')
plt.title('Marital Distribution', color='maroon')
for i, v in enumerate(marital_percentage):
    ax.text(i, v + 1, f'%{v:.2f}', color='black', ha='center', va='bottom')
plt.tight_layout()
plt.show()
st.pyplot()

# Create a cross-tabulation between 'marital' and 'y' columns and see the percentage of 'yes' for them
marital_y_cross_tab = pd.crosstab(ba_cleaned_last['marital'], ba_cleaned_last['y'])
percentage_distribution_marital = marital_y_cross_tab.div(marital_y_cross_tab.sum(axis=1), axis=0) * 100
ax = sns.countplot(data=ba_cleaned_last, x='marital', hue='y', palette=custom_palette)
plt.title('Distribution of "YES/NO" for each Marital and %yes', color='maroon')
plt.legend(title='yes/no')
for i, (yes, no) in enumerate(zip(marital_y_cross_tab['yes'], marital_y_cross_tab['no'])):
    total = yes + no
    percentage_yes = (yes / total) * 100
    ax.text(i, yes + 1, f'%{percentage_yes:.2f}', color='black', ha='center', va='bottom')
plt.tight_layout()
plt.show()
st.pyplot()


# 13. 'education' :
# count and percentage of them
education_counts = ba_cleaned_last['education'].value_counts()
education_percentage = (education_counts / education_counts.sum()) * 100
ax = education_counts.plot(kind='bar', color=colors, edgecolor='black')
plt.title('Education Distribution', color='maroon')
for i, v in enumerate(education_percentage):
    ax.text(i, v + 1, f'%{v:.2f}', color='black', ha='center', va='bottom')
plt.tight_layout()
plt.show()
st.pyplot()

# Create a cross-tabulation between 'education' and 'y' columns and see the percentage of 'yes' for them
education_y_cross_tab = pd.crosstab(ba_cleaned_last['education'], ba_cleaned_last['y'])
percentage_distribution_education = education_y_cross_tab.div(education_y_cross_tab.sum(axis=1), axis=0) * 100
ax = sns.countplot(data=ba_cleaned_last, x='education', hue='y', palette=custom_palette)
plt.title('Distribution of "YES/NO" for each Education and %yes', color='maroon')
plt.legend(title='yes/no')
for i, (yes, no) in enumerate(zip(education_y_cross_tab['yes'], education_y_cross_tab['no'])):
    total = yes + no
    percentage_yes = (yes / total) * 100
    ax.text(i, yes + 1, f'%{percentage_yes:.2f}', color='black', ha='center', va='bottom')
plt.tight_layout()
plt.show()
st.pyplot()


# 14. 'default' :
# count and percentage of them
default_counts = ba_cleaned_last['default'].value_counts()
default_percentage = (default_counts / default_counts.sum()) * 100
ax = default_counts.plot(kind='bar', color=colors, edgecolor='black')
plt.title('Default Distribution', color='maroon')
for i, v in enumerate(default_percentage):
    ax.text(i, v + 1, f'%{v:.2f}', color='black', ha='center', va='bottom')
plt.tight_layout()
plt.show()
st.pyplot()

# Create a cross-tabulation between 'default' and 'y' columns and see the percentage of 'yes' for them
default_y_cross_tab = pd.crosstab(ba_cleaned_last['default'], ba_cleaned_last['y'])
percentage_distribution_default = default_y_cross_tab.div(default_y_cross_tab.sum(axis=1), axis=0) * 100
ax = sns.countplot(data=ba_cleaned_last, x='default', hue='y', palette=custom_palette)
plt.title('Distribution of "YES/NO" for each Default and %yes', color='maroon')
plt.legend(title='yes/no')
for i, (yes, no) in enumerate(zip(default_y_cross_tab['yes'], default_y_cross_tab['no'])):
    total = yes + no
    percentage_yes = (yes / total) * 100
    ax.text(i, yes + 1, f'%{percentage_yes:.2f}', color='black', ha='center', va='bottom')
plt.tight_layout()
plt.show()
st.pyplot()


# 15. 'housing' :
# count and percentage of them
housing_counts = ba_cleaned_last['housing'].value_counts()
housing_percentage = (housing_counts / housing_counts.sum()) * 100
ax = housing_counts.plot(kind='bar', color=colors, edgecolor='black')
plt.title('Housing Distribution', color='maroon')
for i, v in enumerate(housing_percentage):
    ax.text(i, v + 1, f'%{v:.2f}', color='black', ha='center', va='bottom')
plt.tight_layout()
plt.show()
st.pyplot()


# Create a cross-tabulation between 'housing' and 'y' columns and see the percentage of 'yes' for them
housing_y_cross_tab = pd.crosstab(ba_cleaned_last['housing'], ba_cleaned_last['y'])
percentage_distribution_housing = housing_y_cross_tab.div(housing_y_cross_tab.sum(axis=1), axis=0) * 100
ax = sns.countplot(data=ba_cleaned_last, x='housing', hue='y', palette=custom_palette)
plt.title('Distribution of "YES/NO" for each Housing and %yes', color='maroon')
plt.legend(title='yes/no')
for i, (yes, no) in enumerate(zip(housing_y_cross_tab['yes'], housing_y_cross_tab['no'])):
    total = yes + no
    percentage_yes = (yes / total) * 100
    ax.text(i, yes + 1, f'%{percentage_yes:.2f}', color='black', ha='center', va='bottom')
plt.tight_layout()
plt.show()
st.pyplot()


# 16. 'loan' :
# count and percentage of them
loan_counts = ba_cleaned_last['loan'].value_counts()
loan_percentage = (loan_counts / loan_counts.sum()) * 100
ax = loan_counts.plot(kind='bar', color=colors, edgecolor='black')
plt.title('Loan Distribution', color='maroon')
for i, v in enumerate(loan_percentage):
    ax.text(i, v + 1, f'%{v:.2f}', color='black', ha='center', va='bottom')
plt.tight_layout()
plt.show()
st.pyplot()

# Create a cross-tabulation between 'loan' and 'y' columns and see the percentage of 'yes' for them
loan_y_cross_tab = pd.crosstab(ba_cleaned_last['loan'], ba_cleaned_last['y'])
percentage_distribution_loan = loan_y_cross_tab.div(loan_y_cross_tab.sum(axis=1), axis=0) * 100
ax = sns.countplot(data=ba_cleaned_last, x='loan', hue='y', palette=custom_palette)
plt.title('Distribution of "YES/NO" for each Loan and %yes', color='maroon')
plt.legend(title='yes/no')
for i, (yes, no) in enumerate(zip(loan_y_cross_tab['yes'], loan_y_cross_tab['no'])):
    total = yes + no
    percentage_yes = (yes / total) * 100
    ax.text(i, yes + 1, f'%{percentage_yes:.2f}', color='black', ha='center', va='bottom')
plt.tight_layout()
plt.show()
st.pyplot()


# 17. 'contact' :
# count and percentage of them
contact_counts = ba_cleaned_last['contact'].value_counts()
contact_percentage = (contact_counts / contact_counts.sum()) * 100
ax = contact_counts.plot(kind='bar', color=colors, edgecolor='black')
plt.title('Contact Distribution', color='maroon')
for i, v in enumerate(contact_percentage):
    ax.text(i, v + 1, f'%{v:.2f}', color='black', ha='center', va='bottom')
plt.tight_layout()
plt.show()
st.pyplot()


# Create a cross-tabulation between 'contact' and 'y' columns and see the percentage of 'yes' for them
contact_y_cross_tab = pd.crosstab(ba_cleaned_last['contact'], ba_cleaned_last['y'])
percentage_distribution_contact = contact_y_cross_tab.div(contact_y_cross_tab.sum(axis=1), axis=0) * 100
ax = sns.countplot(data=ba_cleaned_last, x='contact', hue='y', palette=custom_palette)
plt.title('Distribution of "YES/NO" for each Contact and %yes', color='maroon')
plt.legend(title='yes/no')
for i, (yes, no) in enumerate(zip(contact_y_cross_tab['yes'], loan_y_cross_tab['no'])):
    total = yes + no
    percentage_yes = (yes / total) * 100
    ax.text(i, yes + 1, f'%{percentage_yes:.2f}', color='black', ha='center', va='bottom')
plt.tight_layout()
plt.show()
st.pyplot()


# 18. 'month' :
# count and percentage of them
month_counts = ba_cleaned_last['month'].value_counts()
month_percentage = (month_counts / month_counts.sum()) * 100
ax = month_counts.plot(kind='bar', color=colors, edgecolor='black')
plt.title('Month Distribution', color='maroon')
for i, v in enumerate(month_percentage):
    ax.text(i, v + 1, f'%{v:.2f}', color='black', ha='center', va='bottom')
plt.tight_layout()
plt.show()
st.pyplot()

# Create a cross-tabulation between 'month' and 'y' columns and see the percentage of 'yes' for them
month_y_cross_tab = pd.crosstab(ba_cleaned_last['month'], ba_cleaned_last['y'])
percentage_distribution_month = month_y_cross_tab.div(month_y_cross_tab.sum(axis=1), axis=0) * 100
ax = sns.countplot(data=ba_cleaned_last, x='month', hue='y', palette=custom_palette)
plt.title('Distribution of "YES/NO" for each Month and %yes', color='maroon')
plt.legend(title='yes/no')
for i, (yes, no) in enumerate(zip(month_y_cross_tab['yes'], month_y_cross_tab['no'])):
    total = yes + no
    percentage_yes = (yes / total) * 100
    ax.text(i, yes + 1, f'%{percentage_yes:.2f}', color='black', ha='center', va='bottom')
plt.tight_layout()
plt.show()
st.pyplot()


# 19. 'day_of_week' :
# count and percentage of them
day_of_week_counts = ba_cleaned_last['day_of_week'].value_counts()
day_of_week_percentage = (day_of_week_counts / day_of_week_counts.sum()) * 100
ax = day_of_week_counts.plot(kind='bar', color=colors, edgecolor='black')
plt.title('Day of Week Distribution', color='maroon')
for i, v in enumerate(day_of_week_percentage):
    ax.text(i, v + 1, f'%{v:.2f}', color='black', ha='center', va='bottom')
plt.tight_layout()
plt.show()
st.pyplot()

# Create a cross-tabulation between 'day_of_week' and 'y' columns and see the percentage of 'yes' for them
day_of_week_y_cross_tab = pd.crosstab(ba_cleaned_last['day_of_week'], ba_cleaned_last['y'])
percentage_distribution_day_of_week = day_of_week_y_cross_tab.div(day_of_week_y_cross_tab.sum(axis=1), axis=0) * 100
ax = sns.countplot(data=ba_cleaned_last, x='day_of_week', hue='y', palette=custom_palette)
plt.title('Distribution of "YES/NO" for each Day of Week and %yes', color='maroon')
plt.legend(title='yes/no')
for i, (yes, no) in enumerate(zip(day_of_week_y_cross_tab['yes'], day_of_week_y_cross_tab['no'])):
    total = yes + no
    percentage_yes = (yes / total) * 100
    ax.text(i, yes + 1, f'%{percentage_yes:.2f}', color='black', ha='center', va='bottom')
plt.tight_layout()
plt.show()
st.pyplot()


# 20. 'poutcome' :
# count and percentage of them
poutcome_counts = ba_cleaned_last['poutcome'].value_counts()
poutcome_percentage = (poutcome_counts / poutcome_counts.sum()) * 100
ax = poutcome_counts.plot(kind='bar', color=colors, edgecolor='black')
plt.title('Previous Outcome Distribution', color='maroon')
for i, v in enumerate(poutcome_percentage):
    ax.text(i, v + 1, f'%{v:.2f}', color='black', ha='center', va='bottom')
plt.tight_layout()
plt.show()
st.pyplot()
 
# Create a cross-tabulation between 'poutcome' and 'y' columns and see the percentage of 'yes' for them
poutcome_y_cross_tab = pd.crosstab(ba_cleaned_last['poutcome'], ba_cleaned_last['y'])
percentage_distribution_poutcome = poutcome_y_cross_tab.div(poutcome_y_cross_tab.sum(axis=1), axis=0) * 100
ax = sns.countplot(data=ba_cleaned_last, x='poutcome', hue='y', palette=custom_palette)
plt.title('Distribution of "YES/NO" for each Previous Outcome and %yes', color='maroon')
plt.legend(title='yes/no')
for i, (yes, no) in enumerate(zip(poutcome_y_cross_tab['yes'], poutcome_y_cross_tab['no'])):
    total = yes + no
    percentage_yes = (yes / total) * 100
    ax.text(i, yes + 1, f'%{percentage_yes:.2f}', color='black', ha='center', va='bottom')
plt.tight_layout()
plt.show()
st.pyplot()


# 21. 'y' :
y_counts = ba_cleaned_last['y'].value_counts()
y_percentage = (y_counts / y_counts.sum()) * 100
ax = y_counts.plot(kind='bar', color=colors, edgecolor='black')
plt.title('Yes/No Distribution', color='maroon')
for i, v in enumerate(y_percentage):
    ax.text(i, v + 1, f'%{v:.2f}', color='black', ha='center', va='bottom')
plt.tight_layout()
plt.show()
st.pyplot()


st.text('Modeling')

st.text('Encoding categorical variables')
ba_dummies = pd.get_dummies(ba_cleaned_last, drop_first=False)
ba_dummies = ba_dummies.astype(int)

# check for new columns
ba_dummies.columns

st.text('we have binary variables like housing_no, housing_yes, loan_no, loan_yes, y_no, y_yes')
st.text('and we have just contact_cellular and contact_telephone, so we can transform it')
st.text('drop "_no" columns and contact_cellular')
ba_dummies.drop(columns=['housing_no', 'loan_no', 'y_no', 'contact_cellular'], inplace=True)

st.text('train-test split')
st.text('split the target data and variables')
x = ba_dummies.drop(columns=['y_yes'])
y = ba_dummies['y_yes']
st.text('split the data train %80 and test %20')
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state= 42)
# store in 'original_ba'
original_ba = {'X_train': X_train, 'y_train':y_train, 'X_test': X_test, 'y_test':y_test}

st.text('Scale Data')
scaler = StandardScaler()
st.text('transform train and test')
scaled_data_train = scaler.fit_transform(X_train)
scaled_data_test = scaler.transform(X_test)
scaled_ba_train = pd.DataFrame(scaled_data_train, columns=x.columns)
scaled_ba_test = pd.DataFrame(scaled_data_test, columns=x.columns)
scaled_ba_train
scaled_ba_test
# lets check how its affect our data
# age before scaleing
sns.distplot(ba_cleaned_last['age'])
plt.title('Age before scaling', color='maroon')
plt.show()
st.pyplot()
# age after scaling
sns.distplot(scaled_ba_train['age'])
plt.title('Age after scaling', color='maroon')
plt.show()
st.pyplot()
st.text('we can see that the data is distributed same, but range is different')
# store in 'scaled_ba'
scaled_ba = {'X_train': scaled_ba_train, 'y_train':y_train, 'X_test': scaled_ba_test, 'y_test':y_test}

st.text('SMOTE for Imbalance Data')
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(scaled_ba_train, y_train)
# Store in 'smote_ba'
smote_ba = {'X_train': X_train_resampled, 'y_train': y_train_resampled, 'X_test': scaled_ba_test, 'y_test': y_test}
st.text('check the 0s and 1s in the target variable after applying SMOTE')
countplot_data = pd.DataFrame({'Target Variable': smote_ba['y_train']})
plt.figure(figsize=(8, 6))
sns.countplot(x='Target Variable', data=countplot_data)
plt.title('Distribution of Target Variable after SMOTE', color='maroon')
plt.xlabel('Target Variable')
plt.ylabel('Count')
plt.show()
st.pyplot()
# they are equal now


st.text('Logistic Regression')

st.text('We will create a Logistic Regression model for Original/Scaled/SMOTE data, make predictions on the Original/Scaled/SMOTE training and testing sets')
st.text('evaluate the model on the Original/Scaled/SMOTE data and create a DataFrame to store the results for the Original/Scaled/SMOTE data (Logistic Regression)')
# create a Logistic Regression model for Original data
lg_model_original = LogisticRegression(max_iter=1000)
lg_model_original.fit(original_ba['X_train'], original_ba['y_train'])

# make predictions on the Original training and testing sets
y_train_pred_original = lg_model_original.predict(original_ba['X_train'])
y_test_pred_original = lg_model_original.predict(original_ba['X_test'])

# evaluate the model on the Original data
train_accuracy_original = accuracy_score(original_ba['y_train'], y_train_pred_original)
test_accuracy_original = accuracy_score(original_ba['y_test'], y_test_pred_original)
precision_original = precision_score(original_ba['y_test'], y_test_pred_original)
recall_original = recall_score(original_ba['y_test'], y_test_pred_original)
f1_original = f1_score(original_ba['y_test'], y_test_pred_original)

# create a DataFrame to store the results for the Original data (Logistic Regression)
lg_model_results_original = pd.DataFrame({
    'model': ['Logistic Regression'],
    'data': ['original'],
    'train_accuracy': [train_accuracy_original],
    'test_accuracy': [test_accuracy_original],
    'precision': [precision_original],
    'recall': [recall_original],
    'f1_score': [f1_original]
})


# create a Logistic Regression model for Scaled data
lg_model_scaled = LogisticRegression(max_iter=1000)
lg_model_scaled.fit(scaled_ba['X_train'], scaled_ba['y_train'])

# make predictions on the Scaled training and testing sets
y_train_pred_scaled = lg_model_scaled.predict(scaled_ba['X_train'])
y_test_pred_scaled = lg_model_scaled.predict(scaled_ba['X_test'])

# evaluate the model on the Scaled data
train_accuracy_scaled = accuracy_score(scaled_ba['y_train'], y_train_pred_scaled)
test_accuracy_scaled = accuracy_score(scaled_ba['y_test'], y_test_pred_scaled)
precision_scaled = precision_score(scaled_ba['y_test'], y_test_pred_scaled)
recall_scaled = recall_score(scaled_ba['y_test'], y_test_pred_scaled)
f1_scaled = f1_score(scaled_ba['y_test'], y_test_pred_scaled)

# create a DataFrame to store the results for the Scaled data (Logistic Regression)
lg_model_results_scaled = pd.DataFrame({
    'model': ['Logistic Regression'],
    'data': ['scaled'],
    'train_accuracy': [train_accuracy_scaled],
    'test_accuracy': [test_accuracy_scaled],
    'precision': [precision_scaled],
    'recall': [recall_scaled],
    'f1_score': [f1_scaled]
})


# create a Logistic Regression model for SMOTE-resampled data
lg_model_smote = LogisticRegression(max_iter=1000)
lg_model_smote.fit(smote_ba['X_train'], smote_ba['y_train'])

# make predictions on the SMOTE-resampled training and testing sets
y_train_pred_smote = lg_model_smote.predict(smote_ba['X_train'])
y_test_pred_smote = lg_model_smote.predict(smote_ba['X_test'])

# evaluate the model on the SMOTE-resampled data
train_accuracy_smote = accuracy_score(smote_ba['y_train'], y_train_pred_smote)
test_accuracy_smote = accuracy_score(smote_ba['y_test'], y_test_pred_smote)
precision_smote = precision_score(smote_ba['y_test'], y_test_pred_smote)
recall_smote = recall_score(smote_ba['y_test'], y_test_pred_smote)
f1_smote = f1_score(smote_ba['y_test'], y_test_pred_smote)

# create a DataFrame to store the results for the SMOTE-resampled data (Logistic Regression)
lg_model_results_smote = pd.DataFrame({
    'model': ['Logistic Regression'],
    'data': ['smote'],
    'train_accuracy': [train_accuracy_smote],
    'test_accuracy': [test_accuracy_smote],
    'precision': [precision_smote],
    'recall': [recall_smote],
    'f1_score': [f1_smote]
})

st.text('Logistic Regression for Original/Scaled/SMOTE data')
# Combine the results for Original, Scaled, and SMOTE-resampled data
combined_lg_model_results = pd.concat([lg_model_results_original, lg_model_results_scaled, lg_model_results_smote], ignore_index=True)
combined_lg_model_results

st.text('Accuracy : Original and Scaled about %90, Smote %71-74.')
st.text('Precision : Original %70.83, Scaled %64.29, Smote %25.22')
st.text('Recall : Original and Scaled about %19, Smote %62')
st.text('F1 Score : Original and Scaled about %30, Smote %36')

st.text('Original Data : high accuracy on training and test sets.')
st.text('relatively low recall and F1 score, indicating challenges in predicting the minority class.')

st.text('Scaled Data : similar performance to the Original data.')
st.text('slight decrease in precision, recall, and F1 score.')

st.text('SMOTE-resampled Data : lower training accuracy but improved generalization to the test set.')
st.text('Significantly improved recall and F1 score, indicating better handling of the minority class.')


st.text('Random Forest')

st.text('We will create a Random Forest model for Original/Scaled/SMOTE data, make predictions on the Original/Scaled/SMOTE training and testing sets')
st.text('evaluate the model on the Original/Scaled/SMOTE data and create a DataFrame to store the results for the Original/Scaled/SMOTE data (Random Forest)')
# create a Random Forest model for Original data
rf_model_original = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_original.fit(original_ba['X_train'], original_ba['y_train'])

# make predictions on the Original training and testing sets
y_train_pred_rf_original = rf_model_original.predict(original_ba['X_train'])
y_test_pred_rf_original = rf_model_original.predict(original_ba['X_test'])

# evaluate the model on the Original data
train_accuracy_rf_original = accuracy_score(original_ba['y_train'], y_train_pred_rf_original)
test_accuracy_rf_original = accuracy_score(original_ba['y_test'], y_test_pred_rf_original)
precision_rf_original = precision_score(original_ba['y_test'], y_test_pred_rf_original)
recall_rf_original = recall_score(original_ba['y_test'], y_test_pred_rf_original)
f1_rf_original = f1_score(original_ba['y_test'], y_test_pred_rf_original)

# create a DataFrame to store the results for the Original data (Random Forest)
rf_model_results_original = pd.DataFrame({
    'model': ['Random Forest'],
    'data': ['original'],
    'train_accuracy': [train_accuracy_rf_original],
    'test_accuracy': [test_accuracy_rf_original],
    'precision': [precision_rf_original],
    'recall': [recall_rf_original],
    'f1_score': [f1_rf_original]
})


# Create a Random Forest model for Scaled data
rf_model_scaled = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_scaled.fit(scaled_ba['X_train'], scaled_ba['y_train'])

# Make predictions on the Scaled training and testing sets
y_train_pred_rf_scaled = rf_model_scaled.predict(scaled_ba['X_train'])
y_test_pred_rf_scaled = rf_model_scaled.predict(scaled_ba['X_test'])

# Evaluate the model on the Scaled data
train_accuracy_rf_scaled = accuracy_score(scaled_ba['y_train'], y_train_pred_rf_scaled)
test_accuracy_rf_scaled = accuracy_score(scaled_ba['y_test'], y_test_pred_rf_scaled)
precision_rf_scaled = precision_score(scaled_ba['y_test'], y_test_pred_rf_scaled)
recall_rf_scaled = recall_score(scaled_ba['y_test'], y_test_pred_rf_scaled)
f1_rf_scaled = f1_score(scaled_ba['y_test'], y_test_pred_rf_scaled)

# Create a DataFrame to store the results for the Scaled data (Random Forest)
rf_model_results_scaled = pd.DataFrame({
    'model': ['Random Forest'],
    'data': ['scaled'],
    'train_accuracy': [train_accuracy_rf_scaled],
    'test_accuracy': [test_accuracy_rf_scaled],
    'precision': [precision_rf_scaled],
    'recall': [recall_rf_scaled],
    'f1_score': [f1_rf_scaled]
})


# create a Random Forest model for SMOTE-resampled data
rf_model_smote = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_smote.fit(smote_ba['X_train'], smote_ba['y_train'])

# make predictions on the SMOTE-resampled training and testing sets
y_train_pred_rf_smote = rf_model_smote.predict(smote_ba['X_train'])
y_test_pred_rf_smote = rf_model_smote.predict(smote_ba['X_test'])

# evaluate the model on the SMOTE-resampled data
train_accuracy_rf_smote = accuracy_score(smote_ba['y_train'], y_train_pred_rf_smote)
test_accuracy_rf_smote = accuracy_score(smote_ba['y_test'], y_test_pred_rf_smote)
precision_rf_smote = precision_score(smote_ba['y_test'], y_test_pred_rf_smote)
recall_rf_smote = recall_score(smote_ba['y_test'], y_test_pred_rf_smote)
f1_rf_smote = f1_score(smote_ba['y_test'], y_test_pred_rf_smote)

# create a DataFrame to store the results for the SMOTE-resampled data (Random Forest)
rf_model_results_smote = pd.DataFrame({
    'model': ['Random Forest'],
    'data': ['smote'],
    'train_accuracy': [train_accuracy_rf_smote],
    'test_accuracy': [test_accuracy_rf_smote],
    'precision': [precision_rf_smote],
    'recall': [recall_rf_smote],
    'f1_score': [f1_rf_smote]
})

st.text('Random Forest for Original/Scaled/SMOTE data')
# display the combined results
combined_rf_model_results = pd.concat([rf_model_results_original, rf_model_results_scaled, rf_model_results_smote], ignore_index=True)
combined_rf_model_results

st.text('Accuracy : all of them are nearly 1.0 at train accuracy and about %87 at test accuracy')
st.text('Precision : Original and Scaled about %44, Smote %39')
st.text('Recall : all of them about %22-24')
st.text('F1 Score : all of them about %29')

st.text('Random Forest model performs well in terms of accuracy, nearly all values ​​are close to each other')
st.text('but may benefit from further optimization to address potential overfitting and improve its ability to capture instances of the positive class.')

st.text('Combined Data')
# Display the combined results for both Logistic Regression and Random Forest
combined_all_model_results = pd.concat([combined_lg_model_results, combined_rf_model_results], ignore_index=True)
combined_all_model_results


st.text('K-Nearest Neighbors')

st.text('We will create a K-Nearest Neighbors model for Original/Scaled/SMOTE data, make predictions on the Original/Scaled/SMOTE training and testing sets')
st.text('evaluate the model on the Original/Scaled/SMOTE data and create a DataFrame to store the results for the Original/Scaled/SMOTE data (K-Nearest Neighbors)')
# create a K-Nearest Neighbors model for Original data
knn_model_original = KNeighborsClassifier(n_neighbors=5)
knn_model_original.fit(original_ba['X_train'], original_ba['y_train'])

# make predictions on the Original training and testing sets
y_train_pred_knn_original = knn_model_original.predict(original_ba['X_train'])
y_test_pred_knn_original = knn_model_original.predict(original_ba['X_test'])

# evaluate the model on the Original data
train_accuracy_knn_original = accuracy_score(original_ba['y_train'], y_train_pred_knn_original)
test_accuracy_knn_original = accuracy_score(original_ba['y_test'], y_test_pred_knn_original)
precision_knn_original = precision_score(original_ba['y_test'], y_test_pred_knn_original)
recall_knn_original = recall_score(original_ba['y_test'], y_test_pred_knn_original)
f1_knn_original = f1_score(original_ba['y_test'], y_test_pred_knn_original)

# create a DataFrame to store the results for the Original data (K-Nearest Neighbors)
knn_model_results_original = pd.DataFrame({
    'model': ['K-Nearest Neighbors'],
    'data': ['original'],
    'train_accuracy': [train_accuracy_knn_original],
    'test_accuracy': [test_accuracy_knn_original],
    'precision': [precision_knn_original],
    'recall': [recall_knn_original],
    'f1_score': [f1_knn_original]
})


# create a K-Nearest Neighbors model for Scaled data
knn_model_scaled = KNeighborsClassifier(n_neighbors=5)
knn_model_scaled.fit(scaled_ba['X_train'], scaled_ba['y_train'])

# make predictions on the Scaled training and testing sets
y_train_pred_knn_scaled = knn_model_scaled.predict(scaled_ba['X_train'])
y_test_pred_knn_scaled = knn_model_scaled.predict(scaled_ba['X_test'])

# evaluate the model on the Scaled data
train_accuracy_knn_scaled = accuracy_score(scaled_ba['y_train'], y_train_pred_knn_scaled)
test_accuracy_knn_scaled = accuracy_score(scaled_ba['y_test'], y_test_pred_knn_scaled)
precision_knn_scaled = precision_score(scaled_ba['y_test'], y_test_pred_knn_scaled)
recall_knn_scaled = recall_score(scaled_ba['y_test'], y_test_pred_knn_scaled)
f1_knn_scaled = f1_score(scaled_ba['y_test'], y_test_pred_knn_scaled)

# create a DataFrame to store the results for the Scaled data (K-Nearest Neighbors)
knn_model_results_scaled = pd.DataFrame({
    'model': ['K-Nearest Neighbors'],
    'data': ['scaled'],
    'train_accuracy': [train_accuracy_knn_scaled],
    'test_accuracy': [test_accuracy_knn_scaled],
    'precision': [precision_knn_scaled],
    'recall': [recall_knn_scaled],
    'f1_score': [f1_knn_scaled]
})


# create a K-Nearest Neighbors model for SMOTE-resampled data
knn_model_smote = KNeighborsClassifier(n_neighbors=5)
knn_model_smote.fit(smote_ba['X_train'], smote_ba['y_train'])

# make predictions on the SMOTE-resampled training and testing sets
y_train_pred_knn_smote = knn_model_smote.predict(smote_ba['X_train'])
y_test_pred_knn_smote = knn_model_smote.predict(smote_ba['X_test'])

# evaluate the model on the SMOTE-resampled data
train_accuracy_knn_smote = accuracy_score(smote_ba['y_train'], y_train_pred_knn_smote)
test_accuracy_knn_smote = accuracy_score(smote_ba['y_test'], y_test_pred_knn_smote)
precision_knn_smote = precision_score(smote_ba['y_test'], y_test_pred_knn_smote)
recall_knn_smote = recall_score(smote_ba['y_test'], y_test_pred_knn_smote)
f1_knn_smote = f1_score(smote_ba['y_test'], y_test_pred_knn_smote)

# create a DataFrame to store the results for the SMOTE-resampled data (K-Nearest Neighbors)
knn_model_results_smote = pd.DataFrame({
    'model': ['K-Nearest Neighbors'],
    'data': ['smote'],
    'train_accuracy': [train_accuracy_knn_smote],
    'test_accuracy': [test_accuracy_knn_smote],
    'precision': [precision_knn_smote],
    'recall': [recall_knn_smote],
    'f1_score': [f1_knn_smote]
})

st.text('K-Nearest Neighbors for Original/Scaled/SMOTE data')
# display the combined results
combined_knn_model_results = pd.concat([knn_model_results_original, knn_model_results_scaled, knn_model_results_smote], ignore_index=True)
combined_knn_model_results


st.text('Accuracy : all of them about %91 at train accuracy. Original and Scaled about %87-89 at test accuracy but Smote %69')
st.text('Precision : Original about %56, Scaled %35 and Smote %22')
st.text('Recall : Original and Scaled about %17, Smote %63')
st.text('F1 Score : Original and Scaled about %23-25, Smote %32')

st.text('Original Data : the model performs reasonably well on the original data,')
st.text('with a good balance between precision and recall.')

st.text('Scaled Data : Scaled data slightly decreases the models performance, especially in precision and recall.')

st.text('SMOTE-resampled Data : SMOTE-resampled data significantly impacts the models accuracy, precision, and recall, suggesting challenges in generalization.')

st.text('All Combined Data')
# Display the combined results for all Logistic Regression, Random Forest and K-Nearest Neighbors
combined_all_model_results = pd.concat([combined_lg_model_results, combined_rf_model_results, combined_knn_model_results], ignore_index=True)
combined_all_model_results


st.text('Model selection')

st.text('1. Accuracy : Identify the best model based on test_accuracy')
best_model_accuracy = combined_all_model_results[['model', 'data', 'test_accuracy']]
best_model_accuracy_sorted = best_model_accuracy.sort_values(by='test_accuracy', ascending=False)
print(best_model_accuracy_sorted)
best_model_top_accuracy = best_model_accuracy_sorted.iloc[0]
print(f"Best: {best_model_top_accuracy['model']}({best_model_top_accuracy['data']}) with Test Accuracy: {best_model_top_accuracy['test_accuracy']:.2%}")
st.text('Best: Logistic Regression(original) with Test Accuracy: %89.66')

st.text('2. Precision : Identify the best model based on precision')
best_model_precision = combined_all_model_results[['model', 'data', 'precision']]
best_model_precision_sorted = best_model_precision.sort_values(by='precision', ascending=False)
print(best_model_precision_sorted)
best_model_top_precision = best_model_precision_sorted.iloc[0]
print(f"Best: {best_model_top_precision['model']}({best_model_top_precision['data']}) with Precision: {best_model_top_precision['precision']:.2%}")
st.text('Best: Logistic Regression(original) with Precision: %70.83')

st.text('3. Recall : Identify the best model based on recall')
best_model_recall = combined_all_model_results[['model', 'data', 'recall']]
best_model_recall_sorted = best_model_recall.sort_values(by='recall', ascending=False)
print(best_model_recall_sorted)
best_model_top_recall = best_model_recall_sorted.iloc[0]
print(f"Best: {best_model_top_recall['model']}({best_model_top_recall['data']}) with Recall: {best_model_top_recall['recall']:.2%}")
st.text('Best: K-Nearest Neighbors(smote) with Recall: %63.04')

st.text('4. F1 Score : Identify the best model based on F1 Score')
best_model_f1 = combined_all_model_results[['model', 'data', 'f1_score']]
best_model_f1_sorted = best_model_f1.sort_values(by='f1_score', ascending=False)
print(best_model_f1_sorted)
best_model_top_f1 = best_model_f1_sorted.iloc[0]
print(f"Best: {best_model_top_f1['model']}({best_model_top_f1['data']}) with F1 Score: {best_model_top_f1['f1_score']:.2%}")
st.text('Best: Logistic Regression(smote) with F1 Score: %35.85')


st.text('given the class imbalance in our target variable (yes and no),')
st.text('its crucial to consider evaluation metrics that are sensitive to imbalanced datasets.')
st.text('we might want to consider the area under the Precision-Recall (PR) curve.')


st.text('Calculate AUC-PR')
# Function to calculate AUC-PR
def calculate_auc_pr(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    auc_pr = auc(recall, precision)
    return auc_pr

# Calculate AUC-PR for Logistic Regression (Original)
y_scores_lr_original = lg_model_original.predict_proba(original_ba['X_test'])[:, 1]
auc_pr_lr_original = calculate_auc_pr(original_ba['y_test'], y_scores_lr_original)
lg_model_results_original['auc_pr'] = auc_pr_lr_original

# Calculate AUC-PR for Logistic Regression (Scaled)
y_scores_lr_scaled = lg_model_scaled.predict_proba(scaled_ba['X_test'])[:, 1]
auc_pr_lr_scaled = calculate_auc_pr(scaled_ba['y_test'], y_scores_lr_scaled)
lg_model_results_scaled['auc_pr'] = auc_pr_lr_scaled

# Calculate AUC-PR for Logistic Regression (SMOTE)
y_scores_lr_smote = lg_model_smote.predict_proba(smote_ba['X_test'])[:, 1]
auc_pr_lr_smote = calculate_auc_pr(smote_ba['y_test'], y_scores_lr_smote)
lg_model_results_smote['auc_pr'] = auc_pr_lr_smote

# Calculate AUC-PR for Random Forest (Original)
y_scores_rf_original = rf_model_original.predict_proba(original_ba['X_test'])[:, 1]
auc_pr_rf_original = calculate_auc_pr(original_ba['y_test'], y_scores_rf_original)
rf_model_results_original['auc_pr'] = auc_pr_rf_original

# Calculate AUC-PR for Random Forest (Scaled)
y_scores_rf_scaled = rf_model_scaled.predict_proba(scaled_ba['X_test'])[:, 1]
auc_pr_rf_scaled = calculate_auc_pr(scaled_ba['y_test'], y_scores_rf_scaled)
rf_model_results_scaled['auc_pr'] = auc_pr_rf_scaled

# Calculate AUC-PR for Random Forest (SMOTE)
y_scores_rf_smote = rf_model_smote.predict_proba(smote_ba['X_test'])[:, 1]
auc_pr_rf_smote = calculate_auc_pr(smote_ba['y_test'], y_scores_rf_smote)
rf_model_results_smote['auc_pr'] = auc_pr_rf_smote

# Calculate AUC-PR for K-Nearest Neighbors (Original)
y_scores_knn_original = knn_model_original.predict_proba(original_ba['X_test'])[:, 1]
auc_pr_knn_original = calculate_auc_pr(original_ba['y_test'], y_scores_knn_original)
knn_model_results_original['auc_pr'] = auc_pr_knn_original

# Calculate AUC-PR for K-Nearest Neighbors (Scaled)
y_scores_knn_scaled = knn_model_scaled.predict_proba(scaled_ba['X_test'])[:, 1]
auc_pr_knn_scaled = calculate_auc_pr(scaled_ba['y_test'], y_scores_knn_scaled)
knn_model_results_scaled['auc_pr'] = auc_pr_knn_scaled

# Calculate AUC-PR for K-Nearest Neighbors (SMOTE)
y_scores_knn_smote = knn_model_smote.predict_proba(smote_ba['X_test'])[:, 1]
auc_pr_knn_smote = calculate_auc_pr(smote_ba['y_test'], y_scores_knn_smote)
knn_model_results_smote['auc_pr'] = auc_pr_knn_smote

st.text('All Combined Data with AUC-PR')
# Combine the results for all models
combined_all_model_results_with_auc_pr = pd.concat([lg_model_results_original, lg_model_results_scaled, lg_model_results_smote,
                                                    rf_model_results_original, rf_model_results_scaled, rf_model_results_smote,
                                                    knn_model_results_original, knn_model_results_scaled, knn_model_results_smote], ignore_index=True)
combined_all_model_results_with_auc_pr


st.text('Let we see the plots of AUC-PR')

st.text('Plot AUC-PR for Logistic Regression')
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
precision_lr_original, recall_lr_original, _ = precision_recall_curve(original_ba['y_test'], y_scores_lr_original)
plt.plot(recall_lr_original, precision_lr_original, label=f'Logistic Regression (Original), AUC-PR: {auc_pr_lr_original:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Logistic Regression (Original)', color='maroon')
plt.legend()

plt.subplot(3, 1, 2)
precision_lr_scaled, recall_lr_scaled, _ = precision_recall_curve(scaled_ba['y_test'], y_scores_lr_scaled)
plt.plot(recall_lr_scaled, precision_lr_scaled, label=f'Logistic Regression (Scaled), AUC-PR: {auc_pr_lr_scaled:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Logistic Regression (Scaled)', color='maroon')
plt.legend()

plt.subplot(3, 1, 3)
precision_lr_smote, recall_lr_smote, _ = precision_recall_curve(smote_ba['y_test'], y_scores_lr_smote)
plt.plot(recall_lr_smote, precision_lr_smote, label=f'Logistic Regression (SMOTE), AUC-PR: {auc_pr_lr_smote:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Logistic Regression (SMOTE)', color='maroon')
plt.legend()

plt.tight_layout()
plt.show()
st.pyplot()

st.text('Plot AUC-PR for Random Forest')
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
precision_rf_original, recall_rf_original, _ = precision_recall_curve(original_ba['y_test'], y_scores_rf_original)
plt.plot(recall_rf_original, precision_rf_original, label=f'Random Forest (Original), AUC-PR: {auc_pr_rf_original:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Random Forest (Original)', color='maroon')
plt.legend()

plt.subplot(3, 1, 2)
precision_rf_scaled, recall_rf_scaled, _ = precision_recall_curve(scaled_ba['y_test'], y_scores_rf_scaled)
plt.plot(recall_rf_scaled, precision_rf_scaled, label=f'Random Forest (Scaled), AUC-PR: {auc_pr_rf_scaled:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Random Forest (Scaled)', color='maroon')
plt.legend()

plt.subplot(3, 1, 3)
precision_rf_smote, recall_rf_smote, _ = precision_recall_curve(smote_ba['y_test'], y_scores_rf_smote)
plt.plot(recall_rf_smote, precision_rf_smote, label=f'Random Forest (SMOTE), AUC-PR: {auc_pr_rf_smote:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Random Forest (SMOTE)', color='maroon')
plt.legend()

plt.tight_layout()
plt.show()
st.pyplot()

st.text('Plot AUC-PR for K-Nearest Neighbors')
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
precision_knn_original, recall_knn_original, _ = precision_recall_curve(original_ba['y_test'], y_scores_knn_original)
plt.plot(recall_knn_original, precision_knn_original, label=f'K-Nearest Neighbors (Original), AUC-PR: {auc_pr_knn_original:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for K-Nearest Neighbors (Original)', color='maroon')
plt.legend()

plt.subplot(3, 1, 2)
precision_knn_scaled, recall_knn_scaled, _ = precision_recall_curve(scaled_ba['y_test'], y_scores_knn_scaled)
plt.plot(recall_knn_scaled, precision_knn_scaled, label=f'K-Nearest Neighbors (Scaled), AUC-PR: {auc_pr_knn_scaled:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for K-Nearest Neighbors (Scaled)', color='maroon')
plt.legend()

plt.subplot(3, 1, 3)
precision_knn_smote, recall_knn_smote, _ = precision_recall_curve(smote_ba['y_test'], y_scores_knn_smote)
plt.plot(recall_knn_smote, precision_knn_smote, label=f'K-Nearest Neighbors (SMOTE), AUC-PR: {auc_pr_knn_smote:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for K-Nearest Neighbors (SMOTE)', color='maroon')
plt.legend()

plt.tight_layout()
plt.show()
st.pyplot()


st.text('In a highly imbalanced dataset like ours, where the positive class is only %10.87, accuracy alone may not be a reliable metric.')
st.text('Precision, recall, and the area under the precision-recall curve (AUC-PR) are often more informative in such cases.')

st.text('5. AUC-PR : Identify the best model based on AUC-PR')
best_model_auc_pr = combined_all_model_results_with_auc_pr[['model', 'data', 'auc_pr']]
best_model_auc_pr_sorted = best_model_auc_pr.sort_values(by='auc_pr', ascending=False)
print(best_model_auc_pr_sorted)
best_model_top_auc_pr = best_model_auc_pr_sorted.iloc[0]
print(f"Best: {best_model_top_auc_pr['model']}({best_model_top_auc_pr['data']}) with AUC-PR: {best_model_top_auc_pr['auc_pr']:.2%}")
st.text('Best: Logistic Regression(scaled) with AUC-PR: %43.79')


st.text('Let we check all list')
combined_all_model_results_with_auc_pr

st.text('we choose Logistic Regression(original) with highest test_accuracy and precision.')
st.text('Its recall is low but auc_pr is 2nd highest value and it reinforcing the models ability to balance precision and recall.')

st.text('Logistic Regression (Original) - Chosen Model:')
st.text('Test Accuracy: It has the highest test accuracy among the Logistic Regression models, indicating good overall performance on the test set.')
st.text('Precision: The precision is relatively high, suggesting that when the model predicts the positive class, it is likely to be correct.')
st.text('Recall: The recall is lower, indicating that the model may miss some positive instances, but this might be acceptable depending on our priorities.')
st.text('AUC-PR: AUC-PR is the second-highest among the models, reinforcing the models ability to balance precision and recall.')
st.text('Depending on the specific requirements of our application and the associated costs of false positives and false negatives, this model strikes a balance between accuracy, precision, and AUC-PR.')

st.text('Future Steps: tune the hyperparameters of the selected model to improve its performance.')


st.text('Hyperparameter tuning')

# Function to calculate AUC-PR for GridSearchCV
def calculate_auc_pr_grid_search(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    auc_pr = auc(recall, precision)
    return auc_pr

# Logistic Regression model
lg_model = LogisticRegression()

# Parameter grid for GridSearchCV
param_grid = {
    'C': [1],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear'],
    'max_iter': [1000, 2000, 5000]
}

# AUC-PR as the scoring metric for GridSearchCV
scorer = make_scorer(calculate_auc_pr_grid_search, greater_is_better=True)

# GridSearchCV
grid_search = GridSearchCV(lg_model, param_grid, cv=5, scoring=scorer, verbose=1, n_jobs=-1)

# Fit the model with the original data
grid_search.fit(original_ba['X_train'], original_ba['y_train'])

# Get the best model from the grid search
best_lg_model = grid_search.best_estimator_

# Make predictions on the training and testing sets using the best model
y_train_pred_best = best_lg_model.predict(original_ba['X_train'])
y_test_pred_best = best_lg_model.predict(original_ba['X_test'])

# Evaluate the best model on the original data
train_accuracy_best = accuracy_score(original_ba['y_train'], y_train_pred_best)
test_accuracy_best = accuracy_score(original_ba['y_test'], y_test_pred_best)
precision_best = precision_score(original_ba['y_test'], y_test_pred_best)
recall_best = recall_score(original_ba['y_test'], y_test_pred_best)
f1_best = f1_score(original_ba['y_test'], y_test_pred_best)

# Create a DataFrame to store the results for the best model
lg_model_results_best = pd.DataFrame({
    'model': ['Logistic Regression (Tuned)'],
    'data': ['original'],
    'train_accuracy': [train_accuracy_best],
    'test_accuracy': [test_accuracy_best],
    'precision': [precision_best],
    'recall': [recall_best],
    'f1_score': [f1_best]
})

# Calculate AUC-PR for the best model
y_scores_best = best_lg_model.predict_proba(original_ba['X_test'])[:, 1]
auc_pr_best = calculate_auc_pr(original_ba['y_test'], y_scores_best)
lg_model_results_best['auc_pr'] = auc_pr_best

# Display the results for the best model
print("Best hyperparameters:", best_lg_model)
print("\nResults for the best model:")
print(lg_model_results_best)
st.text('The tuned Logistic Regression model exhibits improved performance on the original dataset compared to its untuned counterpart.')

# Make predictions on the testing set using the best model
y_test_pred_final = best_lg_model.predict(original_ba['X_test'])

# Evaluate the final model
final_accuracy = accuracy_score(original_ba['y_test'], y_test_pred_final)
final_precision = precision_score(original_ba['y_test'], y_test_pred_final)
final_recall = recall_score(original_ba['y_test'], y_test_pred_final)
final_f1 = f1_score(original_ba['y_test'], y_test_pred_final)
final_auc_pr = calculate_auc_pr(original_ba['y_test'], best_lg_model.predict_proba(original_ba['X_test'])[:, 1])

# Create a DataFrame to store the results for the final model
final_results = pd.DataFrame({
    'model': ['Logistic Regression (Final)'],
    'data': ['original'],
    'test_accuracy': [final_accuracy],
    'precision': [final_precision],
    'recall': [final_recall],
    'f1_score': [final_f1],
    'auc_pr': [final_auc_pr]
})

st.text('Final Results')
# Display the results for the final model
print("Results for the final model:")
print(final_results)
final_results


st.text('Summary:')
st.text('Model Evaluation : Models considered: Logistic Regression, Random Forest, and K-Nearest Neighbors.')
st.text('Evaluation metrics: Accuracy, precision, recall, F1 score, and AUC-PR.')

st.text('Model Selection : Identified the best models based on different metrics (accuracy, precision, recall, F1 score, and AUC-PR).')
st.text('Chose Logistic Regression (original) as the best model based on test accuracy.')

st.text('AUC-PR Analysis : Calculated AUC-PR for each model and visualized Precision-Recall curves.')

st.text('Final Model Selection : Chose Logistic Regression (original) as the final model, considering its test accuracy, precision, recall, and AUC-PR.')

st.text('Hyperparameter Tuning : Used GridSearchCV for hyperparameter tuning of the Logistic Regression model.')
st.text('Tuned parameters include regularization strength (C), penalty type, solver, and maximum iterations.')

st.text('Final Model Evaluation : Evaluated the tuned Logistic Regression model on the testing set.')
st.text('Examined accuracy, precision, recall, F1 score, and AUC-PR for the final model.')
st.text('The final tuned Logistic Regression model is expected to provide improved performance on the original dataset,')
st.text('the hyperparameter tuning process helps optimize the models parameters for better generalization to unseen data.')

