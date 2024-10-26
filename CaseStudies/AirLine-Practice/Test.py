# libraries for data manipulation
import pandas as pd
import numpy as np

# libraries for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# importing LLM models and tokenizers
from transformers import T5Tokenizer, T5ForConditionalGeneration

# importing metric functions
from sklearn.metrics import confusion_matrix, accuracy_score

# importing library to split the data
from sklearn.model_selection import train_test_split

# library for model deployment
import gradio as gr

# mounting Google Drive
#from google.colab import drive
#drive.mount('/content/drive')

#df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/AI Application Case Study - Text Data/Us_Airways.csv')
df = pd.read_csv('D:/Dev/AI-ML-Course/CaseStudies/AirLine/Us_Airways.csv')
df.sample(10, random_state=1)

#df = pd.read_csv('/content/Us_Airways.csv')

#Exploratory Data Analysis (EDA) plays a very important role in an end-to-end AI solution. It enables

#- **Understanding the Data**
#- **Identifying Data Patterns and Insights**
#- **Feature Selection and Engineering**
data = df.copy()
data.describe().T

# function to create labeled barplots

def labeled_barplot(data, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(data[feature])  # length of the column
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n].sort_values(),
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage

    plt.show()  # show the plot

def labeled_barplot(data, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(data[feature])  # length of the column
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n],
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage

    plt.show()  # show the plot

labeled_barplot(data, "airline_sentiment", perc=True)

labeled_barplot(data, "negativereason", perc=True)

sns.histplot(data, x='retweet_count', bins =50);

sns.barplot(data, y='retweet_count', x='airline_sentiment', errorbar=('ci', False));


# **Model Building**
## **Model Training and Evalution**

## Training an AI model is important because it allows machines to learn and perform tasks without explicit programming. It enables the following:

##- **Learning from Data**
##- **Generalization and Adaptability**
##- **Optimization and Performance Improvement**

## Data Preprocessing
# Specify the features (X) and the target variable (y)
X = data.drop('airline_sentiment', axis=1)  # Replace 'target_variable' with the actual name of your target column
y = data['airline_sentiment']

# Further split the temporary data into validation and test sets
X_validation, X_test, y_validation, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

text_column_validation = X_validation['text'].copy()
actual_sentiment_validation = y_validation.copy()

text_column_test = X_test['text'].copy()
actual_sentiment_test = y_test.copy()

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", load_in_8bit=True, device_map="auto")
#model.coef_
# AttributeError: 'T5ForConditionalGeneration' object has no attribute 'coef_'

# defining a function to generate, process, and return the LLM response
def llm_response(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids, max_length=300, do_sample=True, temperature=0.1)
    return tokenizer.decode(outputs[0])[6:-4]

# checking a customer review and it's sentiment
print(text_column_validation[4])
print(actual_sentiment_validation[4])

# predicting the sentiment using the LLM
sys_prompt = """
    Categorize the sentiment of the customer review as positive, negative, or neutral.
"""

pred_sent = llm_response(
    """
        {}
        Review text: '{}'
    """.format(sys_prompt, text_column_validation[4])
)

print(pred_sent)

def predict_sentiment(review_text):
    pred = llm_response(
        """
            {}
            Review text: '{}'
        """.format(sys_prompt, review_text)
    )

    return pred

predicted_sentiment = [predict_sentiment(item) for item in text_column_validation.values]
print(predicted_sentiment[4])


# combining the reviews, actual sentiments, and predicted sentiments together
df_combined = pd.concat([text_column_validation, actual_sentiment_validation], axis=1)
df_combined['predicted_sentiment'] = predicted_sentiment
df_combined.head()

# creating confusion matrix
cnf_mt = confusion_matrix(df_combined['airline_sentiment'], df_combined['predicted_sentiment'], labels=['positive', 'neutral', 'negative'])
# computing accuracy
acc = accuracy_score(df_combined['airline_sentiment'], df_combined['predicted_sentiment'])

# printing accuracy
print("Accuracy:", acc)

# creating a heatmap of the confusion matrix
plt.figure(figsize=(7, 5))
sns.heatmap(
    cnf_mt,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=['positive', 'neutral', 'negative'],
    yticklabels=['positive', 'neutral', 'negative'],
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

sys_prompt = """
    Categorize the sentiment of the customer review as positive, negative, or neutral.
    Leverage your expertise in the aviation industry and deep understanding of industry trends to analyze the nuanced expressions and overall tone.
    It is crucial to accurately identify neutral sentiments, which may indicate a balanced view or neutral stance towards Us Airways. Neutral expressions could involve factual statements without explicit positive or negative opinions.
    Consider the importance of these neutral sentiments in gauging the public sentiment towards the airline company.
    For instance, a positive sentiment might convey satisfaction with the airline's services, a negative sentiment could express dissatisfaction, while neutral sentiment may reflect an impartial observation or a neutral standpoint
"""

pred_sent = llm_response(
    """
        {}
        Review text: '{}'
    """.format(sys_prompt, text_column_validation[4])
)

print(pred_sent)

predicted_sentiment_tuned = [predict_sentiment(item) for item in text_column_validation.values]
print(predicted_sentiment_tuned[4])

# combining the reviews, actual sentiments, and predicted sentiments together
df_combined = pd.concat([text_column_validation, actual_sentiment_validation], axis=1)
df_combined['predicted_sentiment'] = predicted_sentiment_tuned
df_combined.head()

# creating confusion matrix
cnf_mt_tuned = confusion_matrix(df_combined['airline_sentiment'], df_combined['predicted_sentiment'], labels=['positive', 'neutral', 'negative'])
# computing accuracy
acc_tuned = accuracy_score(df_combined['airline_sentiment'], df_combined['predicted_sentiment'])

# printing accuracy
print("Accuracy:", acc_tuned)

# creating a heatmap of the confusion matrix
plt.figure(figsize=(7, 5))
sns.heatmap(
    cnf_mt_tuned,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=['positive', 'neutral', 'negative'],
    yticklabels=['positive', 'neutral', 'negative'],
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

## Observations from Model Evaluation

##- The AI model's performance score has increased a bit to ~84%

##- The model is also able to identify neutral sentiments better now

##- One can try to improvise the prompt further to improve the model performance

## After tuning the car, it's essential to test it thoroughly before using it in real-world situations. Model testing is like taking the car out for a test drive to ensure it performs as expected. In this stage, you simulate different scenarios and evaluate how well the model responds. For the car example, you would assess how the car handles various driving conditions, such as highways, urban roads, and off-road terrains. Testing helps identify any issues or weaknesses in the model that need to be addressed.

# Model testing is important for:

## - **Validating model performance**
## - **Identifying and mitigating errors or flaws**
## - **Assessing model robustness and generalizability**
## - **Building user trust and confidence**

predicted_sentiment_test = [predict_sentiment(item) for item in text_column_test.values]

# combining the reviews, actual sentiments, and predicted sentiments together
df_combined = pd.concat([text_column_test, actual_sentiment_test], axis=1)
df_combined['predicted_sentiment'] = predicted_sentiment_test
df_combined.head()
# creating confusion matrix
cnf_mt_tuned = confusion_matrix(df_combined['airline_sentiment'], df_combined['predicted_sentiment'], labels=['positive', 'neutral', 'negative'])
# computing accuracy
acc_tuned = accuracy_score(df_combined['airline_sentiment'], df_combined['predicted_sentiment'])

# printing accuracy
print("Accuracy:", acc_tuned)

# creating a heatmap of the confusion matrix
plt.figure(figsize=(7, 5))
sns.heatmap(
    cnf_mt_tuned,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=['positive', 'neutral', 'negative'],
    yticklabels=['positive', 'neutral', 'negative'],
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# define a function that will take the necessary inputs and make predictions

def predict_review_sentiment(review):
    sys_prompt = """
        Categorize the sentiment of the customer review as positive, negative, or neutral.
        Leverage your expertise in the aviation industry and deep understanding of industry trends to analyze the nuanced expressions and overall tone.
        It is crucial to accurately identify neutral sentiments, which may indicate a balanced view or neutral stance towards Us Airways. Neutral expressions could involve factual statements without explicit positive or negative opinions.
        Consider the importance of these neutral sentiments in gauging the public sentiment towards the airline company.
        For instance, a positive sentiment might convey satisfaction with the airline's services, a negative sentiment could express dissatisfaction, while neutral sentiment may reflect an impartial observation or a neutral standpoint
    """

    # predicting the sentiment of the review
    pred_sent = llm_response(
        """
            {}
            Review text: '{}'
        """.format(sys_prompt, review)
    )

    # returning the final output
    return pred_sent

# creating the deployment input interface
review_text = gr.Textbox(label="Enter the customer sentiment here.")

# creating the deployment output interface
sentiment = gr.Textbox(label="Sentiment Type")

# defining the structure of the deployment interface and how the components will interact
demo = gr.Interface(
    fn=predict_review_sentiment,
    inputs = review_text,
    outputs = sentiment,
    title="Customer Review Sentiment Analyzer",
    description= "This interface will predict whether the sentiment of a customer is positive, negative, or neutral based on the review text.",
    allow_flagging="never")

# deploying the model
demo.launch(inline=False, share=True, debug=True)

# shutting down the deployed model
demo.close()
##  Metrics and dashboarding are the tools that businesses use to track their performance. Some of the benefits of using metrics and dashboarding:

## - **Improved decision-making**
## - **Increased efficiency**
## - **Increased visibility**

