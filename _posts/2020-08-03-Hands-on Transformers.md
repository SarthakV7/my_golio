---
layout: post
title:  Hands-on Transformers (Kaggle Google QUEST Q&A Labeling).
date:   2020-08-01 16:40:16+0530
description: Part 3/3 of Transformers vs Google QUEST Q&A Labeling (Kaggle top 5%).
published: True
---
#### Part 3/3 of Transformers vs Google QUEST Q&A Labeling (Kaggle top 5%).

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/2400/1*hOohtF4J9u1updZhHNh9ow.jpeg">
    </div>
</div>

*This is a 3 part series where we will be going through Transformers, BERT, and a hands-on Kaggle challenge — [Google QUEST Q&A Labeling](https://www.kaggle.com/c/google-quest-challenge/) to see Transformers in action (top 4.4% on the leaderboard).
In this part (3/3) we will be looking at a hands-on project from Google on Kaggle.
Since this is an NLP challenge, I’ve used transformers in this project. I have not covered transformers in much detail in this part but if you wish you could check out the part 1/3 of this series where I’ve discussed transformers in detail.*

![]()
## Bird's eye view of the blog:

To make the reading easy, I’ve divided the blog into different sub-topics-

* Problem statement and evaluation metrics.

* About the data.

* Exploratory Data Analysis (EDA).

* Modeling (includes data preprocessing).

* Post-modeling analysis.

![]()
## Problem statement and Evaluation metrics:

Computers are really good at answering questions with single, verifiable answers. But, humans are often still better at answering questions about opinions, recommendations, or personal experiences.

Humans are better at addressing subjective questions that require a deeper, multidimensional understanding of context. Questions can take many forms — some have multi-sentence elaborations, others may be simple curiosity or a fully developed problem. They can have multiple intents, or seek advice and opinions. Some may be helpful and others interesting. Some are simple right or wrong.

![]()
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/2664/1*MKNxCI-qlr_YAhcKowSHkQ.png">
    </div>
</div>

![]()
Unfortunately, it’s hard to build better subjective question-answering algorithms because of a lack of data and predictive models. That’s why the [CrowdSource](https://crowdsource.google.com/) team at Google Research, a group dedicated to advancing NLP and other types of ML science via crowdsourcing, has collected data on a number of these quality scoring aspects.

In this competition, we’re challenged to use this new dataset to build predictive algorithms for different subjective aspects of question-answering. The question-answer pairs were gathered from nearly 70 different websites, in a “common-sense” fashion. The raters received minimal guidance and training and relied largely on their subjective interpretation of the prompts. As such, each prompt was crafted in the most intuitive fashion so that raters could simply use their common-sense to complete the task.

Demonstrating these subjective labels can be predicted reliably can shine a new light on this research area. Results from this competition will inform the way future intelligent Q&A systems will get built, hopefully contributing to them becoming more human-like.

**Evaluation metric:** Submissions are evaluated on the mean column-wise [Spearman’s correlation coefficient](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient). The Spearman’s rank correlation is computed for each target column, and the mean of these values is calculated for the submission score.

![]()
## About the data:

The data for this competition includes questions and answers from various StackExchange properties. Our task is to predict the target values of 30 labels for each question-answer pair.
The list of 30 target labels is the same as the column names in the sample_submission.csv file. Target labels with the prefix question_ relate to the question_title and/or question_body features in the data. Target labels with the prefix answer_ relate to the answer feature.
Each row contains a single question and a single answer to that question, along with additional features. The training data contains rows with some duplicated questions (but with different answers). The test data does not contain any duplicated questions.
Target labels can have continuous values in the range [0,1]. Therefore, predictions must also be in that range.
The files provided are:

* train.csv — the training data (target labels are the last 30 columns)

* test.csv — the test set (you must predict 30 labels for each test set row)

* sample_submission.csv — a sample submission file in the correct format; column names are the 30 target labels

You can check out the dataset using [this](https://www.kaggle.com/c/google-quest-challenge/data) link.

![]()
## **Exploratory Data Analysis (EDA)**

***Check-out the notebook with in-depth EDA + Data Scraping ([Kaggle link](https://www.kaggle.com/sarthakvajpayee/top-4-4-in-depth-eda-feature-scraping?scriptVersionId=40263047)).***

The training data contains 6079 listings and each listing has 41 columns. Out of these 41 columns, the first 11 columns/features have to be used as the input and the last 30 columns/features are the target predictions.
Let’s take a look at the input and target labels:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/2456/1*q5AIFvq5vvoWICyjktZtAg.png">
    </div>
</div>

The output features are all of the float types between 0 and 1.

Let's explore the input labels one by one.

![]()
### qa_id

Question answer ID represents the id of a particular data point in the given dataset. Each data point has a unique qa_id. This feature is not to be used for training and will be used later while submitting the output to Kaggle.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/5036/1*AaPKUzHR6jKegZ7ku_INeg.png">
    </div>
</div>

![]()
### question_title

This is a string data type feature that holds the title of the question asked.
For the analysis of question_title, I’ll be plotting a histogram of the number of words in this feature.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/3448/1*Bio022v9rxBg8mhGhLNOjA.png">
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/2760/1*h2VPrcBPqpLrTVOAXO8jMw.png">
    </div>
</div>

From the analysis, it is evident that:
- Most of the question_title features have a word length of around 9.
- The minimum question length is 2.
- The maximum question length is 28.
- 50% of question_title have lengths between 6 and 11.
- 25% of question_title have lengths between 2 and 6.
- 25% of question_title have lengths between 11 and 28.

![]()
### question_body

This is again a string data type feature that holds the detailed text of the question asked.
For the analysis of question_body, I’ll be plotting a histogram of the number of words in this feature.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/3016/1*vck8a5DcxH6JSQIwvW-COA.png">
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/2788/1*J_p2iMsZ8wLfxFlC7acrfQ.png">
    </div>
</div>

![]()
From the analysis, it is evident that:
- Most of the question_body features have a word length of around 93.
- The minimum question length is 1.
- The maximum question length is 4666.
- 50% of question_title have lengths between 55 and 165.
- 25% of question_title have lengths between 1 and 55.
- 25% of question_title have lengths between 165 and 4666.

The distribution looks like a power-law distribution, it can be converted to a gaussian distribution using log and then used as an engineered feature.

![]()
### question_user_name

This is a string data type feature that denotes the name of the user who asked the question.
For the analysis of question_answer, I’ll be plotting a histogram of the number of words in this feature.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/3612/1*y2lMiIxsJae7PqDLCZLruw.png">
    </div>
</div>

![]()
I did not find this feature of much use therefore I won’t be using this for modeling.

![]()
### question_user_page

This is a string data type feature that holds the URL to the profile page of the user who asked the question.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/4196/1*AEIgkWtuReLYpi4vO1jczQ.png">
    </div>
</div>

![]()
On the profile page, I noticed 4 useful features that could be used and should possibly contribute to good predictions. The features are:
- Reputation: Denotes the reputation of the user.
- gold_score: The number of gold medals awarded.
- silver_score: The number of silver medals awarded.
- bronze_score: The number of bronze medals awarded.


 {% highlight python linenos %}
 # code for scraping the data. Since all of the urls are of stackoverflow, they have the same html hie rarchy.
from tqdm.notebook import tqdm
def get_user_rating(url):
  try:
    get = request.urlopen(url).read()
    src = BeautifulSoup(get, 'html.parser')
    reputation, gold = [], []
    silver, bronze = [], []
    template = src.find_all("div", class_ = 'grid--cell fl-shrink0 ws2 overflow-hidden')[0]
    reputation = int(''.join(template.find_all('div', class_='grid--cell fs-title fc-dark')[0].text.strip().split(',')))
    gold = int(''.join(template.find_all('div', class_='grid ai-center s-badge s-badge__gold')[0].text.strip().split(',')))
    silver = int(''.join(template.find_all('div', class_='grid ai-center s-badge s-badge__silver')[0].text.strip().split(',')))
    bronze = int(''.join(template.find_all('div', class_='grid ai-center s-badge s-badge__bronze')[0].text.strip().split(',')))
    output = [reputation, gold, silver, bronze]
  except:
    output = [np.nan]*4 # return np.nan if the code runs into some error like page not found

  return output

data = []
for url in tqdm(train['answer_user_page']):
  data.append(get_user_rating(url))
columns = ['reputation', 'gold', 'silver', 'bronze']
scraped = pd.DataFrame(lens, columns=columns)
scraped.to_csv(f'scraped_score.csv', index=False)
 {% endhighlight %}


![]()
### answer

This is again a string data type feature that holds the detailed text of the answer to the question.
For the analysis of *answer*, I’ll be plotting a histogram of the number of words in this feature.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/3312/1*hoFuOPl3B9cNz1TSeHLzrA.png">
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/2728/1*SPMSqpz78qsCvfOmyD5icg.png">
    </div>
</div>

![]()
From the analysis, it is evident that:
- Most of the question_body features have a word length of around 143.
- The minimum question length is 2.
- The maximum question length is 8158.
- 50% of question_title have lengths between 48 and 170.
- 25% of question_title have lengths between 2 and 48.
- 25% of question_title have lengths between 170 and 8158.

This distribution also looks like a power-law distribution, it can also be converted to a gaussian distribution using log and then used as an engineered feature.

![]()
### answer_user_name

This is a string data type feature that denotes the name of the user who answered the question.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/3024/1*8am6v-oUIg_XKA1d6LpzBQ.png">
    </div>
</div>

![]()
I did not find this feature of much use therefore I won’t be using this for modeling.

![]()
### answer_user_page

This is a string data type feature similar to the feature “question_user_page” that holds the URL to the profile page of the user who asked the question.

I also used the URL in this feature to scrape the external data from the user’s profile page, similar to what I did for the feature ‘question_user_page’.

![]()
### url

This feature holds the URL of the question and answers page on StackExchange or StackOverflow. Below I’ve printed the first 10 *url* data-points from train.csv

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/3656/1*0_RcKZAy01R7OxZ7BaVsCQ.png">
    </div>
</div>

![]()
One thing to notice is that this feature lands us on the question-answer page, and that page may usually contain a lot more data like comments, upvotes, other answers, etc. which can be used for generating more features if the model does not perform well due to fewer data in train.csv
Let’s see the data is present and what additional data can be scraped from the question-answer page.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/3200/1*HikC8L5q8zm8GFasu4gM1Q.png">
    </div>
</div>

![]()
In the snapshot attached above, *Post 1* and *Post 2* contain the answers, upvotes, and comments for the question asked in decreasing order of upvotes. The post with a green tick is the one containing the answer provided in the train.csv file.

Each question may have more than one answer. We can scrape these answers and use them as additional data.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/3096/1*f_qUPbojaSFLJ38d_LPzhA.png">
    </div>
</div>

![]()
The snapshot above defines the anatomy of a post. We can scrape useful features like *upvotes* and *comments* and use them as additional data.

Below is the code for scraping the data from the URL page.


 {% highlight python linenos %}

# Here is the code. Since all of the urls are of stackoverflow, they have the same html hierarchy.
  def get_answers(url):
    try:
      get = request.urlopen(url).read()
      src = BeautifulSoup(get, 'html.parser')
      upvotes, posts = [], []
      correct_ans, comments = [], []
      new_features = []
      post_layout = src.find_all("div", class_ = 'post-layout')
      l = len(post_layout)
      for p in post_layout[:l]:
        posts.append(p.find_all('div', class_='post-text')[0].text.strip())
        upvotes.append(int(p.find_all("div", class_ = 'js-vote-count grid--cell fc-black-500 fs-title g rid fd-column ai-center')[0].get('data-value')))
        correct_ans.append(len(p.find_all("div", class_ = 'js-accepted-answer-indicator grid--cell fc-g reen-500 ta-center py4')))
        comments.append('\n'.join([i.text.strip() for i in p.find_all('span', class_='comment-copy')]))

      idx = np.argmax(correct_ans)
      new_features.append(upvotes.pop(idx))
      new_features.append(comments.pop(idx))
      del posts[idx]
      if l < 3:
        k=l
      else:
        k=3
      for a,b in zip(posts[:k], comments[:k]):
        new_features.append(a)
        new_features.append(b)
      for a,b in zip(posts[:3-k], comments[:3-k]):
        new_features.append('')
        new_features.append('')

      return new_features

    except:
      return [np.nan]*8 # return np.nan if the code runs into some error like page not found

# collecting the data
data = []
for url in tqdm(train['url']):
  data.append(get_answers(url))

# Saving as dataframe
columns = ['upvotes', 'comments_0', 'answer_1', 'comment_1', 'answer_2', 'comment_2', 'answer_3', 'comment_3']
scraped = pd.DataFrame(lens, columns=columns)
scraped.to_csv(f'scraped_posts.csv', index=False)
 {% endhighlight %}

There are 8 new features that I’ve scraped-
- upvotes: The number of upvotes on the provided answer.
- comments_0: Comments to the provided answer.
- answer_1: Most voted answer apart from the one provided.
- comment_1: Top comment to answer_1.
- answer_2: Second most voted answer.
- comment_2: Top comment to answer_2.
- answer_3: Third most voted answer.
- comment_3: Top comment to answer_3.

![]()
### category

This is a categorical feature that tells the categories of question and answers pairs. Below I’ve printed the first 10 *category* data-points from train.csv

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/2724/1*ZVyREoM-KJIonAMNurlWZQ.png">
    </div>
</div>

Below is the code for plotting a Pie chart of category.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/2976/1*sQeMNBXBG_9wjgL5RwMuKA.png">
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/2740/1*raqsI9DfgAi-X9DEteDG-g.png">
    </div>
</div>

![]()
The chart tells us that most of the points belong to the category *TECHNOLOGY *and least belong to *LIFE_ARTS *(709 out of 6079).

![]()
### host

This feature holds the host or domain of the question and answers page on StackExchange or StackOverflow. Below I’ve printed the first 10 *host* data-points from train.csv

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/3052/1*Z2qQIIdk3tS1TTPUjbjesQ.png">
    </div>
</div>

![]()
Below is the code for plotting a bar graph of unique hosts.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/2952/1*MEYLevz8Mb4jEDi45l3CHQ.png">
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/2720/1*AC4MzE30X4w-jOdZUKDWPw.png">
    </div>
</div>

![]()
It seems there are not many but just 63 different subdomains present in the training data. Most of the data points are from StackOverflow.com whereas least from meta.math.stackexchange.com

![]()
### Target values

Let’s analyze the target values that we need to predict. But first, for the sake of a better interpretation, please check out the full dataset on kaggle using [this link](https://www.kaggle.com/c/google-quest-challenge/data?select=train.csv).

Below is the code block displaying the statistical description of the target values. These are only the first 6 features out of all the 30 features.
The values of all the features are of type float and are between 0 and 1.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/4624/1*vIH-FKw4qnebXL4HnYJuSw.png">
    </div>
</div>

![]()
Notice the second code block which displays the unique values present in the dataset. There are just 25 unique values between 0 and 1. This could be useful later while fine-tuning the code.

Finally, let’s check the distribution of the target features and their correlation.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/2740/1*K33iaHeCjBKwQiAE6aW2Yw.png">
    </div>
</div>

![]()
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/2850/1*a6rTLOBC2RuXYBDZExY9Hw.png">
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/2836/1*olpEN_FeEg1ViqLw_jJV7w.png">
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/2190/1*rJGjCI31Tu4yM3bi6vfeAw.png">
    </div>
</div>

![]()
## Modeling

![]()
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/2850/1*QdslcbSWBOdPju1zpqycgQ.jpeg">
    </div>
</div>

![]()
Now that we know our data better through EDA, let’s begin with modeling. Below are the subtopics that we’ll go through in this section-

* **Overview of the architecture:** Quick rundown of the ensemble architecture and it’s different components.

* **Base learners:** Overview of the base learners used in the ensemble.

* **Preparing the data:** Data cleaning and preparation for modeling.

* **Ensembling:** Creating models for training, and predicting. Pipelining the data preparation, model training, and model prediction steps.

* **Getting the scores from Kaggle:** Submitting the predicted target values for test data on Kaggle and generating a leaderboard score to see how well the ensemble did.

I tried various deep neural network architectures with GRU, Conv1D, Dense layers, and with different features for the competition but, an ensemble of 8 transformers (as shown above) seems to work the best.
In this part, we will be focusing on the final architecture of the ensemble used and for the other baseline models that I experimented with, you can check out my github repo.
>  **Overview of the architecture:**

Remember our task was for a given ***question_title, question_body,*** and ***answer***, we had to predict 30 target labels.
Now out of these 30 target labels, the first 21 are related to the ***question_title*** and ***question_body*** and have no connection to the ***answer*** whereas the last 9 target labels are related to the ***answer*** only but out of these 9, some of them also take ***question_title*** and ***question_body*** into the picture.
Eg. features like *answer_relevance* and *answer_satisfaction* can only be rated by looking at both the question and answer.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/2868/1*JsrdmWw2Jwa9uNN8c5LdNw.png">
    </div>
</div>

![]()
With some experimentation, I found that the base-learner (BERT_base) performs exceptionally well in predicting the first 21 target features (related to questions only) but does not perform that well in predicting the last 9 target features. Taking note of this, I constructed 3 dedicated base-learners and 2 different datasets to train them.

 1. The first base-learner was dedicated to predicting the question-related features (first 21) only. The dataset used for training this model consisted of features ***question_title*** and ***question_body*** only.

 2. The second base-learner was dedicated to predicting the answer-related features (last 9) only. The dataset used for training this model consisted of features ***question_title***, ***question_body,*** and ***answer***.

 3. The third base-learner was dedicated to predicting all the 30 features. The dataset used for training this model again consisted of features ***question_title***, ***question_body,*** and ***answer***.

To make the architecture even more robust, I used 3 different types of base learners — **BERT, RoBERTa, and XLNet.**
We will be going through these different transformer models later in this blog.

In the ensemble diagram above, we can see —

* The 2 datasets consisting of **[question_title + question_body]** and **[question_title + question_body + answer]** being used separately to train different base learners.

* Then we can see the 3 different base learners **(BERT, RoBERTa, and XLNet)** dedicated to predicting the **question-related features only** (first 21) colored in blue, using the dataset **[question_title + question_body]**

* Next, we can see the 3 different base learners **(BERT, RoBERTa, and XLNet)** dedicated to predicting the **answer-related features only** (last 9) colored in green, using the dataset **[question_title + question_body + answer].**

* Finally, we can see the 2 different base learners **(BERT, and RoBERTa)** dedicated to predicting **all the 30 features** colored in red, using the dataset **[question_title + question_body + answer].**

In the next step, the predicted data from models dedicated to predicting the **question-related features only** (denoted as ***bert_pred_q, roberta_pred_q, xlnet_pred_q***) and the predicted data from models dedicated to predicting the **answer-related features only** (denoted as ***bert_pred_a, roberta_pred_a, xlnet_pred_a***) is collected and concatenated column-wise which leads to a predicted data with all the 30 features. These concatenated features are denoted as ***xlnet_concat, roberta_concat,*** and ***bert_concat.***

Similarly, the predicted data from models dedicated to predicting **all the 30 features** (denoted as ***bert_qa, roberta_qa***) is collected. Notice that I’ve not used the XLNet model here for predicting all the 30 features because the scores were not up to the mark.

Finally, after collecting all the different predicted data — **[xlnet_concat, roberta_concat, bert_concat, bert_qa, and roberta_qa],** the final value is calculated by taking the average of all the different predicted values.
>  **Base learners**

Now we will take a look at the 3 different transformer models that were used as base learners.

**1. bert_base_uncased:**

[Bert](https://arxiv.org/abs/1810.04805) was proposed by Google AI in late 2018 and since then it has become state-of-the-art for a wide spectrum of NLP tasks.
It uses an architecture derived from transformers pre-trained over a lot of unlabeled text data to learn a language representation that can be used to fine-tune for specific machine learning tasks. BERT outperformed the NLP state-of-the-art on several challenging tasks. This performance of BERT can be ascribed to the transformer’s encoder architecture, unconventional training methodology like the Masked Language Model (MLM), and Next Sentence Prediction (NSP) and the humungous amount of text data (all of Wikipedia and book corpus) that it is trained on. BERT comes in different sizes but for this challenge, I’ve used *bert_base_uncased.*

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/2000/0*nbFb82C1avPQB6aH.png">
    </div>
</div>

The architecture of *bert_base_uncased* consists of 12 encoder cells with 8 attention heads in each encoder cell.
It takes an input of size 512 and returns 2 values by default, the output corresponding to the first input token [CLS] which has a dimension of 786 and another output corresponding to all the 512 input tokens which have a dimension of (512, 768) aka pooled_output.
But apart from these, we can also access the hidden states returned by each of the 12 encoder cells by passing ***output_hidden_states=True*** as one of the parameters.
BERT accepts several sets of input, for this challenge, the input I’ll be using will be of 3 types:

* ***input_ids***: The token embeddings are numerical representations of words in the input sentence. There is also something called sub-word tokenization that BERT uses to first breakdown larger or complex words into simple words and then convert them into tokens. For example, in the above diagram look how the word ‘playing’ was broken into ‘play’ and ‘##ing’ before generating the token embeddings. This tweak in tokenization works wonders as it utilized the sub-word context of a complex word instead of just treating it like a new word.

* ***attention_mask***: The segment embeddings are used to help BERT distinguish between the different sentences in a single input. The elements of this embedding vector are all the same for the words from the same sentence and the value changes if the sentence is different.
Let’s consider an example: Suppose we want to pass the two sentences *“I have a pen”* and *“The pen is red”* to BERT. The tokenizer will first tokenize these sentences as:
**[‘[CLS]’, ‘I’, ‘have’, ‘a’, ‘pen’, ‘[SEP]’, ‘the’, ‘pen’, ‘is’, ‘red’, ‘[SEP]’]**
And the segment embeddings for these will look like:
**[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1].**
Notice how all the elements corresponding to the word in the first sentence have the same element **0** whereas all the elements corresponding to the word in the second sentence have the same element **1**.

* ***token_type_ids:*** The mask tokens that help BERT to understand what all input words are relevant and what all are just there for padding.
Since BERT takes a 512-dimensional input, and suppose we have an input of 10 words only. To make the tokenized words compatible with the input size, we will add padding of size 512–10=502 at the end. Along with the padding, we will generate a mask token of size 512 in which the index corresponding to the relevant words will have **1**s and the index corresponding to padding will have **0**s.

**2. XLNet_base_cased:**

[XLNet](https://arxiv.org/abs/1906.08237) was proposed by Google AI Brain team and researchers at CMU in mid-2019. Its architecture is larger than BERT and uses an improved methodology for training. It is trained on larger data and shows better performance than BERT in many language tasks. The conceptual difference between BERT and XLNet is that while training **BERT**, the words are predicted in an order such that the previous predicted word contributes to the prediction of the next word whereas, **XLNet** learns to predict the words in an arbitrary order but in an autoregressive manner (not necessarily left-to-right).

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/2000/0*V2CD2JYdmFiaPTIa">
    </div>
</div>

![]()
This helps the model to learn bidirectional relationships and therefore better handles dependencies and relations between words.
In addition to the training methodology, XLNet uses Transformer XL based architecture and 2 main key ideas: *relative positional embeddings* and the *recurrence mechanism* which showed good performance even in the absence of permutation-based training.
XLNet was trained with over 130 GB of textual data and 512 TPU chips running for 2.5 days, both of which are much larger than BERT.

For XLNet, I’ll be using only **input_ids** and **attention_mask** as input.

![]()
**3. RoBERTa_base:**

RoBERTa was proposed by Facebook in mid-2019. It is a robustly optimized method for pretraining natural language processing (NLP) systems that improve on BERT’s self-supervised method.
RoBERTa builds on BERT’s language masking strategy, wherein the system learns to predict intentionally hidden sections of text within otherwise unannotated language examples. RoBERTa modifies key hyperparameters in BERT, including removing BERT’s Next Sentence Prediction (NSP) objective, and training with much larger mini-batches and learning rates. This allows RoBERTa to improve on the masked language modeling objective compared with BERT and leads to better downstream task performance. RoBERTa was also trained on more data than BERT and for a longer amount of time. The dataset used was from existing unannotated NLP data sets as well as CC-News, a novel set drawn from public news articles.

For RoBERTa_base, I’ll be using only **input_ids** and **attention_mask** as input.

***Finally here is the comparison of BERT, XLNet, and RoBERTa:***

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/2436/1*lNiXASsDWI86aMKZihMC1Q.png">
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://miro.medium.com/max/1050/0*EfEZgjlXlGl0sXjG.png">
    </div>
</div>

![]()
>  **Preparing the data**

Now that we have gained some idea about the architecture let’s see how to prepare the data for the base learners.

* As a preprocessing step, I have just treated the HTML syntax present in the features. I used html.unescape() to extract the text from HTML DOM elements.
In the code snippet below, the function **get_data()** reads the train and test data and applies the preprocessing to the features ***question_title, question_body,*** and ***answer.***


 {% highlight python linenos %}
def get_data():
    print('getting test and train data...')
    # reading the data into dataframe using pandas
    path = '../input/google-quest-challenge/'
    train = pd.read_csv(path+'train.csv')
    test = pd.read_csv(path+'test.csv')
    submission = pd.read_csv(path+'sample_submission.csv')

    # Selecting data for training and testing
    y = train[train.columns[11:]] # storing the target values in y
    X = train[['question_title', 'question_body', 'answer']]
    X_test = test[['question_title', 'question_body', 'answer']]

    # Cleaning the data
    X.question_body = X.question_body.apply(html.unescape)
    X.question_title = X.question_title.apply(html.unescape)
    X.answer = X.answer.apply(html.unescape)

    X_test.question_body = X_test.question_body.apply(html.unescape)
    X_test.question_title = X_test.question_title.apply(html.unescape)
    X_test.answer = X_test.answer.apply(html.unescape)

    return X, X_test, y, train, test
 {% endhighlight %}

* The next step was to create ***input_ids, attention_masks,*** and ***token_type_ids*** from the input sentence.
In the code snippet below, the function **get_tokenizer()** collects pre-trained tokenizer for the different base_learners.
The second function **fix_length()** goes through the generated question tokens and answer tokens and makes sure that the maximum number of tokens is 512. The steps for fixing the number of tokens are as follows:
- If the input sentence has the number of tokens > 512, the sentence is trimmed down to 512.
- To trim the number of tokens, 256 tokens from the beginning and 256 tokens from the end are kept and the remaining tokens are dropped.
- For example, suppose an answer has 700 tokens, to trim this down to 512, 256 tokens from the beginning are taken and 256 tokens from the end are taken and concatenated to make 512 tokens. The remaining [700-(256+256) = 288] tokens that are in the middle of the answer are dropped.
- The logic makes sense because in a large text, the beginning part usually describes what the text is all about and the end part describes the conclusion of the text.


 {% highlight python linenos %}
 def get_tokenizer(model_name):
     print(f'getting tokenizer for {model_name}...')
     if model_name == 'xlnet-base-cased':
         tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
     elif model_name == 'roberta-base':
         tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
     elif model_name == 'bert-base-uncased':
         tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

     return tokenizer

 def fix_length(tokens, max_sequence_length=512, q_max_len=254, a_max_len=254, model_type='questions'):
     if model_type == 'questions':
         length = len(tokens)
         if length > max_sequence_length:
             tokens = tokens[:max_sequence_length-1]
         return tokens

     else:
         question_tokens, answer_tokens = tokens
         q_len = len(question_tokens)
         a_len = len(answer_tokens)
         if q_len + a_len + 3 > max_sequence_length:
             if a_max_len <= a_len and q_max_len <= q_len:
                 q_new_len_head = q_max_len//2
                 question_tokens = question_tokens[:q_new_len_head] + question_tokens[-q_new_len_head:]
                 a_new_len_head = a_max_len//2
                 answer_tokens = answer_tokens[:a_new_len_head] + answer_tokens[-a_new_len_head:]
             elif q_len <= a_len and q_len < q_max_len:
                 a_max_len = a_max_len + (q_max_len - q_len - 1)
                 a_new_len_head = a_max_len//2
                 answer_tokens = answer_tokens[:a_new_len_head] + answer_tokens[-a_new_len_head:]
             elif a_len < q_len:
                 q_max_len = q_max_len + (a_max_len - a_len - 1)
                 q_new_len_head = q_max_len//2
                 question_tokens = question_tokens[:q_new_len_head] + question_tokens[-q_new_len_head:]

     return question_tokens, answer_tokens
 {% endhighlight %}

Next is the code block for generating the **input_ids, attention_masks,** and **token_type_ids.** I’ve used a condition that checks if the function needs to return the generated data for base learners relying on the dataset **[question_title + question_body]** or the dataset **[question_title + question_body + answer].**


 {% highlight python linenos %}
 def transformer_inputs(title, question, answer, tokenizer, model_type='questions', MAX_SEQUENCE_LENGTH = 512):

     if model_type == 'questions':
         question = f"{title} [SEP] {question}"
         question_tokens = tokenizer.tokenize(question)
         question_tokens = fix_length(question_tokens, model_type=model_type)
         ids_q = tokenizer.convert_tokens_to_ids(["[CLS]"] + question_tokens)
         padded_ids = (ids_q + [tokenizer.pad_token_id] * (MAX_SEQUENCE_LENGTH - len(ids_q)))[:MAX_SEQUENCE_LENGTH]
         token_type_ids = ([0] * MAX_SEQUENCE_LENGTH)[:MAX_SEQUENCE_LENGTH]
         attention_mask = ([1] * len(ids_q) + [0] * (MAX_SEQUENCE_LENGTH - len(ids_q)))[:MAX_SEQUENCE_LENGTH]

         return padded_ids, token_type_ids, attention_mask

     else:
         question = f"{title} [SEP] {question}"
         question_tokens = tokenizer.tokenize(question)
         answer_tokens = tokenizer.tokenize(answer)
         question_tokens, answer_tokens = fix_length(tokens=(question_tokens, answer_tokens), model_type=model_type)
         ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + question_tokens + ["[SEP]"] + answer_tokens + ["[SEP]"])
         padded_ids = ids + [tokenizer.pad_token_id] * (MAX_SEQUENCE_LENGTH - len(ids))
         token_type_ids = [0] * (1 + len(question_tokens) + 1) + [1] * (len(answer_tokens) + 1) + [0] * (MAX_SEQUENCE_LENGTH - len(ids))
         attention_mask = [1] * len(ids) + [0] * (MAX_SEQUENCE_LENGTH - len(ids))

         return padded_ids, token_type_ids, attention_mask
 {% endhighlight %}

Finally, here is the function that makes use of the function initialized above and generates **input_ids, attention_masks,** and **token_type_ids** for each of the instances in the provided data.


 {% highlight python linenos %}
# function for creating the input_ids, masks and segments for the transformer input
def input_data(df, tokenizer, model_type='questions'):
    print(f'generating {model_type} input for transformer...')
    input_ids, input_token_type_ids, input_attention_masks = [], [], []
    for title, body, answer in tqdm(zip(df["question_title"].values, df["question_body"].values, df["answer"].values)):
        ids, type_ids, mask = transformer_inputs(title, body, answer, tokenizer, model_type=model_type)
        input_ids.append(ids)
        input_token_type_ids.append(type_ids)
        input_attention_masks.append(mask)

    return (
        np.asarray(input_ids, dtype=np.int32),
        np.asarray(input_attention_masks, dtype=np.int32),
        np.asarray(input_token_type_ids, dtype=np.int32))
 {% endhighlight %}

To make the model training easy, I also created a class that generates train and cross-validation data based on the fold while using KFlod CV with the help of the functions specified above.


 {% highlight python linenos %}
 class data_generator:
   def __init__(self, X, X_test, tokenizer, type_):
       # test data
       tokens, masks, segments = input_data(X_test, tokenizer, type_)
       self.test_data = {'input_tokens': tokens,
                         'input_mask': masks,
                         'input_segment': segments}

       # Train data
       self.tokens, self.masks, self.segments = input_data(X, tokenizer, type_)
   def generate_data(tr, cv, name='xlnet-base-cased', model_type='questions'):
       if name!='xlnet-base-cased':
           train_data = {'input_tokens': self.tokens[tr],
                         'input_mask': self.masks[tr],
                         'input_segment': self.segments[tr]}

           cv_data = {'input_tokens': self.tokens[cv],
                     'input_mask': self.masks[cv],
                     'input_segment': self.segments[cv]}
       else:
           train_data = {'input_tokens': self.tokens[tr],
                         'input_mask': self.masks[tr]}

           cv_data = {'input_tokens': self.tokens[cv],
                     'input_mask': self.masks[cv]}

       if model_type=='questions':
           y_tr = y.values[tr, 21:]
           y_cv = y.values[cv, 21:]

       elif model_type=='answers':
           y_tr = y.values[tr, 21:]
           y_cv = y.values[cv, 21:]

       else:
           y_tr = y.values[tr]
           y_cv = y.values[cv]  

       return train_data, cv_data, y_tr, y_cv
 {% endhighlight %}
>  **Ensembling**

After data preprocessing, let's create the model architecture starting with base learners.

The code below takes the model name as input, collects the pre-trained model, and its configuration information according to the input name and creates the base learner model. Notice that **output_hidden_states=True** is passed after adding the config data.


 {% highlight python linenos %}
 def get_model(name):
     if name == 'xlnet-base-cased':
         config = XLNetConfig.from_pretrained('xlnet-base-cased', output_hidden_states=True)
         model = TFXLNetModel.from_pretrained('xlnet-base-cased', config=config)
     elif name == 'roberta-base':
         config = RobertaConfig.from_pretrained('roberta-base', output_hidden_states=True)
         model = TFRobertaModel.from_pretrained('roberta-base', config=config)
     elif name == 'bert-base-uncased':
         config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
         model = TFBertModel.from_pretrained('bert-base-uncased', config=config)
     return model
 {% endhighlight %}

The next code block is to create the ensemble architecture. The function accepts 2 parameters name that expects the name of the model that we want to train and model_type that expects the type of model we want to train. The model type can be **bert-base-uncased, roberta-base** or **xlnet-base-cased** whereas the model type can be **questions, answers,** or **question_answers.**
The function **create_model()** takes the model_name and model_type and generates a model that can be trained on the specified data accordingly.

 {% highlight python linenos %}
 def create_model(name='xlnet-base-cased', model_type='questions'):
     print(f'creating model {name}...')
     # Creating the model
     K.clear_session()
     max_seq_length = 512

     input_tokens = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_tokens")
     input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
     input_segment = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_segment")

     model = get_model(name)
     if (name == 'xlnet-base-cased'):
       sequence_output, hidden_states = model([input_tokens, input_mask])
     elif (name=='roberta-base' and model_type!='questions'):
       sequence_output, pooler_output, hidden_states = model([input_tokens, input_mask])
     else:
       sequence_output, pooler_output, hidden_states = model([input_tokens, input_mask, input_segment])

     # Last 4 hidden layers of transformer
     h12 = tf.reshape(hidden_states[-1][:,0],(-1,1,768))
     h11 = tf.reshape(hidden_states[-2][:,0],(-1,1,768))
     h10 = tf.reshape(hidden_states[-3][:,0],(-1,1,768))
     h09 = tf.reshape(hidden_states[-4][:,0],(-1,1,768))
     concat_hidden = tf.keras.layers.Concatenate(axis=2)([h12, h11, h10, h09])

     x = GlobalAveragePooling1D()(concat_hidden)

     x = Dropout(0.2)(x)

     if model_type == 'answers':
       output = Dense(9, activation='sigmoid')(x)
     elif model_type == 'questions':
       output = Dense(21, activation='sigmoid')(x)
     else:
       output = Dense(30, activation='sigmoid')(x)

     if (name == 'xlnet-base-cased') or (name=='roberta-base' and model_type!='questions'):
       model = Model(inputs=[input_tokens, input_mask], outputs=output)
     else:
       model = Model(inputs=[input_tokens, input_mask, input_segment], outputs=output)

     return model
 {% endhighlight %}

Now let's create a function for calculating the evaluation metric [Spearman’s correlation coefficient](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient).


 {% highlight python linenos %}
 # Function to calculate the Spearman's rank correlation coefficient 'rhos' of actual and predicted data.
 def compute_spearmanr_ignore_nan(trues, preds):
     rhos = []
     for tcol, pcol in zip(np.transpose(trues), np.transpose(preds)):
         rhos.append(spearmanr(tcol, pcol).correlation)
     return np.nanmean(rhos)

 # Making the 'rhos' metric to tensorflow graph compatible.
 def rhos(y, y_pred):
     return tf.py_function(compute_spearmanr_ignore_nan, (y, y_pred), tf.double)
 {% endhighlight %}

Now we need a function that can collect the base learner model, data according to the base learner model, and train the model.
I’ve used K-Fold cross-validation with 5 folds for training.


 {% highlight python linenos %}
 def fit_model(model, model_name, model_type, data_gen, file_path, train, use_saved_weights=True):
   path = '../input/google-qna-predicted-data/'
   if use_saved_weights:
     print(f'getting saved weights for {model_name}...')
     model.load_weights(path+file_path)

   else:
     print(f'fitting data on {model_name}...')
     optimizer = tf.keras.optimizers.Adam(learning_rate=0.00002)
     model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[rhos])
     kf = KFold(n_splits=5, random_state=42)
     for tr, cv in kf.split(np.arange(train.shape[0])):
       tr_data, cv_data, y_tr, y_cv = data_gen.generate_data(tr, cv, model_name, model_type)
       model.fit(tr_data, y_tr, epochs=1, batch_size=4, validation_data=(cv_data, y_cv))
       model.save_weights(file_path)

   return model
 {% endhighlight %}

Now once we have trained the models and generated the predicted values, we need a function for calculating the weighted average. Here’s the code for that.
*The weight’s in the weighted average are all 1s.


 {% highlight python linenos %}
 def get_weighted_avg(model_predictions):
   xlnet_q, xlnet_a, roberta_q, roberta_a, roberta_qa, bert_q, bert_a, bert_qa = model_predictions
   xlnet_concat = np.concatenate((xlnet_q, xlnet_a), axis=1)
   bert_concat = np.concatenate((bert_q, bert_a), axis=1)
   roberta_concat = np.concatenate((roberta_q, roberta_a), axis=1)
   predict = (roberta_qa + bert_qa + xlnet_concat + bert_concat + roberta_concat)/5

   return predict
 {% endhighlight %}

Before bringing everything together, there is one more function that I used for processing the final predicted values. Remember in the EDA section there was an analysis of the target values where we noticed that the target values were only 25 unique floats between 0 and 1. To make use of that information, I calculated 61 (a hyperparameter) uniformly distributed percentile values and mapped them to the 25 unique values. This created 61 bins uniformly spaced between the upper and lower range of the target values. Now to process the predicted data, I used those bins to collect the predicted values and put them in the right place/order. This trick helped in improving the score in the final submission to the leaderboard to some extent.


 {% highlight python linenos %}
 # https://www.kaggle.com/markpeng/ensemble-5models-v4-v7-magic/notebook?select=submission.csv#Do-Inference
 def get_exp_labels(train):
     X = train.iloc[:, 11:]
     unique_labels = np.unique(X.values)
     denominator = 60
     q = np.arange(0, 101, 100 / denominator)
     exp_labels = np.percentile(unique_labels, q) # Generating the 60 bins.
     return exp_labels

 def optimize_ranks(preds, unique_labels):
     print(f'optimizing the predicted values...')
     new_preds = np.zeros(preds.shape)
     for i in range(preds.shape[1]):
         interpolate_bins = np.digitize(preds[:, i], bins=unique_labels, right=False)
         if len(np.unique(interpolate_bins)) == 1:
             new_preds[:, i] = preds[:, i]
         else:
             new_preds[:, i] = unique_labels[interpolate_bins]

     return new_preds
 {% endhighlight %}

Finally, to bring the data-preprocessing, model training, and post-processing together, I created the **get_predictions()** function that-
- Collects the data.
- Creates the 8 base_learners.
- Prepares the data for the base_learners.
- Trains the base learners and collects the predicted values from them.
- Calculates the weighted average of the predicted values.
- Processes the weighted average prediction.
- Converts the final predicted values into a dataframe format requested by Kaggle for submission and return it.


 {% highlight python linenos %}
 def get_predictions(predictions_present=True, model_saved_weights_present=True):
   msw = model_saved_weights_present
   X, X_test, y, train, test = get_data()
   path = '../input/google-qna-predicted-data/'
   model_names = ['xlnet-base-cased', 'roberta-base', 'bert-base-uncased']
   model_types = ['questions', 'answers', 'questions_answers']
   saved_weights_names = ['xlnet_q.h5', 'xlnet_a.h5', 'roberta_q.h5', 'roberta_a.h5',
                         'roberta_qa.h5', 'bert_q.h5', 'bert_a.h5', 'bert_qa.h5']

   saved_model_predictions = [path+'xlnet_q.csv', path+'xlnet_a.csv', path+'roberta_q.csv', path+'roberta_a.csv',
                               path+'roberta_qa.csv', path+'bert_q.csv', path+'bert_a.csv', path+'bert_qa.csv']
   model_predictions = []

   if predictions_present:
     model_predictions = [pd.read_csv(file_name).values for file_name in saved_model_predictions]

   else:
     i = 0
     for name_ in model_names:
       for type_ in model_types:
         if name_ == 'xlnet-base-cased' and type_ == 'questions_answers':
           continue
         print('-'*100)
         model = create_model(name_, type_)
         tokenizer = get_tokenizer(name_)
         data_gen = data_generator(X, X_test, tokenizer, type_)
         model = fit_model(model, name_, type_, data_gen, saved_weights_names[i], train, msw)
         print(f'getting target predictions from {name_}...')
         model_predictions.append(model.predict(data_gen.test_data))
         i+=1

   predicted_labels = get_weighted_avg(model_predictions)
   exp_labels = get_exp_labels(train)
   optimized_predicted_labels = optimize_ranks(predicted_labels, exp_labels)
   df = pd.concat([test['qa_id'], pd.DataFrame(optimized_predicted_labels, columns=train.columns[11:])], axis=1)
   print('done...!')

   return df
 {% endhighlight %}
>  **Getting the scores from Kaggle**

Once the code compiles and runs successfully, it generates an output file that can be submitted to Kaggle for **score** calculation. The ranking of the code on the leaderboard is generated using the **score.**
The ensemble model got a public score of **0.43658** which makes it in the top 4.4% on the leaderboard.

![]()
## Post modeling Analysis

***Check-out the notebook with complete post-modeling analysis ([Kaggle link](https://www.kaggle.com/sarthakvajpayee/top-4-4-post-modeling-analysis?scriptVersionId=40262842)).***

![]()
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/4320/1*7gTtCiVIl_oCN6Wpq4R-9g.png">
    </div>
</div>

![]()
Its time for some post-modeling analysis!

In this section, we will go through an analysis of train data to figure out what parts of the data is the model doing well on and what parts of the data it’s not.
The main idea behind this step is to know the capability of the trained model and it works like a charm if applied properly for fine-tuning the model and data.
But we won’t get into the fine-tuning part in this section, we will just be performing some basic EDA on the train data using the predicted target values for the train data.
I’ll be covering the data feature by feature. Here are the top features we’ll be performing analysis on-

* question_title, question_body, and answer.

* Word lengths of question_title, question_body, and answer.

* Host

* Category

First, we will have to divide the data into a spectrum of good data and bad data. Good data will be the data points on which the model achieves a good score and bad data will be the data points on which the model achieves a bad score.
Now for scoring, we will be comparing the actual target values of the train data with the model’s predicted target values on train data. I used **mean squared error (MSE)** as a metric for scoring since it focuses on how close the actual and target values are. Remember the more the MSE-score is, the bad the data point will be.
Calculating the MSE-score is pretty simple. Here’s the code:

    # Generating the MSE-score for each data point in train data.
    from sklearn.metrics import mean_squared_error

    train_score = [mean_squared_error(i,j) for i,j in zip(y_pred, y_true)]

    # sorting the losses from minimum to maximum index wise.
    train_score_args = np.argsort(train_score)

![]()
>  **question_title, question_body, and answer**

Starting with the first set of features, which are all text type features, I’ll be plotting word clouds using them. The plan is to segment out these features from 5 data-points that have the least scores and from another 5 data-points that have the most scores.


 {% highlight python linenos %}
 # function for generating wordcloud
 from wordcloud import WordCloud, STOPWORDS
 import seaborn as sns
 sns.set()

 def generate_wordcloud(indexes, data, color='black'):
   comment_words = ''
   stopwords = set(STOPWORDS)

   title_words = data['question_title'].iloc[indexes]
   body_words = data['question_body'].iloc[indexes]
   answer_words = data['answer'].iloc[indexes]

   title_cloud = WordCloud(width = 400, height = 200, background_color = color,
                         stopwords = stopwords, min_font_size = 10).generate(title_words)

   body_cloud = WordCloud(width = 400, height = 200, background_color = color,
                         stopwords = stopwords, min_font_size = 10).generate(body_words)

   answer_cloud = WordCloud(width = 400, height = 200, background_color = color,
                         stopwords = stopwords, min_font_size = 10).generate(answer_words)

   return title_cloud, body_cloud, answer_cloud
 {% endhighlight %}

Let’s run the code and check what the results look like.


 {% highlight python linenos %}
 # I've picked the top 5 datapoints from train data with lowest loss and plotted the wordcloud of their question_title, question_body and answer.
 print('Top 5 data points from train data that give the "lowest" loss.')
 for i, idx in enumerate(train_score_args[:5]):
   title, body, answer = generate_wordcloud(idx, X_train)
   plt.figure(figsize=(20,12))
   plt.subplot(131)
   plt.imshow(title)
   if i==0: plt.title('question_title')
   plt.ylabel(f'loss: {train_score[idx]}')
   plt.subplot(132)
   plt.imshow(body)
   if i==0: plt.title('question_body')
   plt.subplot(133)
   plt.imshow(answer)
   if i==0: plt.title('answer')
   plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
   plt.show()
 {% endhighlight %}

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/2572/1*5lOfSHKyTTV3YhsU4PxaDQ.png">
    </div>
</div>


 {% highlight python linenos %}
 # I've picked the top 5 datapoints from train data with 'highest' loss and plotted the wordcloud of their question_title, question_body and answer.
 print('Top 5 data points from Train data that give the "highest" loss.')
 for i, idx in enumerate(train_score_args[-5:]):
   title, body, answer = generate_wordcloud(idx, X_train, color='white')
   plt.figure(figsize=(20,12))
   plt.subplot(131)
   plt.imshow(title)
   if i==0: plt.title('question_title')
   plt.ylabel(f'loss: {train_score[idx]}')
   plt.subplot(132)
   plt.imshow(body)
   if i==0: plt.title('question_body')
   plt.subplot(133)
   plt.imshow(answer)
   if i==0: plt.title('answer')
   plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
   plt.show()
 {% endhighlight %}

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/2676/1*pUtWG5KWy_c34dLGH2qJgA.png">
    </div>
</div>
>  **Word lengths of question_title, question_body, and answer**

The next analysis is on the word lengths of question_title, question_body, and answer. For that, I’ll be picking 30 data-points that have the lowest MSE-scores and 30 data-points that have the highest MSE-scores for each of the 3 features question_title, question_body, and answer. Next, I’ll be calculating the word lengths of these 30 data-points for all the 3 features and plot them to see the trend.


 {% highlight python linenos %}
 # I've picked the top 30 datapoints from train and cv data with 'lowest' loss and plotted the word counts of their question_title, question_body and answer.
 print("word counts of the question_title, question_body and answer of top 30 train and cv data with 'lowest' loss.")
 i = 30
 title_train_len = [len(l.split(' ')) for l in X_train.iloc[train_score_args[:i]]['question_title'].values]
 body_train_len = [len(l.split(' ')) for l in X_train.iloc[train_score_args[:i]]['question_body'].values]
 answer_train_len = [len(l.split(' ')) for l in X_train.iloc[train_score_args[:i]]['answer'].values]

 plt.figure(figsize=(20,4))
 plt.subplot(131)
 plt.plot(title_train_len)
 plt.title('question_title (train data)')
 plt.ylabel('number of words')
 plt.xlabel('datapoint (loss: high --> low)')
 plt.subplot(132)
 plt.plot(body_train_len)
 plt.title('question_body (train data)')
 plt.xlabel('datapoint (loss: low --> high)')
 plt.subplot(133)
 plt.plot(answer_train_len)
 plt.title('answer (train data)')
 plt.xlabel('datapoint (loss: high --> low)')
 # plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
 plt.show()
 {% endhighlight %}

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/3916/1*zfhNtTvxqVYPJYX1Emq7rA.png">
    </div>
</div>


 {% highlight python linenos %}
 # I've picked the top 30 datapoints from train data with 'highest' loss and plotted the word counts of their question_title, question_body and answer.
 print("word counts of the question_title, question_body and answer of top 30 train and cv data with 'highest' loss.")
 i = -30
 title_train_len = [len(l.split(' ')) for l in X_train.iloc[train_score_args[i:]]['question_title'].values]
 body_train_len = [len(l.split(' ')) for l in X_train.iloc[train_score_args[i:]]['question_body'].values]
 answer_train_len = [len(l.split(' ')) for l in X_train.iloc[train_score_args[i:]]['answer'].values]

 plt.figure(figsize=(20,4))
 plt.subplot(131)
 plt.plot(title_train_len)
 plt.title('question_title (train data)')
 plt.ylabel('number of words')
 plt.xlabel('datapoint (loss: high --> low)')
 plt.subplot(132)
 plt.plot(body_train_len)
 plt.title('question_body (train data)')
 plt.xlabel('datapoint (loss: high --> low)')
 plt.subplot(133)
 plt.plot(answer_train_len)
 plt.title('answer (train data)')
 plt.xlabel('datapoint (loss: high --> low)')
 # plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
 plt.show()
 {% endhighlight %}


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/3824/1*uF-kg7A_2geR4UBVJty0Nw.png">
    </div>
</div>

*If we look at the number of words in question_title, question_body, and answer we can observe that the data that generates a high loss has a high number of words which means that the questions and answers are kind of thorough. So, the model does a good job when the questions and answers are concise.*
>  **host**

The next analysis is on the feature host. For this feature, I’ll be picking 100 data-points that have the lowest MSE-scores and 100 data-points that have the highest MSE-scores and select the values in the feature host. Then I’ll be plotting a histogram of this categorical feature to see the distributions.


 {% highlight python linenos %}
 # I've picked the top 100 datapoints from train data with 'highest' loss and collected the values of domain names.
 top_url = X_train['host'].iloc[train_score_args[:100]].value_counts()
 bottom_url = X_train['host'].iloc[train_score_args[-100:]].value_counts()

 # Top 10 frequently occuring domain names that lead to minimum loss
 top_url[1:10].plot.bar(figsize=(12,8))
 plt.title('top 10 url domain that produce the minimum loss')
 plt.ylabel('frequency')
 plt.show()
 {% endhighlight %}


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/2556/1*JdLsXYflDINgWEOMhWryUw.png">
    </div>
</div>


 {% highlight python linenos %}
 # Top 10 frequently occuring domain names that lead to maximum loss
 bottom_url[1:10].plot.bar(figsize=(12,8))
 plt.title('top 10 url domain that produce the maximum loss')
 plt.ylabel('frequency')
 plt.show()
 {% endhighlight %}


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/2512/1*_-X1hQl1V8nchYotkUORpg.png">
    </div>
</div>

*We can see that there are a lot of data points from the domain English, biology, sci-fi, physics that contribute to a lesser loss value whereas there are a lot of data points from drupal, programmers, tex that contribute to a higher loss.*

Let’s also take a look at word-clouds of the unique host values that contribute to a low score and a high score. This analysis is again done using the top and bottom 100 data-points.


 {% highlight python linenos %}
 # finding the unique domain names that contribute to low and high losses
 best_url = ' '.join(list(set(top_url.keys()) - set(bottom_url.keys()))) # set of urls that contribute solely to low loss
 worst_url = ' '.join(list(set(bottom_url.keys()) - set(top_url.keys()))) # set of urls that contribute solely to high loss

 best_url_cloud = WordCloud(width = 400, height = 200, background_color ='orange',
                            stopwords = STOPWORDS, min_font_size = 10).generate(best_url)

 worst_url_cloud = WordCloud(width = 400, height = 200, background_color ='cyan',
                             stopwords = STOPWORDS, min_font_size = 10).generate(worst_url)

 plt.figure(figsize=(20,12))
 plt.subplot(121)
 plt.imshow(best_url_cloud)
 plt.title('url domain with well predicted labels (low loss)')
 plt.subplot(122)
 plt.imshow(worst_url_cloud)
 plt.title('url domain with bad predicted labels (high loss)')
 plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
 plt.show()
 {% endhighlight %}

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/3660/1*eC0RkpXcA9jhzioVKIXTfQ.png">
    </div>
</div>
>  **Category**

The final analysis is on the feature category. For this feature, I’ll be picking 100 data-points that have the lowest MSE-scores and 100 data-points that have the highest MSE-scores and select the values in the feature category. Then I’ll be plotting a pie-chart of this categorical feature to see the proportions.


 {% highlight python linenos %}
 # for train data
 plt.figure(figsize=(20,20))
 plt.subplot(121)
 X_train['category'].iloc[train_score_args[:100]].value_counts().plot.pie(autopct='%1.1f%%', explode=(0,0.02,0.04,0.06,0.08), shadow=True)
 plt.ylabel('')
 plt.title('categories of best fitted data points with minimum loss (on train data)')
 plt.subplot(122)
 X_train['category'].iloc[train_score_args[-100:]].value_counts().plot.pie(autopct='%1.1f%%', explode=(0,0.02,0.04,0.06,0.08), shadow=True)
 plt.ylabel('')
 plt.title('categories of worst fitted data points with maximum loss (on train data)')
 plt.show()
 {% endhighlight %}

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-1">
        <img class="img-fluid rounded z-depth-0" src="https://cdn-images-1.medium.com/max/3660/1*cR2JuXG7-r4_Rq5-yjE8kA.png">
    </div>
</div>

![]()
We can notice that datapoints with category as technology make up 50% of the data that the model could not predict well whereas categories like LIFE_ARTS, SCIENCE, and CULTURE contribute much less to bad predictions.
For the good predictions, all the 5 categories contribute almost the same since there is no major difference in the proportion, still, we could say that the data-points with StackOverflow as the category contribute the least.

*With this, we have come to the end of this blog and the 3 part series. Hope the read was pleasant.
You can check the complete notebook on Kaggle using [**this link](https://www.kaggle.com/sarthakvajpayee/top-4-4-bert-roberta-xlnet)** and leave an upvote if found my work useful.
I would like to thank all the creators for creating the awesome content I referred to for writing this blog.*

*Reference links:*

* *Applied AI Course: [https://www.appliedaicourse.com/](https://www.appliedaicourse.com/)*

* [https://www.kaggle.com/c/google-quest-challenge/notebooks](https://www.kaggle.com/c/google-quest-challenge/notebooks)

* [*http://jalammar.github.io/illustrated-transformer/](http://jalammar.github.io/illustrated-transformer/)*

* [*https://medium.com/inside-machine-learning/what-is-a-transformer-d07dd1fbec04](https://medium.com/inside-machine-learning/what-is-a-transformer-d07dd1fbec04)*

* [https://towardsdatascience.com/bert-roberta-distilbert-xlnet-which-one-to-use-3d5ab82ba5f8](https://towardsdatascience.com/bert-roberta-distilbert-xlnet-which-one-to-use-3d5ab82ba5f8)
>  **Final note**

Thank you for reading the blog. I hope it was useful for some of you aspiring to do projects or learn some new concepts in NLP.

In [part 1/3](https://towardsdatascience.com/transformers-state-of-the-art-natural-language-processing-1d84c4c7462b?source=friends_link&sk=4ba3eb424ff59ce765c749819c6b5892) we covered how Transformers became state-of-the-art in various modern natural language processing tasks and their working.

In [part 2/3](https://towardsdatascience.com/understanding-bert-bidirectional-encoder-representations-from-transformers-45ee6cd51eef?source=friends_link&sk=f48ce58edfdf395fe5d86436d8102a61) we went through BERT (Bidirectional Encoder Representations from Transformers).

Kaggle in-depth EDA notebook link: [https://www.kaggle.com/sarthakvajpayee/top-4-4-in-depth-eda-feature-scraping?scriptVersionId=40263047](https://www.kaggle.com/sarthakvajpayee/top-4-4-in-depth-eda-feature-scraping?scriptVersionId=40263047)

Kaggle modeling notebook link: [https://www.kaggle.com/sarthakvajpayee/top-4-4-bert-roberta-xlnet](https://www.kaggle.com/sarthakvajpayee/top-4-4-bert-roberta-xlnet)

Kaggle post-modeling notebook link: [https://www.kaggle.com/sarthakvajpayee/top-4-4-post-modeling-analysis?scriptVersionId=40262842](https://www.kaggle.com/sarthakvajpayee/top-4-4-post-modeling-analysis?scriptVersionId=40262842)

Find me on LinkedIn: [www.linkedin.com/in/sarthak-vajpayee](http://www.linkedin.com/in/sarthak-vajpayee)

Find this project on Github: [https://github.com/SarthakV7/Kaggle_google_quest_challenge](https://github.com/SarthakV7/Kaggle_google_quest_challenge)

Peace! ☮
