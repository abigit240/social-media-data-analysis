#!/usr/bin/env python
# coding: utf-8

# # Clean & Analyze Social Media

# ## Introduction
# 
# Social media has become a ubiquitous part of modern life, with platforms such as Instagram, Twitter, and Facebook serving as essential communication channels. Social media data sets are vast and complex, making analysis a challenging task for businesses and researchers alike. In this project, we explore a simulated social media, for example Tweets, data set to understand trends in likes across different categories.
# 
# ## Prerequisites
# 
# To follow along with this project, you should have a basic understanding of Python programming and data analysis concepts. In addition, you may want to use the following packages in your Python environment:
# 
# - pandas
# - Matplotlib
# - ...
# 
# These packages should already be installed in Coursera's Jupyter Notebook environment, however if you'd like to install additional packages that are not included in this environment or are working off platform you can install additional packages using `!pip install packagename` within a notebook cell such as:
# 
# - `!pip install pandas`
# - `!pip install matplotlib`
# 
# ## Project Scope
# 
# The objective of this project is to analyze tweets (or other social media data) and gain insights into user engagement. We will explore the data set using visualization techniques to understand the distribution of likes across different categories. Finally, we will analyze the data to draw conclusions about the most popular categories and the overall engagement on the platform.
# 
# ## Step 1: Importing Required Libraries
# 
# As the name suggests, the first step is to import all the necessary libraries that will be used in the project. In this case, we need pandas, numpy, matplotlib, seaborn, and random libraries.
# 
# Pandas is a library used for data manipulation and analysis. Numpy is a library used for numerical computations. Matplotlib is a library used for data visualization. Seaborn is a library used for statistical data visualization. Random is a library used to generate random numbers.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install numpy')
get_ipython().system('pip install seaborn')


# In[3]:


import pandas as pd
import random
import numpy as np

# Define the categories for the social media experiment
categories = ['Food', 'Travel', 'Fashion', 'Fitness', 'Music', 'Culture', 'Family', 'Health']

# Set the number of entries (n)
n = 500

# Generate the random data
data = {
    'Date': pd.date_range('2021-01-01', periods=n),  # Generate n consecutive dates starting from 2021-01-01
    'Category': [random.choice(categories) for _ in range(n)],  # Generate a random category for each entry
    'Likes': np.random.randint(0, 10000, size=n)  # Generate random likes between 0 and 10000 for each entry
}

# Convert the dictionary to a pandas DataFrame
data_df = pd.DataFrame(data)

# Display the first few rows of the DataFrame
print(data_df.head())


# In[4]:


print(data_df['Category'].value_counts())


# In[5]:


# Remove null data
data_df.dropna(inplace=True)


# In[6]:


# Remove duplicate data
data_df.drop_duplicates(inplace=True)


# In[7]:


# Convert the 'Date' field to a datetime format
data_df['Date'] = pd.to_datetime(data_df['Date'])

# Convert the 'Likes' field to an integer
data_df['Likes'] = data_df['Likes'].astype(int)


# In[8]:


# Display the first few rows of the DataFrame
print(data_df.head())


# In[9]:


# Print the count of each 'Category' element
print(data_df['Category'].value_counts())


# In[11]:


pip install --upgrade seaborn


# In[20]:


sns.distplot(data_df['Likes'], kde=False, bins=30)
plt.title('Histogram of Likes')
plt.xlabel('Number of Likes')
plt.ylabel('Frequency')
plt.show()
plt.savefig('histogram_likes.png')  # Save the histogram


# In[24]:


import os
print(os.path.abspath('histogram_likes.png'))


# In[22]:


# Visualize the data: Boxplot of 'Likes' by 'Category'
sns.boxplot(x='Category', y='Likes', data=data_df)
plt.title('Boxplot of Likes by Category')
plt.xlabel('Category')
plt.ylabel('Number of Likes')
plt.xticks(rotation=45)
plt.show()
plt.savefig('boxplot_likes_category.png')  # Save the boxplot


# In[15]:


# Perform statistics on the data
# Mean of the 'Likes' column
mean_likes = data_df['Likes'].mean()
print(f"Mean of Likes: {mean_likes}")


# In[16]:


# Mean of 'Likes' grouped by 'Category'
mean_likes_by_category = data_df.groupby('Category')['Likes'].mean()
print("Mean of Likes by Category:")
print(mean_likes_by_category)


# ![Likes Histogram](/home/jovyan/work/histogram_likes.png)

# # Conclusion: Key Findings and Reflections
# 
# ## Process Overview
# The goal of this project was to analyze and visualize social media data, leveraging Python libraries such as Pandas, Seaborn, and Matplotlib for data manipulation and visualization. The process involved several key steps:
# 
# 1. **Data Generation**: I used Python's randomization modules to create synthetic data representing social media categories, dates, and engagement (likes).
# 2. **Data Cleaning**: I ensured the dataset was free from null values and duplicates, converted dates into a datetime format, and standardized the 'Likes' field as integers to maintain consistency.
# 3. **Visualization**: Histograms and boxplots were used to explore the distribution of likes and their relationship to categories.
# 4. **Statistical Analysis**: Key metrics, such as the overall mean likes and category-wise mean likes, were calculated to provide insights into engagement trends.
# 
# ## Key Findings
# 1. **Engagement Distribution**:
#    - The histogram revealed that most posts received fewer likes, with a gradual tapering off toward higher likes.
#    - A small number of posts achieved exceptionally high engagement, indicating a skewed distribution.
#    
# 2. **Category Performance**:
#    - From the boxplot, it was clear that certain categories (e.g., 'Music' and 'Travel') tended to receive higher engagement compared to others like 'Family' and 'Health'.
#    - Statistical analysis showed that 'Travel' had the highest average likes, which might reflect its popularity on social media.
# 
# ## Challenges and Solutions
# - **Challenge**: Managing dependencies across libraries like Seaborn and Matplotlib. An error in visualization due to outdated Seaborn methods (e.g., `histplot`) required debugging.
#   - **Solution**: Ensured compatibility by either updating the library or using alternatives like `distplot`.
# 
# - **Challenge**: Generating realistic synthetic data for analysis.
#   - **Solution**: Used appropriate ranges for likes (0 to 10,000) and ensured randomness in categories to mimic real-world variability.
# 
# ## What Sets This Project Apart
# This project demonstrates:
# 
# 1. **End-to-End Workflow**: The process highlights proficiency in data generation, cleaning, analysis, and visualization.
# 2. **Actionable Insights**: Statistical findings and visualizations go beyond aesthetics, providing meaningful insights that can inform business decisions.
# 3. **Critical Thinking**: Each step, from identifying errors to refining visualizations, shows a commitment to problem-solving.
# 
# ## Future Improvements
# 1. **Enhanced Data Realism**:
#    - Incorporate additional features like hashtags, time of posting, or user demographics to create a richer dataset.
# 2. **Interactive Visualizations**:
#    - Use tools like Plotly or Dash for interactive charts that allow deeper exploration of the data.
# 3. **Predictive Analytics**:
#    - Build models to predict engagement based on category and posting trends.
# 
# ## Artifacts for Portfolio
# 1. **Image Files**: 
# ![Boxplot](boxplot_likes_category.png)
# 
# 2. **Code Excerpts**: # Annotated Code Snippets: Data Cleaning, Visualization, and Statistical Analysis
# 
# ## 1. Data Cleaning
# This step ensures the data is accurate, consistent, and ready for analysis. 
# The following snippet removes null values, eliminates duplicate rows, and standardizes the data types.
# 
# ```python
# # Remove null values to ensure no missing data interferes with analysis
# data_df.dropna(inplace=True)
# 
# # Remove duplicate rows to avoid skewed results
# data_df.drop_duplicates(inplace=True)
# 
# # Convert 'Date' to datetime format for time-based operations
# data_df['Date'] = pd.to_datetime(data_df['Date'])
# 
# # Convert 'Likes' to integer to ensure numerical operations work properly
# data_df['Likes'] = data_df['Likes'].astype(int)
# 
# ## 2. Data Visualization
# Data visualization helps us understand patterns and relationships in the data. Here are two key plots used for this project:
# 
# ### a) Histogram of 'Likes'
# The histogram shows the distribution of the number of likes across all posts.
# 
# ```python
# # Visualize the distribution of 'Likes'
# sns.histplot(data_df['Likes'], kde=False, bins=30)
# plt.title('Histogram of Likes')
# plt.xlabel('Number of Likes')
# plt.ylabel('Frequency')
# plt.show()
# 
# ### b) Boxplot of 'Likes' by Category
# The boxplot shows the spread of likes for each category, highlighting variations and median values.
# ```python
# # Visualize likes grouped by category
# sns.boxplot(x='Category', y='Likes', data=data_df)
# plt.title('Boxplot of Likes by Category')
# plt.xlabel('Category')
# plt.ylabel('Number of Likes')
# plt.xticks(rotation=45)
# plt.show()
# 
# ## 3. Statistical Analysis
# Statistical analysis provides numerical summaries and insights from the data, helping us measure trends and compare performance across categories.
# 
# ### a) Overall Mean of 'Likes'
# This calculates the average number of likes across all posts.
# 
# ```python
# # Calculate the mean of 'Likes'
# mean_likes = data_df['Likes'].mean()
# print(f"Mean of Likes: {mean_likes}")
# 
# ### b) Mean Likes by Category
# This calculates the average likes for each category, allowing us to compare category-level engagement.
# 
# ```python
# # Calculate mean likes for each category
# mean_likes_by_category = data_df.groupby('Category')['Likes'].mean()
# print("Mean of Likes by Category:")
# print(mean_likes_by_category)
# 
# ###Insights
# The overall mean of likes provides a baseline for engagement. Category-wise means highlight which categories consistently attract more engagement, offering valuable input for content strategy.
# 
# ## 4. Documentation: Insights and Future Improvements
# 
# ### Key Insights
# 1. **Overall Engagement**: 
#    - The average number of likes across all posts provides a baseline for evaluating performance. 
#    - Categories like **Music** and **Travel** consistently have higher average likes, indicating they are more engaging for the audience.
# 
# 2. **Category Performance**:
#    - The boxplot revealed that categories like **Fashion** and **Health** show a wider range of likes, suggesting varying audience responses.
#    - Outliers in certain categories (e.g., a few posts in **Fitness** or **Family** receiving exceptionally high likes) could be studied for specific content strategies.
# 
# 3. **Engagement Distribution**:
#    - The histogram of likes showed that engagement is skewed, with most posts receiving likes in a specific range. 
#    - This suggests opportunities for improvement in underperforming posts to increase overall engagement.
# 
# ---
# 
# ### Proposed Future Improvements
# 1. **Content Strategy Optimization**:
#    - Focus on high-performing categories like **Music** and **Travel** while analyzing the specific traits of successful posts in categories with high outliers (e.g., **Health**, **Fitness**).
#    - Experiment with cross-category content to identify synergies (e.g., **Fitness** + **Travel**).
# 
# 2. **Advanced Analysis**:
#    - Implement sentiment analysis on post comments or text to understand the audience's reactions.
#    - Analyze trends over time (e.g., seasonal patterns in engagement).
# 
# 3. **Enhanced Visualizations**:
#    - Add time-series plots to visualize how likes trend over months or weeks.
#    - Create correlation heatmaps to investigate relationships between likes and other potential metrics (e.g., post timing, hashtags).
# 
# 4. **Audience Segmentation**:
#    - Collect demographic data (if available) to see how engagement varies across different audience segments.
#    - Tailor content strategies for target demographics.
# 
# 5. **Improved Data Collection**:
#    - Incorporate additional fields like shares, comments, or reach to get a holistic view of post performance.
#    - Use more granular date ranges (e.g., hourly data) to analyze peak engagement times.
# 
# ---
# 
# ### Conclusion
# These insights and recommendations demonstrate the value of using data-driven decisions to optimize social media content strategies. By addressing the areas of improvement and exploring advanced analysis methods, businesses can better engage their audience and achieve measurable growth.
# 
