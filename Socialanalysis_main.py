###Main of Social Data Analysis
import pandas as pd 
import numpy as np
import tweepy as tw 
from tqdm import tqdm
from typing import List
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
import seaborn as sns
from sklearn.cluster import KMeans


class SocialDataAnalysis():
    """
    SocialDataAnalysis Library
    work with tweepy __version__ == 4.1
    It helps to do query and work with Twitter 
    """
    def __init__(
        self,
        API_KEY_SECRET,
        API_KEY:float
       
    ):
        self.API_KEY_SECRET = API_KEY_SECRET 
        self.API_KEY= API_KEY
    
    def connect_twitter(self):
        """_summary_

        Args:
            API_KEY_SECRET: API_KEY_SECRET of Twitter
            API_KEY: API_KEY of Twitter 

        Returns:
            api: object for make query 
        """
        auth = tw.OAuthHandler(self.API_KEY, self.API_KEY_SECRET)
        self.api = tw.API(auth, wait_on_rate_limit=True)
        return self.api
    
    def __formattation(self,df):
        """
        Formattation of output for Twitter     
        """
        tweets_df = pd.DataFrame(columns=['user_name', 'user_location', 'user_description', 
                                  'user_verified', 'id', 'date', 'text', 
                                  'in_reply_to_status_id', 'in_reply_to_user_id', 
                                  'retweet_original_user', 'retweet_original_tweet_id', 
                                  'hashtags', 'retweet_count', 'favorite_count', 'source'])
        for tweet in tqdm(df):
            hashtags = []
            try:
                for hashtag in tweet.entities["hashtags"]:
                    hashtags.append(hashtag["text"])
            except:
                pass
    
            text = self.api.get_status(id=tweet.id, tweet_mode='extended').full_text

            try:
                original_tweeter = tweet.retweeted_status.user.id
                original_tweet_id = tweet.retweeted_status.id
                df.append(tweet.retweeted_status)
            except:
                original_tweeter = None
                original_tweet_id = None

                tweets_df = tweets_df.append(pd.DataFrame({'user_id': tweet.user.id, 
                                                    'user_name': tweet.user.name, 
                                                    'user_location': tweet.user.location,\
                                                    'user_description': tweet.user.description,
                                                    'user_verified': tweet.user.verified,
                                                    'id': tweet.id,
                                                    'date': tweet.created_at,
                                                    'text': text, 
                                                    'in_reply_to_status_id': [tweet.in_reply_to_status_id if tweet.in_reply_to_status_id else None],
                                                    'in_reply_to_user_id': [tweet.in_reply_to_user_id if tweet.in_reply_to_user_id else None],
                                                    'retweet_original_user': original_tweeter, 
                                                    'retweet_original_tweet_id': original_tweet_id,
                                                    'hashtags': [hashtags if hashtags else None],
                                                    'retweet_count': tweet.retweet_count,
                                                    'favorite_count': tweet.favorite_count,
                                                    'source': tweet.source}))
        tweets_df = tweets_df.reset_index(drop=True)
        return tweets_df
    
    
    
    
    def twitter_search(self,search_query ,fromDate , toDate ,maxResults, Dev_environment_label='staging'):
        """
        Twitter_search
        
        Args:
            search_query: query parameters 
            fromDate: is need to be yyyymmdd
            toDate : is need to be yyyymmdd
            maxResults : the max results you want 
            Dev_environment_label='staging'
        """
        tweets_copy = self.api.search_full_archive(Dev_environment_label, query=search_query,fromDate=fromDate ,toDate=toDate ,maxResults = maxResults)
        df=self.__formattation(tweets_copy)
        return df
        
        
    def twitter_search_30_day(self,search_query ,maxResults, Dev_environment_label='testing'):
        """
        Twitter_search_30_day
        
        Args:
            search_query: query parameters 
            fromDate: is need to be yyyymmdd
            toDate : is need to be yyyymmdd
            maxResults : the max results you want 
            Dev_environment_label='testing'
        """
        tweets_copy = self.api.search_30_day(Dev_environment_label, query=search_query,maxResults = maxResults)
        df=self.__formattation(tweets_copy)
        return df

        
    def twitter_loop(self, search_list: List ,fromDate , toDate ,maxResults, model= "twitter_search" ):
        """
        Twitter_loop

        Args:
            search_list : List of Tweet contenent tou want to query
            fromDate (_type_): _description_
            toDate (_type_): _description_
            maxResults (_type_): _description_

        Returns:
            _type_: _description_
        """        
        if model =="twitter_search":
            query= [ self.twitter_search(i,fromDate,toDate,maxResults=maxResults) for i in search_list]
            result = pd.concat(query).reset_index()
            del(result["index"])
            return result
        
        if model == " twitter_search_30_day":
            query= [ self.twitter_search_30_day(i,maxResults=maxResults) for i in search_list]
            result = pd.concat(query).reset_index()
            del(result["index"])
            return result
            
    def count_location(self,df):
        num=df['user_location'].value_counts()
        return num
    
    
    
class SocialDataAnalysisML():
    """
    Init of the Model
    """
    def __init__(
        self,
        df
      
    ):
        
             
        self.df= df
        nltk.download('stopwords')
        nltk.download('punkt')
    
    def create_hastaglist(self):
        hashtags_col = self.df[~self.df['hashtags'].isna()]['hashtags']
        hashtags_col = hashtags_col.apply(eval)
        hashtags = pd.Series([hashtag for hashtags_list in hashtags_col for hashtag in hashtags_list])
        return hashtags
    
    def __wordcloud(self):
        wordcloud = WordCloud(width=4000, height=2000, background_color="white", mode="RGBA", prefer_horizontal=0.5)
        return wordcloud.fit_words(self.create_hastaglist().value_counts())
    
    def plot_wordcloud(self):
        plt.figure(figsize=(15,10))
        plt.axis("off")
        return plt.imshow(self.__wordcloud())
    
    # def clean_and_create(self):
    #     """Create Filtered and cleaned sentence from the df

    #     Returns:
    #         _type_: _description_
    #     """
    #     nltk.download('stopwords')
    #     nltk.download('punkt')
    #     # rimozione links
    #     example_sent = df2.loc[1, 'text']
    #     self.filtered_sentence = re.sub(r"http\S+", "", example_sent)
        
    #     # lowercase
    #     self.filtered_sentence = self.filtered_sentence.lower()
        
    #     # # stopwords
    #     stop_words = set(stopwords.words('italian'))
    #     word_tokens = word_tokenize(self.filtered_sentence)
    #     self.filtered_sentence = [w for w in word_tokens if not w in stop_words]

    #      # rimuovo rt pattern
    #     if self.filtered_sentence[0] == 'rt':
    #          self.filtered_sentence = self.filtered_sentence[4:]

    #     # # rimuovo punteggiatura
    #     self.filtered_sentence = list(filter(lambda token: token not in string.punctuation, self.filtered_sentence))
    #     return self.filtered_sentence
    # def clean(self):
    #     # rimozione links
    #     example_sent = self.df.loc[1, 'text']
    #     #sentence = self.df['text']
    #     filtered_sentence = re.sub(r"http\S+", "", example_sent)
    #     filtered_sentence = filtered_sentence.lower()
    #     stop_words = set(stopwords.words('italian'))
    #     word_tokens = word_tokenize(filtered_sentence)
    #     filtered_sentence = [w for w in word_tokens if not w in stop_words]
    #     if filtered_sentence[0] == 'rt':
    #         filtered_sentence = filtered_sentence[4:]
    #     filtered_sentence = list(filter(lambda token: token not in string.punctuation, filtered_sentence))
    #     return filtered_sentence

    # def clean_apply(self):
    #     self.df['cleaned'] = self.df.apply(self.clean, axis=1)
    #     return self.df['cleaned']
    # # def clean_df(self,column_name='clean'):
    # #     self.df[column_name] = self.df.apply(self.clean_and_create(self.df), axis=1)
        
    def gensim_init(self,df_cleaned_column):
        sent = [row.split() for row in df_cleaned_column]
        phrases = Phrases(sent, min_count=10)
        bigram = Phraser(phrases)
        self.sentences = bigram[sent]
        return sent
    
    
    def gensim_model(self):
        from gensim.models import Word2Vec
        model = Word2Vec(
                window = 10, 
                min_count = 5,
                sample=6e-5,
                alpha=0.03,
                min_alpha=0.0007,
                negative=10,
                workers=2
            )
        model.build_vocab(self.sentences)
        model.train(self.sentences, total_examples=model.corpus_count, epochs=100)
        return model