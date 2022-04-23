###Main of Social Data Analysis
import pandas as pd 
import numpy as np
import tweepy as tw 
from tqdm import tqdm
from typing import List


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
            
    