from twitterscraper import query_user_info

if __name__ == '__main__':
    userinfo = query_user_info("CNN", True)
    print(userinfo)
    #print the retrieved tweets to the screen:
    # for tweet in query_tweets("Trump OR Clinton", 10):
    #     print(tweet)