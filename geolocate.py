#!/usr/bin/env python2

'''
Design Decisions:
1) Removal of perpheral characters was done. Characters were converted from unicode to ASCII.
2) Stop words were removed from dictionry to get better prediction.
3)Instead of Laplace smoothing (and setting 1 to unseen words), 0.1 has been set for better prediction.

Experimentation:
1) Models with special characters, all the words, laplace smoothing, or unicode characcters deterioted the accuracy.
2) Final Accuracy of around 70% has been obtained.
'''
import sys
import re
import copy
import math
import heapq

#Maintaining a list of cities to use for pre-processing.
city_list = ['Los_Angeles,_CA', 'San_Francisco,_CA', 'Manhattan,_NY', 'San_Diego,_CA', 'Houston,_TX', 'Chicago,_IL', 'Philadelphia,_PA', 'Toronto,_Ontario', 'Atlanta,_GA', 'Washington,_DC', 'Boston,_MA', 'Orlando,_FL']

#This function is used to get tweets for both the train and test file. Thus we have a flag to indicate which file we are working on.
#Flag is True then the file is test file and vice versa.
def get_tweets(file_name, flag):
    tweets = []
    original = []
    with open(file_name) as reader:
        for r in reader:
            if flag == True:
                original.append(r.split(' '))
            temp = r.decode('unicode_escape')
            r = temp.encode('ascii', 'ignore')
            city = r.split(' ')[0]
            #Only taking tweets that have the first word as a city name. This removes lines and subsequent lines of multi-line tweets.
            if city in city_list:
                splitting_point = r.index(' ')
                tweet = clean_tweet(r[splitting_point:])
                r = city + ' ' + tweet
                tweets.append(r.strip().split())
    if flag == True:
        return tweets, original
    else:
        return tweets

#This function cleans the tweets passed to it. It removes special characters, replaces new lines, tabs, etc.
def clean_tweet(tweet):
    only_chars = '[^a-zA-Z\s]+'
    tweet = tweet.replace('\r', '')
    tweet = re.sub(only_chars, '', tweet)
    tweet = tweet.replace('\n', '')
    tweet = re.sub(' +', ' ', tweet)
    cleaned_tweet = tweet.strip().lower()
    return cleaned_tweet

#This function gives us the total number of tweets.
def count_tweets(city_count):
    total_tweets = 0
    for i in city_count:
        total_tweets = total_tweets + city_count[i]
    return total_tweets

#This function gives us a cont of tweets associated with each city.
def city_tweets(tweets):
    city_count = {}
    for r in tweets:
        city = r[0]
        existing = city_count.has_key(r[0])
        if (existing == False):
            city_count[city] = 1
        elif (existing == True):
            city_count[city] = city_count[city] + 1
    return city_count

#This function makes a dictionary of all the words with how many times it has occured in the train file. This serves as a vocabulary.
def make_vocab(cities):
    for r in cities:
        interested = r[1:]
        for word in interested:
            existing = word_dict.has_key(word)
            if (existing == True):
                word_dict[word] = word_dict[word] + 1
            else:
                word_dict[word] = 1

#This function calculates the priors of all cities.
def calc_prior(city_count, total_tweets):
    for i in city_count:
        city_prob[i] = float(city_count[i]) / total_tweets

#This function gets the total words in associated with a given city.
def total_words_city(data):
    total = 0
    for word in data:
        total += data[word]
    return total

#This function gets the frequency of words.
def get_freq():
    data = {}
    for i in city_count:
        data[i] = {}
    for r in range(0, len(cities)):
        for word in cities[r][1:]:
            existing = word_dict.has_key(word)
            if existing == False:
                continue
            else:
                temp = cities[r][0]
                temp_existing = data[temp].has_key(word)
                if temp_existing == False:
                    data[temp][word] = 1
                else:
                    data[temp][word] = data[temp][word] + 1
    return data

#This function calculaes the likelihood for each class.
def calc_likelihood(data, word, city, city_words):
    num = 0
    len_of_dict = len(word_dict)
    exist1 = data[city].has_key(word)
    exist2 = city_words.has_key(city)
    if exist1 == False:
        num += 0.1
    else:
        num += data[city][word]
    if exist2 == False:
        total_words = total_words_city(data[city])
        city_words[city] = total_words
    else:
        total_words = city_words[city]
    denom = total_words + len_of_dict
    return (float(num) / float(denom)), city_words

#This function calculates the conditional probability for each word in the city's words.
def calc_cond_prob(data, city_prob):
    city_words = {}
    likelihood = {}
    for r in city_prob:
        likelihood[r] = {}
        for word in word_dict:
            likelihood[r][word], city_words = calc_likelihood(data, word, r, city_words)
    return likelihood

#This function writes output to the output file.
def write_output(estimated, original):
    with open(output_file, 'w') as outputfile:
        for i in range(len(original)):
            original[i].insert(0, estimated[i])
            outputfile.write(' '.join(map(str, original[i])))

#This function predicts tweet locations.
def predict(file_name):
    count_incorrect = 0
    count_correct = 0
    estimated = []
    test_file, og_tweet = get_tweets(file_name, True)
    for tweet in test_file:
        pred_class = {}
        for city in city_prob:
            mul = city_prob[city]
            for word in tweet[1:]:
                exists = word_dict.has_key(word)
                if exists == False:
                    continue
                else:
                    mul = mul * likelihood_data[city][word]
            pred_class[city] = mul
            prediction = max(pred_class.iterkeys(), key=lambda k: pred_class[k])
        if prediction != tweet[0]:
            count_incorrect += 1
        else:
            count_correct += 1
        estimated.insert(len(estimated), prediction)
    write_output(estimated, og_tweet)
    print 'Correctly classified', count_correct
    print 'Accuracy', (float(count_correct) / len(test_file)) * 100

#Remove stop words from dictionary
def remove_stop_words(word_dict):
    for word in word_dict.keys():
        if word not in stop_words:
            continue
        else:
            del word_dict[word]
    return word_dict

#This function gets the top 5 words for each of the 12 cities.
def top_five_words(likelihood_data):
    for city in likelihood_data:
        print city
        five_words = heapq.nlargest(5, likelihood_data[city], key=likelihood_data[city].get)
        for word in five_words:
            print word
        print

#train_file = 'tweets.train.txt'
#test_file = 'tweets.test1.txt'
#output_file = 'output.txt'
arguments = sys.argv
train_file = arguments[1]
test_file = arguments[2]
output_file = arguments[3]

stop_words = set(
    ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
     'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'all', 'just', 'being', 'over', 'both', 'through',
     'yourselves', 'its', 'before', 'herself', 'had', 'should', 'to', 'only', 'under', 'ours', 'has', 'do', 'them',
     'his', 'very', 'they', 'not', 'during', 'now', 'him', 'nor', 'did', 'this', 'she', 'each', 'further', 'where',
     'few', 'because', 'doing', 'some', 'are', 'our', 'ourselves', 'out', 'what', 'for', 'while', 'does', 'above',
     'between', 'be', 'we', 'who', 'were', 'here', 'hers', 'by', 'on', 'about', 'of', 'against', 'or', 'own', 'into',
     'yourself', 'down', 'your', 'from', 'her', 'their', 'there', 'been', 'whom', 'too', 'themselves', 'was', 'until',
     'more', 'himself', 'that', 'but', 'don', 'with', 'than', 'those', 'he', 'me', 'myself', 'these', 'up', 'will',
     'below', 'can', 'theirs', 'my', 'and', 'then', 'is', 'am', 'it', 'an', 'as', 'itself', 'at', 'have', 'in', 'any',
     'if', 'again', 'no', 'when', 'same', 'how', 'other', 'which', 'you', 'after', 'most', 'such', 'why', 'off',
     'yours', 'so', 'the', 'having', 'once', 'jobs', 'job', 'amp', 'im']
)

city_prob = {}
cities = get_tweets(train_file, False)
city_count = city_tweets(cities)
total_tweets = count_tweets(city_count)

calc_prior(city_count, total_tweets)

word_dict = {}
make_vocab(cities)
word_dict = remove_stop_words(word_dict)

df = get_freq()
likelihood_data = calc_cond_prob(df, city_prob)

top_five_words(likelihood_data)
predict(test_file)