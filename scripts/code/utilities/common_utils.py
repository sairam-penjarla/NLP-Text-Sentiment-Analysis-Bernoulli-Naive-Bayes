from code.logging import logger  # Importing logger module for logging
import re
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer

class CommonUtils:
    def __init__(self) -> None:
        pass
    def preprocess(self, textdata, text_processing_params):
        processedText = []
        emojis = text_processing_params['emojis']
        stopwordlist = text_processing_params['stopwordlist']

        # Create Lemmatizer and Stemmer.
        wordLemm = WordNetLemmatizer()
        
        # Defining regex patterns.
        urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
        userPattern       = r'@[^\s]+'
        alphaPattern      = r"[^a-zA-Z0-9]"
        sequencePattern   = r"(.)\1\1+"
        seqReplacePattern = r"\1\1"
        
        for tweet in tqdm(textdata):
            tweet = tweet.lower()
            
            # Replace all URls with 'URL'
            tweet = re.sub(urlPattern,' URL',tweet)
            # Replace all emojis.
            for emoji in emojis.keys():
                tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])        
            # Replace @USERNAME to 'USER'.
            tweet = re.sub(userPattern,' USER', tweet)        
            # Replace all non alphabets.
            tweet = re.sub(alphaPattern, " ", tweet)
            # Replace 3 or more consecutive letters by 2 letter.
            tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

            tweetwords = ''
            for word in tweet.split():
                # Checking if the word is a stopword.
                if word not in stopwordlist:
                    if len(word)>1:
                        # Lemmatizing the word.
                        word = wordLemm.lemmatize(word)
                        tweetwords += (word+' ')
                
            processedText.append(tweetwords)
            
        return processedText