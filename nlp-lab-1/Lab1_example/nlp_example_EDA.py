# Data Augmentation for Text [with code]
# https://medium.com/analytics-vidhya/data-augmentation-for-text-with-code-6da46aad443d

import random

def random_deletion(sentence, p=0.1):
    words = sentence.split()
    n = len(words)
    
    if n == 1: # return if single word
        return words

    remaining = list(filter(lambda x: random.uniform(0,1) > p,words))

    if len(remaining) == 0: # if not left, choice one word
        return ' '.join([random.choice(words)])
    else:
        return ' '.join(remaining)

#sentence = 'The Best Way To Get Started Is To Quit Talking And Begin Doing.'
sentence = 'W25 W26 W27 W19 W28 W29 W30 W31 W32 W33 W34 W35 W36 W37 W38 W39 W24 W40'

sentence_RD = random_deletion(sentence)
print(sentence_RD)