# coding=utf-8
from __future__ import division
import random

def random_kid():
    return random.choice(["boy","girl"])

both_girls = 0
older_girld = 0
either_girl = 0

random.seed(0)
for _ in range(10000):
    younger = random_kid()
    older = random_kid()
    if older == "girl":
        older_girld += 1
    if older == "girl" and younger == "girl":
        both_girls += 1
    if older == "girl" or younger == "girl":
        either_girl += 1

print "P(both| older):", both_girls / older_girld 
print "P(both| either):", both_girls / either_girl 
