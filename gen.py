#!/usr/bin/env python
import time

def countdown(n):
    while n > 0:
        yield n
        n -= 1

# Create generator function
count_gen = countdown(13)

# Iterate over generator
for num in count_gen:
    print(num)


