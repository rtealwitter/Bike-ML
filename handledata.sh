#!/bin/bash

## Split
#split -l 2000000 data/combined1.csv data/combined1
#split -l 2000000 data/combined2.csv data/combined2

## Compress
#tar -czvf data/1aa.tar.gz data/combined1aa
#tar -czvf data/1ab.tar.gz data/combined1ab
#tar -czvf data/1ac.tar.gz data/combined1ac
#tar -czvf data/2aa.tar.gz data/combined2aa
#tar -czvf data/2ab.tar.gz data/combined2ab
#tar -czvf data/2ac.tar.gz data/combined2ac

# Uncompress
tar -zxvf data/1aa.tar.gz
tar -zxvf data/1ab.tar.gz
tar -zxvf data/1ac.tar.gz
tar -zxvf data/2aa.tar.gz
tar -zxvf data/2ab.tar.gz
tar -zxvf data/2ac.tar.gz

# Combine
cat data/combined1a* > data/combined1.csv
cat data/combined2a* > data/combined2.csv

# Clean
rm data/*tar.gz
rm data/*aa
rm data/*ab
rm data/*ac

