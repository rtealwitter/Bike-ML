#!/bin/bash

## Split and compress
#split -l 1000000 data/data.csv data/data.csv
#tar -czvf data/dataaa.tar.gz data/data.csvaa
#tar -czvf data/dataab.tar.gz data/data.csvab
#tar -czvf data/dataac.tar.gz data/data.csvac

# Combine and uncompress
tar -zxvf data/dataaa.tar.gz
tar -zxvf data/dataab.tar.gz
tar -zxvf data/dataac.tar.gz
cat data/data.csv* > data/data.csv
