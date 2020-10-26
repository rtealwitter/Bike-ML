#!/bin/bash

download () {
    echo "Checking $1..."
    url=https://s3.amazonaws.com/tripdata/$1-citibike-tripdata.zip
    if curl --head -silent --fail $url; then
    if test -f "trips/$2.csv"; then
    echo "$2.csv already exists..."
    else
    curl -o trips/$1-citibike-tripdata.zip $url
    unzip -p trips/$1-citibike-tripdata.zip > trips/$2.csv
    rm trips/$1-citibike-tripdata.zip
    fi fi
}

for year in {2013..2020}
do
for month in {1..12}
do
for name in $year$month "${year}0${month}"
do
filename="${year}${month}"
download $name $filename
done
done
done
