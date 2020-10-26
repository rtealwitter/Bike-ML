#!/bin/bash
rm -r trips
mkdir trips

for year in {2013..2020}
do
for month in 01 02 03 04 05 06 07 08 09 10 11 12
do
name=$year$month
echo $name
url=https://s3.amazonaws.com/tripdata/$name-citibike-tripdata.zip
if curl --head -silent --fail $url
then
curl -o trips/$name-citibike-tripdata.zip $url
unzip -p trips/$name-citibike-tripdata.zip > trips/$name.csv
rm trips/$name-citibike-tripdata.zip
fi
done
done
