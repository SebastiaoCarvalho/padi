#!/bin/bash

group="group45" # Change this to your group number
folder=$1
pdf=$1.pdf
name="padi-$1-$group"
zip=$name.zip

# Create a zip file
cp $folder/$pdf .
cp $folder/padi*.py .

# Zip the folder
zip $zip $pdf *.py

# Remove the files
rm -rf $pdf *.py