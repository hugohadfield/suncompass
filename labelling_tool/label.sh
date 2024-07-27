# If the images folder does not exist, create it
if [ ! -d "images" ]; then
  mkdir images
fi
# Remove everything in the images folder
rm -r images/*.jpg
# Copy data from the user input data folder to the images folder
cp -r $1/*.jpg images/
# Run the python script to fill in the templates
python3 fill_templates.py --lat ${2} --lng ${3}
# Start the server
python3 -m http.server 9000