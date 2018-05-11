
The main problem for me - was absence of any experience with web development and Flask/Django
The process by itself is quite simple

As i have no gpu - I was limited to my laptop
I took the pretrained ResNet for 1000 class classification
An output from the last layer was taken as 2000-vector representation  of each picture
With it I created a dictionary with image name  - 2000 vector representation
Than given a new picture - it is easy to get a 2000 vector representation for it a find 5 closest
pictures out of 25000 based on cosine distance

The results are presented

To use it you need to add flickr 25000 set folder inside static folder
Just run the python app.py in the folder with it
https://www.youtube.com/watch?v=gymnASuGCkc&feature=youtu.be - demo
