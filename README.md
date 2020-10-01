# Guesture controlled music player

This app uses a collection of computer vision techniques to detect guestures and control a music player.
More specifically, it yses skin detection, template matching, object compactness anlysis, motion energy analysis to perform guesture detection.

## Guestures detected

Currently 4 guestures are detected. 

### Pointing up 

Just keep two fingers up as shown in the image below.

![point up](/images/point_up_.jpeg)

### Poiting sideways

Point two fingers sideways as shown in the image below. 

![point sideways](/images/point_sideways.jpeg)


### Vertical Palm

Keep palm straight as shown in the image below.

![vertical palm](/images/vertical_palm_.jpeg)


### Swipe

Perform a swiping motion as shown in the image below.

![swipe](/images/swipe_.jpeg)


## Running instructions

* Download the repo and run the app file from the command line as follows -
`python app.py`

* This would open up a detection window to set skin detection and threshold parameters. 
    * The defaults are pretty good but in case you want to tune it specifically go ahead. The skin color detection is set based on the [HSV color space](https://en.wikipedia.org/wiki/HSL_and_HSV).
    * Set the lower values first and then the upper values
    * Then in a different window set the absolute threshold value.
    * Once happy with settings press `q`

![Initial selction](/images/selection_initial.JPG)

* The GUI gets launched.
    * The "pointing up" guesture controls play/pause
    * The "swipe" guesture increases volume
    * The "vertical palm" guesture decreases volume
    * The "pointing sideways" guesture shuffles and plays a random next song

![app running](/images/app_running.JPG)

## Music library

Currently few sample songs are placed in the `songs` directory and the corresponding album images are placed in `album_images`. Notice how the corresponding song and album image have the same name. To add more songs place the mp3 in the `songs` directory and album image in the `album_images` directory with the same name and then add the name to the songs list in line 465 of the `app.py` file.

## Practical tips for good performance

* Good lighting helps to a degree
* Try to get only your hand in the camera's view and try to keep face out of view

## Future work 

* Try adding more guestures
* Use deep learning based techniques

