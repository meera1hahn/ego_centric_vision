First Person Vision Datasets
-----------------------------
* ADL dataset
    * https://www.csee.umbc.edu/~hpirsiav/papers/ADLdataset/
    * Go Pro ego centric camera with a chest mount
    * 30 frames per second and with 170 degrees of viewing angle
    * 18 actions which are all daily home activities 
    * 20 people filmed in their own apartment and each of these are ~30 minutes long
    * annotated every second annotated with action labeled and object bounding boxes, and which objects are active 
* GTEA GAZE +
    * http://www.cbi.gatech.edu/fpv/
    * 37 total videos, avg length ~10 min, 24 fps
    * 6 subjects
    * 7 different cooking recipes that are being followed in the videos
    * All recorded in the same setting which is an aware home kitchen
    * annotated frame wise with manipulated objects and action segments and the recipe step that the action segment corresponds to 
    * extra annotations (contact me)
* Bristol Ego Object Interaction Dataset
    * https://www.cs.bris.ac.uk/~damen/BEOID/
    * T6 different indoor tasks, each performed in a different location
        * from operating gym equipment to making a cup of coffee. 
    * Using the descriptions given to participants, we created an instruction set (recipe) for each task and labeled the ground truth action segments appropriately. (contact me)
* EgoGesture Dataset
    * http://www.nlpr.ia.ac.cn/iva/yfzhang/datasets/egogesture.html
    * http://www.nlpr.ia.ac.cn/iva/yfzhang/datasets/EgoGesture.pdf
    * 2,081 RGB-D videos
    *  24,161 gesture samples
    *  2,953,224 frames from 50 distinct subjects
    * 83 classes of static or dynamic gestures focused on interaction with wearable devices as shown in Figure 1.
* FPV-O Dataset
    * http://www.eecs.qmul.ac.uk/~andrea/fpvo
    * 20 activities performed by 12 subjects with a chest-mounted camera. 
    * The activities include three person-to-person interaction activities, sixteen person-to-object interaction activities and one proprioceptive activity

3rd Person? Action Datasets
---------------------------
* UCF 101
    * 101 action classes
    * 5 types of actions: Human-Object Interaction, Body-Motion Only, HumanHuman Interaction, Playing Musical Instruments, Sports
    * 13320 total clips
    * 25 fps with 320×240 resolution
    * average clip length 7.21 sec
    * composed of YouTube videos
* HMDB (Human Motion Database)
    * 51 actions, each with at least 101 clips
    * 6766 clips total
    * 5 types of actions: General facial actions, Facial actions with object manipulation, General body movements, Body movements with object interaction, Body movements
    * 30 fps
    * collected from web video, movies, etc. 
* Kinetics
    * 400 action classes at least 400 clips for each class
    * total clips 306,245
    * each clips is  ~10 seconds, 
    * collected from YouTube videos
    * actions include Person Actions (singular), Person-Person Actions, Person-Object Actions
* ActivityNet
    * daily living activities
    * contains hierarchy graph of actions
    * 203 activity classes, average of 137 untrimmed videos per class, 1.41 activity instances per video
    * total of 849 video hours
    * collected from the web
* MPII Cooking Dataset 2
    * 30 subjects over 273 videos
    * 59 composite activities, all kitchen activities 
    * All filmed in the same kitchen with the same camera, from the same angle
    * 30 fps
    * Tacos Dataset: subset of MPII cooking dataset 2 which has human annotations of the video
* Charades Dataset
    * 9,848 videos, avg ~length 30 secs, showing 267 people from 3 continents
    * Collected in 15 types of indoor scenes
    * interactions with 46 object classes, 30 verbs, 157 action classes
    * video annotations  
        * free-text descriptions, action labels, action intervals, classes of interacted objects
        * the localized actions are avg ~length 12.8 sec
    * collected by Amazon Turkers in their own homes following a script given to them
    * depicts daily activities in the home


    

