WHITE = (255, 255, 255) # Yes helmet, Yes jacket
RED = (67, 57, 249)     # No helmet, No jacket
BLUE = (245, 35, 66)    # Yes helmet, No jacket 
GREEN = (33, 182, 68)   # No helmet, Yes jacket

colors = {("yes", "yes"): WHITE,
		  ("no", "no"): RED,
		  ("yes", "no"): BLUE,
		  ("no", "yes"): GREEN}
