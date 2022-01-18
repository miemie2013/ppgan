




import cv2
import ppgan
from ppgan.apps import AnimeGANPredictor


# predictor = AnimeGANPredictor()
# PATH_OF_IMAGE = 'yingbb2.png'
# predictor.run(PATH_OF_IMAGE)



predictor = ppgan.apps.PPMSVSRLargePredictor(output='output', weight_path=None, num_frames=10)
# PATH_OF_IMAGE = 'Peking_input360p_clip6_5s.mp4'
# predictor.run(PATH_OF_IMAGE)

PATH_OF_IMAGE = 'a.png'
# PATH_OF_IMAGE = 'yingbb2.png'
predictor.run_image(PATH_OF_IMAGE)




print()




