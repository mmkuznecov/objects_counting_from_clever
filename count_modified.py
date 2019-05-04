from morph.centroidtracker import CentroidTracker
from morph.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import matplotlib.pyplot as plt

def infographics(time1,time2,name,frames_amount,objects_per_frame):
	list_of_frames = [i for i in range(0,frames_amount,20)]
	if len(list_of_frames)>len(objects_per_frame):
		list_of_frames[:-1]
	elif len(list_of_frames)<len(objects_per_frame):
		list_of_frames.append(list_of_frames[-1]+20)

		
	plt.plot(list_of_frames,objects_per_frame)
	plt.ylabel('Amount of objects')
	plt.xlabel('Number of frames')
	plt.title(time1+' - '+time2)
	plt.savefig(name.split('.')[0]+'.png')
	plt.show()

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str,
	help="type of object to recognize")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
args = vars(ap.parse_args())

tm1 = time.ctime(time.time())

objects_per_frame = []

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')

#vs = VideoStream(src='http://192.168.11.1:8080/snapshot?topic=/main_camera/image_raw').start()
vs = VideoStream(src='http://62.117.66.226:5118/mjpg/video.mjpg?fps=4&resolution=800x600').start()

writer = None

W = None
H = None

ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}


totalFrames = 0
total=0
fps = FPS().start()


while True:

	frame = vs.read()


	frame = imutils.resize(frame, width=500)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	if W is None or H is None:
		(H, W) = frame.shape[:2]


	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,(W, H), True)

	status = "Waiting"
	rects = []

	if totalFrames % 20 == 0: #detect every 20 frames
		objects_per_frame.append(0)
		status = "Detecting"
		trackers = []


		blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
		net.setInput(blob)
		detections = net.forward()

		
		for i in np.arange(0, detections.shape[2]):
			
			confidence = detections[0, 0, i, 2]

			
			if confidence > 0.4:

				idx = int(detections[0, 0, i, 1])

				
				if CLASSES[idx]!=args['type']:
					continue
				
				objects_per_frame[-1]+=1

				
				box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
				(startX, startY, endX, endY) = box.astype("int")

				
				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(startX, startY, endX, endY)
				tracker.start_track(rgb, rect)

				
				trackers.append(tracker)

	
	else:
		for tracker in trackers:
			
			status = "Tracking"

			
			tracker.update(rgb)
			pos = tracker.get_position()

			
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			
			rects.append((startX, startY, endX, endY))

	objects = ct.update(rects)

	for (objectID, centroid) in objects.items():

		to = trackableObjects.get(objectID, None)

		if to is None:
			to = TrackableObject(objectID, centroid)

		
		else:
			
			y = [c[1] for c in to.centroids]
			direction = centroid[1] - np.mean(y)
			to.centroids.append(centroid)

			
			if not to.counted:

				total += 1
				to.counted = True

		trackableObjects[objectID] = to


		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)


	info = [
		('Total',total),
	]

	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)

	if writer is not None:
		writer.write(frame)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break


	totalFrames += 1
	fps.update()


fps.stop()



if writer is not None:
	writer.release()

vs.stop()
cv2.destroyAllWindows()

tm2 = time.ctime(time.time())

infographics(tm1,tm2,args['output'],totalFrames,objects_per_frame)
