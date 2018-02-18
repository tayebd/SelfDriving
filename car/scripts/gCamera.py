#!/usr/bin/env python
import numpy as np
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import gi
import sys
sys.path.insert(1, '/usr/local/lib/python2.7/site-packages')
import cv2
print cv2.__version__
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst

"""
Node that publishes images from the C920 Logitech webcam using Gstreamer

Note: Better performance using gstreamer than cv.capture
      
"""

def gst_to_opencv(gst_buffer):
    return np.ndarray((480,640,3), buffer=gst_buffer.extract_dup(0, gst_buffer.get_size()), dtype=np.uint8)

def gCamera():
    print "gstWebCam"
    bridge = CvBridge()
    video ="video4"
    pub = rospy.Publisher('stream', Image, queue_size=10)
    rospy.init_node('GstWebCam',anonymous=True)
    Gst.init(None)
    pipe = Gst.parse_launch("""v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480,format=(string)BGR ! appsink sync=false max-buffers=2 drop=true name=sink emit-signals=true""")
    sink = pipe.get_by_name('sink')
    pipe.set_state(Gst.State.PLAYING)
    while not rospy.is_shutdown():
        sample = sink.emit('pull-sample')    
        img = gst_to_opencv(sample.get_buffer())
        try:
            pub.publish(bridge.cv2_to_imgmsg(img, "bgr8"))
        except CvBridgeError as e:
            print(e)

if __name__ == '__main__':
    try:
        gCamera()
    except rospy.ROSInterruptException:
        pass

