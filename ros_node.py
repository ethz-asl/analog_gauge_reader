#!/usr/bin/env python3
# coding: utf8
# Copyright (c) 2023 jk-ethz.

import rospy
import os
import numpy as np
import cv2
from cv_bridge import CvBridge
import tempfile
from std_msgs.msg import String, Float64
from sensor_msgs.msg import Image
from analog_gauge_reader.msg import GaugeReading, GaugeReadings
from analog_gauge_reader.srv import GaugeReader as GaugeReaderSrv, GaugeReaderRequest, GaugeReaderResponse
from pipeline import process_image

class AnalogGaugeReaderRos:
    def __init__(self, debug=False):
        self.detection_model_path = rospy.get_param("~detection_model_path")
        self.key_point_model_path = rospy.get_param("~key_point_model_path")
        self.segmentation_model_path = rospy.get_param("~segmentation_model_path")
        self.image_topic = rospy.get_param("~image_topic")
        self.round_decimals = rospy.get_param("~round_decimals", -1)
        self.latch = rospy.get_param("~latch", False)
        self.continuous = rospy.get_param("~continuous", False)
        self.debug = debug

        self.publishers = {}
        self.readings_pub = rospy.Publisher("~readings", GaugeReadings, queue_size=1, latch=self.latch)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self._image_callback, queue_size=1)
        self.trigger_srv = rospy.Service("~read", GaugeReaderSrv, self.read)
        self.image = None

    def _image_callback(self, msg: Image):
        self.image = msg
        if self.continuous:
            self.read(GaugeReaderRequest())

    def read(self, req: GaugeReaderRequest):
        rospy.loginfo("Processing read request...")
        self.clear_visualizations()
        original_image = self.image
        if original_image is None:
            rospy.logwarn("No image received yet, ignoring read request")
            raise rospy.ServiceException("No image received yet, ignoring read request")

        image = self.bridge.imgmsg_to_cv2(original_image)
        self.visualize(np.flip(image, axis=2), original_image, "original_image")
        with tempfile.TemporaryDirectory() as out_path:
            os.removedirs(out_path)
            try:
                gauge_readings = [process_image(image=image,
                                                image_is_raw=True,
                                                detection_model_path=self.detection_model_path,
                                                key_point_model_path=self.key_point_model_path,
                                                segmentation_model_path=self.segmentation_model_path,
                                                run_path=out_path,
                                                debug=self.debug,
                                                eval_mode=False)]
            finally:
                for i in range(2) if len(self.publishers.keys()) <= 1 else [1]:
                    for visual_result in os.listdir(out_path):
                        visual_result_split = os.path.splitext(visual_result)
                        if visual_result_split[1] != ".jpg":
                            continue
                        visual_result_image = cv2.imread(os.path.join(out_path, visual_result))
                        self.visualize(visual_result_image, original_image, visual_result_split[0], i==0)
                    if i==0:
                        rospy.sleep(3)

        res = GaugeReaderResponse()
        for gauge_reading in gauge_readings:
            if gauge_reading["value"] is None:
                raise Exception("Value reading failed")
            reading = GaugeReading()
            value = Float64()
            value.data = gauge_reading["value"] if self.round_decimals < 0 else round(gauge_reading["value"], self.round_decimals)
            unit = String()
            unit.data = gauge_reading["unit"] if gauge_reading["unit"] is not None else ''
            reading.value = value
            reading.unit = unit
            res.result.readings.append(reading)
        rospy.loginfo("Sucessfully processed read request.")
        self.readings_pub.publish(res.result)
        return res

    def visualize(self, img, original_image, name, create_only=False):
        if name not in self.publishers:
            rospy.loginfo(f"Creating a new publisher called ~{name}")
            self.publishers[name] = rospy.Publisher(f"~{name}", Image, queue_size=1, latch=self.latch)
            if not create_only:
                rospy.sleep(1)
        if create_only:
            return
        img_msg = self.bridge.cv2_to_imgmsg(img)
        img_msg.header = original_image.header
        self.publishers[name].publish(img_msg)

    def clear_visualizations(self):
        for publisher in self.publishers.values():
            publisher.publish(Image())

if __name__ == "__main__":
    rospy.init_node("analog_gauge_reader")
    analog_gauge_reader_ros = AnalogGaugeReaderRos(debug=True)
    rospy.loginfo("Analog gauge reader service up and running.")
    rospy.spin()
