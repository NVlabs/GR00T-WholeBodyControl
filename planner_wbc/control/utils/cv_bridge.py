# Software License Agreement (BSD License)
#
# Copyright (c) 2011, Willow Garage, Inc.
# Copyright (c) 2016, Tal Regev.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import sys

import cv2
import sensor_msgs.msg


def CV_MAT_CNWrap(flags):
    return (((flags) & ((63) << 3)) >> 3) + 1


def CV_MAT_DEPTHWrap(flags):
    return (flags) & 7


_CV_CONVERSIONS = {
    ("mono8", "rgb8"): cv2.COLOR_GRAY2RGB,
    ("mono8", "bgr8"): cv2.COLOR_GRAY2BGR,
    ("mono8", "rgba8"): cv2.COLOR_GRAY2RGBA,
    ("mono8", "bgra8"): cv2.COLOR_GRAY2BGRA,
    ("rgb8", "mono8"): cv2.COLOR_RGB2GRAY,
    ("rgb8", "bgr8"): cv2.COLOR_RGB2BGR,
    ("rgb8", "rgba8"): cv2.COLOR_RGB2RGBA,
    ("rgb8", "bgra8"): cv2.COLOR_RGB2BGRA,
    ("bgr8", "mono8"): cv2.COLOR_BGR2GRAY,
    ("bgr8", "rgb8"): cv2.COLOR_BGR2RGB,
    ("bgr8", "rgba8"): cv2.COLOR_BGR2RGBA,
    ("bgr8", "bgra8"): cv2.COLOR_BGR2BGRA,
    ("rgba8", "mono8"): cv2.COLOR_RGBA2GRAY,
    ("rgba8", "rgb8"): cv2.COLOR_RGBA2RGB,
    ("rgba8", "bgr8"): cv2.COLOR_RGBA2BGR,
    ("rgba8", "bgra8"): cv2.COLOR_RGBA2BGRA,
    ("bgra8", "mono8"): cv2.COLOR_BGRA2GRAY,
    ("bgra8", "rgb8"): cv2.COLOR_BGRA2RGB,
    ("bgra8", "bgr8"): cv2.COLOR_BGRA2BGR,
    ("bgra8", "rgba8"): cv2.COLOR_BGRA2RGBA,
    ("yuv422", "mono8"): cv2.COLOR_YUV2GRAY_UYVY,
    ("yuv422", "rgb8"): cv2.COLOR_YUV2RGB_UYVY,
    ("yuv422", "bgr8"): cv2.COLOR_YUV2BGR_UYVY,
    ("yuv422", "rgba8"): cv2.COLOR_YUV2RGBA_UYVY,
    ("yuv422", "bgra8"): cv2.COLOR_YUV2BGRA_UYVY,
    ("bayer_rggb8", "mono8"): cv2.COLOR_BayerBG2GRAY,
    ("bayer_rggb8", "rgb8"): cv2.COLOR_BayerBG2RGB,
    ("bayer_rggb8", "bgr8"): cv2.COLOR_BayerBG2BGR,
    ("bayer_bggr8", "mono8"): cv2.COLOR_BayerRG2GRAY,
    ("bayer_bggr8", "rgb8"): cv2.COLOR_BayerRG2RGB,
    ("bayer_bggr8", "bgr8"): cv2.COLOR_BayerRG2BGR,
    ("bayer_gbrg8", "mono8"): cv2.COLOR_BayerGR2GRAY,
    ("bayer_gbrg8", "rgb8"): cv2.COLOR_BayerGR2RGB,
    ("bayer_gbrg8", "bgr8"): cv2.COLOR_BayerGR2BGR,
    ("bayer_grbg", "mono8"): cv2.COLOR_BayerGB2GRAY,
    ("bayer_grbg", "rgb8"): cv2.COLOR_BayerGB2RGB,
    ("bayer_grbg", "bgr8"): cv2.COLOR_BayerGB2BGR,
}

_CV_TYPES = {
    "rgb8": cv2.CV_8UC3,
    "rgba8": cv2.CV_8UC4,
    "rgb16": cv2.CV_16UC3,
    "rgba16": cv2.CV_16UC4,
    "bgr8": cv2.CV_8UC3,
    "bgra8": cv2.CV_8UC4,
    "bgr16": cv2.CV_16UC3,
    "bgra16": cv2.CV_16UC4,
    "mono8": cv2.CV_8UC1,
    "mono16": cv2.CV_16UC1,
    "8UC1": cv2.CV_8UC1,
    "8UC2": cv2.CV_8UC2,
    "8UC3": cv2.CV_8UC3,
    "8UC4": cv2.CV_8UC4,
    "8SC1": cv2.CV_8SC1,
    "8SC2": cv2.CV_8SC2,
    "8SC3": cv2.CV_8SC3,
    "8SC4": cv2.CV_8SC4,
    "16UC1": cv2.CV_8UC1,
    "16UC2": cv2.CV_8UC2,
    "16UC3": cv2.CV_8UC3,
    "16UC4": cv2.CV_8UC4,
    "16SC1": cv2.CV_16SC1,
    "16SC2": cv2.CV_16SC2,
    "16SC3": cv2.CV_16SC3,
    "16SC4": cv2.CV_16SC4,
    "32SC1": cv2.CV_32SC1,
    "32SC2": cv2.CV_32SC2,
    "32SC3": cv2.CV_32SC3,
    "32SC4": cv2.CV_32SC4,
    "32FC1": cv2.CV_32FC1,
    "32FC2": cv2.CV_32FC2,
    "32FC3": cv2.CV_32FC3,
    "32FC4": cv2.CV_32FC4,
    "64FC1": cv2.CV_64FC1,
    "64FC2": cv2.CV_64FC2,
    "64FC3": cv2.CV_64FC3,
    "64FC4": cv2.CV_64FC4,
    "bayer_rggb8": cv2.CV_8UC1,
    "bayer_bggr8": cv2.CV_8UC1,
    "bayer_gbrg8": cv2.CV_8UC1,
    "bayer_grbg8": cv2.CV_8UC1,
    "bayer_rggb16": cv2.CV_16UC1,
    "bayer_bggr16": cv2.CV_16UC1,
    "bayer_gbrg16": cv2.CV_16UC1,
    "bayer_grbg16": cv2.CV_16UC1,
}


def cvtColor2(img, encoding_in, encoding_out):
    if encoding_in == encoding_out:
        return img

    conversion = _CV_CONVERSIONS[(encoding_in, encoding_out)]
    # depth conversion is not yet implemented
    return cv2.cvtColor(img, conversion)


def getCvType(encoding):
    return _CV_TYPES[encoding]


class CvBridgeError(TypeError):
    """
    This is the error raised by :class:`cv_bridge.CvBridge` methods when they fail.
    """

    pass


class CvBridge(object):
    """
    The CvBridge is an object that converts between OpenCV Images and ROS Image messages.

       .. doctest::
           :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

           >>> import cv2
           >>> import numpy as np
           >>> from cv_bridge import CvBridge
           >>> br = CvBridge()
           >>> dtype, n_channels = br.encoding_as_cvtype2('8UC3')
           >>> im = np.ndarray(shape=(480, 640, n_channels), dtype=dtype)
           >>> msg = br.cv2_to_imgmsg(im)  # Convert the image to a message
           >>> im2 = br.imgmsg_to_cv2(msg) # Convert the message to a new image
           >>> cmprsmsg = br.cv2_to_compressed_imgmsg(im)  # Convert the image to a compress message
           >>> im22 = br.compressed_imgmsg_to_cv2(msg) # Convert the compress message to a new image
           >>> cv2.imwrite("this_was_a_message_briefly.png", im2)

    """

    def __init__(self):
        import cv2

        self.cvtype_to_name = {}
        self.cvdepth_to_numpy_depth = {
            cv2.CV_8U: "uint8",
            cv2.CV_8S: "int8",
            cv2.CV_16U: "uint16",
            cv2.CV_16S: "int16",
            cv2.CV_32S: "int32",
            cv2.CV_32F: "float32",
            cv2.CV_64F: "float64",
        }

        for t in ["8U", "8S", "16U", "16S", "32S", "32F", "64F"]:
            for c in [1, 2, 3, 4]:
                nm = "%sC%d" % (t, c)
                self.cvtype_to_name[getattr(cv2, "CV_%s" % nm)] = nm

        self.numpy_type_to_cvtype = {
            "uint8": "8U",
            "int8": "8S",
            "uint16": "16U",
            "int16": "16S",
            "int32": "32S",
            "float32": "32F",
            "float64": "64F",
        }
        self.numpy_type_to_cvtype.update(
            dict((v, k) for (k, v) in self.numpy_type_to_cvtype.items())
        )

    def dtype_with_channels_to_cvtype2(self, dtype, n_channels):
        return "%sC%d" % (self.numpy_type_to_cvtype[dtype.name], n_channels)

    def cvtype2_to_dtype_with_channels(self, cvtype):
        return self.cvdepth_to_numpy_depth[CV_MAT_DEPTHWrap(cvtype)], CV_MAT_CNWrap(cvtype)

    def encoding_to_cvtype2(self, encoding):
        try:
            return getCvType(encoding)
        except RuntimeError as e:
            raise CvBridgeError(e)

    def encoding_to_dtype_with_channels(self, encoding):
        return self.cvtype2_to_dtype_with_channels(self.encoding_to_cvtype2(encoding))

    def compressed_imgmsg_to_cv2(self, cmprs_img_msg, desired_encoding="passthrough"):
        """
        Convert a sensor_msgs::CompressedImage message to an OpenCV :cpp:type:`cv::Mat`.

        :param cmprs_img_msg:   A :cpp:type:`sensor_msgs::CompressedImage` message
        :param desired_encoding:  The encoding of the image data, one of the following strings:

           * ``"passthrough"``
           * one of the standard strings in sensor_msgs/image_encodings.h

        :rtype: :cpp:type:`cv::Mat`
        :raises CvBridgeError: when conversion is not possible.

        If desired_encoding is ``"passthrough"``, then the returned image has the same format as img_msg.
        Otherwise desired_encoding must be one of the standard image encodings

        This function returns an OpenCV :cpp:type:`cv::Mat` message on success, or raises
        :exc:`cv_bridge.CvBridgeError` on failure.

        If the image only has one channel, the shape has size 2 (width and height)
        """
        import cv2
        import numpy as np

        str_msg = cmprs_img_msg.data
        buf = np.ndarray(shape=(1, len(str_msg)), dtype=np.uint8, buffer=cmprs_img_msg.data)
        im = cv2.imdecode(buf, cv2.IMREAD_ANYCOLOR)

        if desired_encoding == "passthrough":
            return im

        try:
            res = cvtColor2(im, "bgr8", desired_encoding)
        except RuntimeError as e:
            raise CvBridgeError(e)

        return res

    def imgmsg_to_cv2(self, img_msg, desired_encoding="passthrough"):
        """
        Convert a sensor_msgs::Image message to an OpenCV :cpp:type:`cv::Mat`.

        :param img_msg:   A :cpp:type:`sensor_msgs::Image` message
        :param desired_encoding:  The encoding of the image data, one of the following strings:

           * ``"passthrough"``
           * one of the standard strings in sensor_msgs/image_encodings.h

        :rtype: :cpp:type:`cv::Mat`
        :raises CvBridgeError: when conversion is not possible.

        If desired_encoding is ``"passthrough"``, then the returned image has the same format as img_msg.
        Otherwise desired_encoding must be one of the standard image encodings

        This function returns an OpenCV :cpp:type:`cv::Mat` message on success, or raises
        :exc:`cv_bridge.CvBridgeError` on failure.

        If the image only has one channel, the shape has size 2 (width and height)
        """
        import numpy as np

        dtype, n_channels = self.encoding_to_dtype_with_channels(img_msg.encoding)
        dtype = np.dtype(dtype)
        dtype = dtype.newbyteorder(">" if img_msg.is_bigendian else "<")
        if n_channels == 1:
            im = np.ndarray(shape=(img_msg.height, img_msg.width), dtype=dtype, buffer=img_msg.data)
        else:
            im = np.ndarray(
                shape=(img_msg.height, img_msg.width, n_channels), dtype=dtype, buffer=img_msg.data
            )
        # If the byt order is different between the message and the system.
        if img_msg.is_bigendian == (sys.byteorder == "little"):
            im = im.byteswap().newbyteorder()

        if desired_encoding == "passthrough":
            return im

        try:
            res = cvtColor2(im, img_msg.encoding, desired_encoding)
        except RuntimeError as e:
            raise CvBridgeError(e)

        return res

    def cv2_to_compressed_imgmsg(self, cvim, dst_format="jpg"):
        """
        Convert an OpenCV :cpp:type:`cv::Mat` type to a ROS sensor_msgs::CompressedImage message.

        :param cvim:      An OpenCV :cpp:type:`cv::Mat`
        :param dst_format:  The format of the image data, one of the following strings:

           * from http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html
           * from http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#Mat
           imread(const string& filename, int flags)
           * bmp, dib
           * jpeg, jpg, jpe
           * jp2
           * png
           * pbm, pgm, ppm
           * sr, ras
           * tiff, tif

        :rtype:           A sensor_msgs.msg.CompressedImage message
        :raises CvBridgeError: when the ``cvim`` has a type that is incompatible with ``format``


        This function returns a sensor_msgs::Image message on success, or raises
        :exc:`cv_bridge.CvBridgeError` on failure.
        """
        import cv2
        import numpy as np

        if not isinstance(cvim, (np.ndarray, np.generic)):
            raise TypeError("Your input type is not a numpy array")
        cmprs_img_msg = sensor_msgs.msg.CompressedImage()
        cmprs_img_msg.format = dst_format
        ext_format = "." + dst_format
        try:
            cmprs_img_msg.data = np.array(cv2.imencode(ext_format, cvim)[1]).tostring()
        except RuntimeError as e:
            raise CvBridgeError(e)

        return cmprs_img_msg

    def cv2_to_imgmsg(self, cvim, encoding="passthrough"):
        """
        Convert an OpenCV :cpp:type:`cv::Mat` type to a ROS sensor_msgs::Image message.

        :param cvim:      An OpenCV :cpp:type:`cv::Mat`
        :param encoding:  The encoding of the image data, one of the following strings:

           * ``"passthrough"``
           * one of the standard strings in sensor_msgs/image_encodings.h

        :rtype:           A sensor_msgs.msg.Image message
        :raises CvBridgeError: when the ``cvim`` has a type that is incompatible with ``encoding``

        If encoding is ``"passthrough"``, then the message has the same encoding as the image's OpenCV type.
        Otherwise desired_encoding must be one of the standard image encodings

        This function returns a sensor_msgs::Image message on success, or raises
        :exc:`cv_bridge.CvBridgeError`on failure.
        """
        import numpy as np

        if not isinstance(cvim, (np.ndarray, np.generic)):
            raise TypeError("Your input type is not a numpy array")
        img_msg = sensor_msgs.msg.Image()
        img_msg.height = cvim.shape[0]
        img_msg.width = cvim.shape[1]
        if len(cvim.shape) < 3:
            cv_type = self.dtype_with_channels_to_cvtype2(cvim.dtype, 1)
        else:
            cv_type = self.dtype_with_channels_to_cvtype2(cvim.dtype, cvim.shape[2])
        if encoding == "passthrough":
            img_msg.encoding = cv_type
        else:
            img_msg.encoding = encoding
            # # Verify that the supplied encoding is compatible with the type of the OpenCV image
            # if self.cvtype_to_name[self.encoding_to_cvtype2(encoding)] != cv_type:
            #     raise CvBridgeError(
            #         "encoding specified as %s, but image has incompatible type %s"
            #         % (encoding, cv_type)
            #     )
        if cvim.dtype.byteorder == ">":
            img_msg.is_bigendian = True
        img_msg.data = cvim.tostring()
        img_msg.step = len(img_msg.data) // img_msg.height

        return img_msg
