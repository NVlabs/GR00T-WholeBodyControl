#!/usr/bin/env python3
"""
RealSense Camera Streaming Client
Receives and displays camera stream from the G1 robot

on the g1 run : 
source /opt/ros/noetic/setup.bash && roslaunch realsense2_camera rs_camera.launch

and in a separate terminal run (still on the g1) :
source /opt/ros/noetic/setup.bash && python3 /tmp/camera_stream_server.py

note - if the port is still held and in use run pkill -9 -f camera_stream to kill the lock and run the above again

on your desktop run : python3 camera_stream_client.py 192.168.123.164
"""
import cv2
import socket
import pickle
import struct
import sys

class CameraStreamClient:
    def __init__(self, host, port=5000):
        self.host = host
        self.port = port
        self.socket = None
        
    def connect(self):
        """Connect to the streaming server"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            print(f"Connecting to {self.host}:{self.port}...")
            self.socket.connect((self.host, self.port))
            print("Connected!")
        except ConnectionRefusedError:
            print(f"Error: Could not connect to {self.host}:{self.port}")
            sys.exit(1)
            
    def receive_frames(self):
        """Receive and display frames from the server"""
        data = b""
        payload_size = struct.calcsize("Q")
        
        while True:
            # Receive frame size
            while len(data) < payload_size:
                packet = self.socket.recv(4*1024)
                if not packet:
                    print("Connection closed by server")
                    return
                data += packet
                
            # Extract frame size
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q", packed_msg_size)[0]
            
            # Receive frame data
            while len(data) < msg_size:
                data += self.socket.recv(4*1024)
                
            frame_data = data[:msg_size]
            data = data[msg_size:]
            
            # Deserialize and decode frame
            frame = pickle.loads(frame_data)
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            
            # Display frame
            cv2.imshow('G1 Camera Stream', frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    def stop(self):
        """Close connection"""
        if self.socket:
            self.socket.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 camera_stream_client.py <G1_IP_ADDRESS>")
        print("Example: python3 camera_stream_client.py 192.168.123.164")
        sys.exit(1)
        
    host = sys.argv[1]
    client = CameraStreamClient(host=host, port=5000)
    
    try:
        client.connect()
        client.receive_frames()
    except KeyboardInterrupt:
        print("\nDisconnecting...")
    finally:
        client.stop()
