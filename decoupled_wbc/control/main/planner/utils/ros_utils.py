"""Planner-local ROS helpers that are not shared with the rest of the stack."""

from __future__ import annotations

import base64
from typing import Callable, Optional

import msgpack
import msgpack_numpy as mnp
import rclpy
from diagnostic_msgs.srv import AddDiagnostics
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from decoupled_wbc.control.utils.ros_utils import ROSManager


def encode_dict(data: dict) -> str:
    packed = msgpack.packb(data, default=mnp.encode)
    return base64.b64encode(packed).decode("ascii")


def decode_dict(message: str) -> dict:
    decoded = base64.b64decode(message.encode("ascii"))
    return msgpack.unpackb(decoded, object_hook=mnp.decode)


class ROSDictServiceServer:
    """
    Dict-in / dict-out ROS2 service for planner requests.

    Uses diagnostic_msgs/AddDiagnostics as a generic string transport (request string +
    success/message response), matching the msgpack+base64 convention used elsewhere.
    """

    def __init__(self, service_name: str, handler: Callable[[dict], dict]):
        ros_manager = ROSManager()
        self.node = ros_manager.node
        self.handler = handler
        self.server = self.node.create_service(AddDiagnostics, service_name, self._callback)

    def _callback(self, request, response):
        try:
            request_dict = decode_dict(request.load_namespace) if request.load_namespace else {}
            result = self.handler(request_dict)
            response.success = True
            response.message = encode_dict(result if result is not None else {})
        except Exception as e:
            response.success = False
            response.message = str(e)
        return response


class ROSDictServiceClient(Node):
    """Client counterpart to ROSDictServiceServer."""

    def __init__(
        self,
        service_name: str,
        node_name: str = "planner_dict_service_client",
        timeout_sec: float = 60.0,
    ):
        super().__init__(node_name)
        self.timeout_sec = timeout_sec
        self.cli = self.create_client(AddDiagnostics, service_name)
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Waiting for service '{service_name}'...")

    def call(self, request: dict, timeout_sec: Optional[float] = None) -> dict:
        req = AddDiagnostics.Request()
        req.load_namespace = encode_dict(request)
        future = self.cli.call_async(req)
        timeout = self.timeout_sec if timeout_sec is None else timeout_sec
        executor = SingleThreadedExecutor()
        executor.add_node(self)
        executor.spin_until_future_complete(future, timeout_sec=timeout)
        executor.remove_node(self)
        executor.shutdown()
        if not future.done():
            raise TimeoutError(f"Service call timed out after {timeout}s")
        result = future.result()
        if result is None:
            raise RuntimeError("Service call returned no result")
        if not result.success:
            raise RuntimeError(result.message)
        return decode_dict(result.message) if result.message else {}
