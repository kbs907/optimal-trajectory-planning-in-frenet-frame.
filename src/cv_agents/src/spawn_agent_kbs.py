#!/usr/bin/python
#-*- coding: utf-8 -*-

import rospy
import numpy as np
import math
import tf

import rospkg
import sys

from scipy.interpolate import interp1d

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Quaternion, Point
from object_msgs.msg import Object

from stanley import *
from optimal_trajectory_Frenet import *

import pickle
import argparse

rospack = rospkg.RosPack()
path = rospack.get_path("map_server")

rn_id = dict()

rn_id[5] = {
    'left': [18, 2, 11, 6, 13, 8, 15, 10, 26, 0]  # ego route
}

def pi_2_pi(angle):
	return (angle + math.pi) % (2 * math.pi) - math.pi


def interpolate_waypoints(wx, wy, space=0.5):
	_s = 0
	s = [0]
	for i in range(1, len(wx)):
		prev_x = wx[i - 1]
		prev_y = wy[i - 1]
		x = wx[i]
		y = wy[i]

		dx = x - prev_x
		dy = y - prev_y

		_s = np.hypot(dx, dy)
		s.append(s[-1] + _s)

	fx = interp1d(s, wx)
	fy = interp1d(s, wy)
	ss = np.linspace(0, s[-1], num=int(s[-1] / space) + 1, endpoint=True)

	dxds = np.gradient(fx(ss), ss, edge_order=1)
	dyds = np.gradient(fy(ss), ss, edge_order=1)
	wyaw = np.arctan2(dyds, dxds)

	return {
		"x": fx(ss),
		"y": fy(ss),
		"yaw": wyaw,
		"s": ss
}


class State:

	def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, dt=0.1, WB=2.6):
		self.x = x
		self.y = y
		self.yaw = yaw
		self.v = v
		self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
		self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))
		self.dt = dt
		self.WB = WB

	def update(self, a, delta):
		dt = self.dt
		WB = self.WB

		self.x += self.v * math.cos(self.yaw) * dt
		self.y += self.v * math.sin(self.yaw) * dt
		self.yaw += self.v / WB * math.tan(delta) * dt
		self.yaw = pi_2_pi(self.yaw)
		self.v += a * dt
		self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
		self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))

def calc_distance(self, point_x, point_y):
	dx = self.rear_x - point_x
	dy = self.rear_y - point_y
	return math.hypot(dx, dy)


def get_ros_msg(x, y, yaw, v, id):
	quat = tf.transformations.quaternion_from_euler(0, 0, yaw)

	m = Marker()
	m.header.frame_id = "/map"
	m.header.stamp = rospy.Time.now()
	m.id = id
	m.type = m.CUBE

	m.pose.position.x = x + 1.3 * math.cos(yaw)
	m.pose.position.y = y + 1.3 * math.sin(yaw)
	m.pose.position.z = 0.75
	m.pose.orientation = Quaternion(*quat)

	m.scale.x = 4.475
	m.scale.y = 1.850
	m.scale.z = 1.645

	m.color.r = 93 / 255.0
	m.color.g = 122 / 255.0
	m.color.b = 177 / 255.0
	m.color.a = 0.97

	o = Object()
	o.header.frame_id = "/map"
	o.header.stamp = rospy.Time.now()
	o.id = id
	o.classification = o.CLASSIFICATION_CAR
	o.x = x
	o.y = y
	o.yaw = yaw
	o.v = v
	o.L = m.scale.x
	o.W = m.scale.y

	return {
        "object_msg": o,
        "marker_msg": m,
        "quaternion": quat
    }

def get_path_msg(x, y):
	quat = tf.transformations.quaternion_from_euler(0, 0, yaw)

	m = Marker()
	m.header.frame_id = "/map"
	m.header.stamp = rospy.Time.now()
	m.type = m.LINE_STRIP
	m.action = m.ADD

	m.pose.position.x = 0.0
	m.pose.position.y = 0.0
	m.pose.position.z = 0.0

	m.pose.orientation.x = 0.0
	m.pose.orientation.y = 0.0
	m.pose.orientation.z = 0.0
	m.pose.orientation.w = 1.0

	m.scale.x = 0.5
	m.scale.y = 0.5
	m.scale.z = 0.5

	m.color.r = 1.0
	m.color.g = 0.0
	m.color.b = 0.0
	m.color.a = 1.0
	
	m.points = []

	for i in range(len(x)) :
		line_point = Point()
		line_point.x = x[i]
		line_point.y = y[i]
		line_point.z = 0.0
		m.points.append(line_point)

	f_path_pub.publish(m)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Spawn a CV agent')

	parser.add_argument("--id", "-i", type=int, help="agent id", default=1)
	parser.add_argument("--route", "-r", type=int,
                   help="start index in road network. select in [1, 3, 5, 10]", default=5)
	parser.add_argument("--dir", "-d", type=str, default="left", help="direction to go: [left, straight, right]")
	args, unknown = parser.parse_known_args()

	rospy.init_node("three_cv_agents_node_" + str(args.id))

	id = args.id
	tf_broadcaster = tf.TransformBroadcaster()
	marker_pub = rospy.Publisher("/objects/marker/car_" + str(id), Marker, queue_size=1)
	object_pub = rospy.Publisher("/objects/car_" + str(id), Object, queue_size=1)
	f_path_pub = rospy.Publisher("visualization_marker", Marker, queue_size =1)
	start_node_id = args.route
	route_id_list = [start_node_id] + rn_id[start_node_id][args.dir]

	ind = 100

	with open(path + "/src/route.pkl", "rb") as f:
		nodes = pickle.load(f)

	wx = []
	wy = []
	wyaw = []
	for _id in route_id_list:
		wx.append(nodes[_id]["x"][1:])
		wy.append(nodes[_id]["y"][1:])
		wyaw.append(nodes[_id]["yaw"][1:])
	wx = np.concatenate(wx)
	wy = np.concatenate(wy)
	wyaw = np.concatenate(wyaw)
	waypoints = {"x": wx, "y": wy, "yaw": wyaw}
	
	#wx = mapx, wy = mapy
	# static obstacles
	obs = np.array([[45.4, 31.7],
					[25.578, -9.773]
					])
	for i in range(2) :
		obs[i][0],obs[i][1] = get_frenet(obs[i][0],obs[i][1],wx,wy)
    # get maps
	maps = np.zeros(wx.shape)
	for i in range(len(wx)-1):
		x = wx[i]
		y = wy[i]
		sd = get_frenet(x, y, wx, wy)
		maps[i] = sd[0]
	
	# get global position info. of static obstacles
	obs_global = np.zeros(obs.shape)
	for i in range(len(obs[:,0])):
		_s = obs[i,0]
		_d = obs[i,1]
		xy = get_cartesian(_s, _d, wx, wy, maps)
		obs_global[i] = xy[:-1]
	
	# 자챠량 관련 initial condition
	x = -LANE_WIDTH
	y = 0
	yaw = 90 * np.pi/180
	v = 0.5
	a = 0

	s, d = get_frenet(x, y, wx, wy);
	x, y, yaw_road = get_cartesian(s, d, wx, wy, maps)
	yawi = yaw - yaw_road

    # s 방향 초기조건
	si = s
	si_d = v*np.cos(yawi)
	si_dd = a*np.cos(yawi)
	sf_d = TARGET_SPEED
	sf_dd = 0

    # d 방향 초기조건
	di = d
	di_d = v*np.sin(yawi)
	di_dd = a*np.sin(yawi)
	df_d = 0
	df_dd = 0

	opt_d = di
	
	target_speed = 20.0 / 3.6
	state = State(x=waypoints["x"][ind], y=waypoints["y"][ind], yaw=waypoints["yaw"][ind], v=0.1, dt=0.01)
	#state = State(x, y, yaw_road, v=0.1, dt=0.01)
	r = rospy.Rate(100)
	while not rospy.is_shutdown():
		path, opt_ind = frenet_optimal_planning(si, si_d, si_dd,
                                                sf_d, sf_dd, di, di_d, di_dd, df_d, df_dd, obs, wx, wy, maps, opt_d)
		opt_d = path[opt_ind].d[-1]
        # generate acceleration ai, and steering sdi
        # YOUR CODE HERE
		ai = 0.5*(target_speed-state.v)
		sdi = stanley_control(state.x, state.y, state.yaw, state.v, path[opt_ind].x, path[opt_ind].y, path[opt_ind].yaw)

        # update state with acc, delta
		state.update(ai, sdi)
		# update state with si,di
		s, d = get_frenet(state.x, state.y, wx, wy);
		x, y, yaw_road = get_cartesian(s, d, wx, wy, maps)
		yawi = yaw - yaw_road
		si = s
		si_d = v*np.cos(yawi)
		si_dd = a*np.cos(yawi)
		di = d
		di_d = v*np.sin(yawi)
		di_dd = a*np.sin(yawi)
        # vehicle state --> topic msg
		msg = get_ros_msg(state.x, state.y, state.yaw, state.v, id=id)
		# frenet path --> topic msg
		print(len(path[opt_ind].x))
		get_path_msg(path[opt_ind].x, path[opt_ind].y)
        # send tf
		tf_broadcaster.sendTransform(
            (state.x, state.y, 1.5),
            msg["quaternion"],
            rospy.Time.now(),
            "/car_" + str(id), "/map"
        )

        # publish vehicle state in ros msg
		object_pub.publish(msg["object_msg"])

		r.sleep()
