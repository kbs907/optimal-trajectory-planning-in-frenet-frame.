## Programmers project - Do not collide!

```
(terminal1)
roscore

(t2 visualize map)
rosrun map_server map_visualizer.py

(t3 run rviz)
rviz rviz -d rviz.config.rviz

(t4 - generate parking obstacles)
rosrun obstacles parking_car.py

(t5 - collision check algorithm + visualize cars from topics)
rosrun obstacles collision_check_with_visualizing_car.py

(t6 -  spawn the autonomous agent)
rosrun cv_agents spawn_agent.py

```


### Guideline
1. Implement PID speed controller
속도 제어는 pid 제어 중 p게인만을 사용하여 제어를 했습니다.
2. Implement pure pursuit (or stanley, PID, MPC) as a lateral controller
조향각 제어는 stanley기법을 이용하였습니다.
3. Implement optimal frenet planning algorithm
frenet planning을 적용하는데, 계획 경로를 rviz에 표현하기 위해 MAX_T를 늘려봤지만 문제가 생겨서
target_speed를 늘리는 방향으로 적용했습니다.
그리고 경로상 충돌 예상시 경로 설정을 위해 
LANE_WIDTH = 3.0
COL_CHECK = 2.0
으로 설정을 했고
경로는-LANE_WIDTH/2 ~ LANE_WIDTH/2 범위를 7등분한 값을 d로 설정하여 생성하였습니다.
그리고 장애물 회피 후에 원래 목표로 한 경로로 되돌아가기 위하여 d자체를 cost로 설정하였고
이전 경로계획과의 부드러운 움직임을 위하여 K_D를 100으로 설정했습니다.
