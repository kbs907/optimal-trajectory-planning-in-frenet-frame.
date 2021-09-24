 # Optimal trajectory planning in frenet frame

 
 ## Video
 [![Watch the video](https://img.youtube.com/vi/JRviH5MpipM/maxresdefault.jpg)](https://www.youtube.com/watch?v=JRviH5MpipM)
 [![Watch the video](https://img.youtube.com/vi/jDcz4n1qKoo/maxresdefault.jpg)](https://www.youtube.com/watch?v=jDcz4n1qKoo)
 
 ## Goal
 
1. Implement PID speed controller
2. Implement pure pursuit (or stanley, PID, MPC) as a lateral controller
3. Implement optimal frenet planning algorithm

</br>

 ## Environment
 
 * Ubuntu 16.04
 * ROS Kinetic
 * RVIZ
 
 </br>
 
 ## Limitations
 
 경로를 많이, 길게 설정 할수록 진행 속도가 느려짐 

</br>

 ## What I've learned

 1. frenet 좌표계 상에서 차량이 가능한 움직임과 충돌을 고려하여 최적 경로 생성
 2. 생성된 경로를 속도는 PID, 조향은 Stanley를 이용하여 추종 
