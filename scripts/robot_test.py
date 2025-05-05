from Denso_robot.DensoRobotControl import DensoRobotControl
robot = DensoRobotControl(port="/dev/ttyUSB0", baud_rate=19200)
import time
from utils.cvfunc import camera2robot
from utils.transforms import rotation2Euler, rotation2Euler_scale
home_robot = [405, -70, 515.33, 180, 0.0, 180]
posture = [-37.93,-25.38,-324.44,-177.4,0.62,104.69]
pose_robot_inv = camera2robot(posture , home_robot)
pose_robot_euler_inv = rotation2Euler(pose_robot_inv)
print("Pose robot inv", pose_robot_euler_inv)
if robot.isConnected():
    robot.setTimeout(msg_time_ms=100, pos_time_ms=1500000)
    input("enter")
    robot.moveLine(pose=home_robot, speed=15, tool=3,verbose=False)
    time.sleep(1)

