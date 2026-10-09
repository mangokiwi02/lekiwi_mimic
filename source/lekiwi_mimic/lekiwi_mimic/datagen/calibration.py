"""Constants measured with scripts/inspect_robot.py on lekiwi.usd (LeIsaac assets, HF LightwheelAI/leisaac_env)."""

# gripper-link orientation (wxyz, base frame) for a top-down grasp with the
# arm in the shoulder_pan = 0 plane
DOWN_QUAT = (0.6865, -0.1648, 0.1412, -0.6940)
# grasp point (LeIsaac jaw detection frame, gripper closed) relative to the gripper link, in the gripper-link frame (m)
TCP_OFFSET = (0.0030, 0.0446, -0.0881)

# unit vector from the wrist to the grasp point, in the gripper-link / TCP frame (tool approach axis)
APPROACH_AXIS = (0.0190, 0.3247, -0.9456)

# grasp-point correction found with scripts/sweep_sm.py (10/10 lifts): the moving jaw pushes the cube onto the fixed
# finger, so aim 1.5 cm lower and 1.2 cm closer to the shoulder than the calibrated grasp point
GRASP_HEIGHT = -0.015
FORWARD_OFFSET = -0.012

SM_KWARGS = dict(
    down_quat=DOWN_QUAT,
    tcp_offset=TCP_OFFSET,
    approach_axis=APPROACH_AXIS,
    grasp_height=GRASP_HEIGHT,
    forward_offset=FORWARD_OFFSET,
)
