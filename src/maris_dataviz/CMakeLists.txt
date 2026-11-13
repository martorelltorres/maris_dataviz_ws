cmake_minimum_required(VERSION 3.0.2)
project(maris_dataviz)

find_package(catkin REQUIRED)

catkin_python_setup()
catkin_package()

catkin_install_python(PROGRAMS
  scripts/play_mission.py
  scripts/nav_relay.py
  scripts/mission_viz.py
  scripts/camera_projector.py
  scripts/sss_projector.py
  scripts/multibeam_mapper.py
  scripts/satellite_layer.py
  scripts/textured_bathymetry.py
  scripts/rviz_profile.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(PROGRAMS scripts/record_mission.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
