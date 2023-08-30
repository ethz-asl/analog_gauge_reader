cmake_minimum_required(VERSION 3.0.2)
project(analog_gauge_reader)

find_package(catkin REQUIRED COMPONENTS message_generation std_msgs sensor_msgs)

catkin_python_setup()

add_message_files(DIRECTORY msg FILES GaugeReading.msg)
add_message_files(DIRECTORY msg FILES GaugeReadings.msg)
add_service_files(DIRECTORY srv FILES GaugeReader.srv)

generate_messages(
  DEPENDENCIES std_msgs sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs sensor_msgs
)

add_custom_command(
  OUTPUT .venv/pyvenv.cfg
  COMMAND POETRY_VIRTUALENVS_IN_PROJECT=true PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry install --no-ansi
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMMENT "Installing Poetry dependencies"
)

add_custom_target(venv ALL DEPENDS .venv/pyvenv.cfg)
