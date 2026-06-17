# Room Mapper

A low-cost room-mapping system that uses distance sensing and a software pipeline to build a visual map of an area. Once a reference map is established, the system can detect changes in distance readings at any location — objects that move, disappear, or newly appear — making it useful for emergency responders who need to quickly understand an unfamiliar room layout.

## Overview

The system uses a distance sensor mounted on a pan/tilt servo mechanism to sweep a room and build a grid-based map. Readings are filtered with a Kalman filter to reduce noise and angle lag, then visualized in real time. A trained classifier can also estimate object counts from scan data.

## Features

- Pan/tilt servo sweep with distance sensing for full-room scans
- Kalman filtering (`KalmanGrid`) to smooth noisy readings and reduce angle lag
- Change detection between reference and live scans (moved/new/missing objects)
- Real-time Python visualizer for viewing scan data as a 3D map
- Stores old distance readings and compares to new ones to determine if objects have moved/been placed.

## Project Structure
