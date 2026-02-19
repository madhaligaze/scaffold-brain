# CoordinateFrames (FrameSpec v1.0)

This document defines all coordinate frames used by the FramePacket contract and by the reconstruction / planning pipeline.

If any producer (Android) sends frames that do not match this spec, geometry will drift or be scaled/rotated incorrectly.

## Conventions

- Units: meters (m)
- Right-handed coordinate system everywhere.
- Rotation representation in the FramePacket: quaternion **[x, y, z, w]** (xyzw).
- Pose in the FramePacket is **camera -> world** (T_world_from_camera).

## Frames

### camera frame (C)

Attached to the device camera optical center.

- +X: right in the image
- +Y: down in the image
- +Z: forward from the camera (into the scene)

Depth values are measured along +Z in this frame.

### world frame (W)

A stable frame for a scanning session.

- Origin: defined by the first accepted FramePacket in the session.
- Orientation: inherited from the device tracking at session start.
- Units: meters.

All reconstructed geometry (TSDF, Occupancy, ESDF) lives in W.

The FramePacket pose defines:

T_world_from_camera =
| R  t |
| 0  1 |

where t is camera position in W, R is camera orientation in W.

### gravity-aligned frame (G)

A convenience frame derived from W using the gravity vector.

- Origin: same as W
- +Z_G: opposite to gravity (up)
- +X_G, +Y_G: form a right-handed basis in the horizontal plane

Android SHOULD provide a stable gravity vector. If not provided, backend may approximate using dominant plane.

This frame is used for:
- "upright" bounding boxes / cylinders (optional)
- NBV sampling constraints (avoid impossible camera tilts)
- scaffold rules that assume vertical uprights

### target_box frame (B)

A local frame used to define the region-of-interest (ROI).

- Origin: center of the target box in W.
- Axes: aligned with gravity-aligned frame (G) by default.
- Extents: half-sizes in meters.

This frame is produced from user inputs (anchors / boundary points) and is used by:
- readiness score
- next-best-view candidate generation
- unknown-space policy
- scaffold planner scope

## Required fields in FramePacketMeta

- intrinsics: fx, fy, cx, cy, width, height (pixels)
- pose: position [tx, ty, tz] and quaternion [x, y, z, w]
- timestamp: monotonic within session

## Notes on pointcloud_meta.frame

If a point cloud is provided:

- frame="camera": xyz points are in camera frame (C) and must be transformed by pose into W.
- frame="world": xyz points are already in W. Pose is still required for depth integration and consistency checks.
