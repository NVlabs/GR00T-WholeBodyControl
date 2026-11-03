# Camera Image Notes

## Raw depth image

`depth_raw.png` contains the original 16-bit depth measurement for each
pixel.

- Use it to calculate distances, generate point clouds, or perform robot
  perception.
- A value of `0` means that no valid depth measurement was available.
- Convert a pixel value to metres using the camera's depth scale:

  ```python
  distance_meters = raw_depth_value * depth_scale
  ```

- For example, if the depth scale is `0.001`, a raw value of `1176`
  represents approximately `1.176 metres`.
- It can appear black in ordinary image viewers because its values occupy
  only a small part of the full 16-bit display range.

## Grayscale depth image

`depth_gray.jpg` is an 8-bit visualization generated from
`depth_raw.png`.

- Its depth values are rescaled to the display range `0–255`.
- Near and far values are clipped to improve visible contrast.
- JPEG compression can alter pixel values.
- It is useful for visually inspecting the depth image.
- It does **not** preserve the original distance measurements and should
  not be used to calculate distance.

Use `depth_raw.png` for measurements and `depth_gray.jpg` only for
viewing.
