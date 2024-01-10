from math import cos, radians

# Constants
scalex = 0.02  # in meters/pixel
scaley = 0.02  # in meters/pixel
R = 6371000.0  # Earth radius in meters
latitude = 39.533687  # center latitude in decimal degrees
longitude = 2.590453  # center longitude in decimal degrees

# Calculations
degrees_per_pixel_x = scalex / (R * cos(radians(latitude)))
degrees_per_pixel_y = scaley / R
degrees_per_pixel_longitude = degrees_per_pixel_x / cos(radians(latitude))

# Output
print("Degrees per pixel in x direction:", degrees_per_pixel_x)
print("Degrees per pixel in y direction:", degrees_per_pixel_y)
print("Degrees per pixel in longitude direction:", degrees_per_pixel_longitude)

# Calculate the half-width and half-height of the image in degrees
half_width_deg = 0.5 * scalex / (R * cos(radians(latitude)))
half_height_deg = 0.5 * scaley / R

# Calculate the upper-left and lower-right corners
upper_left_x = longitude - half_width_deg
upper_left_y = latitude + half_height_deg
lower_right_x = longitude + half_width_deg
lower_right_y = latitude - half_height_deg

# Output the coordinates of the four corners
print("Upper-Left Corner:", upper_left_x, upper_left_y)
print("Upper-Right Corner:", lower_right_x, upper_left_y)
print("Lower-Left Corner:", upper_left_x, lower_right_y)
print("Lower-Right Corner:", lower_right_x, lower_right_y)
