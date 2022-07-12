from read_images import ImageSeries, ConsideredPictures, RegionOfInterest
import numpy as np

irm_image_path = "C:/Users/sandr/Desktop/Uni/Tuebingen/HiWi/Data/01_26_22/"
considered_pics = ConsideredPictures(min_idx=0, max_idx=500)
roi = RegionOfInterest(x_min=60, x_max=90, y_min=60, y_max=90)
image_series = ImageSeries(path=irm_image_path,
                           fps=100,
                           batch_size=100,
                           region_of_interest=roi, # region_of_interest=roi or region_of_interest=None
                           considered_pictures=considered_pics) # considered_pictures=None or considered_pictures=considered_pics
image_series.get_video()

first_batch = image_series.raw_video[104:204]
second_batch = image_series.raw_video[204:304]
first_batch_average = np.mean(first_batch, axis=0)
second_batch_average = np.mean(second_batch, axis=0)

overall_average = (first_batch_average + second_batch_average) / 2

first_differential_picture_own = second_batch_average - first_batch_average
first_differential_picture_own_normalized = first_differential_picture_own / overall_average

image_series.create_differential_video(with_power_normalization=False)
first_differential_picture_piscat = image_series.differential_video[104]

# Compare both processes:
print(f"Minimum: {np.min(first_differential_picture_piscat)}")
A = first_differential_picture_piscat
np.unravel_index(A.argmin(), A.shape)
print(f"ArgMinimum: {np.unravel_index(A.argmin(), A.shape)}")

print(f"Differential picture made by my own: {first_differential_picture_own_normalized[18, 15]}")
print(f"Differential picture made by PiSCAT: {first_differential_picture_piscat[18, 15]}")

print(f"Ratio: {first_differential_picture_own_normalized[18, 15] / first_differential_picture_piscat[18, 15]}")
