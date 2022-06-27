import os
import pickle
import re
import shutil
from typing import NamedTuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from nptyping import NDArray, Shape

from piscat.BackgroundCorrection import NoiseFloor, DifferentialRollingAverage
from piscat.Preproccessing import Normalization

# plt.style.use("master_presentation")


class ConsideredPictures(NamedTuple):
    min_idx: int
    max_idx: int


class RegionOfInterest(NamedTuple):
    x_min: int
    x_max: int
    y_min: int
    y_max: int


def repair_video(video: NDArray[Shape["Any, Any, Any"], np.int32]) -> NDArray[Shape["Any, Any, Any"], np.int32]:
    """All 0 values within the video are substituted by the mean of the surrounding four pixel values.

    It could happen, that there are pixels that have a gray value of 0. In most cases this is a result of a dead
    pixel in the camera. This error is corrected with this function in a way, that all 0 values are replaced by
    the mean value of the 4 surrounding pixels. This is done for each picture in the whole video. Function: Takes
    video with images of pixel size 128x128. All 0 values are substituted by the mean of the surrounding four
    pixel values:

    Args:
        video (NDArray: IRM image series with Shape[Number of Pictures, pixels in x-direction, pixels in y-direction]

    Returns:
        NDArray: Video with same shape as input video, without any 0 values
    """
    _, x_shape, y_shape = video.shape
    for pic, x, y in zip(*np.where(video == 0)):
        num_pixels = 4
        if x != 0:
            pix1 = video[pic, x - 1, y]
        else:
            num_pixels -= 1
            pix1 = 0
        if x != x_shape - 1:
            pix2 = video[pic, x + 1, y]
        else:
            num_pixels -= 1
            pix2 = 0
        if y != 0:
            pix3 = video[pic, x, y - 1]
        else:
            num_pixels -= 1
            pix3 = 0
        if y != y_shape - 1:
            pix4 = video[pic, x, y + 1]
        else:
            num_pixels -= 1
            pix4 = 0
        mean_value = (pix1 + pix2 + pix3 + pix4) / num_pixels
        video[pic, x, y] = mean_value
    return video


def read_images(path: str,
                considered_pictures: ConsideredPictures = None,
                region_of_interest: RegionOfInterest = None) -> NDArray[Shape["Any, Any, Any"], np.int32]:
    """All tif images in the given directory are stacked in an array and the video is returned.

    Args:
        path (str): directory path with irm images that should be considered.
        considered_pictures (ConsideredPictures): Necessary, when only a certain number of IRM pictures should be
        considered. (optional)
        region_of_interest (RegionOfInterest): Necessary, when not the whole image is of interest, but only a certain
        section of it. (optional)

    Returns:
        NDArray: Array with all pictures, Shape: [Number of IRM Pictures, Number of pixels in x-direction,
        Number of pixels in y-direction]
    """
    files = [f for f in os.listdir(path) if re.search(".tif$", f)]

    if considered_pictures is not None:
        files = files[considered_pictures.min_idx:considered_pictures.max_idx]

    pic = io.imread(path + files[0])
    video = np.ndarray((len(files), len(pic), len(pic[0])))
    for num, file in enumerate(files):
        video[num, :, :] = io.imread(path + file)

    if region_of_interest is not None:
        video = video[
                :,
                region_of_interest.x_min:region_of_interest.x_max,
                region_of_interest.y_min:region_of_interest.y_max]

    return repair_video(video)


class ImageSeries(object):
    """
    Contains all the information of one particular image series.
    """
    def __init__(self,
                 path: str,
                 fps: int,
                 batch_size: int,
                 region_of_interest: RegionOfInterest = None,
                 considered_pictures: ConsideredPictures = None,
                 ):
        """

        Args:
            path (str): directory path with irm images that should be considered.
            fps (int): Frame rate used for the IRM image series
            batch_size (int): Number of averaged images for the differential video
            region_of_interest (RegionOfInterest): Necessary, when not the whole image is of interest, but only a
            certain section of it. (optional)
            considered_pictures (ConsideredPictures): Necessary, when only a certain number of IRM pictures should be
            considered. (optional)
        """
        self.path = path
        self.fps = fps
        self.batch_size = batch_size
        self.region_of_interest = region_of_interest
        self.considered_pictures = considered_pictures
        self.raw_video = None  # Raw IRM images
        self.raw_video_pn = None  # IRM images after power normalization correction
        self.power_fluctuation = None  # Temporal fluctuations of all pixels after power normalization
        self.differential_video = None  # iSCAT Video with differential images
        self.psf = None  # Detected point spread function of all

    def get_video(self) -> None:
        """
        The IRM image series is thread in and saved to raw_video, raw_video_pn and power_fluctuation. After the first
        time calling this function it is recommended to save the arrays via the "save_videos" function. With this a
        lot of computation time is saved, when calling this function again.
        """
        if self.region_of_interest is not None:
            roi_string = f"_roi_x_{self.region_of_interest.x_min}_{self.region_of_interest.x_max}_" \
                         f"y_{self.region_of_interest.y_min}_{self.region_of_interest.y_max}"
        else:
            roi_string = ""
        if os.path.isdir(self.path + f"video{roi_string}"):
            file0 = open(self.path + f"video{roi_string}/video{roi_string}.p", "rb")
            self.raw_video = pickle.load(file0)
            file0.close()
            file1 = open(self.path + f"video{roi_string}/video_pn{roi_string}.p", "rb")
            self.raw_video_pn = pickle.load(file1)
            file1.close()
            file2 = open(self.path + f"video{roi_string}/power_fluctuation.p", "rb")
            self.power_fluctuation = pickle.load(file2)
            file2.close()

        else:
            self.raw_video = read_images(self.path, self.considered_pictures, self.region_of_interest)
            self.raw_video_pn, self.power_fluctuation = Normalization(video=self.raw_video).power_normalized()

    def create_differential_video(self,
                                  with_power_normalization: bool = True) -> None:
        """
        Differential video is created out of the raw IRM images.
        Args:
            with_power_normalization (bool): Takes the power normalized IRM images if True, otherwise it takes the raw
            IRM images without the power normalization
        """
        if with_power_normalization:
            vid = self.raw_video_pn
        else:
            vid = self.raw_video
        DRA = DifferentialRollingAverage(video=vid, batchSize=self.batch_size)
        self.differential_video, _ = DRA.differential_rolling(FPN_flag=False,
                                                              select_correction_axis='Both',
                                                              FFT_flag=False)

    def save_videos(self, saving_list: list[str], delete_raw_images: bool = False) -> None:
        """
        Saves all the arrays named in the saving list.
        Args:
            saving_list (list[str]): List with all videos that should be saved in the video folder.
                                     Possible strings are: "raw", "raw with power normalization",
                                     "power fluctuation" and "differential video".
            delete_raw_images (bool): If True all the raw IRM tif images are deleted.
        """
        if delete_raw_images:
            shutil.rmtree(self.path)
            os.makedirs(self.path)

        roi_string = f"_roi_x_{self.region_of_interest.x_min}_{self.region_of_interest.x_max}_" \
                     f"y_{self.region_of_interest.y_min}_{self.region_of_interest.y_max}"
        os.makedirs(self.path + f"video{roi_string}")

        if "raw" in saving_list:
            pickle.dump(self.raw_video, open(self.path + f"/video{roi_string}/video{roi_string}.p", "wb"))

        if "raw with power normalization" in saving_list:
            pickle.dump(self.raw_video_pn, open(self.path + f"/video{roi_string}/video_pn{roi_string}.p", "wb"))

        if "power fluctuation" in saving_list:
            pickle.dump(self.power_fluctuation, open(self.path + f"/video{roi_string}/power_fluctuation.p", "wb"))

        if "differential video" in saving_list:
            if self.differential_video is not None:
                pickle.dump(self.raw_video,
                            open(self.path + f"/video{roi_string}/differential_video_batch_size_"
                                             f"{self.batch_size}_{roi_string}.p", "wb"))
            else:
                raise TypeError("Differential video is None Type. "
                                "Most likely it is because the differential video was not yet created.")

    def get_noise_floor(self,
                        batch_sizes: list[int],
                        modes: list[int]) -> None:
        """
        The noise floor of an IRM image series is calculated depending on the number of averaged images (batch size).
        The IRM image series can be corrected beforehand by the power normalization correction or different kinds of
        fixed pattern noise correction. The different possibilities are listed in "modes".
        If more information about the different modes are necessary you can take a look into the PiSCAT module.
        The noise floor figure is saved into the subfolder Plots inside the IRM images folder.
        The noise floor data is saved into the subfolder Noise Floor inside the Plots folder.
        Args:
            batch_sizes (list[int]): From each batch size in the list the noise floor is calculated.
            modes (list[int]): There are 8 modes that can be considered (It is also possible to list more than one):
                    - 0: Without any correction
                    - 1: With power normalization correction
                    - 2: With median fixed pattern noise correction
                    - 3: With wavelet fixed pattern noise correction
                    - 4: With FFT (fast Fourier transform) fixed pattern noise correction
                    - 12: Combination of 1 and 2
                    - 13: Combination of 1 and 3
                    - 14: Combination of 1 and 4
        """
        noise_floor_dict = {0: "DRA",  # DRA: Differential Rolling Average
                            1: "DRA + PN",  # PN: Power normalization
                            2: "DRA + mFPN",  # mFPN: median fixed pattern noise correction
                            3: "DRA + wFPN",  # wFPN: waved fixed pattern noise correction
                            4: "DRA + fFPN",  # fFPN: fast Fourier Transform fixed pattern noise correction
                            12: "DRA + PN + mFPN",
                            13: "DRA + PN + wFPN",
                            14: "DRA + PN + fFPN"}

        noise_floor_names = []
        noise_floors = []
        for el in modes:
            if el == 0:
                noise_floor = NoiseFloor(self.raw_video, list_range=batch_sizes, FPN_flag=False)
            elif el == 1:
                noise_floor = NoiseFloor(self.raw_video_pn, list_range=batch_sizes, FPN_flag=False)
            elif el == 2:
                noise_floor = NoiseFloor(self.raw_video, list_range=batch_sizes, FPN_flag=True, mode_FPN="mFPN")
            elif el == 3:
                noise_floor = NoiseFloor(self.raw_video, list_range=batch_sizes, FPN_flag=True, mode_FPN="wFPN")
            elif el == 4:
                noise_floor = NoiseFloor(self.raw_video, list_range=batch_sizes, FPN_flag=True, mode_FPN="fFPN")
            elif el == 12:
                noise_floor = NoiseFloor(self.raw_video_pn, list_range=batch_sizes, FPN_flag=True, mode_FPN="mFPN")
            elif el == 13:
                noise_floor = NoiseFloor(self.raw_video_pn, list_range=batch_sizes, FPN_flag=True, mode_FPN="wFPN")
            elif el == 14:
                noise_floor = NoiseFloor(self.raw_video_pn, list_range=batch_sizes, FPN_flag=True, mode_FPN="fFPN")
            else:
                raise ValueError("One of the modes does not fit the possible modes (0, 1, 2, 3, 4, 12, 13, 14)")

            noise_floor_names.append(noise_floor_dict[el])
            noise_floors.append(noise_floor.mean)

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        ax = axs[0]
        for i in range(len(noise_floors)):
            ax.plot(batch_sizes, noise_floors[i], ".-", label=noise_floor_names[i])
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Noise floor")
        ax.set_title("Linear Plot")
        ax.legend()

        ax = axs[1]
        for i in range(len(noise_floors)):
            ax.loglog(batch_sizes, noise_floors[i], ".-", label=noise_floor_names[i])
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Noise floor")
        ax.set_title("Log-Log Plot")
        ax.legend()

        fig.tight_layout(pad=5.0)

        plot_path = self.path + "Plots"
        Path(plot_path + "/Noise Floor").mkdir(parents=True, exist_ok=True)
        pickle.dump(batch_sizes, open(plot_path + "/Noise Floor/batch_sizes.p", "wb"))
        for nf, name in zip(noise_floors, noise_floor_names):
            pickle.dump(nf, open(plot_path + f"/Noise Floor/{name.replace(' + ', '_')}.p", "wb"))
        path = plot_path + "/noise_floor.png"
        plt.show()
        fig.savefig(path, bbox_inchees="tight")

    def plot_power_fluctuation(self, plot_variants: list[int]) -> None:
        """
        Plots the power fluctuations within an IRM image series.
        4 different axes can be created:
            1: power difference vs. frame number
            2: Power spectrum vs. frequency (linear plot)
            3: power spectrum vs. frequency (semi-log y plot)
            4: power spectrum vs. frequency (log-log plot)
        It is possible to choose more than one plot. Then a figure with all the desired plots is made.
        The power fluctuation figure is saved into the subfolder Plots inside the IRM images folder.
        The data is saved in the subfolder Power Fluctuation inside the Plots folder as follows:
            The y-data of the power fluctuation in real space is saved as "power_fluctuation_real_space",
            the x-data of the power spectrum (Fourier space) is saved as "frequency_axis_fourier_space",
            the y-data of the power spectrum (Fourier space) is saved as "power_spectrum_fourier_space".
        Args:
            plot_variants (list[int]): Defines which plots are made.
        """

        n = len(self.power_fluctuation)  # Number of samples
        dt = 1 / self.fps  # time difference between two neighbor samples (in seconds)
        t = n * dt  # Total measurement time (in seconds)
        df = 1 / t  # frequency resolution
        xf = np.fft.rfft(self.power_fluctuation)
        sxx = (2 / t) * (dt ** 2) * xf * xf.conj()
        frequency_axis = np.arange(len(sxx)) * df  # frequency axis for plot

        def real_space(axis):
            axis.plot(self.power_fluctuation)
            axis.set_xlabel('Frame #')
            axis.set_ylabel(r"$P / \bar P - 1$")

        def freq_space(axis):
            axis.plot(frequency_axis, sxx)
            axis.set_xlabel("Frequency (Hz)")
            axis.set_ylabel('PSD (1/Hz)')

        def freq_space_semilog(axis):
            axis.plot(frequency_axis, 10 * np.log10(sxx / max(sxx)))
            axis.set_xlabel("Frequency (Hz)")
            axis.set_ylabel('PSD (dB)')

        def freq_space_loglog(axis):
            axis.semilogx(frequency_axis, 10 * np.log10(sxx / max(sxx)))
            axis.plot(frequency_axis, sxx)
            axis.set_xlabel("Frequency (Hz)")
            axis.set_ylabel('PSD (dB)')

        func_names = [real_space,
                      freq_space,
                      freq_space_semilog,
                      freq_space_loglog]

        if len(plot_variants) == 1:
            fig, ax = plt.subplots(1, figsize=(6, 6))
            axs = np.array([ax])
        elif len(plot_variants) == 2:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        elif len(plot_variants) in [3, 4]:
            fig, axs = plt.subplots(2, 2, figsize=(12, 12))
            if len(plot_variants) == 3:
                axs[1, 1].remove()
        else:
            raise ValueError("Parameter plot_variants has the wrong length.")

        for i, ax in enumerate(axs.flatten()[0:len(plot_variants)]):
            func_names[plot_variants[i] - 1](ax)
            if i % 2 != 0:
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()

        plot_path = self.path + "Plots"
        Path(plot_path + "/Power Fluctuation").mkdir(parents=True, exist_ok=True)
        pickle.dump(self.power_fluctuation, open(plot_path + "/Power Fluctuation/power_fluctuation_real_space.p", "wb"))
        pickle.dump(frequency_axis, open(plot_path + "/Power Fluctuation/frequency_axis_fourier_space.p", "wb"))
        pickle.dump(sxx, open(plot_path + "/Power Fluctuation/power_spectrum_fourier_space.p", "wb"))
        plt.show()
        fig.savefig(plot_path + "/power_fluctuation.png", bbox_inches="tight")
