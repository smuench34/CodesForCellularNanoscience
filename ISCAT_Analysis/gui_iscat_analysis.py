import sys
import pandas as pd

import pyqtgraph as pg
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from read_images import ImageSeries, RegionOfInterest
from piscat.Localization import particle_localization
from piscat.Trajectory.particle_linking import Linking
from piscat.Localization import localization_filtering
from piscat.Trajectory import TemporalFilter
from piscat.Analysis import PlotProteinHistogram

import numpy as np


class PandasModel(QAbstractTableModel):

    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None

class MyCircleOverlay(pg.CircleROI):
    def __init__(self, pos, size, **args):
        pg.ROI.__init__(self, pos, size, **args)
        self.path = None
        self.aspectLocked = True


class NewMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.main_layout = QGridLayout()


class NewCustomWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.main_layout = QGridLayout()


class PostProcessingWindow(NewMainWindow):
    def __init__(self):
        super().__init__()

        self.image_series = None
        self.detected_particles = None

        # General properties
        self.setWindowTitle("Post-processing routine")

        # input widget:
        input_layout = QGridLayout()
        self.button_start = QPushButton("Start")
        self.button_clear = QPushButton("Clear all")
        self.input_path = QTextEdit("C:/Users/sandr/Desktop/Uni/Tuebingen/HiWi/Data/01_26_22/")
        self.input_choices = QComboBox()
        self.input_choices.addItems(["C:/Users/sandr/Desktop/Uni/Tuebingen/HiWi/Data/",
                                     "C:/Users/sandr/Desktop/Uni/Tuebingen/Master/Masterarbeit/Data/"])

        input_layout.addWidget(self.button_clear, 0, 0)
        input_layout.addWidget(self.button_start, 0, 1)
        input_layout.addWidget(self.input_path, 1, 0)
        input_layout.addWidget(self.input_choices, 2, 0)

        self.main_layout.addLayout(input_layout, 0, 0)

        # roi_widget:
        roi_layout = QGridLayout()
        self.roi_description = QLabel("Region of Interest")
        roi_layout.addWidget(self.roi_description, 0, 0)

        self.x_min_layout = QVBoxLayout()
        self.roi_x_min_label = QLabel("x minimum")
        self.x_min_layout.addWidget(self.roi_x_min_label)
        self.roi_x_min = QSpinBox()
        self.roi_x_min.setMaximum(200)
        self.roi_x_min.setValue(50)
        self.x_min_layout.addWidget(self.roi_x_min)
        roi_layout.addLayout(self.x_min_layout, 1, 0)

        self.x_max_layout = QVBoxLayout()
        self.roi_x_max_label = QLabel("x maximum")
        self.x_max_layout.addWidget(self.roi_x_max_label)
        self.roi_x_max = QSpinBox()
        self.roi_x_max.setMaximum(200)
        self.roi_x_max.setValue(150)
        self.x_max_layout.addWidget(self.roi_x_max)
        roi_layout.addLayout(self.x_max_layout, 1, 1)

        self.y_min_layout = QVBoxLayout()
        self.roi_y_min_label = QLabel("y minimum")
        self.y_min_layout.addWidget(self.roi_y_min_label)
        self.roi_y_min = QSpinBox()
        self.roi_y_min.setMaximum(200)
        self.roi_y_min.setValue(50)
        self.y_min_layout.addWidget(self.roi_y_min)
        roi_layout.addLayout(self.y_min_layout, 2, 0)

        self.y_max_layout = QVBoxLayout()
        self.roi_y_max_label = QLabel("y maximum")
        self.y_max_layout.addWidget(self.roi_y_max_label)
        self.roi_y_max = QSpinBox()
        self.roi_y_max.setMaximum(200)
        self.roi_y_max.setValue(150)
        self.y_max_layout.addWidget(self.roi_y_max)
        roi_layout.addLayout(self.y_max_layout, 2, 1)

        self.check_full_roi = QCheckBox()
        self.check_full_roi.setText("View full image")
        roi_layout.addWidget(self.check_full_roi, 3, 0)
        self.disable = False

        self.main_layout.addLayout(roi_layout, 1, 0)

        # Batch size widget:

        batch_layout = QVBoxLayout()

        batch_layout.addWidget(QLabel("Batch size"))

        self.batch_size = QSpinBox()
        self.batch_size.setMaximum(10000)
        self.batch_size.setValue(100)
        batch_layout.addWidget(self.batch_size)
        batch_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.main_layout.addLayout(batch_layout, 1, 1)

        # fps widget:

        fps_layout = QVBoxLayout()

        fps_layout.addWidget(QLabel("Frames per second"))

        self.fps = QSpinBox()
        self.fps.setMaximum(10000)
        self.fps.setValue(100)
        fps_layout.addWidget(self.fps)
        fps_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.main_layout.addLayout(fps_layout, 1, 2)

        main_widget = QWidget()
        main_widget.setLayout(self.main_layout)
        self.setCentralWidget(main_widget)

        # Plot Window:
        self.plot_window = NewMainWindow()
        self.plot_window.resize(1000, 1000)

        # Toolbar:
        plot_toolbar = QToolBar("Video toolbar")
        self.plot_window.addToolBar(plot_toolbar)

        button_play = QAction(QIcon("play_button.png"), "Play button", self.plot_window)
        button_play.setStatusTip("Video will run with 100 fps")
        button_play.triggered.connect(self.play_video)
        plot_toolbar.addAction(button_play)

        self.plot = pg.PlotItem()
        self.plot.setLabel(axis='left', text="y pixel position")
        self.plot.setLabel(axis='bottom', text="x pixel position")
        self.imv = pg.ImageView(view=self.plot)
        self.imv.roi.setSize(size=(1, 1))
        self.imv.ui.roiBtn.setToolTip("Shows contrast of one pixel over time. With a mouse click inside the picture the considered pixel can be changed.")
        self.imv.scene.sigMouseClicked.connect(self.set_roi_position)

        self.plot_window.main_layout.addWidget(self.imv, 0, 0)

        # Overlay parameters:
        self.overlay_parameters_layout = QGridLayout()

        self.show_overlay = QCheckBox()
        self.show_overlay.setText("Enable particle detection")
        self.overlay_parameters_layout.addWidget(self.show_overlay, 0, 0)

        self.min_sigma_label = QLabel("min_sigma")
        self.min_sigma_value = QDoubleSpinBox()
        self.min_sigma_value.setRange(0.0, 10.0)
        self.min_sigma_value.setSingleStep(0.1)
        self.min_sigma_value.setValue(2.3)
        self.min_sigma_value.setDisabled(True)
        self.overlay_parameters_layout.addWidget(self.min_sigma_label, 1, 0)
        self.overlay_parameters_layout.addWidget(self.min_sigma_value, 2, 0)

        self.max_sigma_label = QLabel("max_sigma")
        self.max_sigma_value = QDoubleSpinBox()
        self.max_sigma_value.setRange(0.0, 10.0)
        self.max_sigma_value.setSingleStep(0.1)
        self.max_sigma_value.setValue(2.3)
        self.max_sigma_value.setDisabled(True)
        self.overlay_parameters_layout.addWidget(self.max_sigma_label, 1, 1)
        self.overlay_parameters_layout.addWidget(self.max_sigma_value, 2, 1)

        self.sigma_ratio_label = QLabel("sigma_ratio")
        self.sigma_ratio_value = QDoubleSpinBox()
        self.sigma_ratio_value.setRange(0.0, 10.0)
        self.sigma_ratio_value.setSingleStep(0.1)
        self.sigma_ratio_value.setValue(4.5)
        self.sigma_ratio_value.setDisabled(True)
        self.overlay_parameters_layout.addWidget(self.sigma_ratio_label, 3, 0)
        self.overlay_parameters_layout.addWidget(self.sigma_ratio_value, 4, 0)

        self.threshold_label = QLabel("threshold")
        self.threshold_value = QDoubleSpinBox()
        self.threshold_value.setDecimals(4)
        self.threshold_value.setRange(0.0, 0.1)
        self.threshold_value.setSingleStep(0.0001)
        self.threshold_value.setValue(0.0004)
        self.threshold_value.setDisabled(True)
        self.overlay_parameters_layout.addWidget(self.threshold_label, 3, 1)
        self.overlay_parameters_layout.addWidget(self.threshold_value, 4, 1)

        self.button_set_new_params = QPushButton()
        self.button_set_new_params.setText("Search for particles")
        self.button_set_new_params.setToolTip("Given parameters will be used to detect particles.")
        self.button_set_new_params.setDisabled(True)
        self.overlay_parameters_layout.addWidget(self.button_set_new_params, 5, 0)

        self.plot_window.main_layout.addLayout(self.overlay_parameters_layout, 1, 0)

        self.button_histogram = QPushButton()
        self.button_histogram.setText("Get histogram")
        self.button_histogram.setDisabled(True)
        self.plot_window.main_layout.addWidget(self.button_histogram, 2, 0)

        # Show plot window:

        plot_main_widget = QWidget()
        plot_main_widget.setLayout(self.plot_window.main_layout)
        self.plot_window.setCentralWidget(plot_main_widget)

        # Actions:

        self.button_start.clicked.connect(self.start_button_clicked)
        self.show_overlay.stateChanged.connect(self.show_overlay_stateChanged)
        self.check_full_roi.stateChanged.connect(self.change_roi_read)
        self.button_set_new_params.clicked.connect(self.set_new_params)
        self.imv.sigTimeChanged.connect(self.show_particle_locations)
        self.button_histogram.clicked.connect(self.get_histogram)

    def set_roi_position(self, evt):
        if self.imv.ui.roiBtn.isChecked():
            x = self.imv.getImageItem().mapFromScene(evt.scenePos()).x()
            y = self.imv.getImageItem().mapFromScene(evt.scenePos()).y()
            self.imv.roi.setSize(size=(1, 1))
            self.imv.roi.setPos(pos=(int(x), int(y)))

    def start_button_clicked(self):
        if self.check_full_roi:
            self.image_series = ImageSeries(path=self.input_path.toPlainText(),
                                       fps=self.fps.value(),
                                       batch_size=self.batch_size.value())
        else:
            roi = ((self.roi_x_min.value(), self.roi_x_max.value()), (self.roi_y_min.value(), self.roi_y_max.value()))
            self.image_series = ImageSeries(path=self.input_path.toPlainText(),
                                       fps=self.fps.value(),
                                       batch_size=self.batch_size.value(),
                                       region_of_interest=RegionOfInterest(*roi))
        self.image_series.get_video()
        self.image_series.create_differential_video()

        # Transpose is necessary to set coordinate origin to the upper left and x-direction to the right
        # and y-direction to the left.
        self.imv.setImage(self.image_series.differential_video.transpose(0, 2, 1))
        cmap = pg.colormap.getFromMatplotlib("gray")
        self.imv.setColorMap(cmap)

        self.plot_window.show()

    def show_particle_locations(self):
        if len(self.plot.items) > 3:
            for el in self.plot.items[3:]:
                self.plot.removeItem(el)
        if self.show_overlay.isChecked() and self.detected_particles is not None:
            current_frame_num = self.imv.currentIndex
            frame_df = self.detected_particles.loc[self.detected_particles["frame"] == current_frame_num]
            pen = QPen(Qt.red, 0.1)
            for x, y, sigma in zip(frame_df.loc[:, "x"], frame_df.loc[:, "y"], frame_df.loc[:, "sigma"]):
                self.imv.addItem(MyCircleOverlay(pos=(x - sigma, y - sigma), size=(2 * sigma), pen=pen, movable=False))

            self.imv.setCurrentIndex(current_frame_num)

    def show_overlay_stateChanged(self):
        if self.show_overlay.isChecked():
            self.min_sigma_value.setDisabled(False)
            self.max_sigma_value.setDisabled(False)
            self.sigma_ratio_value.setDisabled(False)
            self.threshold_value.setDisabled(False)
            self.button_set_new_params.setDisabled(False)
        else:
            self.min_sigma_value.setDisabled(True)
            self.max_sigma_value.setDisabled(True)
            self.sigma_ratio_value.setDisabled(True)
            self.threshold_value.setDisabled(True)
            self.button_set_new_params.setDisabled(True)

        self.show_particle_locations()

    def set_new_params(self):
        self.button_histogram.setDisabled(False)
        self.PSF = particle_localization.PSFsExtraction(video=self.image_series.differential_video)
        self.detected_particles = self.PSF.psf_detection(function='dog', overlap=0, mode='BOTH',
                                                            min_sigma=self.min_sigma_value.value(),
                                                            max_sigma=self.max_sigma_value.value(),
                                                            sigma_ratio=self.sigma_ratio_value.value(),
                                                            threshold=self.threshold_value.value())
        self.show_particle_locations()

    def get_histogram(self):
        PSFs = self.PSF.fit_Gaussian2D_wrapper(PSF_List=self.detected_particles, scale=5, internal_parallel_flag=True)
        linking_ = Linking()
        linked_PSFs = linking_.create_link(psf_position=PSFs,
                                           search_range=2,  # The maximum distance features can move between frames
                                           memory=10)  # The maximum number of frames during which a feature can vanish, then reappear nearby, and be considered the same particle.

        spatial_filters = localization_filtering.SpatialFilter()
        PSFs_filtered = spatial_filters.outlier_frames(linked_PSFs,
                                                       threshold=20)  # images with more detected particles than the threshold are deleted
        PSFs_filtered = spatial_filters.dense_PSFs(PSFs_filtered,
                                                   threshold=10)  # particles that have a short distance to each other are deleted
        PSFs_filtered = spatial_filters.symmetric_PSFs(PSFs_filtered,
                                                       threshold=0.5)  # particles with non-symmetric shape are deleted

        t_filters = TemporalFilter(video=self.image_series.differential_video, batchSize=self.image_series.batch_size)
        all_trajectories, linked_PSFs_filter, his_all_particles = t_filters.v_trajectory(df_PSFs=PSFs_filtered,
                                                                                         threshold=70)  # particles that are present in less images than the threshold value are deleted
        print(f"Number of Particles before filtering: {linking_.trajectory_counter(linked_PSFs)}")
        print(f"Number of Particles after filtering: {len(all_trajectories)}")

        his = PlotProteinHistogram(intersection_display_flag=True, imgSizex=10, imgSizey=5)

        valid_ones = []
        for i in range(len(all_trajectories)):
            number = "#" + str(i)
            try:
                his.plot_contrast_extraction(particles=all_trajectories,
                                             batch_size=self.image_series.batch_size,
                                             video_frame_num=self.image_series.differential_video.shape[0],
                                             MinPeakWidth=100,
                                             MinPeakProminence=0,
                                             pixel_size=0.66,
                                             particles_num=number)
                valid_ones.append(i)
            except:
                pass

        correct_trajectories = [t for i, t in enumerate(all_trajectories) if i in valid_ones]

        his2 = PlotProteinHistogram(intersection_display_flag=True, imgSizex=10, imgSizey=5)
        his2(folder_name='',
             particles=correct_trajectories,
             batch_size=self.image_series.batch_size,
             video_frame_num=self.image_series.differential_video.shape[0],
             MinPeakWidth=90,
             pixel_size=0.66,
             MinPeakProminence=0)

        numbered_particles = ["# " + str(i) for i in range(len(his2.t_contrast_peaks))]
        df = pd.DataFrame(
            [numbered_particles, his2.t_contrast_peaks, his2.t_contrast_intersection, his2.t_contrast_Prominence])
        df_transposed = df.T
        df_transposed.columns = ['Particles', 'Peak', 'Intersection', 'Prominence']
        df_transposed.style.hide_index()
        print(df_transposed)

        model = PandasModel(df_transposed)
        self.view = QTableView()
        self.view.setModel(model)
        self.view.resize(800, 600)
        self.view.show()

    def play_video(self):
        if self.imv.currentIndex == len(self.image_series.differential_video) - 1:
            self.imv.setCurrentIndex(0)
        self.imv.play(rate=100)

    def change_roi_read(self):
        self.roi_x_min.setDisabled(not self.disable)
        self.roi_x_max.setDisabled(not self.disable)
        self.roi_y_min.setDisabled(not self.disable)
        self.roi_y_max.setDisabled(not self.disable)
        self.disable = not self.disable


# driver code
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = PostProcessingWindow()
    w.show()
    app.exec_()

