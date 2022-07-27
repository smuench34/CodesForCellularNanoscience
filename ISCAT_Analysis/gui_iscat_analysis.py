import sys

import pandas as pd
from pathlib import Path

import pyqtgraph as pg
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from read_images import ImageSeries, RegionOfInterest, ConsideredPictures
from read_images import PlotProteinHistogram
from piscat.Localization import particle_localization
from read_images import dog_own
from piscat.Trajectory.particle_linking import Linking
from piscat.Localization import localization_filtering
from piscat.Trajectory import TemporalFilter

import numpy as np

class PandasModel(QAbstractTableModel):
    def __init__(self, df = pd.DataFrame(), parent=None):
        QAbstractTableModel.__init__(self, parent=parent)
        self._df = df.copy()

    def toDataFrame(self):
        return self._df.copy()

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return QVariant()

        if orientation == Qt.Horizontal:
            try:
                return self._df.columns.tolist()[section]
            except (IndexError, ):
                return QVariant()
        elif orientation == Qt.Vertical:
            try:
                # return self.df.index.tolist()
                return self._df.index.tolist()[section]
            except (IndexError, ):
                return QVariant()

    def data(self, index, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return QVariant()

        if not index.isValid():
            return QVariant()

        if index.column() == 0:
            return str(self._df.iloc[index.row(), index.column()])
        else:
            return str(round(self._df.iloc[index.row(), index.column()], 6))

    def setData(self, index, role):
        if role == Qt.DisplayRole:
            value = self._data[index.row()][index.column()]
            return str(value)

    def rowCount(self, parent=QModelIndex()):
        return len(self._df.index)

    def columnCount(self, parent=QModelIndex()):
        return len(self._df.columns)

    def sort(self, column, order):
        colname = self._df.columns.tolist()[column]
        self.layoutAboutToBeChanged.emit()
        self._df.sort_values(colname, ascending= order == Qt.AscendingOrder, inplace=True)
        self._df.reset_index(inplace=True, drop=True)
        self.layoutChanged.emit()

    def flags(self, index):
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable




class MyCircleOverlay(pg.CircleROI):
    def __init__(self, pos, size, **args):
        pg.ROI.__init__(self, pos, size, **args)
        self.path = None
        self.aspectLocked = True


class NewMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.layout_window = QVBoxLayout()


class PostProcessingWindow(NewMainWindow):
    def __init__(self):
        super().__init__()

        # Attributes:
        self.image_series = None
        self.detected_particles = None
        self.trajectories = None
        self.particle_df = None
        self.hist_values = []
        self.nf_values = []
        self.pf_values = []

        # General properties
        self.setWindowTitle("Video parameters")
        self.setWindowIcon(QIcon('app_logo.jpg'))
        self.setGeometry(0, 0, 300, 400)

        # input widget:
        myFont = QFont()
        myFont.setBold(True)

        layout_input = QGridLayout()

        label_data_input = QLabel("Data directory")
        label_data_input.setFont(myFont)
        self.txt_input_path = QLineEdit("C:/Users/sandr/Desktop/Uni/Tuebingen/HiWi/Data/01_26_22/")
        self.btn_search_dir = QPushButton("...")
        self.btn_search_dir.clicked.connect(self.btn_search_dir_is_clicked)

        layout_input.addWidget(self.txt_input_path, 0, 0)
        layout_input.addWidget(self.btn_search_dir, 0, 1)

        self.layout_window.addWidget(label_data_input, 0)
        self.layout_window.addLayout(layout_input, 1)

        # roi_widget:
        roi_layout = QGridLayout()
        self.roi_description = QLabel("Region of Interest")
        self.roi_description.setFont(myFont)
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
        self.check_full_roi.stateChanged.connect(self.change_roi_read)
        roi_layout.addWidget(self.check_full_roi, 3, 0)
        self.disable_check_full_roi = False

        self.layout_window.addLayout(roi_layout, 2)

        # Considered Pictures widget:

        cp_layout = QGridLayout()
        cp_description = QLabel("Considered Pictures")
        cp_description.setFont(myFont)
        cp_layout.addWidget(cp_description, 0, 0)

        self.cp_min_layout = QVBoxLayout()
        self.cp_min_label = QLabel("Starting frame")
        self.cp_min_layout.addWidget(self.cp_min_label)
        self.cp_min_value = QSpinBox()
        self.cp_min_value.setMinimum(0)
        self.cp_min_value.setMaximum(100_000)
        self.cp_min_value.setValue(0)
        self.cp_min_value.setSingleStep(100)
        self.cp_min_value.setDisabled(True)
        self.cp_min_layout.addWidget(self.cp_min_value)
        cp_layout.addLayout(self.cp_min_layout, 1, 0)

        self.cp_max_layout = QVBoxLayout()
        self.cp_max_label = QLabel("Ending frame")
        self.cp_max_layout.addWidget(self.cp_max_label)
        self.cp_max_value = QSpinBox()
        self.cp_max_value.setMinimum(0)
        self.cp_max_value.setMaximum(100_000)
        self.cp_max_value.setValue(1000)
        self.cp_max_value.setSingleStep(100)
        self.cp_max_value.setDisabled(True)
        self.cp_max_layout.addWidget(self.cp_max_value)
        cp_layout.addLayout(self.cp_max_layout, 1, 1)

        self.check_full_cp = QCheckBox()
        self.check_full_cp.setText("Consider all frames")
        self.check_full_cp.stateChanged.connect(self.change_cp_read)
        cp_layout.addWidget(self.check_full_cp, 2, 0)
        self.disable_check_full_cp = False
        self.check_full_cp.setChecked(True)

        self.layout_window.addLayout(cp_layout, 3)

        # General parameters label:

        general_params_label = QLabel("General parameters")
        general_params_label.setFont(myFont)
        self.layout_window.addWidget(general_params_label, 4)

        # Batch size widget:

        batch_layout = QVBoxLayout()

        batch_layout.addWidget(QLabel("Batch size"))

        self.batch_size = QSpinBox()
        self.batch_size.setMaximum(10000)
        self.batch_size.setValue(100)
        batch_layout.addWidget(self.batch_size)
        batch_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.layout_window.addLayout(batch_layout, 5)

        # fps widget:

        fps_layout = QVBoxLayout()

        fps_layout.addWidget(QLabel("Frames per second"))

        self.fps = QSpinBox()
        self.fps.setMaximum(10000)
        self.fps.setValue(100)
        fps_layout.addWidget(self.fps)
        fps_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.layout_window.addLayout(fps_layout, 6)

        # Save video check box:

        self.check_save_videos = QCheckBox("Save created videos")
        self.check_save_videos.setToolTip("Checking this checkbox leads to a much faster computation time by running the same IRM images with identical parameters.\n"
                                          "But it needs roughly 3 times the storage capacity of the raw IRM images.")
        self.layout_window.addWidget(self.check_save_videos, 7)

        # Start button:

        self.button_start = QPushButton("Start")
        self.button_start.clicked.connect(self.start_button_clicked)

        self.layout_window.addWidget(self.button_start, 8)

        main_widget = QWidget()
        main_widget.setLayout(self.layout_window)
        self.setCentralWidget(main_widget)

        # Plot Window:

        self.plot_window = NewMainWindow()
        self.plot_window.setWindowTitle("ISCAT video")
        self.plot_window.setWindowIcon(QIcon('app_logo.jpg'))
        self.plot_window.setGeometry(320,
                                     30,
                                     1000,
                                     1000)

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
        self.imv.roi.removeHandle(0)
        self.imv.roi.removeHandle(0)
        self.imv.roi.setSize(size=(1, 1))
        self.imv.ui.roiBtn.setToolTip(
            "Shows contrast of one pixel over time. With a mouse click inside the picture the considered pixel can be changed.")
        self.imv.scene.sigMouseClicked.connect(self.scene_clicked)
        self.imv.sigTimeChanged.connect(self.show_particle_locations)

        self.plot_window.layout_window.addWidget(self.imv, 0)

        plot_main_widget = QWidget()
        plot_main_widget.setLayout(self.plot_window.layout_window)
        self.plot_window.setCentralWidget(plot_main_widget)

        # Detection parameter window:

        self.edit_window = NewMainWindow()
        self.edit_window.setWindowTitle("Detection parameters")
        self.edit_window.setWindowIcon(QIcon('app_logo.jpg'))
        self.edit_window.setGeometry(0,
                                     700,
                                     300,
                                     300)
        self.overlay_parameters_layout = QGridLayout()

        self.show_overlay = QCheckBox()
        self.show_overlay.setText("Enable particle detection")
        self.show_overlay.stateChanged.connect(self.show_overlay_stateChanged)
        self.show_overlay.clicked.connect(self.show_particle_locations)
        self.overlay_parameters_layout.addWidget(self.show_overlay, 0, 0)

        self.min_sigma_label = QLabel("min_sigma")
        self.min_sigma_value = QDoubleSpinBox()
        self.min_sigma_value.setMinimum(0.0)
        self.min_sigma_value.setSingleStep(0.1)
        self.min_sigma_value.setValue(2.3)
        self.min_sigma_value.setDisabled(True)
        self.min_sigma_value.valueChanged.connect(self.show_particle_locations)
        self.overlay_parameters_layout.addWidget(self.min_sigma_label, 1, 0)
        self.overlay_parameters_layout.addWidget(self.min_sigma_value, 2, 0)

        self.max_sigma_label = QLabel("max_sigma")
        self.max_sigma_value = QDoubleSpinBox()
        self.max_sigma_value.setMaximum(100.0)
        self.max_sigma_value.setSingleStep(0.1)
        self.max_sigma_value.setValue(2.3)
        self.max_sigma_value.setDisabled(True)
        self.max_sigma_value.valueChanged.connect(self.show_particle_locations)
        self.overlay_parameters_layout.addWidget(self.max_sigma_label, 1, 1)
        self.overlay_parameters_layout.addWidget(self.max_sigma_value, 2, 1)

        self.min_sigma_value.setMaximum(self.max_sigma_value.value())
        self.max_sigma_value.setMinimum(self.min_sigma_value.value())
        self.min_sigma_value.valueChanged.connect(self.reset_max_sigma_minimum)
        self.max_sigma_value.valueChanged.connect(self.reset_min_sigma_maximum)

        self.sigma_ratio_label = QLabel("sigma_ratio")
        self.sigma_ratio_value = QDoubleSpinBox()
        self.sigma_ratio_value.setRange(0.0, 10.0)
        self.sigma_ratio_value.setSingleStep(0.1)
        self.sigma_ratio_value.setValue(4.5)
        self.sigma_ratio_value.setDisabled(True)
        self.sigma_ratio_value.valueChanged.connect(self.show_particle_locations)
        self.overlay_parameters_layout.addWidget(self.sigma_ratio_label, 3, 0)
        self.overlay_parameters_layout.addWidget(self.sigma_ratio_value, 4, 0)

        self.combo_dark_white = QComboBox()
        self.combo_dark_white.setToolTip("Bright: Only white spots are detected (particle departures)\n"
                                         "Dark: Only dark spots are detected (particle landings)\n"
                                         "Both: Dark and white spots are detected")
        self.combo_dark_white.addItems(["Both", "Dark", "Bright"])
        self.combo_dark_white.setDisabled(True)
        self.combo_dark_white.currentTextChanged.connect(self.show_particle_locations)
        self.label_dark_white = QLabel("Detection mode")
        self.overlay_parameters_layout.addWidget(self.label_dark_white, 3, 1)
        self.overlay_parameters_layout.addWidget(self.combo_dark_white, 4, 1)

        self.threshold_label = QLabel("DoG threshold value")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(1, 100)
        self.threshold_slider.setSingleStep(1)
        self.threshold_slider.setValue(70)
        self.threshold_slider.setDisabled(True)
        self.threshold_slider.valueChanged.connect(self.threshold_slider_changed)
        self.threshold_slider.valueChanged.connect(self.show_particle_locations)
        self.threshold_value = QLabel()
        self.threshold_value.setText(
            str(np.format_float_scientific(self.threshold_slider.value() / 100_000, precision=2)))
        self.overlay_parameters_layout.addWidget(self.threshold_label, 5, 0)
        self.overlay_parameters_layout.addWidget(self.threshold_slider, 6, 0)
        self.overlay_parameters_layout.addWidget(self.threshold_value, 6, 1)

        self.edit_window.layout_window.addLayout(self.overlay_parameters_layout, 0)

        edit_main_widget = QWidget()
        edit_main_widget.setLayout(self.edit_window.layout_window)
        self.edit_window.setCentralWidget(edit_main_widget)

        # Contrast table window:

        self.table_window = NewMainWindow()
        self.table_window.setGeometry(1320,
                                      30,
                                      500,
                                      500)
        self.table_window.setWindowTitle("Particle contrasts")
        self.table_window.setWindowIcon(QIcon('app_logo.jpg'))
        self.table_particles = QTableView()
        self.table_particles.clicked.connect(self.display_particle_trajectory)
        self.table_window.setCentralWidget(self.table_particles)
        self.table_window_menu = self.table_window.menuBar()

        action_save_particle_contrast = QAction("&Save particle contrasts", self)
        action_save_particle_contrast.setToolTip("The individual contrasts of all selected particles within the table "
                                                 "will be saved. You have to select at least one particle.")
        action_save_particle_contrast.triggered.connect(self.save_contrasts)

        action_save_particle_table = QAction("&Save table", self)
        action_save_particle_table.setToolTip("The whole table will be saved as csv file.")
        action_save_particle_table.triggered.connect(self.save_table)

        file_menu = self.table_window_menu.addMenu("&File")
        file_menu.setToolTipsVisible(True)

        file_submenu = file_menu.addMenu("Saving options")
        file_submenu.addAction(action_save_particle_contrast)
        file_submenu.addAction(action_save_particle_table)
        file_submenu.setToolTipsVisible(True)

        # Parameter windows:

        #   histogram parameters:

        self.hist_parameter_window = QWidget()
        self.hist_parameter_window.setWindowTitle("Additional parameters for histogram calculation")
        self.fill_hist_parameters()

        #   noise_floor parameters:

        self.nf_parameter_widget = QWidget()
        self.nf_parameter_widget.setWindowTitle("Parameters for noise floor calculation")
        self.fill_noise_floor_parameters()

        #   power_fluctuation parameters:

        self.pf_parameter_widget = QWidget()
        self.pf_parameter_widget.setWindowTitle("Parameter for power fluctuation calculation")
        self.fill_power_fluctuation_parameters()

        # Calculation window:

        self.calc_window = NewMainWindow()
        self.calc_window.setWindowIcon(QIcon('app_logo.jpg'))
        self.calc_window.setGeometry(0,
                                     500,
                                     300,
                                     150)
        calc_window_layout = QVBoxLayout()
        self.calc_window.setWindowTitle("Create plots and tables")
        self.calc_window.btn_histogram = QPushButton("Get particle contrasts")
        self.calc_window.btn_histogram.clicked.connect(self.get_histogram)
        self.calc_window.btn_noise_floor = QPushButton("Get noise floor")
        self.calc_window.btn_noise_floor.clicked.connect(self.get_noise_floor)
        self.calc_window.btn_power_fluctuation = QPushButton("Get power fluctuation")
        self.calc_window.btn_power_fluctuation.clicked.connect(self.get_power_fluctuation)

        calc_window_layout.addWidget(self.calc_window.btn_histogram)
        calc_window_layout.addWidget(self.calc_window.btn_noise_floor)
        calc_window_layout.addWidget(self.calc_window.btn_power_fluctuation)

        action_show_hist_parameters = QAction("&Show histogram parameters", self)
        action_show_hist_parameters.triggered.connect(self.show_hist_parameter_window)

        action_show_nf_parameters = QAction("&Show noise floor parameters", self)
        action_show_nf_parameters.triggered.connect(self.show_nf_parameter_widget)

        action_show_pf_parameters = QAction("&Show power fluctuation parameters", self)
        action_show_pf_parameters.triggered.connect(self.show_pf_parameter_widget)

        self.calc_window_menu = self.calc_window.menuBar()
        calc_file_menu = self.calc_window_menu.addMenu("&File")

        calc_file_menu.addAction(action_show_hist_parameters)
        calc_file_menu.addAction(action_show_nf_parameters)
        calc_file_menu.addAction(action_show_pf_parameters)

        calc_main_widget = QWidget()
        calc_main_widget.setLayout(calc_window_layout)
        self.calc_window.setCentralWidget(calc_main_widget)

    # Functions for input window:

    def btn_search_dir_is_clicked(self):
        file = str(QFileDialog.getExistingDirectory(self))
        self.txt_input_path.setText(file + "/")

    def change_roi_read(self):
        self.roi_x_min.setDisabled(not self.disable_check_full_roi)
        self.roi_x_max.setDisabled(not self.disable_check_full_roi)
        self.roi_y_min.setDisabled(not self.disable_check_full_roi)
        self.roi_y_max.setDisabled(not self.disable_check_full_roi)
        self.disable_check_full_roi = not self.disable_check_full_roi

    def change_cp_read(self):
        self.cp_min_value.setDisabled(not self.disable_check_full_cp)
        self.cp_max_value.setDisabled(not self.disable_check_full_cp)
        self.disable_check_full_cp = not self.disable_check_full_cp

    def start_button_clicked(self):
        if self.disable_check_full_roi and self.disable_check_full_cp:
            self.image_series = ImageSeries(path=self.txt_input_path.text(),
                                            fps=self.fps.value(),
                                            batch_size=self.batch_size.value())
        elif self.disable_check_full_roi:
            cp = ConsideredPictures(self.cp_min_value.value(), self.cp_max_value.value())
            self.image_series = ImageSeries(path=self.txt_input_path.text(),
                                            fps=self.fps.value(),
                                            batch_size=self.batch_size.value(),
                                            considered_pictures=ConsideredPictures(*cp))
        elif self.disable_check_full_cp:
            roi = (self.roi_x_min.value(), self.roi_x_max.value(), self.roi_y_min.value(), self.roi_y_max.value())
            self.image_series = ImageSeries(path=self.txt_input_path.text(),
                                            fps=self.fps.value(),
                                            batch_size=self.batch_size.value(),
                                            region_of_interest=RegionOfInterest(*roi))
        else:
            roi = (self.roi_x_min.value(), self.roi_x_max.value(), self.roi_y_min.value(), self.roi_y_max.value())
            cp = ConsideredPictures(self.cp_min_value.value(), self.cp_max_value.value())
            self.image_series = ImageSeries(path=self.txt_input_path.text(),
                                            fps=self.fps.value(),
                                            batch_size=self.batch_size.value(),
                                            region_of_interest=RegionOfInterest(*roi),
                                            considered_pictures=ConsideredPictures(*cp))
        self.image_series.get_video()
        self.image_series.create_differential_video()
        if self.check_save_videos.isChecked():
            self.image_series.save_videos(["raw",
                                           "raw with power normalization",
                                           "power fluctuation",
                                           "differential video"])

        # Transpose is necessary to set coordinate origin to the upper left and x-direction to the right
        # and y-direction to the left.
        self.imv.setImage(self.image_series.differential_video.transpose(0, 2, 1))
        cmap = pg.colormap.getFromMatplotlib("gray")
        self.imv.setColorMap(cmap)

        self.calc_window.show()
        self.plot_window.show()
        self.edit_window.show()

    # Functions for plot window:

    def scene_clicked(self, evt):
        if self.imv.ui.roiBtn.isChecked():
            x = self.imv.getImageItem().mapFromScene(evt.scenePos()).x()
            y = self.imv.getImageItem().mapFromScene(evt.scenePos()).y()
            self.imv.roi.setPos(pos=(int(x), int(y)))

    def play_video(self):
        if self.imv.currentIndex == len(self.image_series.differential_video) - 1:
            self.imv.setCurrentIndex(0)
        self.imv.play(rate=100)

    # Functions for detection parameter window:

    def threshold_slider_changed(self):
        val = np.format_float_scientific(self.threshold_slider.value() / 100_000, precision=2)
        self.threshold_value.setText(str(val))

    def reset_max_sigma_minimum(self):
        self.max_sigma_value.setMinimum(self.min_sigma_value.value())

    def reset_min_sigma_maximum(self):
        self.min_sigma_value.setMaximum(self.max_sigma_value.value())

    def set_particle_locations(self, sign: int):
        current_frame_num = self.imv.currentIndex
        current_threshold = self.threshold_slider.value() / 100_000
        pen = QPen(Qt.red, 0.1)
        pic = sign * self.image_series.differential_video[current_frame_num]
        local_maxima, sigmas = dog_own(image=pic,
                                       min_sigma=self.min_sigma_value.value(),
                                       max_sigma=self.max_sigma_value.value(),
                                       sigma_ratio=self.sigma_ratio_value.value(),
                                       threshold=current_threshold
                                       )
        for lm, sigma in zip(local_maxima, sigmas):
            if lm.any():
                self.imv.addItem(MyCircleOverlay(pos=(lm[1] - sigma[0] + 0.5, lm[0] - sigma[0] + 0.5),
                                                 size=(2 * sigma[0]),
                                                 pen=pen,
                                                 movable=False))
        self.imv.setCurrentIndex(current_frame_num)

    def show_particle_locations(self):
        bright_or_dark = self.combo_dark_white.currentText()
        if len(self.plot.items) > 3:
            for el in self.plot.items[3:]:
                self.plot.removeItem(el)
        if self.show_overlay.isChecked():
            try:
                if bright_or_dark == "Both" or bright_or_dark == "Dark":
                    self.set_particle_locations(-1)
                elif bright_or_dark == "Both" or bright_or_dark == "Bright":
                    self.set_particle_locations(1)
            except:
                raise ValueError("bright_or_dark has to be Dark, Bright or Both")

    def show_overlay_stateChanged(self):
        if self.show_overlay.isChecked():
            self.min_sigma_value.setDisabled(False)
            self.max_sigma_value.setDisabled(False)
            self.sigma_ratio_value.setDisabled(False)
            self.threshold_slider.setDisabled(False)
            self.combo_dark_white.setDisabled(False)
        else:
            self.min_sigma_value.setDisabled(True)
            self.max_sigma_value.setDisabled(True)
            self.sigma_ratio_value.setDisabled(True)
            self.threshold_slider.setDisabled(True)
            self.combo_dark_white.setDisabled(True)

        self.show_particle_locations()

    # Functions for particle contrast window:

    def save_table(self):
        save_dir = self.image_series.path + "Plots/Particle Contrasts/"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.particle_df.to_csv(save_dir + "contrast_table.csv")

    def save_contrasts(self):
        save_dir = self.image_series.path + "Plots/Particle Contrasts/"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        his = PlotProteinHistogram(intersection_display_flag=True, imgSizex=10, imgSizey=5)
        for idx in self.table_particles.selectedIndexes():
            save_path = f"{save_dir}particle_{idx.row()}.png"
            his.plot_contrast_extraction(particles=[self.trajectories[idx.row()]],
                                         batch_size=self.image_series.batch_size,
                                         video_frame_num=self.image_series.differential_video.shape[0],
                                         MinPeakWidth=100,
                                         MinPeakProminence=0,
                                         pixel_size=0.66,
                                         save_path=save_path)

    def display_particle_trajectory(self):
        idx = self.table_particles.selectedIndexes()
        intensities = np.array([self.trajectories[idx[0].row()][2]])
        min_intensity_idx = np.argmin(intensities)
        frame = self.trajectories[idx[0].row()][4][min_intensity_idx]
        x = self.trajectories[idx[0].row()][6][min_intensity_idx]
        y = self.trajectories[idx[0].row()][7][min_intensity_idx]
        if not self.imv.ui.roiBtn.isChecked():
            self.imv.ui.roiBtn.click()
        self.show_overlay.setChecked(False)
        self.imv.roi.setPos(pos=(int(x), int(y)))
        self.imv.setCurrentIndex(frame)

    # Functions for calculation window:

    def fill_hist_parameters(self):
        hist_params_layout = QGridLayout()
        param_names = ["spatial filter (outlier frames)",
                       "spatial filter (dense PSF)", "spatial filter (symmetric PSF)", "temporal filter",
                       "Linking (search range)", "Linking (memory)"]
        param_values = [20, 10, 0.5, 70, 2, 10]
        for i, (name, val) in enumerate(zip(param_names, param_values)):
            hist_params_layout.addWidget(QLabel(name), i, 0)
            temp = QDoubleSpinBox()
            temp.setValue(val)
            self.hist_values.append(temp)
            hist_params_layout.addWidget(self.hist_values[i], i, 1)
        self.hist_parameter_window.setLayout(hist_params_layout)

    def fill_noise_floor_parameters(self):
        layout = QGridLayout()
        param_names = ["batch sizes", "modes"]
        param_values = ["1, 2, 5, 10, 20, 50, 100, 200", "0, 12"]
        for i, (name, val) in enumerate(zip(param_names, param_values)):
            layout.addWidget(QLabel(name), i, 0)
            text = QLineEdit()
            text.setText(val)
            self.nf_values.append(text)
            layout.addWidget(self.nf_values[i], i, 1)
        self.nf_parameter_widget.setLayout(layout)

    def fill_power_fluctuation_parameters(self):
        layout = QGridLayout()
        param_names = ["plot variants"]
        param_values = ["1, 2, 3, 4"]
        for i, (name, val) in enumerate(zip(param_names, param_values)):
            layout.addWidget(QLabel(name), i, 0)
            text = QLineEdit()
            text.setText(val)
            self.pf_values.append(text)
            layout.addWidget(self.pf_values[i], i, 1)
        self.pf_parameter_widget.setLayout(layout)

    def show_hist_parameter_window(self):
        self.hist_parameter_window.show()

    def show_nf_parameter_widget(self):
        self.nf_parameter_widget.show()

    def show_pf_parameter_widget(self):
        self.pf_parameter_widget.show()

    def get_histogram(self):
        PSF = particle_localization.PSFsExtraction(video=self.image_series.differential_video)
        if self.combo_dark_white.currentText() == "Both":
            mode = "BOTH"
        else:
            mode = self.combo_dark_white.currentText()
        detected_particles = PSF.psf_detection(function='dog', overlap=0, mode=mode,
                                               min_sigma=self.min_sigma_value.value(),
                                               max_sigma=self.max_sigma_value.value(),
                                               sigma_ratio=self.sigma_ratio_value.value(),
                                               threshold=self.threshold_slider.value() / 100_000)
        if detected_particles is not None:
            PSFs = PSF.fit_Gaussian2D_wrapper(PSF_List=detected_particles, scale=5, internal_parallel_flag=True)
            linking_ = Linking()
            linked_PSFs = linking_.create_link(psf_position=PSFs,
                                               search_range=int(self.hist_values[4].value()),  # The maximum distance features can move between frames
                                               memory=int(self.hist_values[5].value()))  # The maximum number of frames during which a feature can vanish, then reappear nearby, and be considered the same particle.
            spatial_filters = localization_filtering.SpatialFilter()
            PSFs_filtered = spatial_filters.outlier_frames(linked_PSFs,
                                                           threshold=int(self.hist_values[0].value()))  # images with more detected particles than the threshold are deleted
            PSFs_filtered = spatial_filters.dense_PSFs(PSFs_filtered,
                                                       threshold=int(self.hist_values[1].value()))  # particles that have a short distance to each other are deleted
            PSFs_filtered = spatial_filters.symmetric_PSFs(PSFs_filtered,
                                                           threshold=self.hist_values[2].value())  # particles with non-symmetric shape are deleted

            t_filters = TemporalFilter(video=self.image_series.differential_video,
                                       batchSize=self.image_series.batch_size)
            all_trajectories, linked_PSFs_filter, his_all_particles = t_filters.v_trajectory(df_PSFs=PSFs_filtered,
                                                                                             threshold=int(self.hist_values[3].value()))  # particles that are present in less images than the threshold value are deleted
            print(f"Number of Particles before filtering: {linking_.trajectory_counter(linked_PSFs)}")
            print(f"Number of Particles after filtering: {len(all_trajectories)}")

            his = PlotProteinHistogram(intersection_display_flag=True, imgSizex=10, imgSizey=5)

            valid_ones = []
            for i in range(len(all_trajectories)):
                number = "#" + str(i)
                if his.check_for_v_shape(particles=all_trajectories,
                                         batch_size=self.image_series.batch_size,
                                         video_frame_num=self.image_series.differential_video.shape[0],
                                         MinPeakWidth=100,
                                         MinPeakProminence=0,
                                         pixel_size=0.66,
                                         particles_num=number):
                    valid_ones.append(i)

            self.trajectories = [t for i, t in enumerate(all_trajectories) if i in valid_ones]
            his2 = PlotProteinHistogram(intersection_display_flag=True, imgSizex=10, imgSizey=5)
            his2(folder_name='',
                 particles=self.trajectories,
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
            self.particle_df = df_transposed

            model = PandasModel(df_transposed)
            self.table_particles.setModel(model)
            self.table_window.show()
        else:
            dlg = QDialog()
            dlg.setWindowTitle("Warning")
            dlg_layout = QVBoxLayout()
            dlg_label = QLabel()
            dlg_label.setText("No particles could be found with the given set of parameters!")
            dlg_layout.addWidget(dlg_label, 0)
            dlg.setLayout(dlg_layout)
            dlg.exec_()

    def get_noise_floor(self):
        batch_sizes = [int(batch_size) for batch_size in self.nf_values[0].text().split(", ")]
        modes = [int(mode) for mode in self.nf_values[1].text().split(", ")]
        self.image_series.get_noise_floor(batch_sizes=batch_sizes,
                                          modes=modes)

    def get_power_fluctuation(self):
        plot_variants = [int(plot) for plot in self.pf_values[0].text().split(", ")]
        self.image_series.plot_power_fluctuation(plot_variants=plot_variants)


# driver code
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = PostProcessingWindow()
    w.show()
    app.exec_()
