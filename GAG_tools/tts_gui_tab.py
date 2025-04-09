import sys
import os
import re
import time
import psutil
import shutil
import requests
import sounddevice as sd
import soundfile as sf
from datetime import datetime
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox,
                             QSpinBox, QDoubleSpinBox, QCheckBox, QTextEdit,
                             QFileDialog, QGroupBox, QMessageBox, QStatusBar, QInputDialog, QGridLayout,
                             QProgressBar, QFormLayout, QRadioButton, QButtonGroup)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl, QFileSystemWatcher, QTimer
from GAG_tools.config_manager import ConfigManager
import resources_rc


class ModelSwitchThread(QThread):
    switch_signal = pyqtSignal(bool, str)

    def __init__(self, url, model_type, weights_path):
        super().__init__()
        self.url = url
        self.model_type = model_type  # 'gpt' or 'sovits'
        self.weights_path = weights_path

    def run(self):
        try:
            endpoint = '/set_gpt_weights' if self.model_type == 'gpt' else '/set_sovits_weights'
            response = requests.get(f"{self.url}{endpoint}", params={'weights_path': self.weights_path})

            if response.status_code == 200:
                self.switch_signal.emit(True, self.tr("{} 模型切换成功").format(self.model_type.upper()))
            else:
                error_msg = response.json() if response.headers.get(
                    'content-type') == 'application/json' else response.text
                self.switch_signal.emit(False,
                                        self.tr("{} 模型切换失败:\n {}").format(self.model_type.upper(), error_msg))
        except Exception as e:
            self.switch_signal.emit(False,
                                    self.tr("{} 模型切换时发生错误:\n {}").format(self.model_type.upper(), str(e)))


class TTSThread(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, url, params, cache_dir):
        super().__init__()
        self.url = url
        self.params = params
        self.temp_file = None
        self.cache_dir = cache_dir

    def run(self):
        try:
            print(str(self.params))
            response = requests.post(f"{self.url}/tts", json=self.params)
            if response.status_code == 200:
                temp_file_path = os.path.join(self.cache_dir, f"tmp_{int(time.time())}.wav")
                with open(temp_file_path, 'wb') as f:
                    f.write(response.content)
                self.temp_file = temp_file_path
                self.finished.emit(self.temp_file)
            else:
                self.error.emit(self.tr("执行推理时出现错误:\n {}").format(response.text))
        except Exception as e:
            self.error.emit(self.tr("执行推理时出现错误:\n {}").format(str(e)))


class APICheckThread(QThread):
    status_signal = pyqtSignal(bool, str)

    def __init__(self, url, silent=False, continuous=False, retry_interval=2000):
        super().__init__()
        self.url = url
        self.silent = silent
        self.continuous = continuous
        self.retry_interval = retry_interval
        self.should_continue = True

    def run(self):
        while self.should_continue:
            try:
                response = requests.get(f"{self.url}/control")
                if response.status_code == 400:  # API is ready
                    self.status_signal.emit(True, self.tr("API 就绪"))
                    if not self.continuous:
                        break
                elif not self.silent:
                    self.status_signal.emit(False, self.tr("API 未就绪"))
            except Exception as e:
                if not self.silent:
                    self.status_signal.emit(False, self.tr("API 不可用: {}").format(str(e)))

            # If not continuous or stopped, break the loop
            if not self.continuous or not self.should_continue:
                break

            # Sleep for retry interval before next check
            self.msleep(self.retry_interval)

    def stop(self):
        self.should_continue = False


class TTSGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.param_widgets = None
        self.SPLIT_METHODS = {
            'cut0': self.tr('不切'),
            'cut1': self.tr('凑四句一切'),
            'cut2': self.tr('凑50字一切'),
            'cut3': self.tr('按中文句号。切'),
            'cut4': self.tr('按英文句号.切'),
            'cut5': self.tr('按标点符号切')
        }
        self.LANGUAGES = {
            'all_zh': self.tr('中文'),
            'en': self.tr('英文'),
            'all_ja': self.tr('日文'),
            'all_yue': self.tr('粤语'),
            'all_ko': self.tr('韩文'),
            'zh': self.tr('中英混合'),
            'ja': self.tr('日英混合'),
            'yue': self.tr('粤英混合'),
            'ko': self.tr('韩英混合'),
            'auto': self.tr('多语种混合'),
            'auto_yue': self.tr('多语种混合(粤语)')
        }
        self.config_manager = ConfigManager()
        self.setup_cache_directory()
        self.current_audio_file = None
        self.is_playing = False
        self.current_gpt_model = None
        self.current_sovits_model = None
        self.gpt_switching = False
        self.sovits_switching = False
        self.synthesis_pending = False
        
        # 定义模型版本及对应的目录
        self.MODEL_VERSIONS = {
            'v2': {'gpt': 'GPT_weights_v2', 'sovits': 'SoVITS_weights_v2'},
            'v3': {'gpt': 'GPT_weights_v3', 'sovits': 'SoVITS_weights_v3'}
        }
        # 设置当前模型版本，默认为v2或从配置读取
        self.current_model_version = self.config_manager.get_value('model_version', 'v2')
        
        # 创建模型目录，添加到文件监视器
        for version in self.MODEL_VERSIONS.values():
            os.makedirs(version['gpt'], exist_ok=True)
            os.makedirs(version['sovits'], exist_ok=True)
            
        self.watcher = QFileSystemWatcher(self)
        for version in self.MODEL_VERSIONS.values():
            self.watcher.addPath(version['gpt'])
            self.watcher.addPath(version['sovits'])
        self.watcher.directoryChanged.connect(self.update_model_lists)
        
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAutoFillBackground(True)
        self.setup_background()
        self.initUI()
        self.start_system_monitoring()
        self.api_check_thread = None
        self.autostart_api_check()

    def initUI(self):
        self.setWindowTitle(self.tr('TTS GUI'))
        self.setGeometry(100, 100, 800, 800)

        # Create central widget
        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # API settings group
        api_group = QGroupBox(self.tr("API 设置"))
        api_layout = QHBoxLayout()
        self.api_url_input = QLineEdit(self.config_manager.get_value('api_url'))
        self.api_status_label = QLabel(self.tr("状态: API 未就绪"))
        self.check_api_button = QPushButton(self.tr("检查"))
        self.check_api_button.clicked.connect(self.check_api_status)
        api_layout.addWidget(QLabel(self.tr("API URL:")))
        api_layout.addWidget(self.api_url_input)
        api_layout.addWidget(self.api_status_label)
        api_layout.addWidget(self.check_api_button)
        api_group.setLayout(api_layout)
        main_layout.addWidget(api_group)

        # Parameter settings group
        params_group = QGroupBox(self.tr("待合成文本:"))
        params_group.setFlat(True)
        params_layout = QVBoxLayout()

        # Text input
        self.text_input = QTextEdit()
        self.text_input.setAcceptRichText(False)
        params_layout.addWidget(self.text_input)
        self.text_input.setPlaceholderText(
            self.tr("在这里输入需要合成的文本..."
                    "\n\n使用方法：\n"
                    "1.将本exe放入GPT-SoVITS官方整合包下，双击启动。\n"
                    "2.将读取并使用GPT_weights与SoVITS_weights下的模型，请先完成训练获得模型。\n"
                    "3.保存预设将保存当前所有合成参数设定，可视为一个说话人，后续可快速切换，亦可用于批量合成页面\n"
                    "4.默认使用整合包自带环境来调起并使用API，也可以在API管理页面自定义。\n"
                    "\n此外，如果你使用官方v2-240821整合包，你可能会遇见粤语等语种合成空白音频的问题，这是GPT-SoVITS的一个已知并已解决的问题，更新整合包代码到仓库最新即可。"
                    "\n\nGitHub开源地址: https://github.com/AliceNavigator/GPT-SoVITS-Api-GUI           by  领航员未鸟\n")
        )

        # Create parameter input widgets
        self.create_parameter_inputs(params_layout)

        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)

        # Control button group
        control_group = QGroupBox(self.tr("控制"))
        control_layout = QVBoxLayout()

        # Save path settings
        path_layout = QHBoxLayout()
        self.save_path_input = QLineEdit(self.config_manager.get_value('save_directory'))
        # self.save_path_input.setReadOnly(True)
        self.save_path_input.setPlaceholderText(self.tr("将会把合成的音频保存在该路径下..."))
        self.browse_button = QPushButton(self.tr("浏览"))
        self.browse_button.clicked.connect(self.set_save_path)
        self.open_save_directory_button = QPushButton(self.tr("打开"))
        self.open_save_directory_button.clicked.connect(self.open_save_directory)
        path_layout.addWidget(QLabel(self.tr("保存路径:")))
        path_layout.addWidget(self.save_path_input)
        path_layout.addWidget(self.browse_button)
        path_layout.addWidget(self.open_save_directory_button)
        control_layout.addLayout(path_layout)

        # Existing control buttons
        buttons_layout = QHBoxLayout()
        self.synthesize_button = QPushButton(self.tr("开始合成"))
        self.synthesize_button.setStyleSheet("min-height: 20px;")
        self.synthesize_button.setEnabled(False)
        self.synthesize_button.clicked.connect(self.prepare_synthesis)
        self.play_button = QPushButton(self.tr("播放"))
        self.play_button.setStyleSheet("min-height: 20px;")
        self.play_button.clicked.connect(self.play_audio)
        self.save_button = QPushButton(self.tr("保存音频"))
        self.save_button.setStyleSheet("min-height: 20px;")
        self.save_button.clicked.connect(self.save_audio)
        buttons_layout.addWidget(self.synthesize_button)
        buttons_layout.addWidget(self.play_button)
        buttons_layout.addWidget(self.save_button)
        control_layout.addLayout(buttons_layout)

        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)

        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        # Update model lists
        self.update_model_lists()

        # Load preset
        self.load_preset()

        # Sets the translucent GroupBox style
        self.setup_group_box_styles()

    def create_parameter_inputs(self, layout):
        # All parameters in param_widgets will be used as request parameters.
        self.param_widgets = {}

        # Create main horizontal layout to utilize screen width
        main_param_layout = QHBoxLayout()

        # Left side layout (Presets, Models, and Language settings)
        left_side = QVBoxLayout()

        # Top row layout for presets and models
        top_row = QHBoxLayout()

        # Preset controls
        preset_group = QGroupBox(self.tr("预设"))
        preset_layout = QVBoxLayout()
        preset_layout.setSpacing(8)

        # Preset combo
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(self.config_manager.get_value('presets').keys())
        self.preset_combo.setCurrentText(self.config_manager.get_value('current_preset'))
        self.preset_combo.currentTextChanged.connect(self.load_preset)
        preset_layout.addWidget(self.preset_combo)

        # Preset buttons
        preset_buttons = QHBoxLayout()
        preset_buttons.setSpacing(4)
        self.save_preset_button = QPushButton(self.tr("保存"))
        self.delete_preset_button = QPushButton(self.tr("删除"))
        preset_buttons.addWidget(self.save_preset_button)
        self.save_preset_button.clicked.connect(self.save_preset)
        preset_buttons.addWidget(self.delete_preset_button)
        self.delete_preset_button.clicked.connect(self.delete_preset)
        preset_layout.addLayout(preset_buttons)

        preset_group.setLayout(preset_layout)
        top_row.addWidget(preset_group)
        
        # 模型版本选择
        version_group = QGroupBox(self.tr("模型版本"))
        version_layout = QHBoxLayout()
        
        self.version_button_group = QButtonGroup(self)
        self.v2_radio = QRadioButton("v2")
        self.v3_radio = QRadioButton("v3")
        
        # 根据配置选择当前版本
        if self.current_model_version == 'v2':
            self.v2_radio.setChecked(True)
        else:
            self.v3_radio.setChecked(True)
            
        self.version_button_group.addButton(self.v2_radio, 2)
        self.version_button_group.addButton(self.v3_radio, 3)
        self.version_button_group.buttonClicked.connect(self.change_model_version)
        
        version_layout.addWidget(self.v2_radio)
        version_layout.addWidget(self.v3_radio)
        version_group.setLayout(version_layout)
        
        top_row.addWidget(version_group)

        # Model Selection
        model_group = QGroupBox(self.tr("模型选择"))
        model_layout = QFormLayout()
        model_layout.setSpacing(8)

        # GPT model
        self.gpt_combo = QComboBox()
        self.gpt_combo.setMinimumWidth(175)
        self.gpt_combo.activated.connect(self.update_model_lists)
        model_layout.addRow(self.tr("GPT 模型:"), self.gpt_combo)

        # Sovits model
        self.sovits_combo = QComboBox()
        self.sovits_combo.setMinimumWidth(175)
        self.sovits_combo.activated.connect(self.update_model_lists)
        model_layout.addRow(self.tr("SoVITS 模型:"), self.sovits_combo)

        model_group.setLayout(model_layout)
        top_row.addWidget(model_group)

        # Add top row to left side
        left_side.addLayout(top_row)

        # Language and Text Settings
        lang_group = QGroupBox(self.tr("语言与文本设置"))
        lang_layout = QVBoxLayout()
        lang_layout.setSpacing(8)

        # Language settings in form layout
        lang_form = QFormLayout()
        lang_form.setSpacing(8)

        # Text language
        self.param_widgets['text_lang'] = QComboBox()
        for key, value in self.LANGUAGES.items():
            self.param_widgets['text_lang'].addItem(value, key)
        lang_form.addRow(self.tr("合成文本语种:"), self.param_widgets['text_lang'])

        # Prompt language
        self.param_widgets['prompt_lang'] = QComboBox()
        for key, value in self.LANGUAGES.items():
            self.param_widgets['prompt_lang'].addItem(value, key)
        lang_form.addRow(self.tr("参考音频语种:"), self.param_widgets['prompt_lang'])

        # Text split method
        self.param_widgets['text_split_method'] = QComboBox()
        for key, value in self.SPLIT_METHODS.items():
            self.param_widgets['text_split_method'].addItem(value, key)
        lang_form.addRow(self.tr("文本分割方式:"), self.param_widgets['text_split_method'])

        # Reference audio settings
        ref_audio_widget = QWidget()
        ref_audio_layout = QHBoxLayout(ref_audio_widget)
        ref_audio_layout.setContentsMargins(0, 0, 0, 0)
        ref_audio_layout.setSpacing(4)
        self.param_widgets['ref_audio_path'] = QLineEdit()
        self.param_widgets['ref_audio_path'].setPlaceholderText(self.tr("3-10秒的参考音频，超过会报错"))
        ref_audio_button = QPushButton(self.tr("浏览"))
        ref_audio_button.setFixedWidth(60)
        ref_audio_button.clicked.connect(lambda: self.browse_file('ref_audio_path'))
        ref_audio_layout.addWidget(self.param_widgets['ref_audio_path'])
        ref_audio_layout.addWidget(ref_audio_button)
        lang_form.addRow(self.tr("参考音频:"), ref_audio_widget)

        # Prompt text
        prompt_widget = QWidget()
        prompt_layout = QHBoxLayout(prompt_widget)
        prompt_layout.setContentsMargins(0, 0, 0, 0)
        prompt_layout.setSpacing(4)
        self.param_widgets['prompt_text'] = QLineEdit()
        self.param_widgets['prompt_text'].setPlaceholderText(self.tr("不填则视为使用无参考文本模式"))
        self.param_widgets['no_prompt'] = QCheckBox(self.tr("无参  "))
        prompt_layout.addWidget(self.param_widgets['prompt_text'])
        prompt_layout.addWidget(self.param_widgets['no_prompt'])
        lang_form.addRow(self.tr("参考音频文本:"), prompt_widget)

        # Auxiliary reference audio
        aux_ref_widget = QWidget()
        aux_ref_layout = QHBoxLayout(aux_ref_widget)
        aux_ref_layout.setContentsMargins(0, 0, 0, 0)
        aux_ref_layout.setSpacing(4)
        self.param_widgets['aux_ref_audio_paths'] = QLineEdit()
        self.param_widgets['aux_ref_audio_paths'].setPlaceholderText(self.tr("(可选)以额外的多个音频平均融合音色"))
        aux_ref_button = QPushButton(self.tr("浏览"))
        aux_ref_button.setFixedWidth(60)
        aux_ref_button.clicked.connect(lambda: self.browse_files('aux_ref_audio_paths'))
        aux_ref_layout.addWidget(self.param_widgets['aux_ref_audio_paths'])
        aux_ref_layout.addWidget(aux_ref_button)
        lang_form.addRow(self.tr("辅助参考音频:"), aux_ref_widget)

        lang_layout.addLayout(lang_form)
        lang_group.setLayout(lang_layout)
        left_side.addWidget(lang_group)

        # Add left side to main layout
        main_param_layout.addLayout(left_side, stretch=1)

        # Right side - Generation Parameters
        gen_group = QGroupBox(self.tr("合成参数"))
        gen_layout = QVBoxLayout()

        # Parameter grid
        param_grid = QGridLayout()
        param_grid.setVerticalSpacing(10)

        # Create parameter input widgets with original layout
        self.param_widgets['top_k'] = QSpinBox()
        self.param_widgets['top_k'].setRange(1, 100)
        self.param_widgets['top_p'] = QDoubleSpinBox()
        self.param_widgets['top_p'].setRange(0, 1)
        self.param_widgets['top_p'].setSingleStep(0.05)
        self.param_widgets['temperature'] = QDoubleSpinBox()
        self.param_widgets['temperature'].setRange(0, 1)
        self.param_widgets['temperature'].setSingleStep(0.05)

        param_grid.addWidget(QLabel(self.tr("Top K:")), 0, 0)
        param_grid.addWidget(self.param_widgets['top_k'], 0, 1)
        param_grid.addWidget(QLabel(self.tr("Top P:")), 1, 0)
        param_grid.addWidget(self.param_widgets['top_p'], 1, 1)
        param_grid.addWidget(QLabel(self.tr("Temperature:")), 2, 0)
        param_grid.addWidget(self.param_widgets['temperature'], 2, 1)

        self.param_widgets['speed_factor'] = QDoubleSpinBox()
        self.param_widgets['speed_factor'].setRange(0.1, 3)
        self.param_widgets['speed_factor'].setSingleStep(0.1)
        self.param_widgets['repetition_penalty'] = QDoubleSpinBox()
        self.param_widgets['repetition_penalty'].setRange(0, 5)
        self.param_widgets['repetition_penalty'].setSingleStep(0.05)
        self.param_widgets['seed'] = QSpinBox()
        self.param_widgets['seed'].setRange(-1, 1000000)

        param_grid.addWidget(QLabel(self.tr("语速:")), 0, 2)
        param_grid.addWidget(self.param_widgets['speed_factor'], 0, 3)
        param_grid.addWidget(QLabel(self.tr("重复惩罚:")), 1, 2)
        param_grid.addWidget(self.param_widgets['repetition_penalty'], 1, 3)
        param_grid.addWidget(QLabel(self.tr("种子:")), 2, 2)
        param_grid.addWidget(self.param_widgets['seed'], 2, 3)

        gen_layout.addLayout(param_grid)

        # Checkbox options
        options_grid = QGridLayout()
        self.param_widgets['parallel_infer'] = QCheckBox(self.tr("并行推理"))
        self.param_widgets['split_bucket'] = QCheckBox(self.tr("数据分桶"))
        self.param_widgets['return_fragment'] = QCheckBox(self.tr("返回片段"))
        self.param_widgets['streaming_mode'] = QCheckBox(self.tr("流式模式"))

        options_grid.addWidget(self.param_widgets['parallel_infer'], 0, 0)
        options_grid.addWidget(self.param_widgets['split_bucket'], 0, 1)
        options_grid.addWidget(self.param_widgets['return_fragment'], 1, 0)
        self.param_widgets['return_fragment'].setEnabled(False)
        options_grid.addWidget(self.param_widgets['streaming_mode'], 1, 1)
        self.param_widgets['streaming_mode'].setEnabled(False)

        gen_layout.addLayout(options_grid)

        # Batch settings
        batch_layout = QGridLayout()
        self.param_widgets['batch_size'] = QSpinBox()
        self.param_widgets['batch_size'].setRange(1, 10)
        self.param_widgets['batch_threshold'] = QDoubleSpinBox()
        self.param_widgets['batch_threshold'].setRange(0, 1)
        self.param_widgets['batch_threshold'].setSingleStep(0.05)

        batch_layout.addWidget(QLabel(self.tr("批次大小:")), 0, 0)
        batch_layout.addWidget(self.param_widgets['batch_size'], 0, 1)
        batch_layout.addWidget(QLabel(self.tr("分批阈值:")), 0, 2)
        batch_layout.addWidget(self.param_widgets['batch_threshold'], 0, 3)

        gen_layout.addLayout(batch_layout)

        # System monitoring
        monitor_layout = QVBoxLayout()

        # CPU usage
        cpu_layout = QGridLayout()
        cpu_layout.addWidget(QLabel(self.tr("CPU 占用:")), 0, 0)
        self.cpu_progress = QProgressBar()
        self.cpu_progress.setRange(0, 100)
        cpu_layout.addWidget(self.cpu_progress, 0, 1)
        monitor_layout.addLayout(cpu_layout)

        # Memory usage
        memory_layout = QGridLayout()
        memory_layout.addWidget(QLabel(self.tr("内存占用:")), 1, 0)
        self.memory_progress = QProgressBar()
        self.memory_progress.setRange(0, 100)
        self.memory_progress.setTextVisible(False)
        memory_layout.addWidget(self.memory_progress, 1, 1)
        self.memory_label = QLabel()
        self.memory_label.setAlignment(Qt.AlignCenter)
        memory_layout.addWidget(self.memory_label, 1, 1)
        monitor_layout.addLayout(memory_layout)

        gen_layout.addLayout(monitor_layout)
        gen_group.setLayout(gen_layout)

        # Add right side to main layout
        main_param_layout.addWidget(gen_group, stretch=1)

        # Add the main layout to the parent layout
        layout.addLayout(main_param_layout)

    def open_save_directory(self):
        if os.path.exists(self.config_manager.get_value('save_directory')):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.config_manager.get_value('save_directory')))
        else:
            QMessageBox.warning(self, self.tr("错误"), self.tr("指定的保存文件夹不存在"))

    def browse_file(self, param_name):
        file_name, _ = QFileDialog.getOpenFileName(self, self.tr("选择音频文件"), "", self.tr("Audio Files (*.wav)"))
        if file_name:
            self.param_widgets[param_name].setText(file_name)

    def browse_files(self, param_name):
        files, _ = QFileDialog.getOpenFileNames(self, self.tr("选择音频文件"), "", self.tr("Audio Files (*.wav)"))
        if files:
            self.param_widgets[param_name].setText(';'.join(files))

    def autostart_api_check(self):
        autostart = self.config_manager.get_value('autostart_api', False)
        if not autostart:
            return

        self.check_api_button.setEnabled(False)
        self.api_status_label.setText(self.tr("状态: 检查中..."))
        self.api_check_thread = APICheckThread(
            self.api_url_input.text(),
            silent=True,
            continuous=True
        )
        self.api_check_thread.status_signal.connect(self.handle_autostart_check)
        self.api_check_thread.start()

    def handle_autostart_check(self, is_available, message):
        if is_available:
            self.api_status_label.setText(self.tr("状态: API 就绪"))
            self.check_api_button.setEnabled(True)
            self.synthesize_button.setEnabled(True)
            if self.api_check_thread:
                self.api_check_thread.stop()
                self.api_check_thread = None

    def check_api_status(self):
        self.api_status_label.setText(self.tr("状态: 检查中..."))
        self.check_api_button.setEnabled(False)
        self.synthesize_button.setEnabled(False)

        self.api_check_thread_once = APICheckThread(
            self.api_url_input.text(),
            silent=False,
            continuous=False
        )
        self.api_check_thread_once.status_signal.connect(self.update_api_status)
        self.api_check_thread_once.start()

    def update_api_status(self, is_available, message):
        if is_available:
            self.api_status_label.setText(self.tr("状态: API 就绪"))
            self.check_api_button.setEnabled(True)
            self.synthesize_button.setEnabled(True)
        else:
            self.api_status_label.setText(self.tr("状态: API 未就绪"))
            self.check_api_button.setEnabled(True)
            self.synthesize_button.setEnabled(True)
            QMessageBox.warning(self, self.tr("API 错误"), message)

    def start_system_monitoring(self):
        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self.update_system_monitoring)
        self.monitor_timer.start(1000)  # 1000ms

    def update_system_monitoring(self):
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        memory_percent = memory_info.percent

        self.cpu_progress.setValue(int(cpu_percent))

        memory_used_gb = memory_info.used / (1024 ** 3)
        memory_total_gb = memory_info.total / (1024 ** 3)
        self.memory_label.setText(
            self.tr("{:.2f} GB / {:.2f} GB").format(memory_used_gb, memory_total_gb))
        self.memory_progress.setValue(int(memory_percent))

    @staticmethod
    def get_model_files(directory, extension):
        if not os.path.exists(directory):
            return []
        return [f for f in os.listdir(directory) if f.endswith(extension)]

    def update_model_lists(self):
        # 保存当前选择
        current_gpt = self.gpt_combo.currentText()
        current_sovits = self.sovits_combo.currentText()

        # 获取当前版本的目录
        gpt_dir = self.MODEL_VERSIONS[self.current_model_version]['gpt']
        sovits_dir = self.MODEL_VERSIONS[self.current_model_version]['sovits']

        # 更新模型列表
        self.gpt_combo.clear()
        gpt_models = self.get_model_files(gpt_dir, '.ckpt')
        self.gpt_combo.addItems(gpt_models)

        self.sovits_combo.clear()
        sovits_models = self.get_model_files(sovits_dir, '.pth')
        self.sovits_combo.addItems(sovits_models)

        # 恢复之前的选择（如果仍然存在）
        if current_gpt in gpt_models:
            self.gpt_combo.setCurrentText(current_gpt)
        if current_sovits in sovits_models:
            self.sovits_combo.setCurrentText(current_sovits)
        
        # 更新提示文本信息
        self.text_input.setPlaceholderText(
            self.tr("在这里输入需要合成的文本..."
                    "\n\n使用方法：\n"
                    "1.将本exe放入GPT-SoVITS官方整合包下，双击启动。\n"
                    "2.将读取并使用{}_v{}与{}_v{}下的模型，请先完成训练获得模型。\n"
                    "3.保存预设将保存当前所有合成参数设定，可视为一个说话人，后续可快速切换，亦可用于批量合成页面\n"
                    "4.默认使用整合包自带环境来调起并使用API，也可以在API管理页面自定义。\n"
                    "\n此外，如果你使用官方v2-240821整合包，你可能会遇见粤语等语种合成空白音频的问题，这是GPT-SoVITS的一个已知并已解决的问题，更新整合包代码到仓库最新即可。"
                    "\n\nGitHub开源地址: https://github.com/AliceNavigator/GPT-SoVITS-Api-GUI           by  领航员未鸟\n").format(
            'GPT_weights', self.current_model_version[1:], 
            'SoVITS_weights', self.current_model_version[1:])
        )

    def switch_models_and_synthesize(self):
        # 根据当前选择的版本获取模型路径
        gpt_dir = self.MODEL_VERSIONS[self.current_model_version]['gpt']
        sovits_dir = self.MODEL_VERSIONS[self.current_model_version]['sovits']
        
        gpt_model = os.path.join(gpt_dir, self.gpt_combo.currentText())
        sovits_model = os.path.join(sovits_dir, self.sovits_combo.currentText())

        self.gpt_switching = False
        self.sovits_switching = False

        self.synthesize_button.setEnabled(False)

        if gpt_model != self.current_gpt_model:
            self.gpt_switching = True
            self.gpt_switch_thread = ModelSwitchThread(self.api_url_input.text(), 'gpt', gpt_model)
            self.gpt_switch_thread.switch_signal.connect(self.handle_switch_result)
            self.gpt_switch_thread.start()

        if sovits_model != self.current_sovits_model:
            self.sovits_switching = True
            self.sovits_switch_thread = ModelSwitchThread(self.api_url_input.text(), 'sovits', sovits_model)
            self.sovits_switch_thread.switch_signal.connect(self.handle_switch_result)
            self.sovits_switch_thread.start()

        # If no models need to be switched
        if not self.gpt_switching and not self.sovits_switching:
            self.statusBar.showMessage(self.tr("未检测到模型变更"), 5000)
            self.execute_synthesis()

    def handle_switch_result(self, success, message):
        if not success:
            self.synthesis_pending = False
            self.synthesize_button.setEnabled(True)
            QMessageBox.warning(self, self.tr("模型切换时出错"), message)

            if "GPT" in message:
                self.gpt_switching = False
            elif "SOVITS" in message:
                self.sovits_switching = False
        else:
            self.statusBar.showMessage(message, 5000)
            if "GPT" in message:
                gpt_dir = self.MODEL_VERSIONS[self.current_model_version]['gpt']
                self.current_gpt_model = os.path.join(gpt_dir, self.gpt_combo.currentText())
                self.gpt_switching = False
            elif "SOVITS" in message:
                sovits_dir = self.MODEL_VERSIONS[self.current_model_version]['sovits']
                self.current_sovits_model = os.path.join(sovits_dir, self.sovits_combo.currentText())
                self.sovits_switching = False

        # If all requested switches are complete and there is a pending synthesis task, execute synthesis
        if not self.gpt_switching and not self.sovits_switching and self.synthesis_pending:
            self.execute_synthesis()

    def prepare_synthesis(self):
        if not self.text_input.toPlainText():
            QMessageBox.warning(self, self.tr("错误"), self.tr("请先输入需要合成的文本!"))
            return

        self.synthesis_pending = True
        self.synthesize_button.setEnabled(False)
        self.switch_models_and_synthesize()

    def execute_synthesis(self):
        self.synthesis_pending = False
        self.statusBar.showMessage(self.tr("合成中..."))

        params = self.get_current_parameters()
        api_url = self.api_url_input.text()

        self.tts_thread = TTSThread(api_url, params, self.cache_dir)
        self.tts_thread.finished.connect(self.synthesis_finished)
        self.tts_thread.error.connect(self.synthesis_error)
        self.tts_thread.start()

    def synthesis_finished(self, audio_file):
        self.current_audio_file = audio_file
        self.synthesize_button.setEnabled(True)
        self.statusBar.showMessage(self.tr("合成成功!"))
        self.play_button.setEnabled(True)
        self.save_button.setEnabled(True)

    def synthesis_error(self, error_message):
        self.synthesize_button.setEnabled(True)
        self.statusBar.showMessage(self.tr("合成失败!"))
        QMessageBox.critical(self, self.tr("合成语音失败"), error_message)

    def play_audio(self):
        if not self.current_audio_file:
            return

        if self.is_playing:
            # Stop playback
            if hasattr(self, 'stream') and self.stream:
                self.stream.stop()
                self.stream.close()
                delattr(self, 'stream')
            self.is_playing = False
            self.play_button.setText(self.tr("播放"))
            return

        try:
            data, samplerate = sf.read(self.current_audio_file)

            # Ensure data is 2D (samples, channels)
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)

            # Define callback function
            def callback(outdata, frames, time, status):
                if status:
                    print('Status:', status)
                # When playback is complete
                if len(data) <= self.play_position + frames:
                    self.is_playing = False
                    self.play_button.setText(self.tr("播放"))
                    self.play_position = 0
                    raise sd.CallbackStop()
                # Continue playback
                current_chunk = data[self.play_position:self.play_position + frames]
                # Ensure dimensions match
                if len(current_chunk.shape) == 1:
                    current_chunk = current_chunk.reshape(-1, 1)
                outdata[:] = current_chunk
                self.play_position += frames

            # Reset playback position
            self.play_position = 0

            # Create and save output stream
            self.stream = sd.OutputStream(
                samplerate=samplerate,
                channels=data.shape[1] if len(data.shape) > 1 else 1,
                callback=callback
            )

            # Start playback
            self.stream.start()
            self.is_playing = True
            self.play_button.setText(self.tr("停止"))

        except Exception as e:
            QMessageBox.critical(self, self.tr("错误"), self.tr("尝试播放时出错: {}").format(str(e)))
            self.is_playing = False
            self.play_button.setText(self.tr("播放"))
            if hasattr(self, 'stream') and self.stream:
                self.stream.stop()
                self.stream.close()
                delattr(self, 'stream')

    def __del__(self):
        # Clean up resources
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop()
            self.stream.close()

    def save_audio(self):
        if not self.current_audio_file:
            return

        save_directory = self.save_path_input.text()

        if not save_directory:
            QMessageBox.warning(
                self,
                self.tr("警告"),
                self.tr("请先设置音频保存的路径。")
            )
            return

        if not os.path.exists(save_directory):
            reply = QMessageBox.question(self, self.tr("目录不存在"),
                                         self.tr(" {} 不存在， 是否尝试新建该文件夹？").format(save_directory),
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                os.makedirs(save_directory)
            else:
                return

        try:
            filename = self.generate_filename()
            file_path = os.path.join(save_directory, filename)
            shutil.copy2(self.current_audio_file, file_path)

            self.statusBar.showMessage(self.tr("音频保存至 {}").format(file_path))
        except Exception as e:
            QMessageBox.critical(self, self.tr("错误"), self.tr("保存音频时出错: {}").format(str(e)))

    def set_save_path(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            self.tr("选择保存目录"),
            self.config_manager.get_value('save_directory', '')
        )
        if directory:
            self.save_path_input.setText(directory)

    def generate_filename(self):
        # Get first 10 characters of text, remove special characters
        text = self.text_input.toPlainText().strip()
        clean_text = re.sub(r'[^\w\s]', '', text)
        clean_text = clean_text.replace(' ', '_')
        prefix = clean_text[:10]

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        return f"{prefix}_{timestamp}.wav"

    def load_preset(self):
        preset_name = self.preset_combo.currentText()
        presets = self.config_manager.get_value('presets', {})
        preset_data = presets.get(preset_name, {})

        # 检查是否有保存版本信息，并设置对应的模型版本
        if 'model_version' in preset_data:
            self.current_model_version = preset_data['model_version']
            if self.current_model_version == 'v2':
                self.v2_radio.setChecked(True)
            else:
                self.v3_radio.setChecked(True)
            # 更新模型列表以反映版本变更
            self.update_model_lists()

        if 'gpt_model' in preset_data and preset_data['gpt_model']:
            model_file = os.path.basename(preset_data['gpt_model'])
            index = self.gpt_combo.findText(model_file)
            if index >= 0:
                self.gpt_combo.setCurrentIndex(index)

        if 'sovits_model' in preset_data and preset_data['sovits_model']:
            model_file = os.path.basename(preset_data['sovits_model'])
            index = self.sovits_combo.findText(model_file)
            if index >= 0:
                self.sovits_combo.setCurrentIndex(index)

        for param, value in preset_data.items():
            if param in self.param_widgets:
                widget = self.param_widgets[param]
                if isinstance(widget, QLineEdit):
                    if isinstance(value, list):
                        widget.setText(';'.join(value))
                    else:
                        widget.setText(str(value))
                elif isinstance(widget, QComboBox):
                    index = widget.findData(value)
                    if index >= 0:
                        widget.setCurrentIndex(index)
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    widget.setValue(value)
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(value)

    def save_preset(self):
        preset_name, ok = QInputDialog.getText(self, self.tr("保存预设"), self.tr("输入预设名:"))
        if ok and preset_name:
            # 使用当前版本的目录
            gpt_dir = self.MODEL_VERSIONS[self.current_model_version]['gpt']
            sovits_dir = self.MODEL_VERSIONS[self.current_model_version]['sovits']
            
            preset_data = {
                'model_version': self.current_model_version,
                'gpt_model': os.path.join(gpt_dir, self.gpt_combo.currentText()),
                'sovits_model': os.path.join(sovits_dir, self.sovits_combo.currentText())
            }

            for param, widget in self.param_widgets.items():
                if isinstance(widget, QLineEdit):
                    if param == 'aux_ref_audio_paths':
                        value = widget.text().split(';') if widget.text() else []
                    else:
                        value = widget.text()
                elif isinstance(widget, QComboBox):
                    value = widget.currentData()
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    value = widget.value()
                elif isinstance(widget, QCheckBox):
                    value = widget.isChecked()
                preset_data[param] = value

            update_data = {
                'presets': {
                    preset_name: preset_data
                }
            }
            self.config_manager.update_config(update_data)

            current_items = [self.preset_combo.itemText(i) for i in range(self.preset_combo.count())]
            if preset_name not in current_items:
                self.preset_combo.addItem(preset_name)
                self.preset_combo.setCurrentText(preset_name)

            QMessageBox.information(self, self.tr("成功"),
                                    self.tr("已成功将当前参数保存为 '{}' !").format(preset_name))

    def delete_preset(self):
        current_preset = self.preset_combo.currentText()
        if current_preset == 'Default':
            QMessageBox.warning(self, self.tr('警告'),
                                self.tr('默认预设不能被删除!'))
            return

        reply = QMessageBox.question(self, self.tr('删除预设'),
                                     self.tr('确认删除当前预设 "{}"?').format(current_preset),
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            if self.config_manager.delete_value(f'presets.{current_preset}'):
                # 更新UI
                self.preset_combo.removeItem(self.preset_combo.currentIndex())
                QMessageBox.information(self, self.tr("成功"),
                                        self.tr("预设 '{}' 已被成功删除!").format(current_preset))
            else:
                QMessageBox.warning(self, self.tr("错误"),
                                    self.tr("删除预设 '{}' 失败!").format(current_preset))

    def get_current_parameters(self):
        params = {}
        params['text'] = self.text_input.toPlainText()

        for param, widget in self.param_widgets.items():
            if isinstance(widget, QLineEdit):
                if param == 'aux_ref_audio_paths':
                    value = widget.text().split(';') if widget.text() else []
                elif param == 'prompt_text' and self.param_widgets['no_prompt'].isChecked():
                    value = ""
                else:
                    value = widget.text()
            elif isinstance(widget, QComboBox):
                value = widget.currentData()  # Get current item's data (key)
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                value = widget.value()
            elif isinstance(widget, QCheckBox):
                if param == 'no_prompt':
                    continue
                else:
                    value = widget.isChecked()
            params[param] = value

        return params

    def setup_cache_directory(self):
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        else:
            for file in os.listdir(self.cache_dir):
                try:
                    file_path = os.path.join(self.cache_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(self.tr("清理缓存时出错 {}: {}").format(file, str(e)))

    def setup_background(self):
        background_path = ":/GAG_tools/images/GAG_background.png"
        style = f"""
            QMainWindow {{
                background-image: url({background_path});
            }}
            QWidget#centralWidget {{
                background-color: rgba(255, 255, 255, 180);
            }}
            QGroupBox {{
                background-color: rgba(255, 255, 255, 200);
            }}
            QGroupBox::title {{
                background-color: rgba(255, 255, 255, 200);
            }}
        """
        self.setStyleSheet(style)

    def setup_group_box_styles(self):
        style = """
            QGroupBox {
                background-color: rgba(255, 255, 255, 150);
            }
            QGroupBox::title {
                background-color: rgba(255, 255, 255, 200);
            }
            QTextEdit{
                background-color: rgba(255, 255, 255, 150);
            }

        """
        for widget in self.findChildren(QGroupBox):
            widget.setStyleSheet(style)

    def cleanup(self):
        # Save current configuration
        updates = {
            'api_url': self.api_url_input.text(),
            'current_preset': self.preset_combo.currentText(),
            'save_directory': self.save_path_input.text(),
            'model_version': self.current_model_version
        }
        self.config_manager.update_config(updates)

        # Clean up temporary files
        if self.current_audio_file and os.path.exists(self.current_audio_file):
            try:
                os.remove(self.current_audio_file)
            except:
                pass

        # Stop API check thread if running
        if self.api_check_thread:
            self.api_check_thread.stop()

    def closeEvent(self, event):
        self.cleanup()
        super().closeEvent(event)

    def change_model_version(self, button):
        """
        当用户切换模型版本时更新相关状态和模型列表
        """
        if button == self.v2_radio:
            self.current_model_version = 'v2'
        else:
            self.current_model_version = 'v3'
        
        # 保存当前版本设置到配置
        self.config_manager.update_config({'model_version': self.current_model_version})
        
        # 更新模型列表
        self.update_model_lists()
        
        # 重置当前加载的模型，以便在合成时加载正确版本目录的模型
        self.current_gpt_model = None
        self.current_sovits_model = None
        
        self.statusBar.showMessage(
            self.tr("已切换到{}模型目录").format(self.current_model_version), 
            3000
        )


def main():
    app = QApplication(sys.argv)
    gui = TTSGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
